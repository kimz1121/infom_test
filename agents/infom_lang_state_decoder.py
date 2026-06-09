"""inFOM: multimodal+language ENCODER, state-only DECODER.

Extends the Variant-C idea (state-only decoder) with a task LANGUAGE embedding
that enters ONLY the encoder. The observation vector is laid out as

    obs = [ image_feat (image_feat_dim) | proprio_state (state_dim) | language (rest) ]

so language is just appended to the observation — no extra batch plumbing, no viz
changes. The intention_encoder ingests the FULL obs (image feat + state + language
+ action) to form z, while the flow-occupancy DECODER flow-matches ONLY the middle
proprio-state slice obs[..., image_feat_dim : image_feat_dim+state_dim]. Image
features AND language are encoder-only (never generated). Image+language reach the
occupancy model through z.

agents/infom.py is untouched.
"""
import copy

import flax
import jax
import jax.numpy as jnp
import optax

from agents.infom import InFOMAgent, get_config as infom_get_config
from utils.flax_utils import ModuleDict, TrainState
from utils.networks import VectorField, Actor, IntentionEncoder, Value


class InFOMLangStateDecoderAgent(InFOMAgent):
    """Multimodal+language encoder, state-only decoder inFOM."""

    def _state_slice(self, x):
        f = self.config['image_feat_dim']
        s = self.config['state_dim']
        return x[..., f:f + s]

    def flow_occupancy_loss(self, batch, grad_params, rng):
        batch_size = batch['actions'].shape[0]
        actions = batch['actions']
        next_actions = batch['next_actions']

        # Decoder target = proprio state slice only (image feat + language excluded).
        observations = self._state_slice(batch['observations'])
        next_observations = self._state_slice(batch['next_observations'])

        # z is encoded from the FULL observation (image feat + state + language).
        rng, latent_rng = jax.random.split(rng)
        latent_dist = self.network.select('intention_encoder')(
            batch['next_observations'], next_actions, params=grad_params)
        latents = latent_dist.sample(seed=latent_rng)

        means = latent_dist.mean()
        log_stds = jnp.log(latent_dist.stddev())
        kl_loss = -0.5 * (1 + 2 * log_stds - means ** 2 - jnp.exp(2 * log_stds)).mean()

        rng, time_rng, current_noise_rng, future_noise_rng = jax.random.split(rng, 4)
        times = jax.random.uniform(time_rng, shape=(batch_size,), dtype=observations.dtype)
        current_noises = jax.random.normal(
            current_noise_rng, shape=observations.shape, dtype=observations.dtype)
        current_vf_pred = self.network.select('critic_vf')(
            times[..., None] * observations + (1 - times[..., None]) * current_noises,
            times,
            jax.lax.stop_gradient(observations), actions, latents,
            params=grad_params,
        )
        current_flow_matching_loss = jnp.square(
            jax.lax.stop_gradient(observations - current_noises) - current_vf_pred).mean(axis=-1)

        future_noises = jax.random.normal(
            future_noise_rng, shape=observations.shape, dtype=observations.dtype)
        obs_min = batch.get('observation_min', None)
        obs_max = batch.get('observation_max', None)
        if obs_min is not None:
            obs_min = self._state_slice(obs_min)
            obs_max = self._state_slice(obs_max)
        flow_future_observations = self.compute_fwd_flow_goals(
            future_noises, next_observations, next_actions, jax.lax.stop_gradient(latents),
            observation_min=obs_min, observation_max=obs_max,
            use_target_network=True,
        )
        future_vf_target = self.network.select('target_critic_vf')(
            times[..., None] * flow_future_observations + (1 - times[..., None]) * future_noises,
            times,
            next_observations, next_actions, jax.lax.stop_gradient(latents),
        )
        future_vf_pred = self.network.select('critic_vf')(
            times[..., None] * flow_future_observations + (1 - times[..., None]) * future_noises,
            times,
            jax.lax.stop_gradient(observations), actions, jax.lax.stop_gradient(latents),
            params=grad_params,
        )
        future_flow_matching_loss = jnp.square(future_vf_target - future_vf_pred).mean(axis=-1)

        flow_matching_loss = ((1 - self.config['discount']) * current_flow_matching_loss
                              + self.config['discount'] * future_flow_matching_loss).mean()
        neg_elbo_loss = flow_matching_loss + self.config['kl_weight'] * kl_loss

        return neg_elbo_loss, {
            'neg_elbo_loss': neg_elbo_loss,
            'flow_matching_loss': flow_matching_loss,
            'kl_loss': kl_loss,
            'flow_future_obs_max': flow_future_observations.max(),
            'flow_future_obs_min': flow_future_observations.min(),
            'current_flow_matching_loss': current_flow_matching_loss.mean(),
            'future_flow_matching_loss': future_flow_matching_loss.mean(),
        }

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng, time_rng = jax.random.split(rng, 3)

        fdim = config['image_feat_dim']
        sdim = config['state_dim']
        ex_orig_observations = ex_observations                 # full [feat | state | lang]
        ex_state = ex_observations[..., fdim:fdim + sdim]      # proprio decoder space
        ex_times = ex_actions[..., 0]
        ex_latents = jnp.ones((*ex_actions.shape[:-1], config['latent_dim']))
        action_dim = ex_actions.shape[-1]

        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['value_layer_norm'],
            num_ensembles=2,
        )
        intention_encoder_def = IntentionEncoder(
            hidden_dims=config['intention_encoder_hidden_dims'],
            latent_dim=config['latent_dim'],
            layer_norm=config['intention_encoder_layer_norm'],
        )
        critic_vf_def = VectorField(
            vector_dim=sdim,                                   # decoder generates state only
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['value_layer_norm'],
        )
        actor_def = Actor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            state_dependent_std=False,
            layer_norm=config['actor_layer_norm'],
            const_std=config['const_std'],
        )
        reward_def = Value(
            hidden_dims=config['reward_hidden_dims'],
            layer_norm=config['reward_layer_norm'],
        )

        network_info = dict(
            critic=(critic_def, (ex_orig_observations, ex_actions)),
            critic_vf=(critic_vf_def, (
                ex_state, ex_times, ex_state, ex_actions, ex_latents)),
            target_critic_vf=(copy.deepcopy(critic_vf_def), (
                ex_state, ex_times, ex_state, ex_actions, ex_latents)),
            intention_encoder=(intention_encoder_def, (
                ex_orig_observations, ex_actions)),
            actor=(actor_def, (ex_orig_observations, )),
            reward=(reward_def, (ex_state,)),
        )

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_critic_vf'] = params['modules_critic_vf']

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = infom_get_config()
    config.agent_name = 'infom_lang_state_decoder'
    config.image_feat_dim = 512   # obs[:512]   = resnet image features (encoder-only)
    config.state_dim = 16         # obs[512:528] = proprio state (DECODER target)
    # obs[528:] = language embedding (encoder-only). state_dim slice is the only decoded part.
    return config
