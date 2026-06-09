"""inFOM variant: multimodal ENCODER, state-only DECODER (occupancy in state space).

Motivation: with images, the inFOM flow-occupancy "decoder" must flow-match the
full encoded observation. For our precompute obs = [resnet feat (512) | proprio
state (16)] that means modelling a 512-d image-feature distribution whose values
swing wildly over time — a very hard generative target.

Idea (user): let the intention ENCODER see everything (image feat + state +
action) to form z, but let the flow-occupancy DECODER generate ONLY the smooth,
low-dim proprio state (16-d), not the image features. Image information still
reaches the occupancy model through z (the latent it conditions on).

Concretely vs base inFOM:
  - intention_encoder(full 528-d obs, action) -> z   (unchanged; uses image+state)
  - critic_vf flow-matches the STATE slice obs[..., image_feat_dim:] (16-d),
    conditioned on (state, action, z). vector_dim = state_dim.
  - clip_flow_goals uses the state-space bounds (the [image_feat_dim:] slice of
    observation_min/max), so the flow stays in the stable 16-d proprio box.

encoder is None (the intention MLP already ingests the concatenated feat+state),
so this stays as cheap as the precompute runs. agents/infom.py is untouched.
"""
import copy

import flax
import jax
import jax.numpy as jnp
import optax

from agents.infom import InFOMAgent, get_config as infom_get_config
from utils.flax_utils import ModuleDict, TrainState
from utils.networks import VectorField, Actor, IntentionEncoder, Value


class InFOMStateDecoderAgent(InFOMAgent):
    """Multimodal-encoder, state-decoder inFOM."""

    def flow_occupancy_loss(self, batch, grad_params, rng):
        fdim = self.config['image_feat_dim']
        batch_size = batch['actions'].shape[0]
        actions = batch['actions']
        next_actions = batch['next_actions']

        # Decoder target = proprio state slice only (smooth, low-dim).
        observations = batch['observations'][..., fdim:]
        next_observations = batch['next_observations'][..., fdim:]

        # z is encoded from the FULL multimodal observation (image feat + state).
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
            obs_min = obs_min[..., fdim:]
            obs_max = obs_max[..., fdim:]
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
        ex_orig_observations = ex_observations            # full [feat | state]
        ex_state = ex_observations[..., fdim:]            # proprio decoder space
        ex_times = ex_actions[..., 0]
        ex_latents = jnp.ones((*ex_actions.shape[:-1], config['latent_dim']))
        state_dim = ex_state.shape[-1]
        action_dim = ex_actions.shape[-1]

        # No encoder modules: the intention MLP ingests the concatenated obs directly.
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
            vector_dim=state_dim,                          # decoder generates state only
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
    config.agent_name = 'infom_state_decoder'
    # Split point of the flat observation: first image_feat_dim are image
    # features (decoder-excluded), the rest are proprio state (decoder target).
    config.image_feat_dim = 512
    return config
