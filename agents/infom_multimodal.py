"""Multimodal inFOM agent — additive variant of InFOMAgent (agents/infom.py kept pristine).

Supports fused image+state encoders. Crucially, WHERE the fusion is applied
differs by variant, to keep the flow-occupancy model stable:

  'precompute_state_mlp' (option B, precompute):
      Observations are flat [resnet feat (512) | state (16)] = 528-d. The
      learned state-MLP fusion is applied ONLY in the embedding/actor path
      (intention_encoder, actor, critic). The flow-occupancy model (critic_vf)
      stays in the RAW 528-d space with clip_flow_goals on — exactly the stable
      setup of the encoder=None precompute run. So z = q(z|fused(s), a) while the
      flow target is the proven-stable raw space. (Putting fusion in critic_vf's
      encoded space + disabling clip diverges — KL/flow blow up.)

  'multimodal_resnet34' (option, end-to-end 2nd phase):
      dict {image, state} obs; the resnet fusion IS the flow-space encoder
      (images cannot be flow-matched raw), as in the stock image inFOM.

Only `create` and `flow_occupancy_loss` differ from the base agent; all other
loss/update methods are inherited unchanged.
"""
import copy

import flax
import jax
import jax.numpy as jnp

import optax

from agents.infom import InFOMAgent, get_config as infom_get_config
from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState
from utils.networks import VectorField, Actor, IntentionEncoder, Value

# Encoders whose fusion defines the flow-occupancy (critic_vf) space. For every
# other fusion encoder the flow stays in the raw observation space.
FLOW_SPACE_ENCODERS = ('multimodal_resnet34',)


class InFOMMultiModalAgent(InFOMAgent):
    """InFOM with multimodal (image+state) fusion encoders."""

    def flow_occupancy_loss(self, batch, grad_params, rng):
        batch_size = batch['actions'].shape[0]  # obs may be a dict (end-to-end)
        actions = batch['actions']
        next_actions = batch['next_actions']

        flow_in_encoded_space = self.config['encoder'] in FLOW_SPACE_ENCODERS
        if flow_in_encoded_space:
            observations = self.network.select('critic_vf_encoder')(
                batch['observations'], params=grad_params)
            next_observations = self.network.select('target_critic_vf_encoder')(
                batch['next_observations'])
        else:
            # Flow stays in the raw observation space (stable + clip-compatible).
            observations = batch['observations']
            next_observations = batch['next_observations']

        # z is encoded by the intention_encoder's OWN (fusion) encoder internally.
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
        flow_future_observations = self.compute_fwd_flow_goals(
            future_noises, next_observations, next_actions, jax.lax.stop_gradient(latents),
            observation_min=batch.get('observation_min', None),
            observation_max=batch.get('observation_max', None),
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

    @jax.jit
    def pretrain(self, batch):
        """Same as base pretrain, but only target-update critic_vf_encoder when
        the flow actually runs in an encoded space (it doesn't exist otherwise)."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.pretraining_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        if self.config['encoder'] in FLOW_SPACE_ENCODERS:
            self.target_update(new_network, 'critic_vf_encoder')
        self.target_update(new_network, 'critic_vf')
        return self.replace(network=new_network, rng=new_rng), info

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng, time_rng = jax.random.split(rng, 3)

        ex_orig_observations = ex_observations
        ex_times = ex_actions[..., 0]
        ex_latents = jnp.ones((*ex_actions.shape[:-1], config['latent_dim']))
        is_dict_obs = isinstance(ex_observations, (dict, flax.core.FrozenDict))
        raw_obs_dim = None if is_dict_obs else ex_observations.shape[-1]
        action_dim = ex_actions.shape[-1]
        action_dtype = ex_actions.dtype
        ex_batch_size = ex_actions.shape[0]

        enc_name = config['encoder']
        flow_in_encoded_space = enc_name in FLOW_SPACE_ENCODERS

        encoders = dict()
        # critic_vf flow-space dim + example: encoded only for flow-space encoders.
        critic_vf_obs_dim = raw_obs_dim
        ex_flow_obs = ex_orig_observations
        if enc_name is not None:
            encoder_module = encoder_modules[enc_name]
            # Embedding/actor path always gets a fusion encoder.
            encoders['critic'] = encoder_module()
            encoders['intention'] = encoder_module()
            encoders['actor'] = encoder_module()

            if flow_in_encoded_space:
                # resnet fusion defines the flow space (images can't be raw-matched).
                if enc_name == 'multimodal_resnet34':
                    critic_vf_obs_dim = 512 + ex_observations['state'].shape[-1]
                elif 'mlp_hidden_dims' in encoder_module.keywords:
                    critic_vf_obs_dim = encoder_module.keywords['mlp_hidden_dims'][-1]
                else:
                    critic_vf_obs_dim = encoder_modules['impala'].mlp_hidden_dims[-1]
                encoders['critic_vf'] = encoder_module()
                rng, obs_rng = jax.random.split(rng, 2)
                ex_flow_obs = jax.random.normal(
                    obs_rng, shape=(ex_batch_size, critic_vf_obs_dim), dtype=action_dtype)
            # else (e.g. precompute_state_mlp): flow stays in raw obs space; no
            # critic_vf encoder, critic_vf_obs_dim stays the raw 528-d.

        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['value_layer_norm'],
            num_ensembles=2,
            encoder=encoders.get('critic'),
        )
        intention_encoder_def = IntentionEncoder(
            hidden_dims=config['intention_encoder_hidden_dims'],
            latent_dim=config['latent_dim'],
            layer_norm=config['intention_encoder_layer_norm'],
            encoder=encoders.get('intention'),
        )
        critic_vf_def = VectorField(
            vector_dim=critic_vf_obs_dim,
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['value_layer_norm'],
        )
        actor_def = Actor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            state_dependent_std=False,
            layer_norm=config['actor_layer_norm'],
            const_std=config['const_std'],
            encoder=encoders.get('actor'),
        )
        reward_def = Value(
            hidden_dims=config['reward_hidden_dims'],
            layer_norm=config['reward_layer_norm'],
        )

        network_info = dict(
            critic=(critic_def, (ex_orig_observations, ex_actions)),
            critic_vf=(critic_vf_def, (
                ex_flow_obs, ex_times,
                ex_flow_obs, ex_actions, ex_latents)),
            target_critic_vf=(copy.deepcopy(critic_vf_def), (
                ex_flow_obs, ex_times,
                ex_flow_obs, ex_actions, ex_latents)),
            intention_encoder=(intention_encoder_def, (
                ex_orig_observations, ex_actions)),
            actor=(actor_def, (ex_orig_observations, )),
            reward=(reward_def, (ex_flow_obs,)),
        )
        if flow_in_encoded_space:
            network_info['critic_vf_encoder'] = (
                encoders.get('critic_vf'), (ex_orig_observations,))
            network_info['target_critic_vf_encoder'] = (
                copy.deepcopy(encoders.get('critic_vf')), (ex_orig_observations,))

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        if flow_in_encoded_space:
            params['modules_target_critic_vf_encoder'] = params['modules_critic_vf_encoder']
        params['modules_target_critic_vf'] = params['modules_critic_vf']

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = infom_get_config()
    config.agent_name = 'infom_multimodal'
    return config
