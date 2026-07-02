"""inFOM: DINOv3 patch-TOKEN encoder w/ LEARNABLE attention pool, state-only decoder.

Same state-only-decoder recipe as InFOMLangStateDecoderAgent (the flow-occupancy
DECODER flow-matches only the proprio-state slice; image + language are
encoder-only and reach the occupancy model through the latent z), but the
observation's image half is now a GRID of frozen DINOv3 patch tokens instead of a
single pre-pooled vector:

    obs = [ tokens (n_cam*n_tok*feat_dim) | proprio_state (state_dim) | language (rest) ]

A small LEARNABLE attention pool (utils.encoders.TokenAttnPoolEncoder) collapses
each camera's token set inside the intention/critic/actor networks. The heavy
DINOv3 backbone ran at precompute time (utils.token_dataset streams the tokens),
so training stays cheap. flow_occupancy_loss / behavioral_cloning_loss and the
whole pretraining path are inherited unchanged from the lang-state-decoder — only
create() differs (it attaches the token encoder and derives the obs layout).

agents/infom.py and the other agents are untouched.
"""
import copy

import flax
import jax
import jax.numpy as jnp
import optax

from agents.infom_lang_state_decoder import (
    InFOMLangStateDecoderAgent, get_config as lang_state_get_config)
from utils.encoders import TokenAttnPoolEncoder
from utils.flax_utils import ModuleDict, TrainState
from utils.networks import VectorField, Actor, IntentionEncoder, Value


class InFOMDinoAttnPoolAgent(InFOMLangStateDecoderAgent):
    """Token-grid encoder (learnable attention pool) + state-only decoder inFOM."""

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng, time_rng = jax.random.split(rng, 3)

        # --- derive the observation layout from config + the example obs -----
        n_cam = config['n_cam']
        n_tok = config['n_tok']
        feat_dim = config['token_feat_dim']
        sdim = config['state_dim']
        token_dim = n_cam * n_tok * feat_dim
        obs_dim = ex_observations.shape[-1]
        lang_dim = int(obs_dim - token_dim - sdim)
        assert lang_dim >= 0, (
            f"obs_dim {obs_dim} < token_dim {token_dim} + state_dim {sdim}; "
            f"check n_cam/n_tok/token_feat_dim vs the dataset.")
        # image_feat_dim is the state-slice split point used by _state_slice / the
        # inherited flow_occupancy_loss; for tokens it is the whole token block.
        config['image_feat_dim'] = token_dim
        config['lang_dim'] = lang_dim

        ex_orig_observations = ex_observations
        ex_state = ex_observations[..., token_dim:token_dim + sdim]
        ex_times = ex_actions[..., 0]
        ex_latents = jnp.ones((*ex_actions.shape[:-1], config['latent_dim']))
        action_dim = ex_actions.shape[-1]

        def make_encoder():
            return TokenAttnPoolEncoder(
                n_cam=n_cam, n_tok=n_tok, feat_dim=feat_dim,
                state_dim=sdim, lang_dim=lang_dim,
                num_heads=config['attn_num_heads'],
                num_layers=config['attn_num_layers'],
                layer_norm=config['attn_layer_norm'])

        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['value_layer_norm'],
            num_ensembles=2,
            encoder=make_encoder(),
        )
        intention_encoder_def = IntentionEncoder(
            hidden_dims=config['intention_encoder_hidden_dims'],
            latent_dim=config['latent_dim'],
            layer_norm=config['intention_encoder_layer_norm'],
            encoder=make_encoder(),
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
            encoder=make_encoder(),
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
    config = lang_state_get_config()
    config.agent_name = 'infom_dino_attnpool'
    # DINOv3 token grid layout (must match the token dataset stats.json).
    config.n_cam = 3
    config.n_tok = 65            # pool_grid**2 + 1 CLS (8x8 + CLS)
    config.token_feat_dim = 384  # DINOv3 ViT-S/16 hidden size
    config.state_dim = 16
    # image_feat_dim (state-slice split) and lang_dim are derived in create().
    config.image_feat_dim = 3 * 65 * 384
    config.lang_dim = 0
    # Learnable attention pool.
    config.attn_num_heads = 4
    config.attn_num_layers = 1
    config.attn_layer_norm = True
    return config
