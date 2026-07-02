import functools
from typing import Any, Callable, Sequence, Tuple

import flax.linen as nn
import jax.numpy as jnp

from utils.networks import MLP


class ResnetStack(nn.Module):
    """ResNet stack module."""

    num_features: int
    num_blocks: int
    max_pooling: bool = True

    @nn.compact
    def __call__(self, x):
        initializer = nn.initializers.xavier_uniform()
        conv_out = nn.Conv(
            features=self.num_features,
            kernel_size=(3, 3),
            strides=1,
            kernel_init=initializer,
            padding='SAME',
        )(x)

        if self.max_pooling:
            conv_out = nn.max_pool(
                conv_out,
                window_shape=(3, 3),
                padding='SAME',
                strides=(2, 2),
            )

        for _ in range(self.num_blocks):
            block_input = conv_out
            conv_out = nn.relu(conv_out)
            conv_out = nn.Conv(
                features=self.num_features,
                kernel_size=(3, 3),
                strides=1,
                padding='SAME',
                kernel_init=initializer,
            )(conv_out)

            conv_out = nn.relu(conv_out)
            conv_out = nn.Conv(
                features=self.num_features,
                kernel_size=(3, 3),
                strides=1,
                padding='SAME',
                kernel_init=initializer,
            )(conv_out)
            conv_out += block_input

        return conv_out


class ImpalaEncoder(nn.Module):
    """IMPALA encoder."""

    width: int = 1
    stack_sizes: tuple = (16, 32, 32)
    num_blocks: int = 2
    dropout_rate: float = None
    mlp_hidden_dims: Sequence[int] = (512,)
    layer_norm: bool = False

    def setup(self):
        stack_sizes = self.stack_sizes
        self.stack_blocks = [
            ResnetStack(
                num_features=stack_sizes[i] * self.width,
                num_blocks=self.num_blocks,
            )
            for i in range(len(stack_sizes))
        ]
        if self.dropout_rate is not None:
            self.dropout = nn.Dropout(rate=self.dropout_rate)

    @nn.compact
    def __call__(self, x, train=True):
        x = x.astype(jnp.float32) / 255.0

        conv_out = x

        for idx in range(len(self.stack_blocks)):
            conv_out = self.stack_blocks[idx](conv_out)
            if self.dropout_rate is not None:
                conv_out = self.dropout(conv_out, deterministic=not train)

        conv_out = nn.relu(conv_out)
        if self.layer_norm:
            conv_out = nn.LayerNorm()(conv_out)
        out = conv_out.reshape((*x.shape[:-3], -1))

        out = MLP(self.mlp_hidden_dims, activate_final=True, layer_norm=self.layer_norm)(out)

        return out


class ResNetBlock(nn.Module):
    """ResNet block."""

    filters: int
    conv: Any
    norm: Any
    act: Callable
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x):
        residual = x
        y = self.conv(self.filters, (3, 3), self.strides)(x)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters, (3, 3))(y)
        y = self.norm()(y)

        if residual.shape != y.shape:
            residual = self.conv(self.filters, (1, 1), self.strides, name="conv_proj")(
                residual
            )
            residual = self.norm(name="norm_proj")(residual)

        return self.act(residual + y)


class AddSpatialCoordinates(nn.Module):
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        grid = jnp.array(
            jnp.stack(
                jnp.meshgrid(*[jnp.arange(s) / (s - 1) * 2 - 1 for s in x.shape[-3:-1]]),
                axis=-1,
            ),
            dtype=self.dtype,
        ).transpose(1, 0, 2)

        if x.ndim == 4:
            grid = jnp.broadcast_to(grid, [x.shape[0], *grid.shape])

        return jnp.concatenate([x, grid], axis=-1)


class GroupNorm(nn.GroupNorm):
    def __call__(self, x, **kwargs):
        if x.ndim == 3:
            x = x[jnp.newaxis]
            x = super().__call__(x)
            return x[0]
        else:
            return super().__call__(x)


class ResNetEncoder(nn.Module):
    """ResNetV1."""

    stage_sizes: Sequence[int]
    block_cls: Any
    num_filters: int = 64
    dtype: Any = jnp.float32
    act: str = "swish"
    conv: Any = nn.Conv
    norm: str = "group"
    add_spatial_coordinates: bool = True
    num_spatial_blocks: int = 8

    @nn.compact
    def __call__(self, observations: jnp.ndarray, train=True):
        # put inputs in [-1, 1]
        x = observations.astype(jnp.float32) / 127.5 - 1.0

        if self.add_spatial_coordinates:
            x = AddSpatialCoordinates(dtype=self.dtype)(x)

        conv = functools.partial(
            self.conv,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=nn.initializers.kaiming_normal(),
        )
        if self.norm == "group":
            norm = functools.partial(GroupNorm, num_groups=4, epsilon=1e-5, dtype=self.dtype)
        elif self.norm == "layer":
            norm = functools.partial(nn.LayerNorm, epsilon=1e-5, dtype=self.dtype)
        else:
            raise ValueError("norm not found")

        act = getattr(nn, self.act)

        x = conv(
            self.num_filters, (7, 7), (2, 2),
            padding=[(3, 3), (3, 3)], name="conv_init"
        )(x)

        x = norm(name="norm_init")(x)
        x = act(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")
        for i, block_size in enumerate(self.stage_sizes):
            for j in range(block_size):
                stride = (2, 2) if i > 0 and j == 0 else (1, 1)
                x = self.block_cls(
                    self.num_filters * 2 ** i,
                    strides=stride,
                    conv=conv,
                    norm=norm,
                    act=act,
                )(x)

        x = jnp.mean(x, axis=(-3, -2))

        return x


class GCEncoder(nn.Module):
    """Helper module to handle inputs to goal-conditioned networks.

    It takes in observations (s) and goals (g) and returns the concatenation of `state_encoder(s)`, `goal_encoder(g)`,
    and `concat_encoder([s, g])`. It ignores the encoders that are not provided. This way, the module can handle both
    early and late fusion (or their variants) of state and goal information.
    """

    state_encoder: nn.Module = None
    goal_encoder: nn.Module = None
    concat_encoder: nn.Module = None

    @nn.compact
    def __call__(self, observations, goals=None, goal_encoded=False):
        """Returns the representations of observations and goals.

        If `goal_encoded` is True, `goals` is assumed to be already encoded representations. In this case, either
        `goal_encoder` or `concat_encoder` must be None.
        """
        reps = []
        if self.state_encoder is not None:
            reps.append(self.state_encoder(observations))
        if goals is not None:
            if goal_encoded:
                # Can't have both goal_encoder and concat_encoder in this case.
                assert self.goal_encoder is None or self.concat_encoder is None
                reps.append(goals)
            else:
                if self.goal_encoder is not None:
                    reps.append(self.goal_encoder(goals))
                if self.concat_encoder is not None:
                    reps.append(self.concat_encoder(jnp.concatenate([observations, goals], axis=-1)))
        reps = jnp.concatenate(reps, axis=-1)
        return reps


class MLPConcatFusion(nn.Module):
    """Learned 'option B' late fusion over PRECOMPUTED features.

    The observation is a flat vector [image_feat (image_feat_dim) | proprio_state
    (the rest)] — exactly what data_gen_scripts/extract_mm_features.py stores.
    This splits it, pushes the low-dim state through a small MLP (so it is not
    drowned out by the 512-d image feature), and concatenates the result with the
    raw image feature. The frozen resnet stays OUT of the training loop, so this
    is as cheap as the encoder=None precompute path. Output = image_feat_dim +
    state_hidden_dims[-1]. Used with config.encoder='precompute_state_mlp'.
    """

    image_feat_dim: int = 512
    state_hidden_dims: Sequence[int] = (128, 128)
    layer_norm: bool = True

    @nn.compact
    def __call__(self, observations, train=True):
        feat = observations[..., : self.image_feat_dim]
        state = observations[..., self.image_feat_dim:]
        state_emb = MLP(self.state_hidden_dims, activate_final=True,
                        layer_norm=self.layer_norm)(state)
        return jnp.concatenate([feat, state_emb], axis=-1)


class MultiModalEncoder(nn.Module):
    """Late-fusion encoder: image features concatenated with raw proprio state.

    Expects dict observations with keys ``image`` (B, H, W, 3 uint8) and
    ``state`` (B, D_state). The image encoder normalizes pixels internally
    (ResNetEncoder does x/127.5 - 1), so the image is passed through raw; the
    low-dim state is concatenated unchanged (option A — no separate state MLP).
    Output dim = image_encoder_out + D_state.
    """

    image_encoder_cls: Any
    image_key: str = 'image'
    state_key: str = 'state'

    @nn.compact
    def __call__(self, observations, train=True):
        image = observations[self.image_key]
        state = observations[self.state_key].astype(jnp.float32)
        image_feat = self.image_encoder_cls(name='image_encoder')(image)
        return jnp.concatenate([image_feat, state], axis=-1)


encoder_modules = {
    'mlp': functools.partial(
        MLP,
        hidden_dims=(512, 512, 512, 512),
        layer_norm=True,
    ),
    'impala': ImpalaEncoder,
    'impala_debug': functools.partial(
        ImpalaEncoder,
        num_blocks=1,
        stack_sizes=(4, 4)
    ),
    'impala_small': functools.partial(
        ImpalaEncoder,
        num_blocks=1
    ),
    'impala_large': functools.partial(
        ImpalaEncoder,
        stack_sizes=(64, 128, 128),
        mlp_hidden_dims=(1024,)
    ),
    'resnet_34': functools.partial(
        ResNetEncoder,
        stage_sizes=(3, 4, 6, 3),
        block_cls=ResNetBlock,
        num_spatial_blocks=8
    ),
}

# Multimodal (image + proprio state) late-fusion encoder built on resnet_34.
# resnet_34 emits a 512-d feature (num_filters * 2**(len(stage_sizes)-1)); the
# fused output is 512 + D_state.
encoder_modules['multimodal_resnet34'] = functools.partial(
    MultiModalEncoder,
    image_encoder_cls=encoder_modules['resnet_34'],
)

# Learned 'option B' fusion head over precomputed [resnet34 feat (512) | state]
# vectors — state through an MLP before concat. No resnet in the training loop.
encoder_modules['precompute_state_mlp'] = functools.partial(
    MLPConcatFusion,
    image_feat_dim=512,
    state_hidden_dims=(128, 128),
)


class TokenAttnPoolEncoder(nn.Module):
    """Learnable attention pool over PRECOMPUTED DINOv3 patch-token grids.

    Observation layout (flat, from utils.token_dataset.TokenDataset):
        [ tokens (n_cam*n_tok*feat_dim) | state (state_dim) | lang (lang_dim) ]

    The frozen DINOv3 backbone already ran at precompute time; here we only learn
    a small pool. Per camera (weights SHARED across cameras): ``num_layers`` of
    pre-norm self-attention over the n_tok tokens, then a learned-query attention
    pool that collapses the token set to one feat_dim vector. The pooled per-cam
    vectors are concatenated with the raw state+lang tail and returned, so the
    downstream IntentionEncoder/Value/Actor MLP consumes
        n_cam*feat_dim + state_dim + lang_dim.
    """

    n_cam: int
    n_tok: int
    feat_dim: int
    state_dim: int
    lang_dim: int = 0
    num_heads: int = 4
    num_layers: int = 1
    mlp_ratio: int = 2
    layer_norm: bool = True

    @nn.compact
    def __call__(self, observations, train=True):
        token_dim = self.n_cam * self.n_tok * self.feat_dim
        lead = observations.shape[:-1]                    # supports (B,) and (K,B,)
        tokens = observations[..., :token_dim]
        rest = observations[..., token_dim:]              # state (+ lang)
        # Fold cameras into the batch so the attention weights are shared.
        x = tokens.reshape((-1, self.n_tok, self.feat_dim))  # (M, n_tok, feat)

        for _ in range(self.num_layers):
            h = nn.LayerNorm()(x) if self.layer_norm else x
            h = nn.MultiHeadDotProductAttention(num_heads=self.num_heads)(h, h)
            x = x + h
            h = nn.LayerNorm()(x) if self.layer_norm else x
            h = nn.Dense(self.feat_dim * self.mlp_ratio)(h)
            h = nn.gelu(h)
            h = nn.Dense(self.feat_dim)(h)
            x = x + h

        # Learned-query attention pool -> one vector per (camera in batch).
        query = self.param('pool_query', nn.initializers.normal(0.02),
                           (1, 1, self.feat_dim))
        q = jnp.broadcast_to(query, (x.shape[0], 1, self.feat_dim))
        kv = nn.LayerNorm()(x) if self.layer_norm else x
        pooled = nn.MultiHeadDotProductAttention(num_heads=self.num_heads)(q, kv)
        pooled = pooled.reshape((*lead, self.n_cam * self.feat_dim))
        return jnp.concatenate([pooled, rest.astype(pooled.dtype)], axis=-1)
