"""Shared modules for PET models.

Contains Transformer, Attention, MLP, Linear, ChemicalEmbedding,
shifted_cosine_cutoff, and torch_bias_init.
"""

import jax
import jax.numpy as jnp

from collections.abc import Sequence

import flax.linen as nn
from marathon.utils import masked


class Transformer(nn.Module):
    num_layers: int = 2
    num_heads: int = 4
    num_hidden_feedforward: int = 256

    @nn.compact
    def __call__(self, x, cutoffs, pair_mask):
        num_atoms, num_neighbors, num_features = x.shape
        num_pairs = num_atoms * num_neighbors

        for _ in range(self.num_layers):
            x += Attention(num_heads=self.num_heads)(x, cutoffs, pair_mask)

            x = x.reshape(num_pairs, -1)
            x = masked(nn.LayerNorm(), x, pair_mask)
            x += masked(MLP(features=[self.num_hidden_feedforward, num_features]), x, pair_mask)
            x = masked(nn.LayerNorm(), x, pair_mask)
            x = x.reshape(num_atoms, num_neighbors, num_features)

        return x


class Attention(nn.Module):
    num_heads: int = 4

    @nn.compact
    def __call__(self, x, cutoffs, pair_mask):
        num_heads = self.num_heads
        num_atoms, num_neighbors, num_features = x.shape
        num_features_per_head = num_features // num_heads
        num_pairs = num_atoms * num_neighbors

        qkv = masked(
            Linear(features=3 * num_features, kernel_init="xavier", bias_init="zeros"),
            x.reshape(num_pairs, -1),
            pair_mask,
        )
        q, k, v = jnp.split(qkv, 3, axis=-1)

        q = q.reshape(num_atoms, num_neighbors, num_heads, num_features_per_head)
        k = k.reshape(num_atoms, num_neighbors, num_heads, num_features_per_head)
        v = v.reshape(num_atoms, num_neighbors, num_heads, num_features_per_head)

        # cutoff*exp(qk) = exp(qk + log(cutoff))
        cutoffs = jnp.log(jnp.clip(cutoffs, min=1e-15, max=None))

        attended = jax.nn.dot_product_attention(
            q,
            k,
            v,
            mask=pair_mask.reshape(num_atoms, 1, 1, -1),
            bias=cutoffs.reshape(num_atoms, 1, 1, -1),
        )

        attended = masked(
            Linear(features=num_features, bias_init="zeros"),
            attended.reshape(num_pairs, -1),
            pair_mask,
        ).reshape(num_atoms, num_neighbors, num_features)

        return attended


class MLP(nn.Module):
    features: Sequence[int]
    activation: str = "silu"
    use_bias: bool = True

    @nn.compact
    def __call__(self, x):
        activation = getattr(jax.nn, self.activation)
        num_layers = len(self.features)

        for i, f in enumerate(self.features):
            x = Linear(features=f, use_bias=self.use_bias)(x)
            if i != num_layers - 1:
                x = activation(x)

        return x


def torch_bias_init(fan_in):
    def do_init(key, shape, dtype):
        return jax.random.uniform(key, (shape[-1],), dtype, -1) * jnp.sqrt(1 / fan_in)

    return do_init


class Linear(nn.Module):
    # torch-style Linear: uniform init from [-1/sqrt(fan_in), +1/sqrt(fan_in)]
    features: int
    use_bias: bool = True
    kernel_init: str = "torch"
    bias_init: str = "torch"

    @nn.compact
    def __call__(self, x):
        fan_in = x.shape[-1]
        if self.kernel_init == "torch":
            kernel_init = jax.nn.initializers.variance_scaling(1.0 / 3.0, "fan_in", "uniform")
        elif self.kernel_init == "xavier":
            kernel_init = jax.nn.initializers.xavier_uniform()

        if self.bias_init == "torch":
            bias_init = torch_bias_init(fan_in)
        elif self.bias_init == "zeros":
            bias_init = nn.initializers.zeros_init()

        return nn.Dense(
            features=self.features,
            use_bias=self.use_bias,
            kernel_init=kernel_init,
            bias_init=bias_init,
        )(x)


def shifted_cosine_cutoff(x, cutoff, width=1.0):
    onset = cutoff - width
    left_of_onset = jnp.where(
        x < onset, 1.0, 0.5 * (1 + jnp.cos(jnp.pi * (x - onset) / width))
    )
    return jnp.where(x < cutoff, left_of_onset, 0.0)


class ChemicalEmbedding(nn.Module):
    num_features: int
    total_species: int = 128

    @nn.compact
    def __call__(self, species):
        return nn.Embed(num_embeddings=self.total_species, features=self.num_features)(species)
