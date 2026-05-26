"""PETCore and LRModule: composable building blocks for PET models.

PETCore owns all short-range parameters (under the "sr" key in the param tree).
LRModule owns all long-range Ewald parameters (under "lr").

Both are thin in themselves; outer models (LongRangePET) unpack their
specific batch structures and delegate to these two modules.
"""

import jax
import jax.numpy as jnp

import e3x
import flax.linen as nn
from jaxpme.batched_mixed import Ewald
from marathon.utils import masked

from .modules import (
    MLP,
    ChemicalEmbedding,
    Linear,
    Transformer,
    shifted_cosine_cutoff,
)


class PETCore(nn.Module):
    """Short-range PET body. All SR parameters live under this module's scope."""

    num_hidden: int = 128
    num_hidden_feedforward: int = 256
    num_attention_layers: int = 2
    num_message_passing_layers: int = 2
    num_heads: int = 4
    cutoff: float = 5.0
    cutoff_width: float = 0.2

    @nn.compact
    def __call__(
        self,
        Z_i,
        reverse,
        positions,
        cell,
        centers,
        others,
        cell_shifts,
        pair_mask,
        atom_mask,
        pair_to_structure,
    ):
        R_ij = (
            positions[others]
            - positions[centers]
            + jnp.einsum("pA,pAa->pa", cell_shifts, cell[pair_to_structure])
        )

        num_hidden = self.num_hidden
        num_pairs = R_ij.shape[0]
        num_atoms = Z_i.shape[0]
        num_neighbors = R_ij.reshape(num_atoms, -1, 3).shape[1]

        R_ij = R_ij.reshape(num_pairs, 3)
        r_ij = e3x.ops.norm(R_ij, axis=-1)

        cutoffs = shifted_cosine_cutoff(r_ij, self.cutoff, self.cutoff_width)

        pair_mask_with_central_tokens = pair_mask.reshape(num_atoms, num_neighbors)
        pair_mask_with_central_tokens = pair_mask_with_central_tokens.at[:, -1].set(True)
        pair_mask_with_central_tokens *= atom_mask[..., None]
        pair_mask_with_central_tokens = pair_mask_with_central_tokens.reshape(num_pairs)

        cutoffs *= pair_mask_with_central_tokens
        cutoffs = cutoffs.reshape(num_atoms, num_neighbors)
        cutoffs = cutoffs.at[:, -1].set(1.0)
        cutoffs *= atom_mask[..., None]
        cutoffs = cutoffs.reshape(num_pairs)

        messages = ChemicalEmbedding(num_features=self.num_hidden)(Z_i)
        messages = messages[others] * pair_mask[..., None]

        predictions = 0.0
        for l in range(self.num_message_passing_layers):
            geometric = masked(
                Linear(features=num_hidden),
                jnp.concatenate([R_ij, r_ij[..., None]], axis=-1),
                pair_mask,
            )
            if l == 0:
                tokens = masked(
                    MLP(features=[num_hidden, num_hidden]),
                    jnp.concatenate([geometric, messages], axis=-1),
                    pair_mask,
                )
                tokens = tokens.reshape(num_atoms, num_neighbors, num_hidden)
            else:
                neighbors = ChemicalEmbedding(num_features=self.num_hidden)(Z_i)
                neighbors = neighbors[others] * pair_mask[..., None]
                tokens = masked(
                    MLP(features=[num_hidden, num_hidden]),
                    jnp.concatenate([geometric, messages, neighbors], axis=-1),
                    pair_mask,
                )
                tokens = tokens.reshape(num_atoms, num_neighbors, num_hidden)

            central = (
                ChemicalEmbedding(num_features=self.num_hidden)(Z_i) * atom_mask[..., None]
            )
            tokens = tokens.at[:, -1, :].set(central)

            output_tokens = Transformer(
                num_layers=self.num_attention_layers,
                num_heads=self.num_heads,
                num_hidden_feedforward=self.num_hidden_feedforward,
            )(tokens, cutoffs, pair_mask_with_central_tokens)

            central_tokens = output_tokens[:, -1, :]
            output_tokens = output_tokens.at[:, -1, :].set(0.0)
            output_tokens = output_tokens.reshape(num_pairs, -1)

            predictions += masked(
                MLP(features=[num_hidden, num_hidden, 1]), central_tokens, atom_mask
            )[:, 0]

            pair_predictions = (
                masked(MLP(features=[num_hidden, num_hidden, 1]), output_tokens, pair_mask)
                * cutoffs[..., None]
            ).reshape(num_atoms, num_neighbors, -1)
            predictions += pair_predictions.sum(axis=1)[:, 0]

            messages = 0.5 * (messages + output_tokens[reverse])

        return predictions, central_tokens


class LRModule(nn.Module):
    """Long-range Ewald module. All LR parameters live under this module's scope."""

    num_hidden: int = 128
    num_hidden_feedforward: int = 256
    num_charges: int = 8
    init_prefactor: float = 1.0

    @nn.compact
    def __call__(self, central_tokens, atom_mask, ewald_structure, nopbc, pbc):
        charges = masked(
            MLP(features=[self.num_hidden, self.num_hidden, self.num_charges]),
            central_tokens,
            atom_mask,
        )
        self.sow('intermediates', 'pseudo_charges', charges)

        prefactor = jnp.exp(
            self.param(
                "log_prefactor",
                nn.initializers.constant(jnp.log(self.init_prefactor)),
                (),
            )
        )
        calculator = Ewald(prefactor=prefactor)

        potentials = jax.vmap(
            lambda q: calculator.potentials(q, ewald_structure, nopbc, pbc),
            in_axes=-1,
            out_axes=-1,
        )(charges)

        h = central_tokens + masked(
            MLP(features=[self.num_hidden_feedforward, self.num_hidden]),
            potentials,
            atom_mask,
        )
        h = masked(nn.LayerNorm(), h, atom_mask)
        h = h + masked(
            MLP(features=[self.num_hidden_feedforward, self.num_hidden]),
            h,
            atom_mask,
        )
        h = masked(nn.LayerNorm(), h, atom_mask)

        return masked(
            MLP(features=[self.num_hidden, self.num_hidden, 1]), h, atom_mask
        )[:, 0]
