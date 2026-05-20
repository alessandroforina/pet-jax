"""LongRangePET: PET with coupled short-range and Ewald long-range interactions.

A single jaxpme neighbor list serves both the SR edge transformer and the
Ewald LR module, sharing the same cutoff radius. Use lr=False for pure SR.
"""

import jax
import jax.numpy as jnp

import flax.linen as nn

from .core import LRModule, PETCore
from .transforms import ToBatch, ToSample


class LongRangePET(nn.Module):
    lr: bool = False
    cutoff: float = 5.0
    num_hidden: int = 128
    num_hidden_feedforward: int = 256
    num_attention_layers: int = 2
    num_message_passing_layers: int = 2
    num_heads: int = 4
    cutoff_fn: str = "cosine_cutoff"
    cutoff_width: float = 0.2
    num_charges: int = 8
    init_prefactor: float = 1.0

    @property
    def to_batch(self):
        return ToBatch

    @property
    def to_sample(self):
        return ToSample

    @nn.compact
    def __call__(
        self,
        Z_i,
        reverse,
        sr,
        nopbc,
        pbc,
    ):
        predictions, central_tokens = PETCore(
            num_hidden=self.num_hidden,
            num_hidden_feedforward=self.num_hidden_feedforward,
            num_attention_layers=self.num_attention_layers,
            num_message_passing_layers=self.num_message_passing_layers,
            num_heads=self.num_heads,
            cutoff=self.cutoff,
            cutoff_width=self.cutoff_width,
            name="sr",
        )(
            Z_i,
            reverse,
            sr.positions,
            sr.cell,
            sr.centers,
            sr.others,
            sr.cell_shifts,
            sr.pair_mask,
            sr.atom_mask,
            sr.pair_to_structure,
        )

        if self.lr:
            predictions += LRModule(
                num_hidden=self.num_hidden,
                num_hidden_feedforward=self.num_hidden_feedforward,
                num_charges=self.num_charges,
                init_prefactor=self.init_prefactor,
                name="lr",
            )(central_tokens, sr.atom_mask, sr, nopbc, pbc)

        return predictions

    def dummy_inputs(self):
        from ase.build import bulk

        from .batching import to_batch, to_sample

        sample = to_sample(bulk("Ar") * [2, 2, 2], self.cutoff, keys=(), energy=False, forces=False)
        batch = to_batch([sample], [])
        return jax.tree.map(lambda x: jnp.array(x), batch[:-1])

    def energy(self, params, batch):
        sr = batch.sr
        energies = self.apply(
            params,
            batch.atomic_numbers,
            batch.reverse,
            batch.sr,
            batch.nopbc,
            batch.pbc,
        )
        energies *= sr.atom_mask

        return jnp.sum(energies), energies

    def predict(self, params, batch, stress=False):
        sr = batch.sr

        energy_and_derivatives_fn = jax.value_and_grad(
            self.energy, allow_int=True, has_aux=True, argnums=1
        )
        batch_energy_and_atom_energies, grads = energy_and_derivatives_fn(params, batch)
        _, energies = batch_energy_and_atom_energies

        grads_sr = grads.sr

        energy = (
            jax.ops.segment_sum(energies, sr.atom_to_structure, sr.cell.shape[0])
            * sr.structure_mask
        )

        forces = -grads_sr.positions

        results = {"energy": energy, "forces": forces}

        if stress:
            results["stress"] = (
                jax.ops.segment_sum(
                    jnp.einsum("ia,ib->iab", sr.positions, grads_sr.positions),
                    sr.atom_to_structure,
                    num_segments=sr.cell.shape[0],
                )
                + jnp.einsum("sAa,sAb->sab", sr.cell, grads_sr.cell)
            ) * sr.structure_mask[:, None, None]

        return results
