"""JAX port of the Latent Ewald Summation (LES) algorithm — charge-only.

Reference: https://github.com/ChengUCB/les

Two Ewald paths are provided, matching LES's own split:
  - les_ewald_triclinic : k-space sum for periodic (triclinic) cells
  - les_ewald_realspace : O(N²) real-space direct sum for non-periodic clusters

LESModule wraps both into a Flax module that maps PETCore central tokens to
per-structure long-range energies via a learned MLP + Ewald sum.

Key JAX adaptation: LES determines the k-grid size dynamically from cell norms,
which is incompatible with JIT. Here n_k_max is a static hyperparameter that
pre-allocates the full [-n_k_max, n_k_max]^3 grid; invalid k-vectors are zeroed
out via masking. Set n_k_max >= ceil(max_cell_norm / dl).
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from marathon.utils import masked

from .modules import MLP


def les_ewald_triclinic(q, r, cell, sigma, dl, n_k_max, norm_factor=90.4756):
    """K-space Ewald energy for a single periodic structure (charge-only).

    JAX port of LES Ewald.compute_potential_triclinic, charge channel only.

    Args:
        q          : [n_atoms, n_q]  latent charges (masked atoms must have q=0)
        r          : [n_atoms, 3]    atomic positions
        cell       : [3, 3]          cell matrix (rows = lattice vectors)
        sigma      : Gaussian smearing width (Å)
        dl         : k-space resolution (Å); sets cutoff |k| ≤ 2π/dl
        n_k_max    : max integer k-index per axis (static — determines grid shape)
        norm_factor: 1/(2ε₀) in eV·Å units (LES default: 90.4756)

    Returns:
        scalar energy (summed over n_q channels)
    """
    volume = jnp.abs(jnp.linalg.det(cell))
    G = 2.0 * jnp.pi * jnp.linalg.inv(cell).T  # reciprocal lattice: k = nvec @ G

    # Static k-grid: n ∈ [-n_k_max, n_k_max]^3
    n_range = jnp.arange(-n_k_max, n_k_max + 1)
    n1, n2, n3 = jnp.meshgrid(n_range, n_range, n_range, indexing="ij")
    nvec = jnp.stack([n1.ravel(), n2.ravel(), n3.ravel()], axis=-1).astype(r.dtype)
    kvec = nvec @ G          # [M, 3]
    k_sq = jnp.sum(kvec ** 2, axis=-1)  # [M]

    # Valid k-vectors: |k| > 0  and  |k| ≤ 2π/dl
    k_sq_max = (2.0 * jnp.pi / dl) ** 2
    valid = (k_sq > 1e-10) & (k_sq <= k_sq_max)  # [M]

    # Structure factor  S(k) = Σ_i q_i exp(i k·r_i)
    k_dot_r = r @ kvec.T                                         # [n_atoms, M]
    S_real = jnp.einsum("iq,iM->qM", q, jnp.cos(k_dot_r))      # [n_q, M]
    S_imag = jnp.einsum("iq,iM->qM", q, jnp.sin(k_dot_r))      # [n_q, M]
    S_sq = S_real ** 2 + S_imag ** 2                             # [n_q, M]

    # Per-k prefactor  exp(-σ²k²/2) / k²  (zero for invalid k)
    k_sq_safe = jnp.where(k_sq > 0, k_sq, 1.0)
    kfac = jnp.where(valid, jnp.exp(-0.5 * sigma ** 2 * k_sq) / k_sq_safe, 0.0)

    # Reciprocal-space energy — ÷2 because the grid contains both +k and −k
    pot = norm_factor / volume * jnp.einsum("M,qM->q", kfac, S_sq) / 2.0  # [n_q]

    # Self-interaction correction (matches LES remove_self_interaction=True default)
    self_E = norm_factor / (sigma * (2.0 * jnp.pi) ** 1.5) * jnp.sum(q ** 2, axis=0)
    pot = pot - self_E  # [n_q]

    return jnp.sum(pot)


def les_ewald_realspace(q, r, sigma, norm_factor=90.4756):
    """Real-space direct pairwise energy for a non-periodic structure (charge-only).

    JAX port of LES Ewald.compute_potential_realspace + make_kernels, charge
    channel only. Complexity is O(N²) — suitable for small clusters/molecules.

    Args:
        q          : [n_atoms, n_q]  latent charges (masked atoms must have q=0)
        r          : [n_atoms, 3]    atomic positions
        sigma      : Gaussian smearing width (Å)
        norm_factor: 1/(2ε₀) in eV·Å units

    Returns:
        scalar energy (summed over n_q channels)
    """
    norm_const = norm_factor / (2.0 * jnp.pi)
    a = 1.0 / (sigma * jnp.sqrt(2.0))

    # Pairwise displacement  r_ij[i,j] = r[j] - r[i]
    r_ij = r[None, :, :] - r[:, None, :]     # [n, n, 3]
    r_sq = jnp.sum(r_ij ** 2, axis=-1)       # [n, n]

    # Off-diagonal mask — exclude self-interaction (i == j)
    n = r.shape[0]
    mask_off = (1.0 - jnp.eye(n, dtype=r.dtype))  # [n, n]

    # Guard ALL pairs before sqrt — diagonal has r_sq=0, and padded atoms share
    # position [0,0,0] so off-diagonal padded pairs also have r_sq=0.
    # sqrt(0) has infinite gradient; even inf*0 = NaN in the chain rule.
    r_ij_norm = jnp.sqrt(jnp.maximum(r_sq, 1e-30))  # finite everywhere

    # f_qq[i,j] = erf(r_ij * a) / r_ij * norm_const,  zero on diagonal
    f_qq = jax.scipy.special.erf(r_ij_norm * a) / r_ij_norm * norm_const * mask_off

    # Electric potential at j from all charges i:  φ_j = Σ_i q_i f_qq[i,j]
    e_phi = jnp.einsum("iq,ij->jq", q, f_qq)  # [n, n_q]

    # Energy = ½ Σ_j q_j φ_j
    pot = 0.5 * jnp.einsum("jq,jq->q", q, e_phi)  # [n_q]

    return jnp.sum(pot)


class LESModule(nn.Module):
    """LES long-range module: central tokens → latent charges → Ewald energy.

    Charge-only JAX port of LES (https://github.com/ChengUCB/les).
    Periodic structures use les_ewald_triclinic; non-periodic use
    les_ewald_realspace (detected by |det(cell)| > 1e-6).

    Hyperparameters:
        num_hidden  : MLP hidden width (should match PETCore num_hidden)
        num_charges : number of latent charge channels (LES default: 1)
        sigma       : Gaussian smearing width in Å (LES default: 1.0)
        dl          : k-space resolution in Å, sets cutoff |k| ≤ 2π/dl
        n_k_max     : max k-index per axis; set >= ceil(max_cell_norm / dl)
        norm_factor : 1/(2ε₀) in eV·Å (LES default: 90.4756)
    """

    num_hidden: int = 128
    num_charges: int = 1
    sigma: float = 1.0
    dl: float = 2.0
    n_k_max: int = 10
    norm_factor: float = 90.4756

    @nn.compact
    def __call__(self, central_tokens, positions, cell, atom_mask, atom_to_structure):
        """
        Args:
            central_tokens   : [n_atoms, num_hidden]
            positions        : [n_atoms, 3]
            cell             : [n_structures, 3, 3]
            atom_mask        : [n_atoms] bool
            atom_to_structure: [n_atoms] int

        Returns:
            [n_structures] LES energy per structure
        """
        charges = masked(
            MLP(features=[self.num_hidden, self.num_hidden, self.num_charges]),
            central_tokens,
            atom_mask,
        )  # [n_atoms, n_q]

        n_structures = cell.shape[0]
        sigma = self.sigma
        dl = self.dl
        n_k_max = self.n_k_max
        norm_factor = self.norm_factor

        def energy_for_structure(s):
            s_mask = ((atom_to_structure == s) & atom_mask).astype(charges.dtype)
            q_s = charges * s_mask[:, None]  # zero out atoms from other structures

            volume = jnp.abs(jnp.linalg.det(cell[s]))
            is_periodic = volume > 1e-6

            # Guard cell against det=0 so les_ewald_triclinic stays finite in both
            # branches (jnp.where evaluates both; NaN/Inf would corrupt gradients).
            cell_safe = jnp.where(is_periodic, cell[s], jnp.eye(3, dtype=cell.dtype))

            periodic_E = les_ewald_triclinic(q_s, positions, cell_safe, sigma, dl, n_k_max, norm_factor)
            realspace_E = les_ewald_realspace(q_s, positions, sigma, norm_factor)

            return jnp.where(is_periodic, periodic_E, realspace_E)

        return jax.vmap(energy_for_structure)(jnp.arange(n_structures))
