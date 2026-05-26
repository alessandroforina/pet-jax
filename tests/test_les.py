"""Tests for the LES (Latent Ewald Summation) JAX port.

Covers:
  - les_ewald_triclinic  : k-space Ewald for periodic structures
  - les_ewald_realspace  : direct pairwise sum for non-periodic structures
  - LESModule            : Flax module (tokens → charges → Ewald energy)
  - LESLongRangePET      : full model with LES long-range
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ase.build import bulk


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fcc_cell():
    """FCC Ar conventional cell matrix (rows = lattice vectors)."""
    atoms = bulk("Ar") * [2, 2, 2]
    return jnp.array(atoms.get_cell()[:], dtype=jnp.float32)


@pytest.fixture
def fcc_positions(fcc_cell):
    """Atomic positions for FCC Ar 2x2x2 supercell."""
    atoms = bulk("Ar") * [2, 2, 2]
    return jnp.array(atoms.get_positions(), dtype=jnp.float32)


@pytest.fixture
def small_batch():
    from pet.batching import to_batch, to_sample

    atoms = bulk("Ar") * [2, 2, 2]
    sample = to_sample(atoms, 5.0, keys=(), energy=False, forces=False)
    batch = to_batch([sample], [])
    return jax.tree.map(lambda x: jnp.array(x), batch)


@pytest.fixture
def two_structure_batch():
    """Batch with two different periodic structures."""
    from pet.batching import to_batch, to_sample

    s1 = to_sample(bulk("Ar") * [2, 2, 2], 5.0, keys=(), energy=False, forces=False)
    s2 = to_sample(bulk("NaCl", crystalstructure="rocksalt", a=5.6), 5.0, keys=(), energy=False, forces=False)
    batch = to_batch([s1, s2], [])
    return jax.tree.map(lambda x: jnp.array(x), batch)


# ---------------------------------------------------------------------------
# les_ewald_triclinic
# ---------------------------------------------------------------------------

class TestLesEwaldTriclinic:

    def test_returns_scalar(self, fcc_positions, fcc_cell):
        from pet.les import les_ewald_triclinic

        n = fcc_positions.shape[0]
        q = jnp.ones((n, 1), dtype=jnp.float32) * 0.1
        e = les_ewald_triclinic(q, fcc_positions, fcc_cell, sigma=1.0, dl=2.0, n_k_max=5)
        assert e.shape == ()

    def test_zero_charges_give_zero_energy(self, fcc_positions, fcc_cell):
        from pet.les import les_ewald_triclinic

        n = fcc_positions.shape[0]
        q = jnp.zeros((n, 1), dtype=jnp.float32)
        e = les_ewald_triclinic(q, fcc_positions, fcc_cell, sigma=1.0, dl=2.0, n_k_max=5)
        assert jnp.abs(e) < 1e-6

    def test_energy_is_finite(self, fcc_positions, fcc_cell):
        from pet.les import les_ewald_triclinic

        n = fcc_positions.shape[0]
        q = jnp.ones((n, 1), dtype=jnp.float32) * 0.1
        e = les_ewald_triclinic(q, fcc_positions, fcc_cell, sigma=1.0, dl=2.0, n_k_max=5)
        assert jnp.isfinite(e)

    def test_energy_scales_with_charge_squared(self, fcc_positions, fcc_cell):
        """Ewald energy is quadratic in the charges."""
        from pet.les import les_ewald_triclinic

        n = fcc_positions.shape[0]
        q1 = jnp.ones((n, 1), dtype=jnp.float32) * 0.1
        q2 = q1 * 2.0
        e1 = les_ewald_triclinic(q1, fcc_positions, fcc_cell, sigma=1.0, dl=2.0, n_k_max=5)
        e2 = les_ewald_triclinic(q2, fcc_positions, fcc_cell, sigma=1.0, dl=2.0, n_k_max=5)
        assert jnp.abs(e2 / e1 - 4.0) < 1e-4

    def test_charge_sign_symmetry(self, fcc_positions, fcc_cell):
        """Energy is symmetric under q → -q (charge-charge interaction is even in q)."""
        from pet.les import les_ewald_triclinic

        n = fcc_positions.shape[0]
        q = jnp.ones((n, 1), dtype=jnp.float32) * 0.1
        e_pos = les_ewald_triclinic( q, fcc_positions, fcc_cell, sigma=1.0, dl=2.0, n_k_max=5)
        e_neg = les_ewald_triclinic(-q, fcc_positions, fcc_cell, sigma=1.0, dl=2.0, n_k_max=5)
        assert jnp.abs(e_pos - e_neg) < 1e-5

    def test_translation_invariance(self, fcc_positions, fcc_cell):
        """Energy does not change when all atoms are shifted by a constant vector."""
        from pet.les import les_ewald_triclinic

        n = fcc_positions.shape[0]
        q = jnp.ones((n, 1), dtype=jnp.float32) * 0.1
        shift = jnp.array([0.5, 1.2, -0.3], dtype=jnp.float32)
        e_orig = les_ewald_triclinic(q, fcc_positions, fcc_cell, sigma=1.0, dl=2.0, n_k_max=5)
        e_shifted = les_ewald_triclinic(q, fcc_positions + shift, fcc_cell, sigma=1.0, dl=2.0, n_k_max=5)
        assert jnp.abs(e_orig - e_shifted) < 1e-3

    def test_gradient_wrt_positions_finite(self, fcc_positions, fcc_cell):
        from pet.les import les_ewald_triclinic

        n = fcc_positions.shape[0]
        q = jnp.ones((n, 1), dtype=jnp.float32) * 0.1
        grad = jax.grad(les_ewald_triclinic, argnums=1)(q, fcc_positions, fcc_cell, 1.0, 2.0, 5)
        assert not jnp.any(jnp.isnan(grad))
        assert not jnp.any(jnp.isinf(grad))

    def test_gradient_wrt_positions_zero_when_charges_zero(self, fcc_positions, fcc_cell):
        from pet.les import les_ewald_triclinic

        n = fcc_positions.shape[0]
        q = jnp.zeros((n, 1), dtype=jnp.float32)
        grad = jax.grad(les_ewald_triclinic, argnums=1)(q, fcc_positions, fcc_cell, 1.0, 2.0, 5)
        assert jnp.allclose(grad, 0.0, atol=1e-6)

    def test_multi_channel_charges(self, fcc_positions, fcc_cell):
        """n_q > 1 channels are summed over correctly."""
        from pet.les import les_ewald_triclinic

        n = fcc_positions.shape[0]
        q1 = jnp.ones((n, 1), dtype=jnp.float32) * 0.1
        q2 = jnp.ones((n, 2), dtype=jnp.float32) * 0.1
        e1 = les_ewald_triclinic(q1, fcc_positions, fcc_cell, sigma=1.0, dl=2.0, n_k_max=5)
        e2 = les_ewald_triclinic(q2, fcc_positions, fcc_cell, sigma=1.0, dl=2.0, n_k_max=5)
        assert jnp.abs(e2 - 2.0 * e1) < 1e-4


# ---------------------------------------------------------------------------
# les_ewald_realspace
# ---------------------------------------------------------------------------

class TestLesEwaldRealspace:

    def test_returns_scalar(self, fcc_positions):
        from pet.les import les_ewald_realspace

        n = fcc_positions.shape[0]
        q = jnp.ones((n, 1), dtype=jnp.float32) * 0.1
        e = les_ewald_realspace(q, fcc_positions, sigma=1.0)
        assert e.shape == ()

    def test_zero_charges_give_zero_energy(self, fcc_positions):
        from pet.les import les_ewald_realspace

        n = fcc_positions.shape[0]
        q = jnp.zeros((n, 1), dtype=jnp.float32)
        e = les_ewald_realspace(q, fcc_positions, sigma=1.0)
        assert jnp.abs(e) < 1e-6

    def test_energy_is_finite(self, fcc_positions):
        from pet.les import les_ewald_realspace

        n = fcc_positions.shape[0]
        q = jnp.ones((n, 1), dtype=jnp.float32) * 0.1
        e = les_ewald_realspace(q, fcc_positions, sigma=1.0)
        assert jnp.isfinite(e)

    def test_charge_sign_symmetry(self, fcc_positions):
        """Energy is symmetric under q → -q."""
        from pet.les import les_ewald_realspace

        n = fcc_positions.shape[0]
        q = jnp.ones((n, 1), dtype=jnp.float32) * 0.1
        e_pos = les_ewald_realspace( q, fcc_positions, sigma=1.0)
        e_neg = les_ewald_realspace(-q, fcc_positions, sigma=1.0)
        assert jnp.abs(e_pos - e_neg) < 1e-5

    def test_gradient_wrt_positions_finite(self, fcc_positions):
        """No NaN gradient even for padded atoms sharing position [0,0,0]."""
        from pet.les import les_ewald_realspace

        n = fcc_positions.shape[0]
        # Mix real atoms (first half) with zero-position padded atoms (second half)
        n_real = n // 2
        q = jnp.concatenate([
            jnp.ones((n_real, 1), dtype=jnp.float32) * 0.1,
            jnp.zeros((n - n_real, 1), dtype=jnp.float32),
        ], axis=0)
        positions_with_padding = jnp.concatenate([
            fcc_positions[:n_real],
            jnp.zeros((n - n_real, 3), dtype=jnp.float32),  # padded at origin
        ], axis=0)
        grad = jax.grad(les_ewald_realspace, argnums=1)(q, positions_with_padding, 1.0)
        assert not jnp.any(jnp.isnan(grad))
        assert not jnp.any(jnp.isinf(grad))

    def test_gradient_wrt_positions_zero_when_charges_zero(self, fcc_positions):
        from pet.les import les_ewald_realspace

        n = fcc_positions.shape[0]
        q = jnp.zeros((n, 1), dtype=jnp.float32)
        grad = jax.grad(les_ewald_realspace, argnums=1)(q, fcc_positions, 1.0)
        assert jnp.allclose(grad, 0.0, atol=1e-6)

    def test_gradient_wrt_padded_atoms_is_zero(self, fcc_positions):
        """Gradient is zero for atoms with q=0 (padded atoms)."""
        from pet.les import les_ewald_realspace

        n = fcc_positions.shape[0]
        n_real = n // 2
        q = jnp.concatenate([
            jnp.ones((n_real, 1), dtype=jnp.float32) * 0.1,
            jnp.zeros((n - n_real, 1), dtype=jnp.float32),
        ], axis=0)
        positions_with_padding = jnp.concatenate([
            fcc_positions[:n_real],
            jnp.zeros((n - n_real, 3), dtype=jnp.float32),
        ], axis=0)
        grad = jax.grad(les_ewald_realspace, argnums=1)(q, positions_with_padding, 1.0)
        assert jnp.allclose(grad[n_real:], 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# LESModule
# ---------------------------------------------------------------------------

class TestLESModule:

    def test_output_shape(self, small_batch):
        from pet.les import LESModule

        sr = small_batch.sr
        n_atoms = sr.atom_mask.shape[0]
        n_structures = sr.cell.shape[0]
        central_tokens = jnp.ones((n_atoms, 32), dtype=jnp.float32)

        mod = LESModule(num_hidden=32, num_charges=1, sigma=1.0, dl=2.0, n_k_max=5)
        params = mod.init(jax.random.key(0), central_tokens, sr.positions, sr.cell,
                          sr.atom_mask, sr.atom_to_structure)
        out = mod.apply(params, central_tokens, sr.positions, sr.cell,
                        sr.atom_mask, sr.atom_to_structure)
        assert out.shape == (n_structures,)

    def test_output_is_finite(self, small_batch):
        from pet.les import LESModule

        sr = small_batch.sr
        n_atoms = sr.atom_mask.shape[0]
        central_tokens = jnp.ones((n_atoms, 32), dtype=jnp.float32)

        mod = LESModule(num_hidden=32, num_charges=1, sigma=1.0, dl=2.0, n_k_max=5)
        params = mod.init(jax.random.key(0), central_tokens, sr.positions, sr.cell,
                          sr.atom_mask, sr.atom_to_structure)
        out = mod.apply(params, central_tokens, sr.positions, sr.cell,
                        sr.atom_mask, sr.atom_to_structure)
        assert jnp.all(jnp.isfinite(out))

    def test_padding_structure_gives_zero_energy(self, small_batch):
        """Padded (dummy) structures have zero atoms and must contribute zero energy."""
        from pet.les import LESModule

        sr = small_batch.sr
        n_atoms = sr.atom_mask.shape[0]
        central_tokens = jnp.ones((n_atoms, 32), dtype=jnp.float32)

        mod = LESModule(num_hidden=32, num_charges=1, sigma=1.0, dl=2.0, n_k_max=5)
        params = mod.init(jax.random.key(0), central_tokens, sr.positions, sr.cell,
                          sr.atom_mask, sr.atom_to_structure)
        out = mod.apply(params, central_tokens, sr.positions, sr.cell,
                        sr.atom_mask, sr.atom_to_structure)
        # batch has 1 real structure + padding; padding energies must be 0
        n_real = int(sr.structure_mask.sum())
        assert jnp.allclose(out[n_real:], 0.0, atol=1e-5)


# ---------------------------------------------------------------------------
# LESLongRangePET model
# ---------------------------------------------------------------------------

class TestLESLongRangePET:

    def test_init_les_false_has_only_sr_params(self):
        from pet.model import LESLongRangePET

        model = LESLongRangePET(les=False, cutoff=5.0, num_hidden=32,
                                num_message_passing_layers=1)
        params = model.init(jax.random.key(0), *model.dummy_inputs())
        assert set(params["params"].keys()) == {"sr"}

    def test_init_les_true_has_sr_and_les_params(self):
        from pet.model import LESLongRangePET

        model = LESLongRangePET(les=True, cutoff=5.0, num_hidden=32,
                                num_message_passing_layers=1, n_k_max=5)
        params = model.init(jax.random.key(0), *model.dummy_inputs())
        assert "sr" in params["params"]
        assert "les" in params["params"]

    def test_dummy_inputs_length(self):
        from pet.model import LESLongRangePET

        model = LESLongRangePET(cutoff=5.0)
        dummy = model.dummy_inputs()
        assert len(dummy) == 5  # (atomic_numbers, reverse, sr, nopbc, pbc)

    def test_predict_output_shapes(self, small_batch):
        from pet.model import LESLongRangePET

        model = LESLongRangePET(les=True, cutoff=5.0, num_hidden=32,
                                num_message_passing_layers=1, n_k_max=5)
        params = model.init(jax.random.key(0), *model.dummy_inputs())
        results = model.predict(params, small_batch)

        n_structures = small_batch.sr.cell.shape[0]
        n_atoms = small_batch.sr.positions.shape[0]
        assert results["energy"].shape == (n_structures,)
        assert results["forces"].shape == (n_atoms, 3)

    def test_predict_forces_finite(self, small_batch):
        from pet.model import LESLongRangePET

        model = LESLongRangePET(les=True, cutoff=5.0, num_hidden=32,
                                num_message_passing_layers=1, n_k_max=5)
        params = model.init(jax.random.key(0), *model.dummy_inputs())
        results = model.predict(params, small_batch)
        assert not jnp.any(jnp.isnan(results["forces"]))
        assert not jnp.any(jnp.isinf(results["forces"]))

    def test_predict_with_stress(self, small_batch):
        from pet.model import LESLongRangePET

        model = LESLongRangePET(les=True, cutoff=5.0, num_hidden=32,
                                num_message_passing_layers=1, n_k_max=5)
        params = model.init(jax.random.key(0), *model.dummy_inputs())
        results = model.predict(params, small_batch, stress=True)

        n_structures = small_batch.sr.cell.shape[0]
        assert "stress" in results
        assert results["stress"].shape == (n_structures, 3, 3)
        assert not jnp.any(jnp.isnan(results["stress"]))

    def test_les_false_matches_longrangepet_sr(self, small_batch):
        """LESLongRangePET(les=False) is identical to LongRangePET(lr=False)."""
        from pet.model import LESLongRangePET, LongRangePET

        kwargs = dict(cutoff=5.0, num_hidden=32, num_message_passing_layers=1)
        les_model = LESLongRangePET(les=False, **kwargs)
        lr_model = LongRangePET(lr=False, **kwargs)

        key = jax.random.key(0)
        les_params = les_model.init(key, *les_model.dummy_inputs())
        lr_params = lr_model.init(key, *lr_model.dummy_inputs())

        les_results = les_model.predict(les_params, small_batch)
        lr_results = lr_model.predict(lr_params, small_batch)
        assert jnp.allclose(les_results["energy"], lr_results["energy"], atol=1e-5)

    def test_predict_two_structures(self, two_structure_batch):
        """Forces and energy are finite for a batch with two different structures."""
        from pet.model import LESLongRangePET

        model = LESLongRangePET(les=True, cutoff=5.0, num_hidden=32,
                                num_message_passing_layers=1, n_k_max=5)
        params = model.init(jax.random.key(0), *model.dummy_inputs())
        results = model.predict(params, two_structure_batch)

        assert not jnp.any(jnp.isnan(results["forces"]))
        assert not jnp.any(jnp.isnan(results["energy"]))

    def test_to_batch_to_sample_properties(self):
        from pet.model import LESLongRangePET
        from pet.transforms import ToBatch, ToSample

        model = LESLongRangePET(cutoff=5.0)
        assert model.to_batch is ToBatch
        assert model.to_sample is ToSample

    def test_jit_compatible(self, small_batch):
        """predict can be wrapped in jax.jit without retracing issues."""
        from pet.model import LESLongRangePET

        model = LESLongRangePET(les=True, cutoff=5.0, num_hidden=32,
                                num_message_passing_layers=1, n_k_max=5)
        params = model.init(jax.random.key(0), *model.dummy_inputs())
        predict_jit = jax.jit(lambda p, b: model.predict(p, b))
        r1 = predict_jit(params, small_batch)
        r2 = predict_jit(params, small_batch)
        assert jnp.allclose(r1["energy"], r2["energy"])
