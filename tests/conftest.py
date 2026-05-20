import jax
import jax.numpy as jnp
import pytest


@pytest.fixture
def small_batch():
    from ase.build import bulk

    from pet.batching import to_batch, to_sample

    atoms = bulk("Ar") * [2, 2, 2]
    sample = to_sample(atoms, 5.0, keys=(), energy=False, forces=False)
    batch = to_batch([sample], [])
    # Convert all arrays to JAX arrays so tests work without JIT
    return jax.tree.map(lambda x: jnp.array(x), batch)
