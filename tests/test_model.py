"""Tests for LongRangePET model."""

import jax
import jax.numpy as jnp
import pytest
from ase.build import bulk

from pet.model import LongRangePET


def test_model_init():
    model = LongRangePET(cutoff=5.0)
    assert model.cutoff == 5.0
    assert model.lr is False


def test_model_lr_flag():
    model = LongRangePET(cutoff=5.0, lr=True)
    assert model.lr is True


def test_dummy_inputs(small_batch):
    model = LongRangePET(cutoff=5.0)
    dummy = model.dummy_inputs()
    # dummy_inputs returns batch[:-1]: (atomic_numbers, reverse, sr, nopbc, pbc)
    assert len(dummy) == 5


def test_model_params_init():
    model = LongRangePET(cutoff=5.0, num_hidden=32, num_message_passing_layers=1)
    params = model.init(jax.random.key(0), *model.dummy_inputs())
    assert "params" in params


def test_predict_sr(small_batch):
    model = LongRangePET(cutoff=5.0, num_hidden=32, num_message_passing_layers=1)
    params = model.init(jax.random.key(0), *model.dummy_inputs())

    results = model.predict(params, small_batch)
    assert "energy" in results
    assert "forces" in results
    assert results["energy"].shape == (small_batch.sr.cell.shape[0],)
    assert results["forces"].shape == (small_batch.sr.positions.shape[0], 3)


def test_predict_with_stress(small_batch):
    model = LongRangePET(cutoff=5.0, num_hidden=32, num_message_passing_layers=1)
    params = model.init(jax.random.key(0), *model.dummy_inputs())

    results = model.predict(params, small_batch, stress=True)
    assert "stress" in results
    assert results["stress"].shape == (small_batch.sr.cell.shape[0], 3, 3)


def test_predict_lr(small_batch):
    model = LongRangePET(cutoff=5.0, lr=True, num_hidden=32, num_message_passing_layers=1)
    params = model.init(jax.random.key(0), *model.dummy_inputs())

    results = model.predict(params, small_batch)
    assert "energy" in results
    assert "forces" in results


def test_to_batch_to_sample_properties():
    """LongRangePET exposes to_batch and to_sample as class properties."""
    from pet.transforms import ToBatch, ToSample

    model = LongRangePET(cutoff=5.0)
    assert model.to_batch is ToBatch
    assert model.to_sample is ToSample
