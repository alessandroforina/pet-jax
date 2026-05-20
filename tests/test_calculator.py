"""Tests for the PET ASE Calculator."""

import numpy as np
import pytest
from ase.build import bulk

from pet.calculator import Calculator
from pet.model import LongRangePET


@pytest.fixture
def model_and_params():
    import jax

    model = LongRangePET(cutoff=5.0, num_hidden=32, num_message_passing_layers=1)
    params = model.init(jax.random.key(0), *model.dummy_inputs())
    return model, params


@pytest.fixture
def atoms():
    return bulk("Ar") * [2, 2, 2]


def test_from_model(model_and_params, atoms):
    model, params = model_and_params
    calc = Calculator.from_model(model, params=params)
    assert calc.cutoff == model.cutoff
    assert calc.add_offset is False  # no species_weights → offset disabled


def test_calculate_energy(model_and_params, atoms):
    model, params = model_and_params
    calc = Calculator.from_model(model, params=params)
    calc.calculate(atoms)
    assert "energy" in calc.results
    assert isinstance(calc.results["energy"], float)


def test_calculate_forces(model_and_params, atoms):
    model, params = model_and_params
    calc = Calculator.from_model(model, params=params)
    calc.calculate(atoms)
    assert "forces" in calc.results
    assert calc.results["forces"].shape == (len(atoms), 3)


def test_calculate_stress(model_and_params, atoms):
    model, params = model_and_params
    calc = Calculator.from_model(model, params=params, stress=True)
    calc.calculate(atoms)
    assert "stress" in calc.results
    assert calc.results["stress"].shape == (6,)  # Voigt notation


def test_calculate_no_stress_by_default(model_and_params, atoms):
    model, params = model_and_params
    calc = Calculator.from_model(model, params=params)
    assert "stress" not in calc.implemented_properties


def test_get_potential_energy(model_and_params, atoms):
    model, params = model_and_params
    calc = Calculator.from_model(model, params=params)
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    assert isinstance(energy, float)


def test_get_forces(model_and_params, atoms):
    model, params = model_and_params
    calc = Calculator.from_model(model, params=params)
    atoms.calc = calc
    forces = atoms.get_forces()
    assert forces.shape == (len(atoms), 3)


def test_update_no_recompute_same_atoms(model_and_params, atoms):
    """Calling calculate twice on the same atoms reuses results."""
    model, params = model_and_params
    calc = Calculator.from_model(model, params=params)
    calc.calculate(atoms)
    first_energy = calc.results["energy"]
    calc.calculate(atoms)
    assert calc.results["energy"] == first_energy


def test_energy_offset(model_and_params, atoms):
    """Energy offset is added when species_weights are provided."""
    model, params = model_and_params
    species_weights = {18: -10.0}  # Ar has Z=18
    calc = Calculator.from_model(model, params=params, species_weights=species_weights)
    calc.calculate(atoms)
    # offset = -10.0 * len(atoms)
    expected_offset = -10.0 * len(atoms)

    calc_no_offset = Calculator.from_model(model, params=params)
    calc_no_offset.calculate(atoms)

    assert abs(calc.results["energy"] - calc_no_offset.results["energy"] - expected_offset) < 1e-4


def test_lr_model(atoms):
    """Calculator works with lr=True model."""
    import jax

    model = LongRangePET(cutoff=5.0, lr=True, num_hidden=32, num_message_passing_layers=1)
    params = model.init(jax.random.key(0), *model.dummy_inputs())
    calc = Calculator.from_model(model, params=params)
    calc.calculate(atoms)
    assert "energy" in calc.results
    assert "forces" in calc.results
