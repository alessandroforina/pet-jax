import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_default_matmul_precision", "float32")

from ase.calculators.calculator import (
    BaseCalculator,
    PropertyNotImplementedError,
    compare_atoms,
)
from marathon.emit.checkpoint import read_msgpack
from marathon.io import from_dict, read_yaml


class Calculator(BaseCalculator):
    name = "pet"
    parameters = {}

    def todict(self):
        return self.parameters

    implemented_properties = [
        "energy",
        "forces",
        "stress",
    ]

    def __init__(
        self,
        pred_fn,
        species_weights,
        params,
        cutoff,
        atoms=None,
        stress=False,
        add_offset=True,
        lr_wavelength=None,
        smearing=None,
    ):
        self.params = params
        self.cutoff = cutoff
        self.add_offset = add_offset

        # jaxpme Ewald parameters: derived from cutoff if not given
        self.lr_wavelength = lr_wavelength if lr_wavelength is not None else cutoff / 8.0
        self.smearing = smearing if smearing is not None else self.lr_wavelength * 2.0

        if not stress:
            self.implemented_properties = ["energy", "forces"]

        predict_fn = lambda params, batch: pred_fn(params, batch, stress=stress)
        self.predict_fn = jax.jit(predict_fn)
        self.species_weights = species_weights

        self.atoms = None
        self.batch = None
        self.results = {}

        if atoms is not None:
            self.setup(atoms)

    @classmethod
    def from_model(cls, model, params=None, species_weights=None, **kwargs):
        """Create a Calculator from a LongRangePET model instance."""
        if params is None:
            params = model.init(jax.random.key(0), *model.dummy_inputs())
        if species_weights is None:
            species_weights = {}
            kwargs.setdefault("add_offset", False)
        return cls(model.predict, species_weights, params, model.cutoff, **kwargs)

    @classmethod
    def from_checkpoint(cls, folder, **kwargs):
        """Load a Calculator from a marathon checkpoint directory.

        Expected layout:
            folder/
              model/
                model.yaml      model architecture + optional baseline
                baseline.yaml   per-element energy offsets
                model.msgpack   trained parameters
        """
        from pathlib import Path

        folder = Path(folder)

        model = from_dict(read_yaml(folder / "model/model.yaml"))
        _ = model.init(jax.random.key(1), *model.dummy_inputs())

        baseline = read_yaml(folder / "model/baseline.yaml")
        species_to_weight = baseline["elemental"]

        params = read_msgpack(folder / "model/model.msgpack")

        return cls(model.predict, species_to_weight, params, model.cutoff, **kwargs)

    def update(self, atoms):
        changes = compare_atoms(self.atoms, atoms)
        if len(changes) > 0:
            self.results = {}
            self.atoms = atoms.copy()
            self.setup(atoms)

    def setup(self, atoms):
        from pet.batching import to_batch, to_sample

        sample = to_sample(
            atoms,
            self.cutoff,
            keys=(),
            energy=False,
            forces=False,
            stress=False,
            lr_wavelength=self.lr_wavelength,
            smearing=self.smearing,
        )
        batch = to_batch([sample], [])
        self.batch = jax.tree.map(lambda x: jnp.array(x), batch)

    def calculate(self, atoms=None, properties=None, system_changes=None, **kwargs):
        self.update(atoms)

        results = self.predict_fn(self.params, self.batch)

        sr = self.batch.sr
        actual_results = {}

        for key in self.implemented_properties:
            if key == "energy":
                actual_results[key] = float(results[key][sr.structure_mask].squeeze())
            elif key == "forces":
                actual_results[key] = np.array(
                    results[key][sr.atom_mask].reshape(-1, 3), dtype=np.float32
                )
            elif key == "stress":
                virial = np.array(
                    results[key][sr.structure_mask].reshape(3, 3), dtype=np.float32
                )
                volume = atoms.get_volume()
                from ase.stress import full_3x3_to_voigt_6_stress

                actual_results[key] = full_3x3_to_voigt_6_stress(virial / volume)

        if self.add_offset:
            energy_offset = np.sum(
                [self.species_weights[Z] for Z in atoms.get_atomic_numbers()]
            )
            actual_results["energy"] += energy_offset

        self.results = actual_results
        return actual_results

    def get_property(self, name, atoms=None, allow_calculation=True):
        if name not in self.implemented_properties:
            raise PropertyNotImplementedError(f"{name} property not implemented")

        self.update(atoms)

        if name not in self.results:
            if not allow_calculation:
                return None
            self.calculate(atoms=atoms)

        if name not in self.results:
            raise PropertyNotImplementedError(f"{name} property not present in results!")

        result = self.results[name]
        if isinstance(result, np.ndarray):
            result = result.copy()
        return result

    def get_potential_energy(self, atoms=None, force_consistent=True):
        return self.get_property(name="energy", atoms=atoms)
