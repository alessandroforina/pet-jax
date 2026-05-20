"""PET driver for i-PI."""

try:
    from ipi.pes._ase import ASEDriver
except ImportError:
    from ipi.pes.ase import ASEDriver

from pet.calculator import Calculator

__DRIVER_NAME__ = "pet"
__DRIVER_CLASS__ = "PET_driver"


class PET_driver(ASEDriver):
    def __init__(self, template, model_path, *args, **kwargs):
        self.model_path = model_path
        super().__init__(template, *args, **kwargs)

    def check_parameters(self):
        super().check_parameters()
        has_stress = "stress" in self.capabilities
        self.ase_calculator = Calculator.from_checkpoint(
            self.model_path, stress=has_stress
        )
