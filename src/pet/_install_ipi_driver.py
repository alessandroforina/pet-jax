"""Helper to install the PET driver into i-PI."""

import shutil
from importlib.util import find_spec
from pathlib import Path


def install_ipi_driver():
    """Copy the installed `pet.py` into the i-PI `pes` directory."""

    ipi_spec = find_spec("ipi")
    if ipi_spec is None or not ipi_spec.submodule_search_locations:
        raise RuntimeError(
            "i-PI installation not found. Install i-PI and rerun "
            "`pet-install-ipi-driver`.",
        )
    pes_dir = Path(ipi_spec.submodule_search_locations[0]) / "pes"

    source_spec = find_spec("pet.ipi")
    if source_spec is None or source_spec.origin is None:
        raise FileNotFoundError("Could not locate pet.ipi module")
    source_path = Path(source_spec.origin)

    target_path = pes_dir / "pet.py"
    shutil.copy(source_path, target_path)
    print(f"Installed PET driver to {target_path}")


if __name__ == "__main__":
    install_ipi_driver()
