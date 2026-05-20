"""Tests for the i-PI driver module."""

import pytest


def test_ipi_module_importable():
    import pet.ipi as ipi_module

    assert hasattr(ipi_module, "__DRIVER_NAME__")
    assert hasattr(ipi_module, "__DRIVER_CLASS__")


def test_driver_name():
    from pet.ipi import __DRIVER_NAME__

    assert __DRIVER_NAME__ == "pet"


def test_driver_class_name():
    from pet.ipi import __DRIVER_CLASS__

    assert __DRIVER_CLASS__ == "PET_driver"


def test_driver_class_exists():
    from pet.ipi import PET_driver

    assert PET_driver is not None


def test_driver_subclasses_asedriver():
    """PET_driver must subclass ASEDriver for i-PI compatibility."""
    try:
        from ipi.pes._ase import ASEDriver
    except ImportError:
        from ipi.pes.ase import ASEDriver

    from pet.ipi import PET_driver

    assert issubclass(PET_driver, ASEDriver)


def test_install_script_importable():
    from pet._install_ipi_driver import install_ipi_driver

    assert callable(install_ipi_driver)
