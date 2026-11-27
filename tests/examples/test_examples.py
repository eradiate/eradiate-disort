"""Regression tests for example notebooks."""

import sys

import numpy as np
import pytest

from eradiate_disort.testing.plotting import er_plt  # noqa: F401

pytestmark = pytest.mark.order(-1)  # negative order value: execute last


def test_01_noatmo(er_plt):  # noqa: F811
    from .example_01_noatmo import result

    np.testing.assert_allclose(result["disort"], result["mitsuba"])


def test_02_single_layer(er_plt):  # noqa: F811
    from .example_02_single_layer import results

    # TODO: Currently only a smoke test, add regression assertions
    assert results


def test_03_two_layers(er_plt):  # noqa: F811
    from .example_03_two_layers import results

    # TODO: Currently only a smoke test, add regression assertions
    assert results


def test_04_molecular_atmosphere(er_plt):  # noqa: F811
    from .example_04_molecular_atmosphere import results

    # TODO: Currently only a smoke test, add regression assertions
    assert results


def test_05_aerosols(er_plt):  # noqa: F811
    from .example_05_aerosols import results

    # TODO: Currently only a smoke test, add regression assertions
    assert results


def test_06_full_atmo(er_plt):  # noqa: F811
    from .example_06_full_atmo import results

    # TODO: Currently only a smoke test, add regression assertions
    assert results


def test_07_spectrum(er_plt):  # noqa: F811
    from .example_07_spectrum import result_disort

    # TODO: Currently only a smoke test, add assert
    print(result_disort)


# shortcut: the Mitsuba spherical-shell run in this example crashes Python at
# interpreter shutdown on Windows (heap corruption in mitsuba::Shape destructor,
# eradiate-mitsuba 0.6.1); unskip once fixed upstream.
@pytest.mark.skipif(
    sys.platform == "win32",
    reason="eradiate-mitsuba crashes at interpreter shutdown on Windows",
)
def test_10_spherical(er_plt):  # noqa: F811
    from .example_10_spherical import results

    # TODO: Currently only a smoke test, add regression assertions
    assert results
