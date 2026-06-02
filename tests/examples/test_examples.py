"""Regression tests for example notebooks."""

import numpy as np
import pytest

from eradiate_disort.testing.plotting import er_plt  # noqa: F401

pytestmark = pytest.mark.order(-1)  # negative order value: execute last


def test_01_noatmo(er_plt):  # noqa: F811
    from .example_01_noatmo import result

    np.testing.assert_allclose(result.disort, result.mitsuba)


def test_02_single_layer(er_plt):  # noqa: F811
    from .example_02_single_layer import result

    # TODO: Currently only a smoke test, add assert
    print(result)


def test_03_two_layers(er_plt):  # noqa: F811
    from .example_03_two_layers import result

    # TODO: Currently only a smoke test, add assert
    print(result)


def test_04_molecular_atmosphere(er_plt):  # noqa: F811
    from .example_04_molecular_atmosphere import result

    # TODO: Currently only a smoke test, add assert
    print(result)


def test_05_aerosols(er_plt):  # noqa: F811
    from .example_05_aerosols import result

    # TODO: Currently only a smoke test, add assert
    print(result)


def test_06_full_atmo(er_plt):  # noqa: F811
    from .example_06_full_atmo import result

    # TODO: Currently only a smoke test, add assert
    print(result)


def test_07_spectrum(er_plt):  # noqa: F811
    from .example_07_spectrum import result_disort

    # TODO: Currently only a smoke test, add assert
    print(result_disort)
