# SPDX-FileCopyrightText: 2026 Rayference
#
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr


class XarrayRegressionFixture:
    """
    Regression helper for xarray DataArray and Dataset objects.

    Converts the xarray object to a flat dict of NumPy arrays (data values
    plus all coordinates, prefixed with ``coord_``) and delegates to
    ``ndarrays_regression`` from pytest-regressions.

    The mapping is deterministic:

    * **DataArray** → ``{"<name>": values, "coord_<dim>": coord_values, …}``
      where ``<name>`` is the DataArray's ``.name`` attribute (or ``"data"``
      when unnamed).
    * **Dataset** → ``{"<var>": values, …, "coord_<coord>": coord_values, …}``

    Use ``--force-regen`` (from pytest-regressions) to regenerate reference
    files in bulk.
    """

    def __init__(self, ndarrays_regression) -> None:
        self._ndarrays = ndarrays_regression

    @staticmethod
    def _to_arrays(data_object) -> dict[str, np.ndarray]:
        """Convert a DataArray or Dataset to a flat ``{name: ndarray}`` dict."""
        if isinstance(data_object, xr.DataArray):
            name = data_object.name or "data"
            arrays: dict[str, np.ndarray] = {str(name): np.asarray(data_object)}
            for coord_name, coord in data_object.coords.items():
                arrays[f"coord_{coord_name}"] = np.asarray(coord)
        elif isinstance(data_object, xr.Dataset):
            arrays = {}
            for var_name in data_object.data_vars:
                arrays[str(var_name)] = np.asarray(data_object[var_name])
            for coord_name, coord in data_object.coords.items():
                arrays[f"coord_{coord_name}"] = np.asarray(coord)
        else:
            raise TypeError(
                f"xarray_regression.check() expects a DataArray or Dataset, "
                f"got {type(data_object).__name__}"
            )
        return arrays

    def check(
        self,
        data_object: xr.DataArray | xr.Dataset,
        default_tolerance: dict | None = None,
        tolerances: dict[str, dict] | None = None,
    ) -> None:
        """
        Compare *data_object* against stored reference data.

        On first run (or when ``--force-regen`` is passed) the reference is
        saved as an ``.npz`` file next to the test module.  On subsequent runs
        the stored reference is loaded and compared via ``numpy.allclose``.

        Parameters
        ----------
        data_object:
            Result to compare.  Must be a DataArray or Dataset.
        default_tolerance:
            Tolerance applied to every array unless overridden by *tolerances*.
            Example: ``{"atol": 1e-4, "rtol": 1e-4}``.
        tolerances:
            Per-array tolerance overrides, keyed by the array name as it
            appears in the flattened dict (e.g. ``{"brf": {"atol": 1e-5},
            "coord_vza": {"atol": 0}}``).
        """
        arrays = self._to_arrays(data_object)
        kwargs: dict = {}
        if default_tolerance is not None:
            kwargs["default_tolerance"] = default_tolerance
        if tolerances is not None:
            kwargs["tolerances"] = tolerances
        self._ndarrays.check(arrays, **kwargs)


@pytest.fixture
def xarray_regression(ndarrays_regression):
    """
    Regression fixture for xarray DataArray and Dataset objects.

    Wraps ``ndarrays_regression`` from pytest-regressions.  Converts xarray
    objects to a flat dict of NumPy arrays and delegates comparison and
    reference-file management to the underlying fixture.

    Examples
    --------
    ::

        def test_radiance(xarray_regression, mode_mono):
            result = run_experiment()
            xarray_regression.check(
                result.brf,
                default_tolerance=dict(atol=1e-4, rtol=1e-4),
            )

    Pass ``--force-regen`` on the command line to regenerate all reference
    files in one run.
    """
    return XarrayRegressionFixture(ndarrays_regression)
