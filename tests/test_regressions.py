"""Tests for the xarray_regression fixture helper."""

import numpy as np
import pytest
import xarray as xr

from eradiate_disort.testing.regressions import XarrayRegressionFixture

pytestmark = pytest.mark.order(2)


class TestToArrays:
    def test_unnamed_dataarray(self):
        da = xr.DataArray(
            np.array([1.0, 2.0, 3.0]),
            dims=["x"],
            coords={"x": [10.0, 20.0, 30.0]},
        )
        arrays = XarrayRegressionFixture._to_arrays(da)
        assert set(arrays.keys()) == {"data", "coord_x"}

    def test_named_dataarray_uses_name_as_key(self):
        # Named arrays must not fall back to the generic "data" key
        da = xr.DataArray(
            np.array([1.0, 2.0]),
            dims=["vza"],
            coords={"vza": [-30.0, 30.0]},
            name="brf",
        )
        arrays = XarrayRegressionFixture._to_arrays(da)
        assert "brf" in arrays
        assert "data" not in arrays

    def test_multiple_coords(self):
        da = xr.DataArray(
            np.ones((3, 2)),
            dims=["vza", "vaa"],
            coords={"vza": [0.0, 30.0, 60.0], "vaa": [0.0, 180.0]},
            name="radiance",
        )
        arrays = XarrayRegressionFixture._to_arrays(da)
        assert set(arrays.keys()) == {"radiance", "coord_vza", "coord_vaa"}

    def test_dataset(self):
        ds = xr.Dataset(
            {
                "brf": (["vza"], np.array([0.1, 0.2, 0.3])),
                "irradiance": (["vza"], np.array([1.0, 1.0, 1.0])),
            },
            coords={"vza": [-30.0, 0.0, 30.0]},
        )
        arrays = XarrayRegressionFixture._to_arrays(ds)
        assert set(arrays.keys()) == {"brf", "irradiance", "coord_vza"}

    def test_no_coords(self):
        arrays = XarrayRegressionFixture._to_arrays(xr.DataArray(np.array([1.0, 2.0])))
        assert "data" in arrays

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError, match="DataArray or Dataset"):
            XarrayRegressionFixture._to_arrays(np.array([1, 2, 3]))


class TestXarrayRegressionFixture:
    # These tests write reference files on first run and compare on subsequent
    # runs (pytest-regressions behaviour). Re-generate by deleting the files
    # under tests/test_regressions/.

    def test_check_dataarray(self, xarray_regression):
        da = xr.DataArray(
            np.array([1.0, 2.0, 3.0]),
            dims=["x"],
            coords={"x": [0.0, 1.0, 2.0]},
            name="signal",
        )
        xarray_regression.check(da, default_tolerance={"atol": 1e-6, "rtol": 1e-6})

    def test_check_dataset(self, xarray_regression):
        ds = xr.Dataset(
            {"flux": (["z"], np.array([1.0, 0.9, 0.8]))},
            coords={"z": [0.0, 500.0, 1000.0]},
        )
        xarray_regression.check(ds, default_tolerance={"atol": 1e-6})
