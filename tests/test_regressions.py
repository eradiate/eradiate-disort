"""Tests for the xarray_regression fixture helper."""

import numpy as np
import pytest
import xarray as xr

from eradiate_disort.testing.regressions import XarrayRegressionFixture


class TestToArrays:
    def test_dataarray_unnamed(self):
        da = xr.DataArray(
            np.array([1.0, 2.0, 3.0]),
            dims=["x"],
            coords={"x": [10.0, 20.0, 30.0]},
        )
        arrays = XarrayRegressionFixture._to_arrays(da)
        assert "data" in arrays
        assert "coord_x" in arrays
        np.testing.assert_array_equal(arrays["data"], da.values)
        np.testing.assert_array_equal(arrays["coord_x"], da.coords["x"].values)

    def test_dataarray_named(self):
        da = xr.DataArray(
            np.array([1.0, 2.0]),
            dims=["vza"],
            coords={"vza": [-30.0, 30.0]},
            name="brf",
        )
        arrays = XarrayRegressionFixture._to_arrays(da)
        assert "brf" in arrays
        assert "coord_vza" in arrays
        assert "data" not in arrays

    def test_dataarray_multiple_coords(self):
        da = xr.DataArray(
            np.ones((3, 2)),
            dims=["vza", "vaa"],
            coords={"vza": [0.0, 30.0, 60.0], "vaa": [0.0, 180.0]},
            name="radiance",
        )
        arrays = XarrayRegressionFixture._to_arrays(da)
        assert set(arrays.keys()) == {"radiance", "coord_vza", "coord_vaa"}
        assert arrays["radiance"].shape == (3, 2)

    def test_dataset(self):
        ds = xr.Dataset(
            {
                "brf": (["vza"], np.array([0.1, 0.2, 0.3])),
                "irradiance": (["vza"], np.array([1.0, 1.0, 1.0])),
            },
            coords={"vza": [-30.0, 0.0, 30.0]},
        )
        arrays = XarrayRegressionFixture._to_arrays(ds)
        assert "brf" in arrays
        assert "irradiance" in arrays
        assert "coord_vza" in arrays

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError, match="DataArray or Dataset"):
            XarrayRegressionFixture._to_arrays(np.array([1, 2, 3]))

    def test_dataarray_no_coords(self):
        da = xr.DataArray(np.array([1.0, 2.0]))
        arrays = XarrayRegressionFixture._to_arrays(da)
        assert "data" in arrays


class TestXarrayRegressionFixture:
    def test_check_creates_reference(self, xarray_regression):
        da = xr.DataArray(
            np.array([1.0, 2.0, 3.0]),
            dims=["x"],
            coords={"x": [0.0, 1.0, 2.0]},
            name="signal",
        )
        xarray_regression.check(da, default_tolerance=dict(atol=1e-6, rtol=1e-6))

    def test_check_dataset(self, xarray_regression):
        ds = xr.Dataset(
            {"flux": (["z"], np.array([1.0, 0.9, 0.8]))},
            coords={"z": [0.0, 500.0, 1000.0]},
        )
        xarray_regression.check(ds, default_tolerance=dict(atol=1e-6))
