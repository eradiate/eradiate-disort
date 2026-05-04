import numpy as np
import pytest
from eradiate.scenes.measure import (
    AngleLayout,
    AzimuthRingLayout,
    GridLayout,
    HemispherePlaneLayout,
    measure_factory,
)

import eradiate_disort as ed


class TestDisortMeasure:
    def test_construct_flux_only(self, mode_mono):
        m = ed.DisortMeasure()
        assert m.direction_layout is None

    def test_construct_radiance_hplane(self, mode_mono):
        m = ed.DisortMeasure.hplane(zeniths=np.linspace(-75, 75, 11), azimuth=0.0)
        assert isinstance(m.direction_layout, HemispherePlaneLayout)

    def test_construct_radiance_aring(self, mode_mono):
        m = ed.DisortMeasure.aring(zenith=30.0, azimuths=np.linspace(0, 360, 9)[:-1])
        assert isinstance(m.direction_layout, AzimuthRingLayout)

    def test_construct_radiance_grid(self, mode_mono):
        m = ed.DisortMeasure.grid(
            zeniths=np.linspace(0, 75, 4), azimuths=np.linspace(0, 270, 4)
        )
        assert isinstance(m.direction_layout, GridLayout)

    def test_factory(self, mode_mono):
        m = measure_factory.create("disort")
        assert isinstance(m, ed.DisortMeasure)

    def test_invalid_layout(self, mode_mono):
        layout = AngleLayout(angles=np.array([[30.0, 0.0]]))
        with pytest.raises(TypeError, match="HemispherePlaneLayout"):
            ed.DisortMeasure(direction_layout=layout)

    def test_z_levels_utau_exclusive(self, mode_mono):
        from eradiate.units import unit_registry as ureg

        with pytest.raises(ValueError, match="mutually exclusive"):
            ed.DisortMeasure(
                z_levels=ureg.Quantity([1000.0], "m"), utau=np.array([0.0])
            )
