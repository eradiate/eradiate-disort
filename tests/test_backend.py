# SPDX-FileCopyrightText: 2026 Rayference
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Integration tests for :class:`eradiate_disort.DisortBackend`.

The backend translates an Eradiate ``AtmosphereExperiment`` into the
encapsulated :class:`nanodisort.DisortState` it solves. These tests check that
translation (``_setup_global`` / ``_setup_spectral``), the ``validate`` guards,
and a handful of end-to-end runs that previously crashed.
"""

from __future__ import annotations

import numpy as np
import pytest
from eradiate.experiments import AtmosphereExperiment
from eradiate.scenes.bsdfs import RPVBSDF
from eradiate.scenes.illumination import ConstantIllumination
from eradiate.units import unit_registry as ureg

import eradiate_disort as ed
from eradiate_disort._phase import get_phase

pytestmark = pytest.mark.order(1)


# ------------------------------------------------------------------------------
#                                   Helpers
# ------------------------------------------------------------------------------


def plane_parallel(zgrid, toa_altitude) -> dict:
    """Plane-parallel geometry dict for the given altitude grid and TOA."""
    return {
        "type": "plane_parallel",
        "toa_altitude": toa_altitude,
        "zgrid": zgrid,
    }


def spherical_shell(zgrid, toa_altitude, **kwargs) -> dict:
    """Spherical-shell geometry dict for the given altitude grid and TOA."""
    return {
        "type": "spherical_shell",
        "toa_altitude": toa_altitude,
        "zgrid": zgrid,
        **kwargs,
    }


def make_exp(**overrides) -> AtmosphereExperiment:
    """Build a minimal valid DISORT experiment, overriding selected fields."""
    cfg = {
        "geometry": plane_parallel(np.linspace(0, 1, 3) * ureg.km, 1.0 * ureg.km),
        "surface": {"type": "lambertian", "reflectance": 0.3},
        "atmosphere": {"type": "homogeneous"},
        "illumination": {"type": "directional", "zenith": 30.0, "azimuth": 0.0},
        "measures": {"type": "disort"},
    }
    cfg.update(overrides)
    return AtmosphereExperiment(**cfg)


def configure_state(backend: ed.DisortBackend, exp: AtmosphereExperiment):
    """
    Drive the backend's setup steps (global + first spectral) without solving.

    Solving is skipped on purpose: a flux-only ``solve()`` makes CDISORT
    overwrite ``numu`` with ``nstr``, which would mask the configured value.
    The returned state therefore reflects exactly what the experiment maps to.
    """
    exp.init()
    ctxs = backend._get_contexts(exp, 0)
    run_ctx = backend._setup_global(exp, ref_ctx=ctxs[0])
    backend._setup_spectral(exp, ctxs[0], run_ctx, first_call=True)
    return backend._state, run_ctx, ctxs[0]


# ------------------------------------------------------------------------------
#                       Experiment -> DisortState mapping
# ------------------------------------------------------------------------------


class TestStateConfiguration:
    @pytest.mark.parametrize("n_levels", [2, 3, 11])
    def test_layer_count(self, mode_mono, n_levels):
        exp = make_exp(
            geometry=plane_parallel(
                np.linspace(0, 1, n_levels) * ureg.km, 1.0 * ureg.km
            )
        )
        state, _, _ = configure_state(ed.DisortBackend(), exp)
        assert state.nlyr == n_levels - 1

    def test_angular_resolution(self, mode_mono):
        backend = ed.DisortBackend(nstr=8, nmom=12)
        state, _, _ = configure_state(backend, make_exp())
        assert state.nstr == 8
        assert state.nmom == 12

    def test_static_flags(self, mode_mono):
        state, _, _ = configure_state(ed.DisortBackend(verbose=False), make_exp())
        assert state.lamber is True
        assert state.planck is False
        assert state.usrtau is True
        assert state.quiet is True
        assert state.spher is False  # plane-parallel geometry

    def test_spherical_shell_flags(self, mode_mono):
        # A spherical-shell geometry enables the pseudo-spherical correction
        exp = make_exp(
            geometry=spherical_shell(np.linspace(0, 100, 11) * ureg.km, 100.0 * ureg.km)
        )
        state, _, _ = configure_state(ed.DisortBackend(), exp)
        assert state.spher is True
        assert state.radius == pytest.approx(exp.geometry.planet_radius.m_as("km"))

        # Level heights above the ground, top-to-bottom, in the unit of radius
        zd = np.array(state.zd)
        assert len(zd) == 11
        assert zd[0] == pytest.approx(100.0)
        assert zd[-1] == 0.0  # nanodisort checks this exactly
        assert np.all(np.diff(zd) < 0.0)

    def test_spherical_shell_ground_altitude(self, mode_mono):
        # zd = 0 sits on the ground surface, whose radius includes the ground
        # altitude
        exp = make_exp(
            geometry=spherical_shell(
                np.linspace(1, 11, 11) * ureg.km,
                11.0 * ureg.km,
                ground_altitude=1.0 * ureg.km,
            )
        )
        state, _, _ = configure_state(ed.DisortBackend(), exp)
        assert state.radius == pytest.approx(
            exp.geometry.planet_radius.m_as("km") + 1.0
        )
        zd = np.array(state.zd)
        assert zd[0] == pytest.approx(10.0)
        assert zd[-1] == 0.0

    def test_verbose_controls_quiet(self, mode_mono):
        state, _, _ = configure_state(ed.DisortBackend(verbose=True), make_exp())
        assert state.quiet is False

    def test_flux_only_mode(self, mode_mono):
        # No direction layout -> flux-only solve, single dummy umu/phi
        state, _, _ = configure_state(ed.DisortBackend(), make_exp())
        assert state.onlyfl is True
        assert state.usrang is False
        assert state.numu == 1
        assert state.nphi == 1

    def test_radiance_mode_dimensions(self, mode_mono):
        zeniths = [-30.0, 0.0, 30.0]
        exp = make_exp(
            measures={
                "type": "disort",
                "construct": "hplane",
                "azimuth": 0.0,
                "zeniths": zeniths,
            }
        )
        state, run_ctx, _ = configure_state(ed.DisortBackend(), exp)
        assert state.onlyfl is False
        assert state.usrang is True
        # numu/nphi follow the unique viewing cosines / azimuths
        assert state.numu == len(run_ctx["mes_mu"])
        assert state.nphi == len(run_ctx["mes_phi"])
        assert state.numu == len(np.unique(np.cos(np.deg2rad(zeniths))))

    def test_illumination_mapping(self, mode_mono):
        # Non-trivial azimuth exercises the source -> travel-direction shift
        exp = make_exp(
            illumination={"type": "directional", "zenith": 40.0, "azimuth": 160.0}
        )
        state, _, _ = configure_state(ed.DisortBackend(), exp)
        assert state.umu0 == pytest.approx(np.cos(np.deg2rad(40.0)))
        assert state.phi0 == pytest.approx((160.0 + 180.0) % 360.0)

    def test_surface_and_source_mapping(self, mode_mono):
        exp = make_exp(surface={"type": "lambertian", "reflectance": 0.42})
        state, _, ctx = configure_state(ed.DisortBackend(), exp)
        assert state.albedo == pytest.approx(0.42)
        assert state.fisot == 0.0
        assert state.fluor == 0.0
        expected_fbeam = exp.illumination.irradiance.eval(ctx.si).m_as("W/m^2/nm")
        assert state.fbeam == pytest.approx(float(expected_fbeam))

    @pytest.mark.parametrize(
        "method, old_ic",
        [("buras_emde", False), ("nakajima_tanaka", True)],
    )
    def test_intensity_correction(self, mode_mono, method, old_ic):
        backend = ed.DisortBackend(nstr=8, nmom=8, intensity_correction=method)
        exp = make_exp()
        state, _, ctx = configure_state(backend, exp)
        assert state.intensity_correction is True
        assert state.old_intensity_correction is old_ic
        if method == "buras_emde":
            mu_grid, _ = get_phase(exp.atmosphere, backend.nstr, ctx)
            assert state.nphase == len(mu_grid) + 2
        else:
            assert state.nphase == 0

    def test_utau_default_toa_boa(self, mode_mono):
        state, _, _ = configure_state(ed.DisortBackend(), make_exp())
        utau = np.array(state.utau)
        assert utau.shape == (2,)
        assert utau[0] == pytest.approx(0.0)  # TOA
        assert utau[-1] > 0.0  # BOA: total optical depth
        assert state.ntau == 2

    def test_utau_from_z_levels(self, mode_mono):
        exp = make_exp(
            geometry=plane_parallel(np.linspace(0, 1, 11) * ureg.km, 1.0 * ureg.km),
            measures={"type": "disort", "z_levels": [0.0, 0.5, 1.0] * ureg.km},
        )
        state, _, _ = configure_state(ed.DisortBackend(), exp)
        assert state.ntau == 3
        assert np.array(state.utau)[0] == pytest.approx(0.0)

    def test_utau_explicit(self, mode_mono):
        exp = make_exp(measures={"type": "disort", "utau": [0.0, 0.02, 0.05]})
        state, _, _ = configure_state(ed.DisortBackend(), exp)
        assert np.allclose(np.array(state.utau), [0.0, 0.02, 0.05])
        assert state.ntau == 3


# ------------------------------------------------------------------------------
#                              validate() guards
# ------------------------------------------------------------------------------


class TestValidation:
    def test_accepts_valid_experiment(self, mode_mono):
        ed.DisortBackend().validate(make_exp())  # must not raise

    def test_rejects_non_directional_illumination(self, mode_mono):
        exp = make_exp()
        exp.illumination = ConstantIllumination()
        with pytest.raises(TypeError):
            ed.DisortBackend().validate(exp)

    def test_rejects_non_disort_measure(self, mode_mono):
        exp = make_exp(
            measures={
                "type": "mdistant",
                "construct": "hplane",
                "azimuth": 0.0,
                "zeniths": [0.0],
            }
        )
        with pytest.raises(TypeError):
            ed.DisortBackend().validate(exp)

    def test_rejects_multiple_radiance_measures(self, mode_mono):
        exp = make_exp(
            measures=[
                {
                    "id": "a",
                    "type": "disort",
                    "construct": "hplane",
                    "azimuth": 0.0,
                    "zeniths": [0.0],
                },
                {
                    "id": "b",
                    "type": "disort",
                    "construct": "hplane",
                    "azimuth": 0.0,
                    "zeniths": [10.0],
                },
            ]
        )
        with pytest.raises(TypeError):
            ed.DisortBackend().validate(exp)

    def test_rejects_non_lambertian_surface(self, mode_mono):
        exp = make_exp()
        exp.surface.bsdf = RPVBSDF()
        with pytest.raises(TypeError):
            ed.DisortBackend().validate(exp)


# ------------------------------------------------------------------------------
#                       End-to-end runs (regression guards)
# ------------------------------------------------------------------------------

_FLUX_FIELDS = ["rfldir", "rfldn", "flup", "dfdt", "uavg", "uavgdn", "uavgup", "uavgso"]


class TestRun:
    def test_flux_only_ckd_runs(self, mode_ckd):
        """
        Flux-only measure over a multi-bin CKD grid.

        Reproduces a failure detected when building example_09_fluxes.py:
        a usrang=False solve makes CDISORT overwrite numu, which crashes the
        second spectral iteration before the fix.
        """
        srf = {"type": "uniform", "wmin": 600.0, "wmax": 610.0}
        exp = AtmosphereExperiment(
            geometry=plane_parallel(np.arange(0, 121, 1) * ureg.km, 120.0 * ureg.km),
            surface={"type": "lambertian", "reflectance": 0.5},
            atmosphere={
                "type": "heterogeneous",
                "molecular_atmosphere": {"absorption_data": "mycena"},
            },
            illumination={"type": "directional", "zenith": 30.0, "azimuth": 0.0},
            measures={"type": "disort", "srf": srf},
        )
        result = ed.DisortBackend().run(exp)
        ds = result["measure"].ds
        assert set(_FLUX_FIELDS) <= set(ds.data_vars)
        assert set(ds["flup"].dims) == {"w", "z"}
        assert np.all(np.isfinite(ds["flup"].values))

    def test_homogeneous_multilayer_runs(self, mode_mono):
        """
        Homogeneous atmosphere on a multi-layer grid.

        Guards the scalar-optical-property broadcast fix: previously crashed
        with a dtauc size mismatch (length-1 vs nlyr).
        """
        exp = make_exp(
            geometry=plane_parallel(np.linspace(0, 100, 11) * ureg.km, 100.0 * ureg.km),
        )
        result = ed.DisortBackend().run(exp)
        ds = result["measure"].ds
        assert set(_FLUX_FIELDS) <= set(ds.data_vars)
        assert np.all(np.isfinite(ds["flup"].values))

    def test_flux_matches_between_flux_only_and_radiance(self, mode_mono):
        """Enabling a radiance layout must not perturb the flux quantities."""
        geometry = plane_parallel(np.linspace(0, 100, 11) * ureg.km, 100.0 * ureg.km)
        common = {
            "geometry": geometry,
            "surface": {"type": "lambertian", "reflectance": 0.3},
            "atmosphere": {"type": "homogeneous"},
            "illumination": {"type": "directional", "zenith": 30.0, "azimuth": 0.0},
        }
        backend = ed.DisortBackend(nstr=8, nmom=8)

        exp_flux = AtmosphereExperiment(measures={"type": "disort"}, **common)
        flux = backend.run(exp_flux)["measure"].ds

        exp_rad = AtmosphereExperiment(
            measures={
                "type": "disort",
                "construct": "hplane",
                "azimuth": 0.0,
                "zeniths": [0.0, 30.0],
            },
            **common,
        )
        rad = ed.DisortBackend(nstr=8, nmom=8).run(exp_rad)["measure"].ds

        for field in ("rfldir", "rfldn", "flup"):
            np.testing.assert_allclose(
                flux[field].values, rad[field].values, rtol=1e-6, atol=1e-9
            )

    def test_sza_equals_quadpoint(self):
        """
        Using an SZA value that matches an angular quadrature point should not
        crash the backend.
        """
        sza = 65.90299907 * ureg.deg
        exp = AtmosphereExperiment(
            geometry=plane_parallel(
                np.linspace(0, 100.0, 11) * ureg.km, 100.0 * ureg.km
            ),
            surface={"type": "lambertian", "reflectance": 0.5},
            atmosphere={"type": "molecular"},
            illumination={"type": "directional", "zenith": sza, "azimuth": 0.0},
            measures={"type": "disort"},
        )

        backend = ed.DisortBackend(nstr=16)
        assert backend.run(exp) is not None


# ------------------------------------------------------------------------------
#                                DATA CONSISTENCY
# ------------------------------------------------------------------------------


class TestOutput:
    def test_output_illumination(self, mode_ckd):
        """
        Check if the illumination geometry mentioned in the output is consistent
        with Eradiate's conventions.
        """
        sza = 30.0
        saa = 180.0
        exp = AtmosphereExperiment(
            geometry=plane_parallel(np.arange(0, 120.1, 1) * ureg.km, 120.0 * ureg.km),
            surface={"type": "lambertian", "reflectance": 0.5},
            atmosphere={
                "type": "molecular",
                "absorption_data": "mycena",
            },
            illumination={"type": "directional", "zenith": sza, "azimuth": saa},
            measures={"type": "disort"},
        )
        result = ed.DisortBackend().run(exp)
        ds = result["measure"].ds
        np.testing.assert_approx_equal(ds["saa"].values, saa)
        np.testing.assert_approx_equal(ds["sza"].values, sza)


# ------------------------------------------------------------------------------
#                           Numerical regressions
# ------------------------------------------------------------------------------

_SRF = {"type": "delta", "wavelengths": [550.0]}


class TestRegression:
    """
    Pin numerical outputs to catch silent drift.

    Cases are kept small (Rayleigh-only atmosphere, single wavelength, few
    viewing angles) for fast, deterministic, compact reference files.
    Regenerate with ``pixi run test tests/test_backend.py --force-regen``.
    """

    def test_molecular_radiance(self, mode_mono, xarray_regression):
        exp = AtmosphereExperiment(
            geometry=plane_parallel(np.linspace(0, 100, 51) * ureg.km, 100.0 * ureg.km),
            surface={"type": "lambertian", "reflectance": 0.2},
            atmosphere={"type": "molecular", "has_absorption": False},
            illumination={"type": "directional", "zenith": 30.0, "azimuth": 0.0},
            measures={
                "type": "disort",
                "construct": "hplane",
                "azimuth": 0.0,
                "zeniths": [-60.0, -30.0, 0.0, 30.0, 60.0],
                "srf": _SRF,
            },
        )
        result = ed.DisortBackend(nstr=8, nmom=8).run(exp)
        xarray_regression.check(
            result["measure"].ds, default_tolerance={"atol": 1e-6, "rtol": 1e-5}
        )

    def test_homogeneous_flux(self, mode_mono, xarray_regression):
        exp = make_exp(
            geometry=plane_parallel(np.linspace(0, 100, 11) * ureg.km, 100.0 * ureg.km),
            measures={"type": "disort", "srf": _SRF},
        )
        result = ed.DisortBackend(nstr=8, nmom=8).run(exp)
        xarray_regression.check(
            result["measure"].ds, default_tolerance={"atol": 1e-6, "rtol": 1e-5}
        )
