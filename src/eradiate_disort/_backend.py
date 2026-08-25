# SPDX-FileCopyrightText: 2026 Rayference
#
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import logging
from collections.abc import Generator, Hashable
from typing import TYPE_CHECKING

import attrs
import eradiate
import nanodisort as nd
import numpy as np
import tqdm.auto as tqdm
import xarray as xr
from eradiate import KernelContext, config
from eradiate.exceptions import UnsupportedModeError
from eradiate.experiments import AtmosphereExperiment
from eradiate.scenes.atmosphere import (
    HeterogeneousAtmosphere,
    HomogeneousAtmosphere,
    MolecularAtmosphere,
    ParticleLayer,
)
from eradiate.scenes.bsdfs import LambertianBSDF
from eradiate.scenes.geometry import SphericalShellGeometry
from eradiate.scenes.illumination import DirectionalIllumination
from eradiate.units import unit_registry as ureg

from ._measurements import DisortMeasure
from ._phase import get_phase, get_pmom
from ._pipeline import build_disort_pipeline, compute_measures_info

if TYPE_CHECKING:
    from eradiate.spectral import CKDSpectralGrid, MonoSpectralGrid, SpectralIndex

logger = logging.getLogger(__name__)


@attrs.define
class DisortBackend:
    """
    Eradiate DISORT backend.

    This class implements an experimental Eradiate radiometric backend that uses
    the CDISORT implementation of the DISORT algorithm. It supports 1D scenes
    with atmospheres featuring an arbitrary number of components and can
    generally be used as a fast alternative to the Monte Carlo ray tracing
    backend. Plane-parallel and spherical-shell geometries are both supported;
    the latter activates CDISORT's pseudo-spherical correction, which applies
    the Chapman airmass to the direct beam (the diffuse field remains
    plane-parallel).

    Parameters
    ----------
    nstr : int, default: 16
        Number of streams (angular discretization).

    nmom : int, default: 16
        Number of Legendre moments used to represent scattering distributions
        (phase functions and BRDFs).

    verbose : bool, default: False
        If ``False``, silence CDISORT terminal output.

    intensity_correction : {"nakajima_tanaka", "buras_emde"}, default: "buras_emde"
        Intensity correction method. ``"nakajima_tanaka"`` uses only Legendre
        moments and is always available. ``"buras_emde"`` additionally requires
        the actual phase function values and is more accurate for sharply peaked
        phase functions.

    See Also
    --------
    nanodisort.DisortState
    """

    nstr: int = attrs.field(default=16, repr=False)
    nmom: int = attrs.field(default=16, repr=False)
    intensity_correction: str = attrs.field(
        default="buras_emde",
        validator=attrs.validators.in_(["nakajima_tanaka", "buras_emde"]),
        repr=False,
    )
    verbose: bool = attrs.field(default=False, repr=False)
    _state: nd.DisortState = attrs.field(factory=nd.DisortState, repr=False)
    _results: dict[Hashable, dict] = attrs.field(factory=dict, repr=False)
    # Populated at the end of process(); consumed by postprocess()
    _postprocess_ctx: dict = attrs.field(factory=dict, repr=False)
    _name: str = "CDISORT"

    def validate(self, exp: AtmosphereExperiment):
        """
        Check internal state consistency and compatibility with the passed
        Experiment configuration.

        Parameters
        ----------
        exp : AtmosphereExperiment
            Processed experiment configuration.

        Raises
        ------
        TypeError
            If validation fails.
        """
        # Illumination: only directional illumination is supported
        if not isinstance(exp.illumination, DirectionalIllumination):
            raise TypeError(
                "EradiateDisortBackend requires a DirectionalIllumination, "
                f"got {type(exp.illumination).__name__}"
            )

        # Measures: only DisortMeasure instances
        for measure in exp.measures:
            if not isinstance(measure, DisortMeasure):
                raise TypeError(
                    "EradiateDisortBackend requires DisortMeasure instances, "
                    f"got {type(measure).__name__}"
                )

        # At most one measure with a direction layout (DISORT has a single umu/phi grid)
        radiance_measures = [m for m in exp.measures if m.direction_layout is not None]
        if len(radiance_measures) > 1:
            raise TypeError(
                "EradiateDisortBackend supports at most one radiance-mode "
                f"DisortMeasure per run (found {len(radiance_measures)})"
            )

        # Surface: only Lambertian (diffuse) BSDF is supported
        if not isinstance(exp.surface.bsdf, LambertianBSDF):
            raise TypeError(
                "EradiateDisortBackend requires a Lambertian surface BSDF, "
                f"got {type(exp.surface.bsdf).__name__}"
            )

        # Atmosphere: only heterogeneous atmospheres are supported
        allowed = (
            MolecularAtmosphere,
            ParticleLayer,
            HeterogeneousAtmosphere,
            HomogeneousAtmosphere,
        )
        if exp.atmosphere is not None and not isinstance(exp.atmosphere, allowed):
            raise TypeError(
                "EradiateDisortBackend requires one of "
                f"[{', '.join([f'{x.__name__}' for x in allowed])}], "
                f"got {type(exp.atmosphere).__name__}"
            )

    def _setup_global(
        self, exp: AtmosphereExperiment, ref_ctx: KernelContext | None = None
    ) -> dict:
        """
        Perform global setup that does not depend on the spectral dimension.
        Called once at the beginning of :meth:`.process`.

        Parameters
        ----------
        exp : AtmosphereExperiment
            Processed experiment configuration.

        ref_ctx : KernelContext, optional
            Reference spectral context used for initialization. Required only
            when using the Buras-Emde intensity correction.

        Returns
        -------
        dict
            Run context with transient processing state. Keys:

            - ``has_radiance`` : bool
            - ``active_measures`` : list
            - ``mes_mu`` : ndarray — sorted unique cosines for DISORT
            - ``mes_phi`` : ndarray — azimuth angles [deg] for DISORT
            - ``ill_mu`` : float — illumination cosine
            - ``ill_phi`` : float — illumination azimuth [deg]
        """
        logger.debug("EradiateDisortBackend: Global setup")
        ds = self._state

        # Classify active measures
        measures = list(exp.measures)
        has_radiance = any(m.direction_layout is not None for m in measures)

        # Illumination angles
        illumination = exp.illumination
        ill_mu = np.cos(illumination.zenith.m_as("rad"))
        # DISORT rejects a beam cosine that coincides with one of its
        # computational angles. These are the double-Gauss abscissae:
        # nstr/2 Gauss-Legendre nodes mapped from [-1, 1] onto (0, 1).
        # Dither away from any colliding node; no-op otherwise, so unaffected
        # SZAs are unchanged.
        nodes = (np.polynomial.legendre.leggauss(self.nstr // 2)[0] + 1.0) / 2.0
        if np.any(np.abs(nodes - ill_mu) / np.abs(ill_mu) < 1e-4):
            ill_mu = np.clip(ill_mu * (1.0 + 2e-4), -1.0, 1.0)
        # DISORT phi0 is the azimuth of the beam's travel direction; Eradiate's
        # illumination.azimuth is the sun's source direction.
        ill_phi = (illumination.azimuth.m_as("deg") + 180.0) % 360.0

        # Control flags
        ds.quiet = not self.verbose
        ds.usrtau = True
        ds.lamber = True
        ds.planck = False
        ds.usrang = has_radiance
        ds.onlyfl = not has_radiance

        # Pseudo-spherical correction, triggered by a spherical-shell geometry
        geometry = exp.geometry
        ds.spher = isinstance(geometry, SphericalShellGeometry)
        if ds.spher:
            # zd is measured from the ground surface, whose radius is the planet
            # radius offset by the ground altitude (cf. SphereShape.surface)
            ds.radius = (geometry.planet_radius + geometry.ground_altitude).m_as(
                ureg.km
            )

        # Intensity correction method
        if self.intensity_correction == "buras_emde":
            if ref_ctx is None:
                raise RuntimeError(
                    "Buras-Emde correction requires a reference spectral context "
                    "to size the phase angle grid before allocation."
                )
            mu_grid, _ = get_phase(exp.atmosphere, self.nstr, ref_ctx)
            # +2 accounts for sentinel points added at both ends in _setup_spectral
            ds.nphase = len(mu_grid) + 2
            ds.intensity_correction = True
            ds.old_intensity_correction = False
        else:  # "nakajima_tanaka"
            ds.nphase = 0
            ds.intensity_correction = True
            ds.old_intensity_correction = True

        # Atmosphere layer count
        ds.nlyr = exp.atmosphere.geometry.zgrid.n_layers if exp.atmosphere else 1
        ds.nstr = self.nstr
        ds.nmom = self.nmom

        # Viewing angle setup (for radiance measure)
        if has_radiance:
            rad_measure = next(m for m in measures if m.direction_layout is not None)
            mes_angles = rad_measure.direction_layout.angles
            mask = mes_angles[:, 0] < 0
            mes_angles = mes_angles.copy()
            mes_angles[mask, 0] *= -1.0
            mes_angles[mask, 1] = mes_angles[mask, 1] + 180.0 * ureg.deg
            mes_angles[:, 1] %= 360.0 * ureg.deg
            mes_mu = np.sort(np.unique(np.cos(mes_angles[:, 0].m_as("rad"))))
            mes_phi = np.sort(np.unique(mes_angles[:, 1].m_as("deg")))
            ds.numu = len(mes_mu)
            ds.nphi = len(mes_phi)
        else:
            # DISORT needs at least one umu/phi even with onlyfl=True
            ds.numu = 1
            ds.nphi = 1
            mes_mu = np.array([1.0])  # nadir
            mes_phi = np.array([0.0])

        return {
            "has_radiance": has_radiance,
            "active_measures": measures,
            "mes_mu": mes_mu,
            "mes_phi": mes_phi,
            "ill_mu": ill_mu,
            "ill_phi": ill_phi,
        }

    def _setup_spectral(
        self,
        exp: AtmosphereExperiment,
        ctx: KernelContext,
        run_ctx: dict,
        first_call: bool = False,
    ) -> None:
        """
        Perform setup that depends on the spectral dimension.
        Called at each iteration of the spectral loop.

        Parameters
        ----------
        exp : AtmosphereExperiment
            Processed experiment configuration.

        ctx : KernelContext
            Current spectral context.

        run_ctx : dict
            Run context produced by :meth:`._setup_global`.

        first_call : bool
            If ``True``, allocates DISORT memory (must be called exactly once,
            before :meth:`._solve`).
        """
        logger.debug("EradiateDisortBackend: Spectral loop setup")
        ds = self._state
        atmosphere = exp.atmosphere

        # --- Compute values (no array assignments yet — memory may not be allocated)
        if atmosphere is not None:
            h = atmosphere.geometry.zgrid.layer_height
            sigma_t = atmosphere.eval_sigma_t(ctx.si)
            tau_btt = np.atleast_1d((sigma_t * h).m_as("dimensionless"))

            _dither = 100.0 * np.finfo(float).eps
            ssalb = np.atleast_1d(atmosphere.eval_albedo(ctx.si).m_as("dimensionless"))
            ssalb = np.minimum(ssalb, 1.0 - _dither)

            zgrid = atmosphere.geometry.zgrid

            # Homogeneous atmospheres return scalar optical properties; broadcast
            # them across all layers so the arrays match DISORT's nlyr.
            n_layers = zgrid.n_layers
            if tau_btt.shape[0] != n_layers:
                tau_btt = np.broadcast_to(tau_btt, (n_layers,)).copy()
            if ssalb.shape[0] != n_layers:
                ssalb = np.broadcast_to(ssalb, (n_layers,)).copy()
        else:
            tau_btt = np.array([0.0])
            ssalb = np.array([0.0])
            # Minimal two-level zgrid for altitude resolution when no atmosphere
            from eradiate.radprops import ZGrid

            zgrid = ZGrid([0.0, 1.0] * ureg.m)

        pmom = get_pmom(atmosphere, ds.nmom, ctx)[:, ::-1]

        buras_emde_arrays = None
        if self.intensity_correction == "buras_emde" and atmosphere is not None:
            mu_grid, phase_tbt = get_phase(atmosphere, ds.nstr, ctx)
            _eps = 1e-10
            mu_padded = np.concatenate([[-1.0 - _eps], mu_grid, [1.0 + _eps]])
            phase_padded = np.hstack([phase_tbt[:, :1], phase_tbt, phase_tbt[:, -1:]])
            buras_emde_arrays = (mu_padded, np.ascontiguousarray(phase_padded[::-1, :]))

        # Merged utau must be computed before allocate() so ntau is known
        merged_utau, measures_info = compute_measures_info(
            run_ctx["active_measures"],
            tau_btt,
            zgrid,
        )

        irradiance = exp.illumination.irradiance.eval(ctx.si).m_as("W/m^2/nm")
        albedo = exp.surface.bsdf.reflectance.eval(ctx.si).m_as("dimensionless")

        # --- Allocate on first call, then assign all arrays
        if first_call:
            ds.ntau = len(merged_utau)
            ds.allocate()
            # Save structural info for post-processing (fixed across spectral loop)
            self._postprocess_ctx["measures_info"] = measures_info

        if atmosphere is not None:
            ds.dtauc = tau_btt[::-1]  # DISORT expects top-to-bottom
            ds.ssalb = ssalb[::-1]
        else:
            ds.dtauc = tau_btt
            ds.ssalb = ssalb

        ds.pmom = pmom

        if buras_emde_arrays is not None:
            ds.mu_phase, ds.phase = buras_emde_arrays

        # A flux-only solve (usrang=False) makes CDISORT overwrite numu with
        # nstr internally; restore the user dimensions before re-assigning the
        # angular arrays on subsequent spectral iterations.
        ds.numu = len(run_ctx["mes_mu"])
        ds.nphi = len(run_ctx["mes_phi"])
        ds.umu = run_ctx["mes_mu"]
        ds.phi = run_ctx["mes_phi"]
        ds.utau = merged_utau

        if ds.spher:
            # Level heights above the ground, top-to-bottom, in the unit of ds.radius
            ds.zd = (zgrid.levels - zgrid.levels[0]).m_as(ureg.km)[::-1]

        ds.fbeam = irradiance
        ds.umu0 = run_ctx["ill_mu"]
        ds.phi0 = run_ctx["ill_phi"]
        ds.albedo = albedo
        ds.fisot = 0.0
        ds.fluor = 0.0

    def _solve(self) -> None:
        """Run the DISORT solver."""
        logger.debug("EradiateDisortBackend: Running DISORT solver")
        self._state.solve()

    def _collect_results(self, run_ctx: dict) -> dict:
        """Collect all relevant outputs from the DISORT state."""
        ds = self._state
        return {
            "uu": np.array(ds.uu) if run_ctx["has_radiance"] else None,
            "rfldir": np.array(ds.rfldir),
            "rfldn": np.array(ds.rfldn),
            "flup": np.array(ds.flup),
            "dfdt": np.array(ds.dfdt),
            "uavg": np.array(ds.uavg),
            "uavgdn": np.array(ds.uavgdn),
            "uavgup": np.array(ds.uavgup),
            "uavgso": np.array(ds.uavgso),
        }

    def _get_spectral_indices(
        self, exp: AtmosphereExperiment, measure_index: int
    ) -> Generator[SpectralIndex]:
        if eradiate.mode().is_mono:
            spectral_grid: MonoSpectralGrid = exp.spectral_grids[measure_index]

            def generator():
                yield from spectral_grid.walk_indices()

        elif eradiate.mode().is_ckd:
            spectral_grid: CKDSpectralGrid = exp.spectral_grids[measure_index]
            quad_config = exp.ckd_quad_config
            try:
                abs_db = exp.atmosphere.abs_db
            except (
                AttributeError
            ):  # There is either no atmosphere or no absorption database
                abs_db = None

            def generator():
                yield from spectral_grid.walk_indices(quad_config, abs_db)
        else:
            raise UnsupportedModeError

        yield from generator()

    def _get_contexts(
        self, exp: AtmosphereExperiment, measure_index: int = 0
    ) -> list[KernelContext]:
        """
        Return the list of spectral contexts for the given measure.

        Uses the spectral grid directly (no Mitsuba kernel required).
        """
        return [
            KernelContext(si) for si in self._get_spectral_indices(exp, measure_index)
        ]

    def process(
        self, exp: AtmosphereExperiment, measure: None | int | str = None
    ) -> None:
        """
        Run the processing step for a given Experiment configuration.

        All measures in the experiment are processed together in a single
        spectral loop. The spectral grid is taken from ``measure`` (default:
        the first measure).

        Parameters
        ----------
        exp : AtmosphereExperiment
            Processed experiment configuration.

        measure : int or str, optional
            Index or string ID of the measure whose spectral grid drives the
            loop. Defaults to the first measure.
        """
        exp.init()

        if measure is None:
            measure_index = 0
        else:
            m = exp.measures.resolve(measure)
            measure_index = exp.measures.get_index(m.id)

        ctxs = self._get_contexts(exp, measure_index)
        ref_ctx = ctxs[0] if ctxs else None
        run_ctx = self._setup_global(exp, ref_ctx=ref_ctx)

        results = {}
        with tqdm.tqdm(
            initial=0,
            total=len(ctxs),
            unit_scale=1.0,
            leave=True,
            bar_format="{desc}{n:g}/{total:g}|{bar}| {elapsed}, ETA={remaining}",
            disable=(config.settings.progress < config.ProgressLevel.SPECTRAL_LOOP)
            or len(ctxs) <= 1,
        ) as pbar:
            for i, ctx in enumerate(ctxs):
                pbar.set_description(
                    f"Spectral loop — {self._name} [{ctx.index_formatted}]"
                )
                self._setup_spectral(exp, ctx, run_ctx, first_call=(i == 0))
                self._solve()
                results[ctx.si.as_hashable] = self._collect_results(run_ctx)
                pbar.update()

        self._results = results

        # Store everything postprocess() needs
        self._postprocess_ctx.update(
            {
                "geometry": {
                    "umu": np.array(self._state.umu),
                    "phi": np.array(self._state.phi),
                    "umu0": float(self._state.umu0),
                    "phi0": float(self._state.phi0),
                },
                "spectral_grid": exp.spectral_grids[measure_index],
                "ckd_quads": exp.ckd_quads[measure_index],
            }
        )

    def postprocess(
        self, exp: AtmosphereExperiment, measure: None | int | str = None
    ) -> xr.DataTree:
        """
        Run the postprocessing step and return a DataTree.

        Returns a DataTree with one subtree per measure, keyed by measure ID.
        The ``measure`` argument is accepted for API compatibility but ignored;
        all measures are always included in the output.

        Parameters
        ----------
        exp : AtmosphereExperiment
            Processed experiment configuration.

        measure : int or str, optional
            Ignored. Present for API compatibility.

        Returns
        -------
        DataTree
            One subtree per measure, keyed by ``/{measure.id}/``.
        """
        mode = eradiate.get_mode()

        if mode.is_mono:
            mode_id = "mono"
        elif mode.is_ckd:
            mode_id = "ckd"
        else:
            raise UnsupportedModeError

        ctx = self._postprocess_ctx
        pipeline = build_disort_pipeline()
        result = pipeline.execute(
            outputs=["datatree"],
            inputs={
                "raw_results": self._results,
                "mode": mode_id,
                "spectral_grid": ctx["spectral_grid"],
                "ckd_quads": ctx["ckd_quads"],
                "geometry": ctx["geometry"],
                "measures_info": ctx["measures_info"],
            },
        )
        return result["datatree"]

    def run(
        self, exp: AtmosphereExperiment, measure: None | int | str = None
    ) -> xr.DataTree:
        """
        Run validation, processing, and postprocessing in sequence.

        Parameters
        ----------
        exp : AtmosphereExperiment
            Processed experiment configuration.

        measure : int or str, optional
            Index or string ID of the measure whose spectral grid drives the
            processing loop. Defaults to the first measure.

        Returns
        -------
        DataTree
            Post-processed results, one subtree per measure.
        """
        self.validate(exp)
        self.process(exp, measure=measure)
        return self.postprocess(exp, measure=measure)
