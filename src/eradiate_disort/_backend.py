# SPDX-FileCopyrightText: 2026 Rayference
#
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import logging
from typing import Hashable

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
from eradiate.scenes.illumination import DirectionalIllumination
from eradiate.units import unit_registry as ureg

from ._measurements import (
    DisortIrradianceMeasure,
    DisortRadianceMeasure,
)
from ._phase import get_phase, get_pmom
from ._pipeline import build_disort_pipeline, compute_measures_info

logger = logging.getLogger(__name__)


@attrs.define
class EradiateDisortBackend:
    """
    Eradiate DISORT backend.

    This class implements an experimental Eradiate radiometric backend that uses
    the CDISORT implementation of the DISORT algorithm. It supports 1D scenes
    with atmospheres featuring an arbitrary number of components and can
    generally be used as a fast alternative to the Monte Carlo ray tracing
    backend on plane-parallel geometries.

    Parameters
    ----------
    nstr : int, default: 16
        Number of streams (angular discretization).

    nmom : int, default: 16
        Number of Legendre moments used to represent scattering distributions
        (phase functions and BRDFs).

    verbose : bool, default: False
        If ``False``, silence CDISORT terminal output.

    intensity_correction : {"nakajima_tanaka", "buras_emde"}, default: "nakajima_tanaka"
        Intensity correction method. ``"nakajima_tanaka"`` uses only Legendre
        moments and is always available. ``"buras_emde"`` additionally requires
        the actual phase function values and is more accurate for sharply peaked
        phase functions (e.g. particles/aerosols).
    """

    nstr: int = attrs.field(default=16, repr=False)
    nmom: int = attrs.field(default=16, repr=False)
    intensity_correction: str = attrs.field(
        default="nakajima_tanaka",
        validator=attrs.validators.in_(["nakajima_tanaka", "buras_emde"]),
        repr=False,
    )
    verbose: bool = attrs.field(default=False, repr=False)
    _state: nd.DisortState = attrs.field(factory=nd.DisortState, repr=False)
    _results: dict[Hashable, dict] = attrs.field(factory=dict, repr=False)
    _measures_info: list[dict] = attrs.field(factory=list, repr=False)
    _geometry: dict = attrs.field(factory=dict, repr=False)
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
                f"EradiateDisortBackend requires a DirectionalIllumination, "
                f"got {type(exp.illumination).__name__}"
            )

        # Measures: only DisortRadianceMeasure and DisortIrradianceMeasure
        allowed_measure_types = (DisortRadianceMeasure, DisortIrradianceMeasure)
        for measure in exp.measures:
            if not isinstance(measure, allowed_measure_types):
                raise TypeError(
                    f"EradiateDisortBackend requires DisortRadianceMeasure or "
                    f"DisortIrradianceMeasure instances, got "
                    f"{type(measure).__name__}"
                )

        # At most one DisortRadianceMeasure (DISORT has a single umu/phi grid)
        radiance_measures = [
            m for m in exp.measures if isinstance(m, DisortRadianceMeasure)
        ]
        if len(radiance_measures) > 1:
            raise TypeError(
                "EradiateDisortBackend supports at most one DisortRadianceMeasure "
                f"per run (found {len(radiance_measures)})"
            )

        # Surface: only Lambertian (diffuse) BSDF is supported
        if not isinstance(exp.surface.bsdf, LambertianBSDF):
            raise TypeError(
                f"EradiateDisortBackend requires a Lambertian surface BSDF, "
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
    ) -> None:
        """
        Perform global setup that does not depend on the spectral dimension.
        Called once at the beginning of :meth:`.process`.
        """
        logger.debug("EradiateDisortBackend: Global setup")
        ds = self._state

        # Classify active measures
        measures = list(exp.measures)
        radiance_measures = [
            m for m in measures if isinstance(m, DisortRadianceMeasure)
        ]
        irradiance_measures = [
            m for m in measures if isinstance(m, DisortIrradianceMeasure)
        ]
        has_radiance = len(radiance_measures) > 0
        has_fluxes = len(irradiance_measures) > 0

        self._has_radiance = has_radiance
        self._has_fluxes = has_fluxes
        self._active_measures = measures

        # Illumination angles
        illumination = exp.illumination
        ill_mu = np.cos(illumination.zenith.m_as("rad"))
        ill_phi = illumination.azimuth.m_as("deg")

        # Control flags
        ds.quiet = not self.verbose
        ds.usrtau = True
        ds.lamber = True
        ds.planck = False
        ds.usrang = has_radiance
        ds.onlyfl = not has_radiance

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
            rad_measure = radiance_measures[0]
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
            self._mes_mu = mes_mu
            self._mes_phi = mes_phi
        else:
            # DISORT needs at least one umu/phi even with onlyfl=True
            ds.numu = 1
            ds.nphi = 1
            self._mes_mu = np.array([1.0])  # nadir
            self._mes_phi = np.array([0.0])

        # ntau is determined from the number of unique user-requested levels
        # across all measures. We compute a rough count here assuming BOA+TOA
        # defaults, then refine in _setup_spectral where tau is known.
        # The actual ntau is set in _setup_spectral on the first call, before
        # allocate().
        self._exp = exp  # stash for _setup_spectral access
        self._ill_mu = ill_mu
        self._ill_phi = ill_phi

    def _setup_spectral(
        self, exp: AtmosphereExperiment, ctx: KernelContext, first_call: bool = False
    ) -> None:
        """
        Perform setup that depends on the spectral dimension.
        Called at each iteration of the spectral loop.
        """
        logger.debug("EradiateDisortBackend: Spectral loop setup")
        ds = self._state
        atmosphere = exp.atmosphere

        # --- Atmospheric optical properties ---
        if atmosphere is not None:
            h = atmosphere.geometry.zgrid.layer_height
            sigma_t = atmosphere.eval_sigma_t(ctx.si)
            tau_btt = np.atleast_1d((sigma_t * h).m_as("dimensionless"))

            _dither = 100.0 * np.finfo(float).eps
            ssalb = np.atleast_1d(atmosphere.eval_albedo(ctx.si).m_as("dimensionless"))
            ssalb = np.minimum(ssalb, 1.0 - _dither)

            ds.dtauc = tau_btt[::-1]  # DISORT expects top-to-bottom
            ds.ssalb = ssalb[::-1]
            zgrid = atmosphere.geometry.zgrid
        else:
            tau_btt = np.array([0.0])
            ds.dtauc = tau_btt
            ds.ssalb = tau_btt
            # Minimal two-level zgrid for altitude resolution when no atmosphere
            from eradiate.radprops import ZGrid

            zgrid = ZGrid([0.0, 1.0] * ureg.m)

        # --- Phase function ---
        ds.pmom = get_pmom(atmosphere, ds.nmom, ctx)[:, ::-1]

        # Buras-Emde phase function values
        if self.intensity_correction == "buras_emde" and atmosphere is not None:
            mu_grid, phase_tbt = get_phase(atmosphere, ds.nstr, ctx)
            _eps = 1e-10
            mu_padded = np.concatenate([[-1.0 - _eps], mu_grid, [1.0 + _eps]])
            phase_padded = np.hstack([phase_tbt[:, :1], phase_tbt, phase_tbt[:, -1:]])
            ds.mu_phase = mu_padded
            ds.phase = np.ascontiguousarray(phase_padded[::-1, :])

        # --- Merged utau from all measures ---
        merged_utau, measures_info = compute_measures_info(
            self._active_measures,
            tau_btt,
            zgrid,
            self._mes_mu,
            self._mes_phi,
        )

        if first_call:
            # ntau can only be set before allocate(); lock it on the first call
            ds.ntau = len(merged_utau)
            ds.allocate()
            # Now that memory is allocated, set viewing angles
            if self._has_radiance:
                ds.umu = self._mes_mu
                ds.phi = self._mes_phi
            else:
                ds.umu = self._mes_mu
                ds.phi = self._mes_phi

        ds.utau = merged_utau
        self._measures_info = measures_info

        # --- Illumination & surface ---
        irradiance = exp.illumination.irradiance.eval(ctx.si).m_as("W/m^2/nm")
        ds.fbeam = irradiance
        ds.umu0 = self._ill_mu
        ds.phi0 = self._ill_phi

        albedo = exp.surface.bsdf.reflectance.eval(ctx.si).m_as("dimensionless")
        ds.albedo = albedo
        ds.fisot = 0.0
        ds.fluor = 0.0

    def _solve(self) -> None:
        """Run the DISORT solver."""
        logger.debug("EradiateDisortBackend: Running DISORT solver")
        self._state.solve()

    def _collect_results(self) -> dict:
        """Collect all relevant outputs from the DISORT state."""
        ds = self._state
        return {
            "uu": np.array(ds.uu) if self._has_radiance else None,
            "rfldir": np.array(ds.rfldir),
            "rfldn": np.array(ds.rfldn),
            "flup": np.array(ds.flup),
            "dfdt": np.array(ds.dfdt),
            "uavg": np.array(ds.uavg),
            "uavgdn": np.array(ds.uavgdn),
            "uavgup": np.array(ds.uavgup),
            "uavgso": np.array(ds.uavgso),
        }

    def _get_contexts(
        self, exp: AtmosphereExperiment, measure_idx: int
    ) -> list[KernelContext]:
        sis = list(exp.spectral_indices(measure_idx))
        return [KernelContext(si) for si in sis]

    def process(
        self, exp: AtmosphereExperiment, measure: None | int | str = None
    ) -> None:
        """
        Run the processing step for a given Experiment configuration.

        Parameters
        ----------
        exp : AtmosphereExperiment
            Processed experiment configuration.

        measure : int or str, optional
            Index or string ID of the processed measure. If unset, defaults to
            the first measure defined in the experiment configuration.
        """
        exp.init()

        if measure is None:
            measure = exp.measures[0]
        else:
            measure = exp.measures.resolve(measure)

        measure_idx = exp.measures.get_index(measure.id)
        ctxs = self._get_contexts(exp, measure_idx)

        ref_ctx = ctxs[0] if ctxs else None
        self._setup_global(exp, ref_ctx=ref_ctx)

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
                self._setup_spectral(exp, ctx, first_call=(i == 0))
                self._solve()
                results[ctx.si.as_hashable] = self._collect_results()
                # Update measures_info from the last spectral call for postprocessing
                self._last_measures_info = self._measures_info
                pbar.update()

        self._results = results

    def postprocess(
        self, exp: AtmosphereExperiment, measure: None | int | str = None
    ) -> xr.DataTree:
        """
        Run the postprocessing step and return a DataTree.

        Parameters
        ----------
        exp : AtmosphereExperiment
            Processed experiment configuration.

        measure : int or str, optional
            Index or string ID of the processed measure.

        Returns
        -------
        DataTree
            One subtree per measure, keyed by ``/{measure.id}/``.
        """
        mode = eradiate.get_mode()

        if measure is None:
            measure = exp.measures[0]
        else:
            measure = exp.measures.resolve(measure)
        measure_idx = exp.measures.get_index(measure.id)

        if mode.is_mono:
            mode_id = "mono"
        elif mode.is_ckd:
            mode_id = "ckd"
        else:
            raise UnsupportedModeError

        geometry = {
            "umu": self._state.umu,
            "phi": self._state.phi,
            "umu0": self._state.umu0,
            "phi0": self._state.phi0,
        }

        pipeline = build_disort_pipeline()
        result = pipeline.execute(
            outputs=["datatree"],
            inputs={
                "raw_results": self._results,
                "mode": mode_id,
                "spectral_grid": exp.spectral_grids[measure_idx],
                "ckd_quads": exp.ckd_quads[measure_idx],
                "geometry": geometry,
                "measures_info": self._last_measures_info,
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
            Index or string ID of the processed measure.

        Returns
        -------
        DataTree
            Post-processed results, one subtree per measure.
        """
        self.validate(exp)
        self.process(exp, measure=measure)
        return self.postprocess(exp, measure=measure)
