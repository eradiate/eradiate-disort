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
from eradiate.scenes.phase import IsotropicPhaseFunction, RayleighPhaseFunction
from eradiate.units import unit_registry as ureg
from nanodisort.utils import phase_functions as pf

from .io import normalize_metadata

logger = logging.getLogger(__name__)


@attrs.define
class EradiateDisortBackend:
    # This is a prototype interface for a more general Backend class. Backends
    # are responsible for checking their input (both internal state and
    # additional parameters provided by an Experiment object) through the
    # validate() method. After validation is performed, the computation can
    # start safely using the run() method, which chains the processing and
    # post-processing steps.

    # By default, coupled nstr and nmom values
    # nmom depends on the discretization of the phase function
    # More than 128 streams usually not needed
    # nmom = nstr + 1 by default, must be greater than nstr
    nstr: int = attrs.field(default=16, repr=False)  # TODO: Discuss default
    nmom: int = attrs.field(default=16, repr=False)  # TODO: Discuss default
    verbose: bool = attrs.field(default=False, repr=False)
    _state: nd.DisortState = attrs.field(factory=nd.DisortState, repr=False)
    _results: dict[Hashable, xr.DataArray] = attrs.field(factory=dict, repr=False)
    _name: str = "CDISORT"

    def validate(self, exp: AtmosphereExperiment):
        """
        Check consistency between self parameters and experiment configuration.
        """
        pass

    def _setup_global(self, exp: AtmosphereExperiment) -> None:
        """
        Perform global setup operations that do not depend on the spectral
        dimension. This method is called once at the beginning of the process()
        method.
        """
        logger.debug("EradiateDisortBackend: Global setup")
        ds = self._state

        # Collect sensor data
        mes = exp.measures[0]
        mes_angles = mes.direction_layout.angles
        mask = mes_angles[:, 0] < 0
        mes_angles[mask, 0] *= -1.0
        mes_angles[mask, 1] += 180.0 * ureg.deg
        mes_angles[:, 1] %= 360.0 * ureg.deg
        mes_mu = np.sort(np.unique(np.cos(mes_angles[:, 0].m_as("rad"))))
        mes_phi = np.sort(np.unique(mes_angles[:, 1].m_as("deg")))

        # Collect illumination data
        illumination = exp.illumination
        ill_mu = np.cos(illumination.zenith.m_as("rad"))
        ill_phi = illumination.azimuth.m_as("deg")

        # Control flags (except thermal emission, see below)
        ds.quiet = not self.verbose  # Suppress output
        ds.usrtau = True  # Return radiant quantities at user-specified optical depths
        ds.usrang = True  # Return radiant quantities at user-specified polar angles
        ds.lamber = True  # Lambertian bottom boundary
        ds.onlyfl = False  # Return intensity in addition to fluxes
        ds.planck = False  # No thermal emission

        # Set dimensions
        ds.nlyr = (
            exp.atmosphere.geometry.zgrid.n_layers if exp.atmosphere is not None else 1
        )
        ds.nstr = self.nstr  # Number of streams
        ds.nmom = self.nmom  # Phase function moments

        ds.ntau = 2  # Output at boundaries (top and bottom)
        ds.numu = len(mes_mu)  # User polar angles
        ds.nphi = len(mes_phi)  # User azimuthal angles

        # Allocate dynamic memory buffers
        ds.allocate()

        # Thermal emission
        # ds.temper = ...  # Temperature at each level (needed only if planck is True)

        # Set beam parameters
        ds.umu0 = ill_mu  # Zenith angle
        ds.phi0 = ill_phi  # Azimuth angle

        # Set sensor parameters
        # DISORT requires umu strictly ascending; sort unique positive cosines
        ds.umu = mes_mu
        ds.phi = mes_phi  # DISORT expects azimuth in degrees

    def _setup_spectral(self, exp: AtmosphereExperiment, ctx: KernelContext):
        """
        Perform setup operations that depend on the spectral dimension. This
        method is called multiple times in the process() method (at each
        iteration of the spectral loop).
        """
        logger.debug("EradiateDisortBackend: Spectral loop setup")
        ds = self._state
        atmosphere = exp.atmosphere

        # Set atmospheric properties
        if atmosphere is not None:
            h = atmosphere.geometry.zgrid.layer_height
            sigma_t = atmosphere.eval_sigma_t(ctx.si)
            tau = np.atleast_1d((sigma_t * h).m_as("dimensionless"))
            ssalb = np.atleast_1d(atmosphere.eval_albedo(ctx.si).m_as("dimensionless"))
            ds.dtauc = tau[::-1]
            ds.ssalb = ssalb[::-1]
        else:
            tau = np.array([0])
            ds.dtauc = tau
            ds.ssalb = tau

        # Phase function setup
        phase = atmosphere.phase if atmosphere is not None else None

        if phase is None or isinstance(phase, IsotropicPhaseFunction):
            pmom_1d = pf.isotropic(ds.nmom)
        elif isinstance(phase, RayleighPhaseFunction):
            pmom_1d = pf.rayleigh(ds.nmom)
        else:
            raise NotImplementedError(
                f"Phase function type {type(phase).__name__!r} is not supported by "
                "EradiateDisortBackend"
            )

        ds.pmom = np.tile(pmom_1d.reshape(-1, 1), (1, ds.nlyr))

        # Illumination setup
        irradiance = exp.illumination.irradiance.eval(ctx.si).m_as("W/m^2/nm")
        ds.fbeam = irradiance  # Incident beam flux

        # Bottom boundary albedo
        albedo = exp.surface.bsdf.reflectance.eval(ctx.si).m_as("dimensionless")
        ds.albedo = albedo  # Reflection from bottom

        # Other boundary conditions
        ds.fisot = 0.0  # No isotropic top illumination
        ds.fluor = 0.0  # No bottom illumination

        # Output optical depths (boundaries)
        ds.utau = np.array([0, np.sum(tau)])

    def _solve(self):
        """Run the DISORT solver."""
        logger.debug("EradiateDisortBackend: Running DISORT solver")
        self._state.solve()

    def process(
        self, exp: AtmosphereExperiment, measure: None | int | str = None
    ) -> None:
        # Normalize list of processed measures
        if measure is None:
            measure = exp.measures[0]
        else:
            measure = exp.measures.resolve(measure)

        # Perform global setup
        self._setup_global(exp)

        # Get all spectral loop indices
        measure_idx = exp.measures.get_index(measure.id)
        ctxs = exp.contexts(measure_idx)

        # Run the spectral loop
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
            for ctx in ctxs:
                pbar.set_description(
                    f"Spectral loop — {self._name} [{ctx.index_formatted}]"
                )
                self._setup_spectral(exp, ctx)
                self._solve()
                results[ctx.si.as_hashable] = np.array(self._state.uu)
                pbar.update()

        # Store results in array
        self._results = results

    def postprocess(
        self, exp: AtmosphereExperiment, measure: None | int | str = None
    ) -> xr.DataArray:
        mode = eradiate.get_mode()

        if measure is None:
            measure = exp.measures[0]
        else:
            measure = exp.measures.resolve(measure)
        measure_idx = exp.measures.get_index(measure.id)

        if mode.is_mono:
            result = self._postprocess_mono(exp, measure_idx)
        elif mode.is_ckd:
            result = self._postprocess_ckd(exp, measure_idx)
        else:
            raise UnsupportedModeError

        return result

    def _postprocess_mono(
        self, exp: AtmosphereExperiment, measure_idx: int
    ) -> xr.DataArray:
        results = self._results
        w_coords = np.sort(np.unique(list(results.keys())))

        value_shape = next(iter(results.values())).shape
        dense_shape = (len(w_coords),) + value_shape
        dense_data = np.full(dense_shape, np.nan)

        w_idx = {x: i for i, x in enumerate(w_coords)}

        keys = list(results.keys())
        i_indices = np.array([w_idx[k] for k in keys])
        values = np.array(list(results.values()))

        dense_data[i_indices] = values

        # Basic data assembly
        mes_mu = self._state.umu
        mes_theta = np.rad2deg(np.arccos(mes_mu))
        mes_phi = self._state.phi
        mes_tau = self._state.utau
        vza, vaa = np.meshgrid(mes_theta, mes_phi, indexing="ij")

        result = xr.DataArray(
            dense_data,
            coords={
                "w": ("w", w_coords),
                "y_index": ("y_index", np.arange(len(mes_theta))),
                "tau": ("tau", mes_tau),
                "x_index": ("x_index", np.arange(len(mes_phi))),
                "vza": (("y_index", "x_index"), vza),
                "vaa": (("y_index", "x_index"), vaa),
            },
            dims=["w", "y_index", "tau", "x_index"],
            name="radiance_raw",
        )

        # Drop unused taus, add solar angles
        ill_mu = self._state.umu0
        ill_theta = np.rad2deg(np.arccos(ill_mu))
        ill_phi = self._state.phi0

        result = result.isel(tau=0, drop=True)  # tau = 0 means TOA
        result = result.expand_dims(("saa", "sza"), axis=(-2, -1)).assign_coords(
            {"saa": ("saa", [ill_phi]), "sza": [ill_theta]}
        )

        # Apply metadata
        result = normalize_metadata(result)
        return result

    def _postprocess_ckd(
        self, exp: AtmosphereExperiment, measure_idx: int
    ) -> xr.DataArray:
        from eradiate.pipelines import logic as pplogic

        results = self._results
        w_coords = np.sort(np.unique(list(k[0] for k in results.keys())))
        g_coords = np.sort(np.unique(list(k[1] for k in results.keys())))

        value_shape = next(iter(results.values())).shape
        dense_shape = (len(w_coords), len(g_coords)) + value_shape
        dense_data = np.full(dense_shape, np.nan)

        w_idx = {x: i for i, x in enumerate(w_coords)}
        g_idx = {x: i for i, x in enumerate(g_coords)}

        keys = list(results.keys())
        i_indices = np.array([w_idx[k[0]] for k in keys])
        j_indices = np.array([g_idx[k[1]] for k in keys])
        values = np.array(list(results.values()))

        dense_data[i_indices, j_indices] = values

        # Basic data assembly
        mes_mu = self._state.umu
        mes_theta = np.rad2deg(np.arccos(mes_mu))
        mes_phi = self._state.phi
        mes_tau = self._state.utau
        vza, vaa = np.meshgrid(mes_theta, mes_phi, indexing="ij")

        result = xr.DataArray(
            dense_data,
            coords={
                "w": ("w", w_coords),
                "g": ("g", g_coords),
                "y_index": ("y_index", np.arange(len(mes_theta))),
                "tau": ("tau", mes_tau),
                "x_index": ("x_index", np.arange(len(mes_phi))),
                "vza": (("y_index", "x_index"), vza),
                "vaa": (("y_index", "x_index"), vaa),
            },
            dims=["w", "g", "y_index", "tau", "x_index"],
            name="radiance_raw",
        )

        # Drop unused taus, add solar angles
        ill_mu = self._state.umu0
        ill_theta = np.rad2deg(np.arccos(ill_mu))
        ill_phi = self._state.phi0

        result = result.isel(tau=0, drop=True)  # tau = 0 means TOA
        result = result.expand_dims(("saa", "sza"), axis=(-2, -1)).assign_coords(
            {"saa": ("saa", [ill_phi]), "sza": [ill_theta]}
        )

        # Apply metadata
        result = normalize_metadata(result)

        # Aggregate the CKD data
        spectral_grid = exp.spectral_grids[measure_idx]
        ckd_quads = exp.ckd_quads[measure_idx]

        result = pplogic.aggregate_ckd_quad(
            raw_data=result,
            mode_id="ckd",
            spectral_grid=spectral_grid,
            ckd_quads=ckd_quads,
            is_variance=False,
        )

        return result

    def run(
        self, exp: AtmosphereExperiment, measure: None | int | str = None
    ) -> xr.DataArray:
        self.process(exp, measure=measure)
        return self.postprocess(exp, measure=measure)
