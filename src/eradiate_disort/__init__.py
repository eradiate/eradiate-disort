from __future__ import annotations

import logging
from typing import Hashable

import attrs
import nanodisort as nd
import numpy as np
import tqdm.auto as tqdm
import xarray as xr
from eradiate.experiments import AtmosphereExperiment


@attrs.define
class EradiateDisortBackend:
    nstr: int = attrs.field(default=8, repr=False)
    nmom: int = attrs.field(default=8, repr=False)
    _state: nd.DisortState = attrs.field(factory=nd.DisortState, repr=False)
    _results: dict[Hashable, xr.DataArray] = attrs.field(factory=dict, repr=False)

    def _setup(self, exp: AtmosphereExperiment) -> None:
        ds = self._state

        # Set control flags
        ds.usrtau = True  # User optical depths
        ds.usrang = False  # No user angles
        ds.lamber = True  # Lambertian bottom boundary
        ds.planck = False  # No thermal emission
        ds.onlyfl = True  # Only fluxes (no intensities)
        ds.quiet = True  # Suppress output

        # Set dimensions
        ds.nstr = self.nstr  # Number of streams
        ds.nmom = self.nmom  # Phase function moments
        ds.nlyr = 1  # Single layer
        ds.ntau = 2  # Output at boundaries (top and bottom)
        ds.numu = 0  # No user polar angles (only fluxes)
        ds.nphi = 0  # No azimuthal angles

    def process(
        self,
        exp: AtmosphereExperiment,
        measures: None | int | str | list[int | str] = None,
    ) -> None:
        # Normalize list of processed measures
        if measures is None:
            measures = exp.measures
        else:
            if isinstance(measures, (int, str)):
                measures = [measures]
            measures = [exp.measures.resolve(i) for i in measures]

        # Get all spectral loop indices
        measure_idxs = [exp.measures.get_index(measure.id) for measure in measures]
        ctxs = exp.contexts(measure_idxs)

        # Set up DISORT state for this computation
        logging.info("Setting up DISORT backend")

        # Run DISORT sequence through the spectral loop
        ds = self._state

        with tqdm.tqdm(desc="Spectral loop", total=len(ctxs)) as pbar:
            for ctx in ctxs:
                # Allocate memory
                ds.allocate()

                # Set optical properties
                # Single layer with optical depth 1.0, pure scattering
                ds.dtauc = np.array([1.0])
                ds.ssalb = np.array(
                    [1.0]
                )  # Single scattering albedo = 1 (no absorption)

                # Isotropic phase function (all moments = 0 except first)
                pmom = np.zeros((ds.nmom + 1, ds.nlyr))
                pmom[0, 0] = 1.0  # Normalization
                ds.pmom = pmom

                # Set output optical depths (boundaries)
                ds.utau = np.array([0.0, 1.0])

                # Set beam parameters
                ds.fbeam = np.pi  # Incident beam flux
                ds.umu0 = 1.0  # Normal incidence (cos(0) = 1)
                ds.phi0 = 0.0  # Azimuth angle

                # Bottom boundary albedo
                ds.albedo = 0.0  # No reflection from bottom

                # Other boundary conditions
                ds.fisot = 0.0  # No isotropic top illumination
                ds.fluor = 0.0  # No bottom illumination

                # Run solver
                ds.solve()

                pbar.update()

        # Store results in array
