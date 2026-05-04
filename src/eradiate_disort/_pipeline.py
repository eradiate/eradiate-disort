# SPDX-FileCopyrightText: 2026 Rayference
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Post-processing pipeline for the DISORT backend.

The pipeline is built with Eradiate's :class:`~eradiate.pipelines.engine.Pipeline`
DAG engine. It converts raw per-spectral-key DISORT output dictionaries into an
:class:`xarray.DataTree` keyed by measure ID.

Pipeline structure
------------------

Virtual inputs (injected at execution time):

- ``raw_results``: ``dict[spectral_key → dict[field → ndarray]]``
- ``mode``: Eradiate mode string (``"mono"`` or ``"ckd"``)
- ``spectral_grid``: :class:`~eradiate.spectral.grid.SpectralGrid`
- ``ckd_quads``: list of :class:`~eradiate.quad.Quad`
- ``geometry``: dict with keys ``umu``, ``phi``, ``umu0``, ``phi0`` sourced
  from the encapsulated :class:`nanodisort.DisortState` after solving
- ``measures_info``: list of per-measure metadata dicts

Computation nodes:

1. ``stacked_data``: stacks raw per-spectral dicts into DataArrays with
   spectral dimensions ``(w,)`` or ``(w, g)`` plus field-specific dimensions.

2. ``aggregated_data``: applies CKD quadrature to each field (no-op in mono).

3. ``datatree``: builds one :class:`xarray.Dataset` per measure and assembles
   the final :class:`xarray.DataTree`.
"""

from __future__ import annotations

import numpy as np
import pint
import xarray as xr
from eradiate.pipelines import logic as pplogic
from eradiate.pipelines.engine import Pipeline

# ------------------------------------------------------------------------------
#                            Field dimension metadata
# ------------------------------------------------------------------------------

# Maps raw field name → list of non-spectral dimension labels
_FIELD_DIMS: dict[str, list[str]] = {
    "uu": ["umu_idx", "utau_idx", "phi_idx"],
    "rfldir": ["utau_idx"],
    "rfldn": ["utau_idx"],
    "flup": ["utau_idx"],
    "dfdt": ["utau_idx"],
    "uavg": ["utau_idx"],
    "uavgdn": ["utau_idx"],
    "uavgup": ["utau_idx"],
    "uavgso": ["utau_idx"],
}

_FLUX_FIELDS = ["rfldir", "rfldn", "flup", "dfdt", "uavg", "uavgdn", "uavgup", "uavgso"]

_FLUX_META = {
    "rfldir": ("direct-beam downward irradiance", "W m-2 nm-1"),
    "rfldn": ("diffuse downward irradiance", "W m-2 nm-1"),
    "flup": ("diffuse upward irradiance", "W m-2 nm-1"),
    "dfdt": ("flux divergence d(net flux)/d(tau)", "dimensionless"),
    "uavg": ("mean intensity (direct + diffuse)", "W m-2 sr-1 nm-1"),
    "uavgdn": ("mean diffuse downward intensity", "W m-2 sr-1 nm-1"),
    "uavgup": ("mean diffuse upward intensity", "W m-2 sr-1 nm-1"),
    "uavgso": ("mean direct-beam intensity", "W m-2 sr-1 nm-1"),
}


# ------------------------------------------------------------------------------
#               Node 1 — stack per-spectral results into DataArrays
# ------------------------------------------------------------------------------


def stacked_data(raw_results: dict, mode: str) -> dict[str, xr.DataArray]:
    """
    Stack per-spectral-key DISORT output dicts into DataArrays.

    Parameters
    ----------
    raw_results : dict
        Mapping from spectral keys to DISORT output dicts. In mono mode,
        keys are wavelength floats; in CKD mode, keys are ``(w, g)`` tuples.

    mode : str
        Eradiate mode identifier: ``"mono"`` or ``"ckd"``.

    Returns
    -------
    dict
        Mapping from field name to DataArray with dimensions
        ``("w", <field_dims>)`` (mono) or ``("w", "g", <field_dims>)`` (CKD).
        Fields whose values are ``None`` in the raw results are omitted.
    """
    if not raw_results:
        return {}

    first = next(iter(raw_results.values()))
    present_fields = [f for f, v in first.items() if v is not None]

    if mode == "mono":
        w_coords = np.array(sorted(raw_results.keys()))
        result = {}
        for field in present_fields:
            arrays = np.stack([raw_results[w][field] for w in w_coords], axis=0)
            da = xr.DataArray(
                arrays,
                dims=["w"] + _FIELD_DIMS[field],
                name=field,
            ).assign_coords(w=w_coords)
            result[field] = da
        return result

    else:  # ckd
        keys = sorted(raw_results.keys())
        w_coords = np.array(sorted(set(k[0] for k in keys)))
        g_coords = np.array(sorted(set(k[1] for k in keys)))
        w_idx = {w: i for i, w in enumerate(w_coords)}
        g_idx = {g: i for i, g in enumerate(g_coords)}

        result = {}
        for field in present_fields:
            val_shape = np.asarray(first[field]).shape
            dense = np.full((len(w_coords), len(g_coords)) + val_shape, np.nan)
            for (w, g), d in raw_results.items():
                if d[field] is not None:
                    dense[w_idx[w], g_idx[g]] = d[field]
            da = xr.DataArray(
                dense,
                dims=["w", "g"] + _FIELD_DIMS[field],
                name=field,
            ).assign_coords(w=w_coords, g=g_coords)
            result[field] = da
        return result


# ------------------------------------------------------------------------------
#                            Node 2 — CKD aggregation
# ------------------------------------------------------------------------------


def aggregated_data(
    stacked_data: dict[str, xr.DataArray], mode: str, spectral_grid, ckd_quads
) -> dict[str, xr.DataArray]:
    """
    Apply CKD quadrature to each field; pass through unchanged in mono mode.

    Parameters
    ----------
    stacked_data : dict
        Output of :func:`stacked_data`.

    mode : str
        Eradiate mode identifier.

    spectral_grid : SpectralGrid
        Spectral grid for CKD bin definitions.

    ckd_quads : list of Quad
        Quadrature rules, one per CKD bin.

    Returns
    -------
    dict
        Same structure as ``stacked_data`` with the ``g`` dimension removed
        in CKD mode.
    """
    if mode != "ckd":
        return stacked_data

    result = {}
    for field, da in stacked_data.items():
        result[field] = pplogic.aggregate_ckd_quad(
            mode_id=mode,
            raw_data=da,
            spectral_grid=spectral_grid,
            ckd_quads=ckd_quads,
            is_variance=False,
        )
    return result


# ------------------------------------------------------------------------------
#          Node 3 — per-measure dataset builders and DataTree assembly
# ------------------------------------------------------------------------------


def _build_flux_vars(
    flux_fields: dict[str, xr.DataArray],
    utau_idxs: np.ndarray,
    altitudes: pint.Quantity,
) -> dict[str, xr.DataArray]:
    """Slice and label flux field DataArrays for a single measure."""
    data_vars = {}
    for field, da in flux_fields.items():
        sliced = da.isel(utau_idx=utau_idxs)
        sliced = sliced.assign_coords(
            utau_idx=("utau_idx", altitudes.m_as("m"))
        ).rename({"utau_idx": "z"})
        long_name, units = _FLUX_META.get(field, (field, ""))
        sliced.attrs.update({"long_name": long_name, "units": units})
        data_vars[field] = sliced
    return data_vars


def _build_radiance_dataset(
    uu: xr.DataArray,
    flux_fields: dict[str, xr.DataArray],
    measure_info: dict,
    geometry: dict,
) -> xr.Dataset:
    """
    Build an xr.Dataset for a radiance-mode DisortMeasure.

    Includes the full angular radiance field ``uu`` and all flux quantities.
    Viewing angles are sourced from ``geometry["umu"]`` and ``geometry["phi"]``,
    which are read directly from the encapsulated DisortState after solving.

    Parameters
    ----------
    uu : DataArray
        Full radiance field, dims ``(w[, g], umu_idx, utau_idx, phi_idx)``.

    flux_fields : dict
        Flux field DataArrays with ``utau_idx`` dimension.

    measure_info : dict
        Per-measure metadata: ``utau_indices``, ``altitudes``.

    geometry : dict
        Solver geometry sourced from DisortState: ``umu``, ``phi``, ``umu0``, ``phi0``.

    Returns
    -------
    Dataset
        Variables ``uu``, ``rfldir``, ``rfldn``, ``flup``, ``dfdt``,
        ``uavg``, ``uavgdn``, ``uavgup``, ``uavgso`` with appropriate dims.
        Coordinates: ``z`` [m], ``vza`` [deg], ``vaa`` [deg], ``sza`` [deg],
        ``saa`` [deg].
    """
    utau_idxs = measure_info["utau_indices"]
    altitudes: pint.Quantity = measure_info["altitudes"]
    umu: np.ndarray = geometry["umu"]
    phi: np.ndarray = geometry["phi"]
    umu0: float = geometry["umu0"]
    phi0: float = geometry["phi0"]

    # Slice radiance to this measure's utau levels
    uu_slice = uu.isel(utau_idx=utau_idxs)

    vza = np.rad2deg(np.arccos(umu))
    uu_slice = uu_slice.assign_coords(
        umu_idx=("umu_idx", vza),
        utau_idx=("utau_idx", altitudes.m_as("m")),
        phi_idx=("phi_idx", phi),
    ).rename({"umu_idx": "vza", "utau_idx": "z", "phi_idx": "vaa"})

    data_vars = {"uu": uu_slice}
    data_vars["uu"].attrs.update(
        {
            "long_name": "spectral radiance",
            "standard_name": "radiance_per_unit_wavelength",
            "units": "W m-2 sr-1 nm-1",
        }
    )

    flux_vars = _build_flux_vars(flux_fields, utau_idxs, altitudes)
    data_vars.update(flux_vars)

    ds = xr.Dataset(data_vars)
    ds = ds.assign_coords(
        z=("z", altitudes.m_as("m"), {"long_name": "altitude", "units": "m"}),
        vza=("vza", vza, {"long_name": "viewing zenith angle", "units": "deg"}),
        vaa=("vaa", phi, {"long_name": "viewing azimuth angle", "units": "deg"}),
        sza=(
            [],
            float(np.rad2deg(np.arccos(umu0))),
            {"long_name": "solar zenith angle", "units": "deg"},
        ),
        saa=([], float(phi0), {"long_name": "solar azimuth angle", "units": "deg"}),
    )
    return ds


def _build_flux_dataset(
    flux_fields: dict[str, xr.DataArray], measure_info: dict, geometry: dict
) -> xr.Dataset:
    """
    Build an xr.Dataset for a flux-only DisortMeasure.

    Parameters
    ----------
    flux_fields : dict
        Mapping of field name → DataArray with dims ``(w[, g], utau_idx)``.

    measure_info : dict
        Per-measure metadata: ``utau_indices``, ``altitudes``.

    geometry : dict
        Solver geometry sourced from DisortState: ``umu0``, ``phi0``.

    Returns
    -------
    Dataset
        Variables ``rfldir``, ``rfldn``, ``flup``, ``dfdt``, ``uavg``,
        ``uavgdn``, ``uavgup``, ``uavgso`` with dims ``(w, z)``.
        Coordinates: ``z`` [m], ``sza`` [deg], ``saa`` [deg].
    """
    utau_idxs = measure_info["utau_indices"]
    altitudes: pint.Quantity = measure_info["altitudes"]
    umu0: float = geometry["umu0"]
    phi0: float = geometry["phi0"]

    data_vars = _build_flux_vars(flux_fields, utau_idxs, altitudes)
    ds = xr.Dataset(data_vars)
    ds = ds.assign_coords(
        z=("z", altitudes.m_as("m"), {"long_name": "altitude", "units": "m"}),
        sza=(
            [],
            float(np.rad2deg(np.arccos(umu0))),
            {"long_name": "solar zenith angle", "units": "deg"},
        ),
        saa=([], float(phi0), {"long_name": "solar azimuth angle", "units": "deg"}),
    )
    return ds


def datatree(
    aggregated_data: dict[str, xr.DataArray], geometry: dict, measures_info: list[dict]
) -> xr.DataTree:
    """
    Build an :class:`xarray.DataTree` with one subtree per measure.

    Parameters
    ----------
    aggregated_data : dict
        Aggregated field DataArrays from :func:`aggregated_data`.

    geometry : dict
        Solver geometry sourced from DisortState (``umu``, ``phi``, ``umu0``, ``phi0``).

    measures_info : list of dict
        One entry per measure, each with keys:

        - ``id``: measure string ID
        - ``type``: ``"radiance"`` or ``"flux"``
        - ``utau_indices``: indices into the merged utau array
        - ``altitudes``: pint.Quantity of output altitudes

    Returns
    -------
    DataTree
        One subtree per measure, keyed by ``/{measure.id}/``.
    """
    subtrees: dict[str, xr.Dataset] = {}

    uu = aggregated_data.get("uu")
    flux_fields = {f: aggregated_data[f] for f in _FLUX_FIELDS if f in aggregated_data}

    for minfo in measures_info:
        mid = minfo["id"]
        if minfo["type"] == "radiance" and uu is not None:
            subtrees[f"/{mid}"] = _build_radiance_dataset(
                uu, flux_fields, minfo, geometry
            )
        else:
            subtrees[f"/{mid}"] = _build_flux_dataset(flux_fields, minfo, geometry)

    return xr.DataTree.from_dict(subtrees)


# ------------------------------------------------------------------------------
#                                 Pipeline builder
# ------------------------------------------------------------------------------


def build_disort_pipeline() -> Pipeline:
    """
    Build the DISORT post-processing pipeline.

    Virtual inputs that must be provided at :meth:`.Pipeline.execute` time:

    - ``raw_results`` : dict mapping spectral keys to DISORT output dicts
    - ``mode`` : Eradiate mode string (``"mono"`` or ``"ckd"``)
    - ``spectral_grid`` : :class:`~eradiate.spectral.grid.SpectralGrid`
    - ``ckd_quads`` : list of :class:`~eradiate.quad.Quad`
    - ``geometry`` : dict with ``umu``, ``phi``, ``umu0``, ``phi0`` sourced
      from the encapsulated :class:`nanodisort.DisortState` after solving
    - ``measures_info`` : list of per-measure metadata dicts

    Returns
    -------
    Pipeline
        Configured pipeline with three nodes: ``stacked_data``,
        ``aggregated_data``, ``datatree``.
    """
    p = Pipeline()

    p.add_node(
        "stacked_data",
        stacked_data,
        dependencies=["raw_results", "mode"],
        description="Stack per-spectral DISORT outputs into DataArrays",
    )

    p.add_node(
        "aggregated_data",
        aggregated_data,
        dependencies=["stacked_data", "mode", "spectral_grid", "ckd_quads"],
        description="Apply CKD quadrature (no-op in mono mode)",
    )

    p.add_node(
        "datatree",
        datatree,
        dependencies=["aggregated_data", "geometry", "measures_info"],
        description="Assemble per-measure DataTree",
    )

    return p


# ------------------------------------------------------------------------------
#       Backend helper: compute merged utau and per-measure index metadata
# ------------------------------------------------------------------------------


def compute_measures_info(
    measures: list, tau_btt: np.ndarray, zgrid
) -> tuple[np.ndarray, list[dict]]:
    """
    Compute the merged utau array and per-measure index metadata.

    This function is called once per spectral loop iteration from the backend's
    ``_setup_spectral`` method, since ``tau_btt`` (and therefore utau values)
    is spectrally varying.

    Viewing angles (``umu``, ``phi``) for radiance-mode measures are sourced
    from the encapsulated :class:`nanodisort.DisortState` via the ``geometry``
    dict passed to the post-processing pipeline, not stored here.

    Parameters
    ----------
    measures : list of DisortMeasure
        Active DISORT measures.

    tau_btt : ndarray
        Per-layer optical depths, bottom-to-top, shape ``(n_layers,)``.

    zgrid : ZGrid
        Atmospheric altitude grid.

    Returns
    -------
    merged_utau : ndarray
        Sorted, deduplicated utau values covering all measures, shape ``(ntau,)``.

    measures_info : list of dict
        One entry per measure with keys ``id``, ``type`` (``"radiance"`` or
        ``"flux"``), ``utau_indices``, ``altitudes``.
    """
    from ._measurements import _utau_from_spec

    per_measure_utau = []

    for m in measures:
        utau_vals, alts = _utau_from_spec(m.z_levels, m.utau, tau_btt, zgrid)
        per_measure_utau.append((m, utau_vals, alts))

    # Merge and deduplicate (preserve sort order)
    all_utau = np.concatenate([u for _, u, _ in per_measure_utau])
    merged_utau = np.unique(all_utau)

    # Map each measure's utau values to indices in the merged array
    measures_info = []
    for m, utau_vals, alts in per_measure_utau:
        idxs = np.searchsorted(merged_utau, utau_vals)
        measures_info.append(
            {
                "id": m.id,
                "type": "radiance" if m.direction_layout is not None else "flux",
                "utau_indices": idxs,
                "altitudes": alts,
            }
        )

    return merged_utau, measures_info
