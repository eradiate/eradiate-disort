from __future__ import annotations

from collections.abc import Sequence

import xarray as xr

_METADATA: dict[str, dict] = {
    "w": {
        "long_name": "wavelength",
        "standard_name": "radiation_wavelength",
        "units": "nm",
    },
    "g": {
        "long_name": "g",
        "standard_name": "g",
        "units": "dimensionless",
    },
    "vza": {
        "long_name": "viewing zenith angle",
        "standard_name": "viewing_zenith_angle",
        "units": "deg",
    },
    "vaa": {
        "long_name": "azimuth zenith angle",
        "standard_name": "azimuth_zenith_angle",
        "units": "deg",
    },
}


def normalize_metadata(
    da: xr.DataArray, vars: Sequence[str] | None = None, inplace=True
) -> xr.DataArray:
    """
    Update a DataArray's metadata with normalized values.

    Parameters
    ----------
    da : DataArray
        Data whose metadata will be normalized.

    vars : sequence of str, optional
        List of variables which will have their metadata normalized. If unset,
        all variables present in ``da`` are processed.

    inplace : bool, default: True
        If ``True``, mutate ``da``. Otherwise, process a copy.

    Returns
    -------
    DataArray
    """
    if not inplace:
        da = da.copy()

    available = set(_METADATA.keys())
    present = set(da.coords) if not vars else set(vars)
    applied = available & present

    for var in applied:
        da[var].attrs.update(_METADATA[var])

    if da.name in _METADATA:
        da.attrs.update(_METADATA[da.name])

    return da
