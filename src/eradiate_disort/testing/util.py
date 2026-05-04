# SPDX-FileCopyrightText: 2026 Rayference
#
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import attrs
import xarray as xr


@attrs.define
class Result:
    disort: xr.DataArray | None = None
    mitsuba: xr.DataArray | None = None


def reshape_pplane(dt: xr.DataTree) -> xr.DataArray:
    """
    Extract and reshape principal-plane radiance from a DISORT DataTree.

    Finds the first subtree containing ``uu`` (directional radiance), then
    reconstructs a signed-zenith principal-plane DataArray by treating
    ``vaa ≈ 0°`` as the forward half-plane (positive vza) and ``vaa ≈ 180°``
    as the backward half-plane (negative vza).

    Parameters
    ----------
    dt : DataTree
        Output of :meth:`.EradiateDisortBackend.run`.

    Returns
    -------
    DataArray
        Radiance dataset indexed by signed viewing zenith angle.

    Notes
    -----
    Following Eradiate's principal plane orientation convention,
    the back-scattering half-plane maps to positive zenith angles.
    """
    # Locate the first radiance subtree
    for node in dt.children.values():
        if "uu" in node.ds:
            da = node.ds["uu"]
            break
    else:
        raise ValueError("No radiance dataset ('uu') found in DataTree")

    da = da.squeeze(drop=True)  # drop scalar dims (single z, single w)

    # vaa=0°: forward half-plane (positive vza)
    # vaa=180°: backward half-plane (sign-flip vza)
    forward = da.sel(vaa=0.0, method="nearest").drop_vars("vaa")
    backward = (
        da.sel(vaa=180.0, method="nearest")
        .drop_vars("vaa")
        .assign_coords(vza=-da["vza"])
    )

    # Drop nadir from backward to avoid a duplicate vza=0 point
    backward = backward.sel(vza=backward["vza"] < 0)

    return xr.concat([backward, forward], dim="vza").sortby("vza")
