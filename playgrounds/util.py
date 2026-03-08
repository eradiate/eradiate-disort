from __future__ import annotations

import attrs
import xarray as xr


@attrs.define
class Result:
    disort: xr.DataArray | None = None
    mitsuba: xr.DataArray | None = None


def reshape_pplane(da: xr.DataArray) -> xr.DataArray:
    """
    Reshape and reindex a DataArray that contains data in the principal plane.
    """
    result = da.stack(i=("x_index", "y_index")).drop_vars(("i", "x_index", "y_index"))
    mask_negative = result["vaa"] == 0.0
    mask_nadir = result["vza"] == 0.0
    neg = result.where(mask_negative & ~mask_nadir).dropna("i")
    neg["vza"] *= -1.0
    pos = result.where(~mask_negative).dropna("i")
    result = xr.concat((neg, pos), dim="i").sortby("vza")
    result = result.squeeze()
    return result
