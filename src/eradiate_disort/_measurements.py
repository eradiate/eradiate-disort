# SPDX-FileCopyrightText: 2026 Rayference
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
DISORT-specific measure type for Eradiate experiments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import attrs
import numpy as np
import pint
from eradiate import unit_registry as ureg
from eradiate.scenes.measure import (
    AzimuthRingLayout,
    GridLayout,
    HemispherePlaneLayout,
    Measure,
    measure_factory,
)
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from eradiate.radprops import ZGrid


def _extract_kwargs(kwargs: dict, keys: list[str]) -> dict:
    """Pop and return named keys from a kwargs dict (mutates in-place)."""
    return {key: kwargs.pop(key) for key in keys if key in kwargs}


def _utau_from_spec(
    z_levels: pint.Quantity | None,
    utau: np.ndarray | None,
    tau_btt: np.ndarray,
    zgrid: ZGrid,
) -> tuple[np.ndarray, pint.Quantity]:
    """
    Resolve a user altitude or optical-depth specification to DISORT utau values.

    Cumulative optical depth (utau) is measured from the top of the atmosphere
    (TOA), so utau = 0 at TOA and utau = total_optical_depth at BOA.

    Parameters
    ----------
    z_levels : pint.Quantity or None
        User-specified output altitudes. Each value is snapped to the nearest
        zgrid level boundary. Mutually exclusive with ``utau``.

    utau : array-like or None
        Direct optical-depth specification (from TOA). Mutually exclusive with
        ``z_levels``. If both are ``None``, return the default ``[0.0, total_tau]``
        pair (TOA and BOA).

    tau_btt : ndarray
        Per-layer optical depths in bottom-to-top order, shape ``(n_layers,)``.

    zgrid : ZGrid
        Atmospheric altitude grid.

    Returns
    -------
    utau_values : ndarray
        Sorted ascending utau values for DISORT, shape ``(n,)``.

    level_altitudes : pint.Quantity
        Altitudes corresponding to each utau value, shape ``(n,)``.

    Notes
    -----
    This measurement adheres to DISORT conventions. This is a deliberate
    choice made to simplify debugging and to make the effort of
    transitioning from another DISORT-powered RTM easier.
    """
    # Cumulative tau from TOA at each level boundary, indexed top-to-bottom
    # tau_cumsum_tob[0] = 0 (TOA), tau_cumsum_tob[-1] = total tau (BOA)
    tau_cumsum_tob = np.concatenate([[0.0], np.cumsum(tau_btt[::-1])])
    # Reindex to bottom-to-top to match zgrid.levels ordering
    # tau_at_levels[0] = total tau (ground), tau_at_levels[-1] = 0 (TOA)
    tau_at_levels = tau_cumsum_tob[::-1]
    z_levels_grid = zgrid.levels  # bottom-to-top Quantity, shape (n_levels,)
    n_levels = len(z_levels_grid)

    if z_levels is not None:
        z_m = z_levels.m_as("m")
        z_grid_m = z_levels_grid.m_as("m")
        # Snap each altitude to the nearest level boundary
        idxs = np.clip(
            np.round(np.interp(z_m, z_grid_m, np.arange(n_levels))).astype(int),
            0,
            n_levels - 1,
        )
        utau_values = tau_at_levels[idxs]
        alt_values = z_levels_grid[idxs]

    elif utau is not None:
        utau_values = np.asarray(utau, dtype=float)
        # Interpolate altitudes from tau values for coordinate labelling.
        # tau_at_levels is decreasing → reverse to get ascending xp for np.interp.
        alt_m = np.interp(
            utau_values,
            tau_at_levels[::-1],  # ascending: 0 (TOA) … total (BOA)
            z_levels_grid.m_as("m")[::-1],  # descending: z_TOA … z_ground
        )
        alt_values = alt_m * z_levels_grid.u

    else:
        # Default: TOA and BOA
        utau_values = np.array([0.0, tau_cumsum_tob[-1]])
        alt_values = z_levels_grid[[-1, 0]]  # [z_TOA, z_ground]

    # Sort ascending (DISORT requirement)
    order = np.argsort(utau_values)

    return utau_values[order], alt_values[order]


def _validate_direction_layout(instance, attribute, value):
    if value is not None and not isinstance(
        value, (HemispherePlaneLayout, AzimuthRingLayout, GridLayout)
    ):
        raise TypeError(
            f"DisortMeasure.direction_layout must be a HemispherePlaneLayout, "
            f"AzimuthRingLayout, or GridLayout (or None for flux-only mode); "
            f"got {type(value).__name__}"
        )


@attrs.define(eq=False, slots=False)
class DisortMeasure(Measure):
    """
    DISORT measurement [``disort``].

    Records irradiance (flux) and intensity quantities at user-specified
    altitudes or optical depths. When a direction layout is specified, also
    records the full spectral radiance field ``uu(umu, utau, phi)``.

    When no ``direction_layout`` is given (the default), DISORT runs with
    ``onlyfl = True``, skipping angular radiance computation for speed.

    The following irradiance and intensity quantities are always recorded:

    - ``rfldir``: direct-beam downward irradiance
    - ``rfldn``: diffuse downward irradiance
    - ``flup``: diffuse upward irradiance
    - ``dfdt``: flux divergence d(net flux)/d(tau)
    - ``uavg``: intensity (direct + diffuse)
    - ``uavgdn``: diffuse downward intensity
    - ``uavgup``: diffuse upward intensity
    - ``uavgso``: direct-beam intensity

    Parameters
    ----------
    direction_layout : HemispherePlaneLayout or AzimuthRingLayout or GridLayout, \
                       optional
        Viewing direction specification. Only structured layouts are accepted:

        - :class:`~eradiate.scenes.measure.HemispherePlaneLayout` (hemisphere plane cut)
        - :class:`~eradiate.scenes.measure.AzimuthRingLayout` (azimuth ring)
        - :class:`~eradiate.scenes.measure.GridLayout` (zenith × azimuth grid)

        Defaults to ``None`` (flux-only mode). Use the :meth:`hplane`,
        :meth:`aring`, or :meth:`grid` class-method constructors for convenience.

    z_levels : quantity, optional
        Output altitudes. Each value is snapped to the nearest zgrid level
        boundary. Mutually exclusive with ``utau``. If neither is set,
        defaults to TOA and BOA.

    utau : array-like, optional
        Output optical depths from TOA. Mutually exclusive with ``z_levels``.
        If neither is set, defaults to TOA and BOA.

    Notes
    -----
    The ``kernel_type`` and ``template`` properties are not implemented;
    this measure type is only usable with the DISORT backend.
    """

    direction_layout: HemispherePlaneLayout | AzimuthRingLayout | GridLayout | None = (
        attrs.field(
            kw_only=True,
            default=None,
            validator=_validate_direction_layout,
        )
    )

    z_levels: pint.Quantity | None = attrs.field(kw_only=True, default=None)

    utau: np.ndarray | None = attrs.field(kw_only=True, default=None)

    def __attrs_post_init__(self):
        if self.z_levels is not None and self.utau is not None:
            raise ValueError("DisortMeasure: z_levels and utau are mutually exclusive")

    @property
    def origin(self) -> pint.Quantity:
        # Dummy origin required by the Measure interface (measure_inside_atmosphere).
        # Not used by the DISORT backend.
        return ureg.Quantity([0.0, 0.0, 0.0], "m")

    @property
    def film_resolution(self) -> tuple[int, int]:
        if self.direction_layout is None:
            raise NotImplementedError(
                "DisortMeasure in flux-only mode does not map to a Mitsuba film"
            )
        return (self.direction_layout.n_directions, 1)

    @property
    def template(self) -> dict:
        return {}

    # --------------------------------------------------------------------------
    #                          Convenience constructors
    # --------------------------------------------------------------------------

    @classmethod
    def hplane(
        cls, zeniths: ArrayLike, azimuth: float | pint.Quantity, **kwargs
    ) -> DisortMeasure:
        """
        Construct using a hemisphere-plane viewing direction layout.

        Parameters
        ----------
        zeniths : array-like
            Zenith angle values. Negative values map to the
            ``azimuth + 180°`` half-plane. Unitless values are converted to
            ``ucc['angle']``.

        azimuth : float or quantity
            Azimuth of the hemisphere plane cut.

        azimuth_convention : AzimuthConvention or str, optional
            Azimuth convention for the layout.

        **kwargs
            Forwarded to :class:`DisortMeasure`.
        """
        layout = HemispherePlaneLayout(
            zeniths=zeniths,
            azimuth=azimuth,
            **_extract_kwargs(kwargs, ["azimuth_convention"]),
        )
        return cls(direction_layout=layout, **kwargs)

    @classmethod
    def aring(
        cls, zenith: float | pint.Quantity, azimuths: ArrayLike, **kwargs
    ) -> DisortMeasure:
        """
        Construct using an azimuth-ring viewing direction layout.

        Parameters
        ----------
        zenith : float or quantity
            Ring zenith angle. Unitless values are converted to ``ucc['angle']``.

        azimuths : array-like
            Azimuth values. Unitless values are converted to ``ucc['angle']``.

        azimuth_convention : AzimuthConvention or str, optional
            Azimuth convention for the layout.

        **kwargs
            Forwarded to :class:`DisortMeasure`.
        """
        layout = AzimuthRingLayout(
            zenith=zenith,
            azimuths=azimuths,
            **_extract_kwargs(kwargs, ["azimuth_convention"]),
        )
        return cls(direction_layout=layout, **kwargs)

    @classmethod
    def grid(cls, zeniths: ArrayLike, azimuths: ArrayLike, **kwargs) -> DisortMeasure:
        """
        Construct using a gridded (Cartesian product) viewing direction layout.

        Parameters
        ----------
        zeniths : array-like
            Zenith values.

        azimuths : array-like
            Azimuth values.

        azimuth_convention : AzimuthConvention or str, optional
            Azimuth convention for the layout.

        **kwargs
            Forwarded to :class:`DisortMeasure`.
        """
        layout = GridLayout(
            zeniths=zeniths,
            azimuths=azimuths,
            **_extract_kwargs(kwargs, ["azimuth_convention"]),
        )
        return cls(direction_layout=layout, **kwargs)


measure_factory.register(DisortMeasure, type_id="disort", aliases=["disort_measure"])
