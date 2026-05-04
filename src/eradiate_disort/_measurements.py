# SPDX-FileCopyrightText: 2026 Rayference
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
DISORT-specific measure types for Eradiate experiments.
"""

from __future__ import annotations

import attrs
import numpy as np
import pint
from eradiate import unit_registry as ureg
from eradiate.scenes.measure import (
    AngleLayout,
    AzimuthRingLayout,
    DirectionLayout,
    GridLayout,
    HemispherePlaneLayout,
    Layout,
    Measure,
    measure_factory,
)


def _extract_kwargs(kwargs: dict, keys: list[str]) -> dict:
    """Pop and return named keys from a kwargs dict (mutates in-place)."""
    return {key: kwargs.pop(key) for key in keys if key in kwargs}


def _utau_from_spec(
    z_levels: pint.Quantity | None, utau: np.ndarray | None, tau_btt: np.ndarray, zgrid
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
        # Snap each altitude to the nearest level boundary by linear interpolation
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


@attrs.define(eq=False, slots=False)
class DisortRadianceMeasure(Measure):
    """
    DISORT directional radiance measurement [``disort_radiance``].

    Records the full spectral radiance field ``uu(umu, utau, phi)`` at
    user-specified altitudes or optical depths.

    Viewing directions are specified with the same ``direction_layout``
    interface as :class:`~eradiate.scenes.measure.MultiDistantMeasure`, and all
    convenience class-method constructors are available.

    Parameters
    ----------
    direction_layout : Layout or array-like or dict, optional
        Viewing direction specification. Accepts the same forms as
        :class:`~eradiate.scenes.measure.MultiDistantMeasure`:

        - a :class:`~eradiate.scenes.measure.Layout` instance;
        - a (N, 2) array → :class:`~eradiate.scenes.measure.AngleLayout`;
        - a (N, 3) array → :class:`~eradiate.scenes.measure.DirectionLayout`;
        - a dict ``{"type": ..., **kwargs}``.

        Defaults to nadir (straight down).

    z_levels : quantity, optional
        Output altitudes. Each value is snapped to the nearest zgrid level
        boundary. Mutually exclusive with ``utau``. If neither is set,
        defaults to TOA only (``utau = [0.0]``).

    utau : array-like, optional
        Output optical depths from TOA. Mutually exclusive with ``z_levels``.
        If neither is set, defaults to TOA only (``[0.0]``).

    Notes
    -----
    The ``kernel_type`` and ``template`` properties are not implemented;
    this measure type is only usable with the DISORT backend.
    """

    direction_layout: Layout = attrs.field(
        kw_only=True,
        factory=lambda: DirectionLayout(directions=[0, 0, 1]),
        converter=Layout.convert,
        validator=attrs.validators.instance_of(Layout),
    )

    z_levels: pint.Quantity | None = attrs.field(kw_only=True, default=None)

    utau: np.ndarray | None = attrs.field(kw_only=True, default=None)

    def __attrs_post_init__(self):
        if self.z_levels is not None and self.utau is not None:
            raise ValueError(
                "DisortRadianceMeasure: z_levels and utau are mutually exclusive"
            )

    @property
    def origin(self) -> pint.Quantity:
        # Dummy origin required by the Measure interface (measure_inside_atmosphere).
        # Not used by the DISORT backend.
        return ureg.Quantity([0.0, 0.0, 0.0], "m")

    @property
    def film_resolution(self) -> tuple[int, int]:
        return (self.direction_layout.n_directions, 1)

    @property
    def template(self) -> dict:
        return {}

    # --------------------------------------------------------------------------
    #                          Convenience constructors
    # --------------------------------------------------------------------------

    @classmethod
    def hplane(
        cls, zeniths: np.typing.ArrayLike, azimuth: float | pint.Quantity, **kwargs
    ) -> DisortRadianceMeasure:
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
            Forwarded to :class:`DisortRadianceMeasure`.
        """
        layout = HemispherePlaneLayout(
            zeniths=zeniths,
            azimuth=azimuth,
            **_extract_kwargs(kwargs, ["azimuth_convention"]),
        )
        return cls(direction_layout=layout, **kwargs)

    @classmethod
    def aring(
        cls, zenith: float | pint.Quantity, azimuths: np.typing.ArrayLike, **kwargs
    ) -> DisortRadianceMeasure:
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
            Forwarded to :class:`DisortRadianceMeasure`.
        """
        layout = AzimuthRingLayout(
            zenith=zenith,
            azimuths=azimuths,
            **_extract_kwargs(kwargs, ["azimuth_convention"]),
        )
        return cls(direction_layout=layout, **kwargs)

    @classmethod
    def grid(
        cls, zeniths: np.typing.ArrayLike, azimuths: np.typing.ArrayLike, **kwargs
    ) -> DisortRadianceMeasure:
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
            Forwarded to :class:`DisortRadianceMeasure`.
        """
        layout = GridLayout(
            zeniths=zeniths,
            azimuths=azimuths,
            **_extract_kwargs(kwargs, ["azimuth_convention"]),
        )
        return cls(direction_layout=layout, **kwargs)

    @classmethod
    def from_angles(
        cls, angles: np.typing.ArrayLike, **kwargs
    ) -> DisortRadianceMeasure:
        """
        Construct from explicit (zenith, azimuth) pairs.

        Parameters
        ----------
        angles : array-like
            (N, 2) array of (zenith, azimuth) pairs.
        azimuth_convention : AzimuthConvention or str, optional
            Azimuth convention for the layout.
        **kwargs
            Forwarded to :class:`DisortRadianceMeasure`.
        """
        layout = AngleLayout(
            angles=angles,
            **_extract_kwargs(kwargs, ["azimuth_convention"]),
        )
        return cls(direction_layout=layout, **kwargs)

    @classmethod
    def from_directions(
        cls, directions: np.typing.ArrayLike, **kwargs
    ) -> DisortRadianceMeasure:
        """
        Construct from explicit outward-pointing direction vectors.

        Parameters
        ----------
        directions : array-like
            (N, 3) array of direction vectors (pointing outward from target).
        azimuth_convention : AzimuthConvention or str, optional
            Azimuth convention for the layout.
        **kwargs
            Forwarded to :class:`DisortRadianceMeasure`.
        """
        layout = DirectionLayout(
            directions=directions,
            **_extract_kwargs(kwargs, ["azimuth_convention"]),
        )
        return cls(direction_layout=layout, **kwargs)


@attrs.define(eq=False, slots=False)
class DisortIrradianceMeasure(Measure):
    """
    DISORT irradiance and mean-intensity measurement [``disort_irradiance``].

    Records irradiance (flux) and mean-intensity quantities at user-specified
    altitudes or optical depths:

    - ``rfldir``: direct-beam downward irradiance
    - ``rfldn``: diffuse downward irradiance
    - ``flup``: diffuse upward irradiance
    - ``dfdt``: flux divergence d(net flux)/d(tau)
    - ``uavg``: mean intensity (direct + diffuse)
    - ``uavgdn``: mean diffuse downward intensity
    - ``uavgup``: mean diffuse upward intensity
    - ``uavgso``: mean direct-beam intensity

    When this is the only active measure type, DISORT runs with
    ``onlyfl = True``, which skips angular radiance computation for speed.

    Parameters
    ----------
    z_levels : quantity, optional
        Output altitudes. Each value is snapped to the nearest zgrid level
        boundary. Mutually exclusive with ``utau``. If neither is set,
        defaults to TOA and BOA.

    utau : array-like, optional
        Output optical depths from TOA. Mutually exclusive with ``z_levels``.
        If neither is set, defaults to TOA and BOA.

    Notes
    -----
    The ``kernel_type``, ``template``, and ``film_resolution`` properties are
    not implemented; this measure type is only usable with the DISORT backend.
    """

    z_levels: pint.Quantity | None = attrs.field(kw_only=True, default=None)

    utau: np.ndarray | None = attrs.field(kw_only=True, default=None)

    def __attrs_post_init__(self):
        if self.z_levels is not None and self.utau is not None:
            raise ValueError(
                "DisortIrradianceMeasure: z_levels and utau are mutually exclusive"
            )

    @property
    def origin(self) -> pint.Quantity:
        # Dummy origin required by the Measure interface (measure_inside_atmosphere).
        # Not used by the DISORT backend.
        return ureg.Quantity([0.0, 0.0, 0.0], "m")

    @property
    def film_resolution(self) -> tuple[int, int]:
        raise NotImplementedError(
            "DisortIrradianceMeasure does not map to a Mitsuba film"
        )


measure_factory.register(
    DisortRadianceMeasure, type_id="disort_radiance", aliases=["disoradiance"]
)
measure_factory.register(
    DisortIrradianceMeasure, type_id="disort_irradiance", aliases=["disoflux"]
)
