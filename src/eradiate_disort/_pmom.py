# SPDX-FileCopyrightText: 2026 Rayference
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Functions to collect Legendre moments and phase function data for DISORT.
"""

from __future__ import annotations

import numpy as np
from eradiate.contexts import KernelContext
from eradiate.scenes.atmosphere import HeterogeneousAtmosphere
from eradiate.scenes.phase import (
    IsotropicPhaseFunction,
    ParticlePhaseFunction,
    PhaseFunction,
    RayleighPhaseFunction,
)
from eradiate.units import unit_context_config as ucc
from nanodisort.utils import phase_functions as pf


def _get_pmom_particle_layer(
    phase: ParticlePhaseFunction, nmom: int, ctx: KernelContext
) -> np.ndarray:
    pmom_raw = phase.eval_pmom(ctx.si, phamat=0)
    # The particle data stores (2l+1)*f_l; DISORT expects f_l directly.
    # Divide by (2l+1) to convert to DISORT convention.
    factors = 2 * np.arange(len(pmom_raw)) + 1
    pmom_raw = pmom_raw / factors
    # Truncate or zero-pad to nmom+1 to match DISORT's expected shape
    n = nmom + 1
    if len(pmom_raw) >= n:
        pmom_1d = pmom_raw[:n]
    else:
        pmom_1d = np.zeros(n)
        pmom_1d[: len(pmom_raw)] = pmom_raw
    return pmom_1d


def _get_pmom_phase(
    phase: PhaseFunction | None, nmom: int, ctx: KernelContext
) -> np.ndarray:
    """Return 1-D Legendre moment array (nmom+1,) for a single phase function."""
    if phase is None or isinstance(phase, IsotropicPhaseFunction):
        return pf.isotropic(nmom)
    elif isinstance(phase, RayleighPhaseFunction):
        return pf.rayleigh(nmom)
    elif isinstance(phase, ParticlePhaseFunction):
        return _get_pmom_particle_layer(phase, nmom, ctx)
    else:
        raise NotImplementedError(
            f"Phase function type {type(phase).__name__!r} is not supported by "
            "EradiateDisortBackend"
        )


def _get_pmom_blend(
    atmosphere: HeterogeneousAtmosphere, nmom: int, ctx: KernelContext
) -> np.ndarray:
    """
    Blend per-component phase moments weighted by scattering coefficient.

    Parameters
    ----------
    atmosphere : HeterogeneousAtmosphere
        Multi-component atmosphere. Must have ``len(components) > 1``.
    nmom : int
        Number of Legendre moments (DISORT ``nmom`` parameter).
    ctx : KernelContext
        Spectral context.

    Returns
    -------
    np.ndarray
        Shape ``(nmom+1, n_layers)``, top-to-bottom layer order.
        At layers where total scattering is zero, the isotropic phase function
        is used as a fallback.
    """
    zgrid = atmosphere.geometry.zgrid
    n_layers = zgrid.n_layers
    n = nmom + 1
    sigma_units = ucc.get("collision_coefficient")

    pmom_sum = np.zeros((n, n_layers))
    sigma_s_sum = np.zeros(n_layers)

    for component in atmosphere.components:
        sigma_s = component.eval_sigma_s(ctx.si, zgrid).m_as(sigma_units)
        pmom_1d = _get_pmom_phase(component.phase, nmom, ctx)
        pmom_sum += pmom_1d[:, np.newaxis] * sigma_s[np.newaxis, :]
        sigma_s_sum += sigma_s

    # Weighted average; fall back to isotropic where total scattering is zero
    result = np.zeros((n, n_layers))
    nonzero = sigma_s_sum > 0
    result[:, nonzero] = pmom_sum[:, nonzero] / sigma_s_sum[np.newaxis, nonzero]
    result[0, ~nonzero] = 1.0  # isotropic: f_0 = 1, rest remain 0

    return result


def get_pmom(atmosphere, nmom: int, ctx: KernelContext) -> np.ndarray:
    """
    Collect DISORT-convention Legendre phase moments for an atmosphere.

    Parameters
    ----------
    atmosphere : Atmosphere or None
        Atmosphere object. If ``None``, an isotropic phase function is used.
        For a :class:`.HeterogeneousAtmosphere` with more than one component,
        moments are blended per layer using scattering coefficients as weights.
    nmom : int
        Number of Legendre moments (DISORT ``nmom`` parameter).
    ctx : KernelContext
        Spectral context used to evaluate spectral quantities.

    Returns
    -------
    np.ndarray
        Shape ``(nmom+1, n_layers)`` in Eradiate's top-to-bottom layer order.
        For ``atmosphere=None``, ``n_layers=1``.
    """
    if atmosphere is None:
        return pf.isotropic(nmom).reshape(-1, 1)

    if (
        isinstance(atmosphere, HeterogeneousAtmosphere)
        and len(atmosphere.components) > 1
    ):
        return _get_pmom_blend(atmosphere, nmom, ctx)

    n_layers = atmosphere.geometry.zgrid.n_layers
    pmom_1d = _get_pmom_phase(atmosphere.phase, nmom, ctx)
    return np.tile(pmom_1d.reshape(-1, 1), (1, n_layers))


# ---------------------------------------------------------------------------
# Phase function evaluation for Buras-Emde intensity correction
# ---------------------------------------------------------------------------


def _find_particle_phase_function(atmosphere) -> ParticlePhaseFunction | None:
    """Return the first ParticlePhaseFunction found in the atmosphere, or None."""
    if isinstance(atmosphere, HeterogeneousAtmosphere):
        for comp in atmosphere.components:
            if isinstance(comp.phase, ParticlePhaseFunction):
                return comp.phase
    elif isinstance(getattr(atmosphere, "phase", None), ParticlePhaseFunction):
        return atmosphere.phase
    return None


def _eval_phase_1d(
    phase: PhaseFunction | None, mu_grid: np.ndarray, ctx: KernelContext
) -> np.ndarray:
    """
    Evaluate a single phase function at given cos(theta) values.

    Parameters
    ----------
    phase : PhaseFunction or None
        Phase function to evaluate. ``None`` and ``IsotropicPhaseFunction``
        both yield 1.0 everywhere.
    mu_grid : ndarray, shape (nphase,)
        Cosines of scattering angles (must be sorted ascending).
    ctx : KernelContext
        Spectral context.

    Returns
    -------
    ndarray, shape (nphase,)
    """
    if phase is None or isinstance(phase, IsotropicPhaseFunction):
        return np.ones(len(mu_grid))
    elif isinstance(phase, RayleighPhaseFunction):
        return 0.75 * (1.0 + mu_grid**2)
    elif isinstance(phase, ParticlePhaseFunction):
        comp_mu = phase.eval_mu(ctx.si)
        comp_p = phase.eval_phase(ctx.si, phamat=0)
        return np.interp(mu_grid, comp_mu, comp_p)
    else:
        raise NotImplementedError(
            f"Phase function {type(phase).__name__!r} is not supported by "
            "the Buras-Emde correction"
        )


def _get_phase_blend(
    atmosphere: HeterogeneousAtmosphere,
    mu_grid: np.ndarray,
    ctx: KernelContext,
) -> np.ndarray:
    """
    Compute blended phase function array for a multi-component atmosphere.

    Parameters
    ----------
    atmosphere : HeterogeneousAtmosphere
    mu_grid : ndarray, shape (nphase,)
        Common angle grid (ascending cosines).
    ctx : KernelContext

    Returns
    -------
    ndarray, shape (nlyr, nphase)
        Phase function per layer, top-to-bottom order. Falls back to isotropic
        (1.0) where total scattering is zero.
    """
    zgrid = atmosphere.geometry.zgrid
    n_layers = zgrid.n_layers
    nphase = len(mu_grid)
    sigma_units = ucc.get("collision_coefficient")

    phase_sum = np.zeros((n_layers, nphase))
    sigma_s_sum = np.zeros(n_layers)

    for comp in atmosphere.components:
        sigma_s = comp.eval_sigma_s(ctx.si, zgrid).m_as(sigma_units)
        p_1d = _eval_phase_1d(comp.phase, mu_grid, ctx)
        phase_sum += sigma_s[:, np.newaxis] * p_1d[np.newaxis, :]
        sigma_s_sum += sigma_s

    result = np.ones((n_layers, nphase))
    nonzero = sigma_s_sum > 0
    result[nonzero] = phase_sum[nonzero] / sigma_s_sum[nonzero, np.newaxis]
    return result


def get_phase(
    atmosphere, nstr: int, ctx: KernelContext
) -> tuple[np.ndarray, np.ndarray]:
    """
    Collect phase function data for the Buras-Emde intensity correction.

    The angle grid is taken from the native grid of the first
    :class:`.ParticlePhaseFunction` found in the atmosphere (if any), or a
    uniform grid of *nstr* points in [-1, 1] otherwise.

    Parameters
    ----------
    atmosphere : Atmosphere or None
        Atmosphere object. If ``None``, a single isotropic layer is returned.
    nstr : int
        Number of streams. Used to size the fallback uniform angle grid.
    ctx : KernelContext
        Spectral context.

    Returns
    -------
    mu_grid : ndarray, shape (nphase,)
        Scattering angle cosines in ascending order.
    phase : ndarray, shape (nlyr, nphase)
        Phase function values per layer in top-to-bottom layer order.
    """
    # Determine the reference angle grid
    pf_particle = _find_particle_phase_function(atmosphere)
    if pf_particle is not None:
        mu_grid = np.sort(pf_particle.eval_mu(ctx.si))
    else:
        mu_grid = np.linspace(-1.0, 1.0, nstr)

    if atmosphere is None:
        return mu_grid, np.ones((1, len(mu_grid)))

    if (
        isinstance(atmosphere, HeterogeneousAtmosphere)
        and len(atmosphere.components) > 1
    ):
        return mu_grid, _get_phase_blend(atmosphere, mu_grid, ctx)

    n_layers = atmosphere.geometry.zgrid.n_layers
    p_1d = _eval_phase_1d(atmosphere.phase, mu_grid, ctx)
    return mu_grid, np.tile(p_1d[np.newaxis, :], (n_layers, 1))
