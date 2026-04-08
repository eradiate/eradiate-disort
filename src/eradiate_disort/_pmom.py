"""
Functions to collect Legendre moments for various phase functions.
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
