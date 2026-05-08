# SPDX-FileCopyrightText: 2026 Rayference
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
DISORT backend performance benchmarks.

Run with:
    pixi run bench
or manually:
    pytest tests/benchmarks/ --benchmark-autosave --benchmark-json=benchmarks/results.json

Compare against a saved run:
    pytest tests/benchmarks/ --benchmark-compare=<run-id> --benchmark-compare-fail=mean:10%
"""

import pytest

import eradiate_disort as ed
from eradiate_disort.testing import cases


# ---------------------------------------------------------------------------
# Mono-mode cases  (test_01 – test_03 notebooks)
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
class TestDisortMonoMode:
    """Benchmarks for scenes that run in mono spectral mode."""

    def test_no_atmo(self, benchmark, mode_mono):
        """No-atmosphere case (test_01): pure surface reflection, no layers."""
        backend = ed.EradiateDisortBackend()
        exp = cases.no_atmo(sza=30.0, backend="disort")
        benchmark.extra_info.update({"case": "no_atmo", "sza": 30.0})
        benchmark(backend.run, exp)

    @pytest.mark.parametrize("phase", ["isotropic", "rayleigh"])
    def test_single_layer(self, benchmark, mode_mono, phase):
        """Single homogeneous layer (test_02): one scattering layer."""
        backend = ed.EradiateDisortBackend()
        exp = cases.single_layer(sza=30.0, phase=phase, backend="disort")
        benchmark.extra_info.update({"case": "single_layer", "phase": phase, "sza": 30.0})
        benchmark(backend.run, exp)

    def test_two_layers(self, benchmark, mode_mono):
        """Two-layer stepped profile (test_03): absorption + scattering split."""
        backend = ed.EradiateDisortBackend()
        exp = cases.two_layers(sza=30.0, backend="disort")
        benchmark.extra_info.update({"case": "two_layers", "sza": 30.0})
        benchmark(backend.run, exp)


# ---------------------------------------------------------------------------
# CKD-mode cases  (test_04 – test_06 notebooks)
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
class TestDisortCkdMode:
    """Benchmarks for scenes that require CKD spectral mode."""

    def test_molecular(self, benchmark, mode_ckd):
        """Molecular atmosphere (test_04): Rayleigh scattering + gas absorption."""
        backend = ed.EradiateDisortBackend()
        exp = cases.molecular(sza=30.0, backend="disort")
        benchmark.extra_info.update({"case": "molecular", "sza": 30.0})
        benchmark(backend.run, exp)

    def test_aerosols(self, benchmark, mode_ckd):
        """Aerosol particle layer (test_05): single particle layer."""
        backend = ed.EradiateDisortBackend()
        exp = cases.aerosols(sza=30.0, backend="disort")
        benchmark.extra_info.update({"case": "aerosols", "sza": 30.0})
        benchmark(backend.run, exp)

    def test_full_atmo(self, benchmark, mode_ckd):
        """Full heterogeneous atmosphere (test_06): molecular + aerosol layers."""
        backend = ed.EradiateDisortBackend()
        exp = cases.full_atmo(sza=30.0, backend="disort")
        benchmark.extra_info.update({"case": "full_atmo", "sza": 30.0})
        benchmark(backend.run, exp)
