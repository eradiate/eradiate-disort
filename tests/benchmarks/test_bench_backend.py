"""
DISORT backend performance benchmarks.

Run with:
    pixi run bench

Compare:
    pytest tests/benchmarks/ --benchmark-compare=<run-id> --benchmark-compare-fail=mean:10%
"""

import pytest

import eradiate_disort as ed
from eradiate_disort.testing import cases


@pytest.mark.benchmark
class TestDisortMonoMode:
    def test_no_atmo(self, benchmark, mode_mono):
        benchmark(ed.DisortBackend().run, cases.no_atmo(sza=30.0, backend="disort"))

    @pytest.mark.parametrize("phase", ["isotropic", "rayleigh"])
    def test_single_layer(self, benchmark, mode_mono, phase):
        benchmark(
            ed.DisortBackend().run,
            cases.single_layer(sza=30.0, phase=phase, backend="disort"),
        )

    def test_two_layers(self, benchmark, mode_mono):
        benchmark(ed.DisortBackend().run, cases.two_layers(sza=30.0, backend="disort"))


@pytest.mark.benchmark
class TestDisortCkdMode:
    def test_molecular(self, benchmark, mode_ckd):
        benchmark(ed.DisortBackend().run, cases.molecular(sza=30.0, backend="disort"))

    def test_aerosols(self, benchmark, mode_ckd):
        benchmark(ed.DisortBackend().run, cases.aerosols(sza=30.0, backend="disort"))

    def test_full_atmo(self, benchmark, mode_ckd):
        benchmark(ed.DisortBackend().run, cases.full_atmo(sza=30.0, backend="disort"))
