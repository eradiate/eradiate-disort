"""
DISORT backend performance benchmarks.
"""

import pytest

import eradiate_disort as ed
from eradiate_disort.testing import cases


@pytest.mark.benchmark
class BenchDisortMonoMode:
    def bench_no_atmo(self, benchmark, mode_mono):
        benchmark(ed.DisortBackend().run, cases.no_atmo(sza=30.0, backend="disort"))

    @pytest.mark.parametrize("phase", ["isotropic", "rayleigh"])
    def bench_single_layer(self, benchmark, mode_mono, phase):
        benchmark(
            ed.DisortBackend().run,
            cases.single_layer(sza=30.0, phase=phase, backend="disort"),
        )

    def bench_two_layers(self, benchmark, mode_mono):
        benchmark(ed.DisortBackend().run, cases.two_layers(sza=30.0, backend="disort"))


@pytest.mark.benchmark
class BenchDisortCkdMode:
    def bench_molecular(self, benchmark, mode_ckd):
        benchmark(ed.DisortBackend().run, cases.molecular(sza=30.0, backend="disort"))

    def bench_aerosols(self, benchmark, mode_ckd):
        benchmark(ed.DisortBackend().run, cases.aerosols(sza=30.0, backend="disort"))

    def bench_full_atmo(self, benchmark, mode_ckd):
        benchmark(ed.DisortBackend().run, cases.full_atmo(sza=30.0, backend="disort"))
