"""
DISORT backend performance benchmarks.
"""

import eradiate
import pytest

import eradiate_disort as ed
from eradiate_disort.testing import cases


class BenchMonoNoAtmo:
    @pytest.mark.benchmark(group="no_atmo")
    def bench_no_atmo_disort(self, benchmark, mode_mono):
        exp = cases.no_atmo(sza=30.0, backend="disort")
        backend = ed.DisortBackend()
        benchmark(backend.run, exp)

    @pytest.mark.benchmark(group="no_atmo")
    def bench_no_atmo_mitsuba(self, benchmark, mode_mono):
        exp = cases.no_atmo(sza=30.0, backend="mitsuba")
        benchmark(eradiate.run, exp)


@pytest.mark.parametrize("phase", ["isotropic", "rayleigh"])
class BenchMonoSingleLayer:
    @pytest.mark.benchmark(group="single_layer")
    def bench_single_layer_disort(self, benchmark, mode_mono, phase):
        backend = ed.DisortBackend()
        exp = cases.single_layer(sza=30.0, phase=phase, backend="disort")
        benchmark(backend.run, exp)

    @pytest.mark.benchmark(group="single_layer")
    def bench_single_layer_mitsuba(self, benchmark, mode_mono, phase):
        exp = cases.single_layer(sza=30.0, phase=phase, backend="mitsuba")
        benchmark(eradiate.run, exp)


class BenchMonoTwoLayers:
    @pytest.mark.benchmark(group="two_layers")
    def bench_two_layers_disort(self, benchmark, mode_mono):
        backend = ed.DisortBackend()
        exp = cases.two_layers(sza=30.0, backend="disort")
        benchmark(backend.run, exp)

    @pytest.mark.benchmark(group="two_layers")
    def bench_two_layers_mitsuba(self, benchmark, mode_mono):
        exp = cases.two_layers(sza=30.0, backend="mitsuba")
        benchmark(eradiate.run, exp)


@pytest.mark.benchmark(group="molecular")
class BenchCkdMolecular:
    def bench_molecular_disort(self, benchmark, mode_ckd):
        backend = ed.DisortBackend()
        exp = cases.molecular(sza=30.0, backend="disort")
        benchmark(backend.run, exp)

    def bench_molecular_mitsuba(self, benchmark, mode_ckd):
        exp = cases.molecular(sza=30.0, backend="mitsuba")
        benchmark(eradiate.run, exp)


@pytest.mark.benchmark(group="aerosols")
class BenchCkdAerosols:
    def bench_aerosols_disort(self, benchmark, mode_ckd):
        backend = ed.DisortBackend()
        exp = cases.aerosols(sza=30.0, backend="disort")
        benchmark(backend.run, exp)

    def bench_aerosols_mitsuba(self, benchmark, mode_ckd):
        exp = cases.aerosols(sza=30.0, backend="mitsuba")
        benchmark(eradiate.run, exp)


@pytest.mark.benchmark(group="full")
class BenchCkdFull:
    def bench_full_atmo_disort(self, benchmark, mode_ckd):
        backend = ed.DisortBackend()
        exp = cases.full_atmo(sza=30.0, backend="disort")
        benchmark(backend.run, exp)

    def bench_bench_full_atmo_mitsuba(self, benchmark, mode_ckd):
        exp = cases.full_atmo(sza=30.0, backend="mitsuba")
        benchmark(eradiate.run, exp)
