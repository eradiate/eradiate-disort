"""Smoke tests for the er_plt fixture."""

import pytest

from eradiate_disort.testing.plotting import ErPlt, PlotNull

pytestmark = pytest.mark.order(2)


class TestPlotNull:
    def test_null_object_protocol(self):
        p = PlotNull()
        assert not p
        assert p.figures == []
        assert p.saveas is None

    def test_chaining_is_noop(self):
        # __getattr__ must return PlotNull (not None) so chained calls don't raise
        result = PlotNull().figure("label").plot([1, 2, 3])
        assert isinstance(result, PlotNull)

    def test_subplots_unpacks(self):
        # Callers do `fig, ax = er_plt.subplots()` — must yield exactly two items
        fig, ax = PlotNull().subplots()
        assert isinstance(fig, PlotNull)
        assert isinstance(ax, PlotNull)


class TestErPlt:
    def test_safe_calls(self, er_plt):
        # Must not raise regardless of --plots flag
        er_plt.plot([1, 2, 3])
        er_plt.xlabel("x")
        er_plt.ylabel("y")
        er_plt.title("test")

    def test_saves_single_figure(self, tmp_path):
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        wrapper = ErPlt(node_id="test__saves_single", plots_dir=tmp_path)
        plt.figure()
        plt.plot([1, 2, 3])
        paths = wrapper._teardown()
        assert len(paths) == 1
        assert paths[0].suffix == ".png"

    def test_saves_labelled_figures(self, tmp_path):
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        wrapper = ErPlt(node_id="test__labelled", plots_dir=tmp_path)
        wrapper.figure("forward")
        plt.plot([1, 2, 3])
        wrapper.figure("backward")
        plt.plot([4, 5, 6])
        paths = wrapper._teardown()
        names = {p.name for p in paths}
        assert any("forward" in n for n in names)
        assert any("backward" in n for n in names)

    def test_saveas_override(self, tmp_path):
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        wrapper = ErPlt(node_id="test__saveas", plots_dir=tmp_path)
        wrapper.saveas = "custom.png"
        plt.figure()
        plt.plot([1])
        paths = wrapper._teardown()
        assert paths[0].name == "custom.png"
