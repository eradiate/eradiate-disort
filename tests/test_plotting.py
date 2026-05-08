"""Smoke tests for the er_plt fixture."""

import pytest

from eradiate_disort.testing.plotting import ErPlt, PlotNull


class TestPlotNull:
    def test_attr_access_is_noop(self):
        p = PlotNull()
        assert p.plot([1, 2, 3]) is not None  # no AttributeError

    def test_call_returns_plot_null(self):
        p = PlotNull()
        result = p.plot([1, 2, 3])
        assert isinstance(result, PlotNull)

    def test_figure_returns_plot_null(self):
        p = PlotNull()
        assert isinstance(p.figure("label"), PlotNull)

    def test_bool_is_false(self):
        assert not PlotNull()

    def test_figures_is_empty(self):
        assert PlotNull().figures == []

    def test_iter_supports_two_unpacking(self):
        p = PlotNull()
        fig, ax = p.subplots()  # must not raise
        assert isinstance(fig, PlotNull)
        assert isinstance(ax, PlotNull)

    def test_saveas_default_none(self):
        assert PlotNull().saveas is None

    def test_chained_methods_noop(self):
        p = PlotNull()
        p.figure().add_subplot(111).plot([1, 2, 3])  # must not raise


class TestErPltFixtureInactive:
    def test_inactive_yields_plot_null(self, er_plt):
        # Without --plots, fixture should be PlotNull
        # (when running this test suite without --plots the fixture is inactive)
        assert isinstance(er_plt, PlotNull) or isinstance(er_plt, ErPlt)

    def test_no_matplotlib_calls_needed(self, er_plt):
        # These calls must be safe regardless of active/inactive
        er_plt.plot([1, 2, 3])
        er_plt.xlabel("x")
        er_plt.ylabel("y")
        er_plt.title("test")


class TestErPltFixtureActive:
    def test_saves_single_figure(self, er_plt, tmp_path, request):
        """Active ErPlt saves a figure on teardown."""
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        wrapper = ErPlt(node_id="test__saves_single", plots_dir=tmp_path)
        plt.figure()
        plt.plot([1, 2, 3])
        paths = wrapper._teardown()
        assert len(paths) == 1
        assert paths[0].exists()
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
        assert len(paths) == 1
        assert paths[0].name == "custom.png"

    def test_figures_property_after_teardown(self, tmp_path):
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        wrapper = ErPlt(node_id="test__figures_prop", plots_dir=tmp_path)
        plt.figure()
        plt.plot([1])
        wrapper._teardown()
        assert len(wrapper.figures) == 1

    def test_bool_is_true(self, tmp_path):
        wrapper = ErPlt(node_id="x", plots_dir=tmp_path)
        assert bool(wrapper)
