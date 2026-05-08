# SPDX-FileCopyrightText: 2026 Rayference
#
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pytest


def _safe_filename(node_id: str) -> str:
    """Convert a pytest node ID to a filesystem-safe stem."""
    return re.sub(r"[^\w\-.]", "_", node_id)


class PlotNull:
    """
    Lightweight no-op returned by ``er_plt`` when ``--plots`` is not active.

    All attribute access returns a callable that silently ignores its arguments.
    No matplotlib import occurs, so tests that never pass ``--plots`` pay zero
    import overhead.
    """

    saveas: Optional[str] = None

    def __getattr__(self, name: str) -> "PlotNull":
        return self

    def __call__(self, *args, **kwargs) -> "PlotNull":
        return PlotNull()

    def figure(self, label: Optional[str] = None) -> "PlotNull":
        return PlotNull()

    def __iter__(self):
        # Support common tuple-unpacking patterns, e.g. ``fig, ax = er_plt.subplots()``
        yield PlotNull()
        yield PlotNull()

    def __bool__(self) -> bool:
        return False

    @property
    def figures(self) -> list[Path]:
        return []


class ErPlt:
    """
    Active matplotlib.pyplot wrapper returned by ``er_plt`` when ``--plots`` is
    active.

    Delegates all attribute access to :mod:`matplotlib.pyplot`. On fixture
    teardown every open figure is saved to the configured output directory and
    closed.

    Usage::

        def test_something(er_plt):
            er_plt.plot([1, 2, 3])  # single implicit figure

        def test_named(er_plt):
            fig = er_plt.figure("fwd")  # labelled figure
            fig.add_subplot(111).plot(...)

        def test_custom_name(er_plt):
            er_plt.saveas = "comparison.pdf"
            er_plt.plot([1, 2, 3])
    """

    def __init__(self, node_id: str, plots_dir: Path) -> None:
        self._node_id = node_id
        self._plots_dir = plots_dir
        self._plots_dir.mkdir(parents=True, exist_ok=True)
        self._labels: dict[int, str] = {}
        self._saved_paths: list[Path] = []
        self.saveas: Optional[str] = None

    def figure(self, label: Optional[str] = None):
        """Create a new matplotlib figure and optionally attach a label to it."""
        import matplotlib.pyplot as plt

        fig = plt.figure()
        if label is not None:
            self._labels[fig.number] = label
        return fig

    def __getattr__(self, name: str):
        import matplotlib.pyplot as plt

        return getattr(plt, name)

    def __bool__(self) -> bool:
        return True

    def _teardown(self) -> list[Path]:
        """Save every open matplotlib figure, close it, and return saved paths."""
        import matplotlib.pyplot as plt

        fignums = plt.get_fignums()
        safe_id = _safe_filename(self._node_id)
        saved = []

        for i, fignum in enumerate(fignums):
            fig = plt.figure(fignum)
            label = self._labels.get(fignum)

            if self.saveas is not None and len(fignums) == 1:
                path = self._plots_dir / self.saveas
            elif label:
                path = self._plots_dir / f"{safe_id}_{label}.png"
            elif len(fignums) > 1:
                path = self._plots_dir / f"{safe_id}_{i}.png"
            else:
                path = self._plots_dir / f"{safe_id}.png"

            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)
            saved.append(path)

        self._saved_paths = saved
        return saved

    @property
    def figures(self) -> list[Path]:
        """Paths of figures saved during the last teardown."""
        return self._saved_paths


@pytest.fixture
def er_plt(request):
    """
    Conditional plotting fixture.

    Returns an :class:`ErPlt` wrapper around :mod:`matplotlib.pyplot` when
    ``--plots`` is active, or a :class:`PlotNull` no-op otherwise.  All open
    figures are saved to the output directory on teardown; no figures are
    generated (and matplotlib is never imported) when ``--plots`` is absent.

    Examples
    --------
    ::

        def test_radiance(er_plt):
            er_plt.plot(vza, radiance, label="DISORT")
            er_plt.xlabel("VZA (deg)")
            er_plt.ylabel("Radiance")

        def test_multi(er_plt):
            fig1 = er_plt.figure("forward")
            fig1.add_subplot(111).plot(vza_fwd, rad_fwd)

            fig2 = er_plt.figure("backward")
            fig2.add_subplot(111).plot(vza_bwd, rad_bwd)
    """
    plots_dir = request.config.getoption("--plots")
    if plots_dir is None:
        yield PlotNull()
        return

    wrapper = ErPlt(
        node_id=request.node.nodeid,
        plots_dir=Path(plots_dir),
    )
    yield wrapper
    wrapper._teardown()
