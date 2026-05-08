# SPDX-FileCopyrightText: 2026 Rayference
#
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import os


class TestMode:
    """
    Runtime mode selector for notebook scripts shared across tutorials, tests,
    and benchmarks.

    The active mode is read from the ``ERADIATE_TEST_MODE`` environment
    variable.  When the variable is absent (e.g. in an interactive JupyterLab
    session), the mode defaults to ``"tutorial"``.

    Modes
    -----
    ``"tutorial"``
        Low sample counts, interactive plotting.  Default in Jupyter.
    ``"test"``
        Full sample counts, no interactive plots.  Set by the test runner.
    ``"benchmark"``
        Sample counts optimised for timing stability.  Set by the bench task.

    Examples
    --------
    Typical usage in a notebook setup cell::

        from eradiate_disort.testing import TestMode

        SPP = TestMode.spp(tutorial=256, test=4_096)
    """

    @staticmethod
    def get() -> str:
        """Return the current mode string."""
        return os.environ.get("ERADIATE_TEST_MODE", "tutorial")

    @staticmethod
    def is_tutorial() -> bool:
        """Return ``True`` when running in tutorial (interactive) mode."""
        return TestMode.get() == "tutorial"

    @staticmethod
    def is_test() -> bool:
        """Return ``True`` when running under the test runner."""
        return TestMode.get() == "test"

    @staticmethod
    def is_benchmark() -> bool:
        """Return ``True`` when running as a benchmark."""
        return TestMode.get() == "benchmark"

    @staticmethod
    def spp(*, tutorial: int = 256, test: int = 4_096, benchmark: int = 4_096) -> int:
        """
        Return the samples-per-pixel count appropriate for the current mode.

        Parameters
        ----------
        tutorial:
            SPP for interactive tutorial runs (default: 256).
        test:
            SPP for automated test runs (default: 4 096).
        benchmark:
            SPP for benchmark runs (default: 4 096).
        """
        return {"tutorial": tutorial, "test": test, "benchmark": benchmark}[
            TestMode.get()
        ]
