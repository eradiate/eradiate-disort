Developer guide
===============

.. note::

   Outline draft. Each section below is a stub to be filled with prose.
   Audience: contributors and maintainers of the DISORT backend. For usage,
   see the :doc:`User guide <../user_guide/index>`.

Development environment
-----------------------

.. todo::

   - pixi workflow and environments (``default``, ``dev``, ``docs``).
   - Git submodules under ``ext/`` (``eradiate``, ``nanodisort``) and editable
     installs; ``git submodule update --init --recursive``.
   - Activation: ``setpath.sh``, ``ERADIATE_PATH``, ``.envrc`` clang pinning.
   - Building the Mitsuba kernel (``kernel-configure`` / ``kernel-build``).
   - Common tasks (``test``, ``bench``, ``lint``, ``docs``, ...).

Architecture
------------

.. todo::

   - High-level picture: a DISORT radiometric backend reached through
     ``nanodisort`` bindings to CDISORT.
   - Execution flow: ``DisortBackend.run()`` → ``validate`` → ``process``
     (``_setup_global`` → spectral loop: ``_setup_spectral`` → ``_solve`` →
     ``_collect_results``) → ``postprocess``.
   - The single shared spectral loop driven by one measure.

Components
----------

.. todo::

   - ``_backend.py`` — ``DisortBackend`` entry point.
   - ``_measurements.py`` — ``DisortMeasure`` (``measure_factory`` registration).
   - ``_phase.py`` — ``get_pmom`` / ``get_phase``.
   - ``_pipeline.py`` — ``build_disort_pipeline`` DAG.
   - ``io.py`` — metadata normalization.
   - ``testing/`` — shared helpers (``TestMode``, ``cases``, fixtures).

DISORT ↔ Eradiate translation
-----------------------------

.. todo::

   - Layer/moment ordering (top-to-bottom vs. bottom-to-top).
   - Azimuth convention (``phi0`` travel vs. source direction, 180° offset).
   - Cumulative optical depth (``utau``) measured from TOA.
   - Flux-only mode: re-assigning ``numu``/``nphi``/``umu``/``phi`` each
     spectral iteration.
   - Intensity correction (``buras_emde`` sentinel padding, the ``+2`` to
     ``nphase``).
   - Phase moment conventions ((2l+1) scaling, truncation/zero-padding).
   - Homogeneous-atmosphere broadcasting; allocation ordering
     (``ds.allocate()`` once, before any array assignment).

Spectral modes
--------------

.. todo::

   - ``mono`` vs. ``ckd``; ``_get_spectral_indices`` branches; CKD quadrature
     in pipeline aggregation.

Testing
-------

.. todo::

   - Test layout and fixtures (``conftest.py``, ``er_plt``,
     ``xarray_regression``).
   - Test modes via ``ERADIATE_TEST_MODE`` (``test`` / ``benchmark`` /
     ``tutorial``) and ``TestMode``.
   - Examples as regression tests (``tests/examples``, jupytext pairing,
     ``order(-1)``).
   - Regression reference data and ``--force-regen``.
   - Benchmarks (separate config, ``bench_`` prefix).

Documentation
-------------

.. todo::

   - Sphinx + shibuya + MyST-NB; committed example outputs (nbstripout
     exclusion).
   - Building locally (``docs`` / ``docs-serve`` / ``docs-clean``).
   - ``docs/requirements.txt`` regeneration (``docs-lock``).

Conventions and tooling
-----------------------

.. todo::

   - Ruff (rules, format), Numpydoc docstrings, type hints.
   - Private modules underscore-prefixed; public API re-exported from
     ``__init__.py``.
   - SPDX headers / REUSE compliance.
   - pre-commit hooks (ruff, taplo, nbstripout).

Releasing
---------

.. todo::

   - Versioning, changelog, CI workflows, publishing to PyPI.

AI / agentic programming
------------------------

.. todo::

   - Revisit with `the Ghostty policy <https://github.com/ghostty-org/ghostty/blob/main/AI_POLICY.md>`__
     in mind.
   - Actually link to AI_POLICY.md?
   - Scope: how AI coding assistants (e.g. Claude Code) are expected to be used
     on this project, and what stays a human responsibility.
   - Project context for agents: ``CLAUDE.md`` is the source of truth for
     conventions, architecture and gotchas; keep it in sync with the code.
   - High-risk areas where AI output must be reviewed carefully: the
     DISORT ↔ Eradiate translation (ordering, azimuth, ``utau``, allocation
     ordering, phase-moment scaling) — numerical correctness is not obvious
     from a diff.
   - Verification expectations: agent-authored changes must pass the test suite
     and, where physics is touched, be checked against regression references or
     the Monte Carlo backend before merging.
   - Regression data discipline: never regenerate ``.npz`` references with
     ``--force-regen`` to make a test pass without understanding the change.
   - Provenance and attribution: commit co-authorship trailers for
     AI-assisted commits; disclosure in PRs.
   - Licensing caution: GPLv3 project — do not paste in code of unknown or
     incompatible provenance.
   - Tooling guardrails: rely on pre-commit (ruff, taplo, nbstripout) and CI
     rather than trusting formatting/lint claims from the agent.
