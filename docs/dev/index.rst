Developer guide
===============

This guide is intended to contributors of all sorts — developers, maintainers,
users willing to report issues, with or without AI programming assistance.

.. _dev-ai_programming:

AI programming
--------------

Usage of AI coding assistants (*e.g.* Claude Code, Codex, Cursor) is allowed in
this project, but only as a tool under human supervision. All changes are
authored by a human responsible for them regardless of how they were produced;
"the agent wrote it" is not a defence for a regression.

AI policy
    Be sure to read, understand and comply to our
    `AI policy <https://github.com/eradiate/eradiate-disort/blob/main/AI_POLICY.md>`__.

Project context for agents
    ``AGENTS.md`` at the repository root is the source of truth for conventions,
    architecture and gotchas, and is loaded automatically by AI tools. Keep it
    in sync with the code: when you change a convention or a translation detail,
    update ``AGENTS.md`` (and this guide) in the same change.

Provenance and attribution
    Disclose AI involvement in the pull request description and / or commit
    messages under an "AI disclosure" section. Do **not** list your AI tool as a
    co-author of your commits.

.. _dev-general_guidelines:

General guidelines
------------------

The following, general guidelines, apply to everyone, regardless of whether they
use AI tools.

Verification expectations
    All changes must pass the full test suite. Where physics is touched, they
    must additionally be checked against the regression references or the Monte
    Carlo backend before merging — a green suite alone is not enough if the
    references themselves were regenerated.

Regression data discipline
    Never regenerate regression test references with ``--force-regen`` just to
    make a test pass. Regenerate only when you understand why the expected
    values changed and have confirmed the new values are correct.

High-risk areas
    Pay particular attention to parts of the code falling under the
    :ref:`Eradiate-to-DISORT translation <dev-translation>` section
    (ordering, azimuth, ``utau``, allocation ordering, phase-moment scaling).
    This is where AI output must be reviewed most carefully. Such issues do not
    show up as obvious errors in a diff and a plausible-looking change can
    silently break.

Licensing caution
    This is a GPLv3 project. Do not paste in code of unknown or incompatible
    provenance; an agent suggesting a verbatim block from elsewhere is a
    licensing risk.

Tooling guardrails
    Rely on pre-commit (ruff, taplo) and CI as the source of truth for
    formatting and lint, not on an agent's — or your own — claim that a file is
    clean.

Development environment
-----------------------

The project is managed with `Pixi <https://pixi.sh/>`__. Install Pixi, clone the
repository, and initialize the submodule::

    git submodule update --init --recursive

`Eradiate <https://github.com/eradiate/eradiate>`__ is pulled from PyPI with the
``kernel`` and ``recommended`` extras, so the Mitsuba kernel ships as a prebuilt
wheel and there is nothing to compile locally. Its Monte Carlo backend is what
the DISORT results are validated against.

One dependency lives under ``ext/`` as a git submodule and is installed as an
editable Pixi dependency:

``ext/nanodisort``
    The ``nanodisort`` bindings to CDISORT (the C port of DISORT) through which
    the solver is reached.

Because it is an editable install, changes made in the submodule tree are
picked up without reinstalling.

Activation
^^^^^^^^^^

The ``dev`` feature's activation runs ``scripts/set-macos-deployment-target.sh``
on environment entry (a no-op off macOS; pins ``MACOSX_DEPLOYMENT_TARGET`` so
locally built ``nanodisort`` wheel tags match uv's acceptance ceiling).
Activation also adds ``tests/data`` to ``ERADIATE_PATH``, so that Eradiate's path
resolver picks up testing data.

Features and environments
^^^^^^^^^^^^^^^^^^^^^^^^^

The Pixi manifest defines feature sets — ``pyXXX`` (interpreter pins),
``test``, ``docs``, ``dev`` and ``kernel`` — composed into environments:

``default``
    The day-to-day working environment: ``py312`` plus ``test``, ``docs`` and
    ``dev``. Running ``pixi run <task>`` uses this environment unless told
    otherwise.

``ghapy310`` ... ``ghapy313``
    CI environments, one per supported interpreter, combining the corresponding
    Python pin with ``kernel`` and ``test``.

Task cheatsheet
^^^^^^^^^^^^^^^

All tasks run as ``pixi run <task>`` in the ``default`` environment.

+------------------------------------------------------------------------------+
| **Testing and benchmarks**                                                   |
+------------------------+-----------------------------------------------------+
| ``test``               | Run the test suite (``MPLBACKEND=Agg``,             |
|                        | ``ERADIATE_TEST_MODE=test``; benchmarks excluded).  |
|                        | Extra arguments pass through to ``pytest``.         |
+------------------------+-----------------------------------------------------+
| ``bench``              | Run the benchmarks in ``tests/benchmarks``          |
|                        | (``ERADIATE_TEST_MODE=benchmark``).                 |
+------------------------+-----------------------------------------------------+
| **Linting**                                                                  |
+------------------------+-----------------------------------------------------+
| ``lint``               | Ruff with the base rule set.                        |
+------------------------+-----------------------------------------------------+
| ``lint-ext``           | Ruff with the extended rule set.                    |
+------------------------+-----------------------------------------------------+
| ``lint-reuse``         | REUSE license-compliance check.                     |
+------------------------+-----------------------------------------------------+
| **Documentation**                                                            |
+------------------------+-----------------------------------------------------+
| ``docs``               | Build the HTML docs into ``docs/_build/html``.      |
+------------------------+-----------------------------------------------------+
| ``docs-serve``         | Live-serve the docs with sphinx-autobuild.          |
+------------------------+-----------------------------------------------------+
| ``docs-clean``         | Remove the ``docs/_build/`` tree.                   |
+------------------------+-----------------------------------------------------+
| ``docs-lock``          | Regenerate ``docs/requirements.txt``.               |
+------------------------+-----------------------------------------------------+
| **Notebooks**                                                                |
+------------------------+-----------------------------------------------------+
| ``nb-kernel-register`` | Register the Pixi environment as a Jupyter kernel   |
|                        | (idempotent).                                       |
+------------------------+-----------------------------------------------------+
| ``nb-execute``         | Sync and execute one example notebook               |
|                        | (``ERADIATE_TEST_MODE=tutorial``).                  |
+------------------------+-----------------------------------------------------+
| ``nb-execute-all``     | Sync and execute all                                |
|                        | ``tests/examples/example_*.py`` notebooks.          |
+------------------------+-----------------------------------------------------+
| **Releasing**                                                                |
+------------------------+-----------------------------------------------------+
| ``bump``               | Bump the version to ``$RELEASE_VERSION``.           |
+------------------------+-----------------------------------------------------+
| ``bump-show``          | List the possible next versions.                    |
+------------------------+-----------------------------------------------------+
| ``bump-dry``           | Verbose dry run of the version bump.                |
+------------------------+-----------------------------------------------------+

Architecture
------------

``eradiate-disort`` is a DISORT radiometric backend for Eradiate. It computes
radiances and fluxes for 1D plane-parallel scenes using CDISORT — reached through
the ``nanodisort`` Python bindings — as a fast alternative to Eradiate's Monte
Carlo ray-tracing backend.

Execution flow
^^^^^^^^^^^^^^

``DisortBackend.run()`` (in ``_backend.py``) drives three stages that mirror
Eradiate's experiment lifecycle:

1. :meth:`validate() <.DisortBackend.validate>` — Rejects unsupported
   configurations early.

2. :meth:`process() <.DisortBackend.process>` —
   First, ``_setup_global`` does  the spectral-independent setup. Then, the
   spectral loop runs  ``_setup_spectral`` → ``_solve`` → ``_collect_results``
   once per spectral context. The raw per-spectral DISORT output is accumulated
   in ``self._results``.

3. :meth:`postprocess() <.DisortBackend.postprocess>` —
   Runs the post-processing pipeline to turn the raw results into an
   :class:`xarray.DataTree` with one subtree per measure, keyed by measure ID.

Main components
---------------

:class:`~eradiate_disort.DisortBackend`
    The entry point that drives the aforementioned execution flow.

:class:`~eradiate_disort.DisortMeasure`
    A custom Eradiate :class:`~eradiate.scenes.measure.Measure` registered in
    :class:`~eradiate.scenes.measure.measure_factory` under ``"disort"``. It
    records fluxes and intensities at specified altitudes or optical depths.
    The convenience constructors ``hplane``, ``aring`` and ``grid`` build common
    direction layouts.

:mod:`~eradiate_disort._phase` *(private)*
    Evaluation routines for various phase functions (:func:`get_phase`) and their
    Legendre coefficients (:func:`get_pmom`).

:mod:`~eradiate_disort._pipeline` *(private)*
    Specific post-processing pipeline that processes the raw ``nanodisort``
    output into a :class:`~xarray.DataTree`.

:mod:`~eradiate_disort.io`
    Various data formatting and I/O support components, *e.g.* metadata
    normalization.

:mod:`~eradiate_disort.testing`
    Shared helpers used for development and testing, such as the
    :class:`.TestMode` selector (driven by the ``ERADIATE_TEST_MODE`` environment
    variable), the :mod:`.cases` a benchmarking case library, the
    :func:`.xarray_regression` fixture, and plotting helpers.

.. _dev-translation:

Eradiate-to-DISORT translation
------------------------------

These are the subtle, error-prone parts of the backend. The points below are the
conventions that must be respected when changing it. Bugs here are numerical,
not structural, and rarely visible in a diff — see
:ref:`AI / agentic programming <dev-ai_programming>`.

Layer ordering
    CDISORT expects atmospheric layers ordered **top-to-bottom**, whereas
    Eradiate works **bottom-to-top**. Arrays are reversed at the boundary
    (``dtauc[::-1]``, ``ssalb[::-1]``, ``pmom[:, ::-1]``).

Azimuth convention
    CDISORT's ``phi0`` is the beam's *travel* direction, while Eradiate's
    ``illumination.azimuth`` is the *source* direction. They differ by 180°.

Cumulative optical depth (``utau``)
    Measured from the top of the atmosphere (0 at TOA, total τ at BOA).
    :class:`.DisortMeasure` accepts either ``z_levels`` (snapped to grid
    boundaries) or ``utau`` — mutually exclusive.

Flux-only mode
    When ``onlyfl=True`` (no ``direction_layout``), CDISORT internally
    overwrites ``numu`` with ``nstr``. The user ``numu``/``nphi``/``umu``/``phi``
    must therefore be re-assigned on **every** spectral iteration after the first.

Intensity correction
    ``buras_emde`` (the default) needs actual phase-function values and pads the
    μ-phase grid with sentinel points at both ends (``±(1+eps)``); the ``+2`` to
    ``nphase`` in ``_setup_global`` accounts for these. ``nakajima_tanaka`` uses
    only Legendre moments and needs no padding.

Phase-moment conventions
    Eradiate's particle data stores ``(2l+1)·f_l``, whereas DISORT wants
    ``f_l``. Components in :mod:`._phase` divide by ``(2l+1)`` and truncate or
    zero-pad to ``nmom+1``.

Homogeneous atmospheres
    These return scalar optical properties; broadcast them to ``nlyr`` before
    assigning.

Allocation ordering
    ``ds.allocate()`` is called exactly once (``first_call=True``), after ``ntau``
    is known and before any array assignment. Never assign DISORT arrays before
    allocation.

Spectral modes
--------------

The backend supports Eradiate's ``mono`` and ``ckd`` spectral modes, selected with
:func:`eradiate.set_mode`. Mode-dependent behaviour is confined to two places:

* spectral index generation (:func:`DisortBackend._get_spectral_indices`);
* aggregation step in the post-processing pipeline (:func:`.aggregated_data`).

The rest of the backend is mode-agnostic.

Testing
-------

Layout and fixtures
^^^^^^^^^^^^^^^^^^^

Tests live in ``tests/``. The ``tests/conftest.py`` configuration file pulls in
the shared fixtures, including ``er_plt`` (plotting fixture) and
``xarray_regression`` (which wraps pytest-regressions' ``ndarrays_regression``).

Test modes
^^^^^^^^^^

The ``ERADIATE_TEST_MODE`` environment variable, surfaced through
:class:`.TestMode`, selects how shareable code runs:

``test``
    Used by test tasks. Full sample counts, plots output to disk.

``benchmark``
    Used by benchmarking tasks. Full sample counts (possibly different from
    ``test``), no plots.

``tutorial``
    Used by notebook tasks. Low sample counts, plots enabled (interactive window
    in ).

:meth:`.TestMode.spp` and :meth:`.TestMode.plt` let a single notebook serve all
three modes. Hidden cells can bypass default settings (*e.g.* sample count) with
values output by :class:`.TestMode`.

Examples as regression tests
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The example notebooks double as regression tests.Source files, stored as
``tests/examples/example_*.py``, are Jupytext percent-format scripts paired to
``docs/examples/*.ipynb``. The ``tests/examples/test_examples.py`` test file
imports each example's results and asserts against them. These run **last**,
under ``pytest.mark.order(-1)``.

.. seealso::

    * `Percent-format scripts <https://jupytext.org/formats/scripts/>`__
    * `Jupytext notebook pairing <https://jupytext.org/using/paired-notebooks/>`__
    * `pytest-order documentation <https://pytest-order.readthedocs.io>`__

Regression reference data
^^^^^^^^^^^^^^^^^^^^^^^^^^

Regression references  are stored in ``tests/data/regression_references/``,
under a subdirectory tree that reflects test the module layout. They can be
regenerated them by appending ``--force-regen`` to the test invocation. Do this
only when you understand *why* the reference changed — see
:ref:`AI / agentic programming <dev-ai_programming>`.

Benchmarks
^^^^^^^^^^

Benchmarks live in ``tests/benchmarks`` and have specific configuration defined
in a separate ``pytest.ini``. They use
`pytest-benchmark <https://pytest-benchmark.readthedocs.io/>`__.
Benchmark definition files, functions and classes use ``bench_`` and ``Bench``
prefixes. They are excluded from the default suite and run with ``pixi run bench``.

Documentation
-------------

The documentation is built with Sphinx, the
`Shibuya <https://shibuya.lepture.com/>`__ theme and
`MyST-NB <https://myst-nb.readthedocs.io/>`__, and is deployed on Read the Docs.

Example notebook outputs are committed. The nbstripout pre-commit hook excludes
``docs/examples/`` so MyST-NB renders them without re-executing. This keeps the
docs build fast and reproducible, at the cost of having to regenerate and commit
outputs when an example changes (``pixi run nb-execute-all``).

Build the docs locally with ``pixi run docs``; ``pixi run docs-serve``
live-rebuilds with sphinx-autobuild, and ``pixi run docs-clean`` removes the
build tree.

The pinned doc dependencies in ``docs/requirements.txt`` are regenerated with
``pixi run docs-lock`` (a ``uv pip compile`` invocation); commit the result when
the doc dependency set changes.

Conventions and tooling
-----------------------

Code style
^^^^^^^^^^

Python code is formatted linted with `Ruff <https://docs.astral.sh/ruff/>`__.
Activate pre-commit hooks to enforce formatting and the basic linting rules.

The basic lint rule set is intentionally small and can be applied with
``pixi run lint``. A more extensive rule set is available and should be applied
to detect issues early during a refactoring pass, prior to opening a PR or
merging a branch into ``main``. The extended rule set is applied by the
``pixi run lint-ext`` task.

Public functions, classes and methods carry `Numpydoc
<https://numpydoc.readthedocs.io/en/latest/format.html>`__-style docstrings, and
all signatures are type-hinted. In-code documentation is the primary
documentation surface; keep it comprehensive and current rather than deferring
explanation to external prose.

Module layout and public API
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Implementation modules are private and underscore-prefixed (*e.g.* ``_backend.py``,
``_measurements.py``, ``_phase.py``). The supported public API is the small
surface re-exported from ``eradiate_disort/__init__.py``. Treat anything not
re-exported there as internal and subject to change.

In tests and examples, when relevant, import from the package root rather than
reaching into private modules.

Licensing and REUSE
^^^^^^^^^^^^^^^^^^^

The project is licensed under ``GPL-3.0-or-later`` (consistent with
``nanodisort`` and CDISORT) and is `REUSE <https://reuse.software/>`__-compliant.
Every source file under ``src/`` begins with an SPDX header::

    # SPDX-FileCopyrightText: 2026 Rayference
    #
    # SPDX-License-Identifier: GPL-3.0-or-later

Files that do not carry their own header inherit copyright and license from the
``**`` default annotation in ``REUSE.toml``, and the license texts live under
``LICENSES/``. When adding a new source file, copy the SPDX header above; when
adding a file of a different license, register it in ``REUSE.toml`` and drop the
corresponding text in ``LICENSES/``. Verify compliance with
``pixi run lint-reuse``.

.. important::

    CDISORT code is licensed under the terms of the GPL, while Eradiate is
    licensed under terms of the LGPL. For this reason, this backend must be
    only be downstream (in terms of dependency graph) of Eradiate:
    ``eradiate-disort`` can import ``eradiate``, but not the reverse, unless
    done on an entirely optional and opt-in basis, *e.g.* through a plugin
    system.

pre-commit
^^^^^^^^^^

Formatting and lint conventions are enforced automatically through pre-commit
hooks, defined in ``.pre-commit-config.yaml``. Install the hooks once with
`pre-commit <https://pre-commit.com/>`__ or `prek <https://prek.j178.dev/>`__
(recommended). If necessary, ``pre-commit run --all-files`` can run them across
the entire project.

Because the hooks run on commit, do not rely on an agent's or your own claim
that a file is formatted or lint-clean — let pre-commit and CI be the source of
truth.

.. _dev-releasing:

Releasing
---------

Versioning
^^^^^^^^^^

The project losely follows semantic versioning with an optional pre-release
suffix (``dev`` / ``rc`` / ``final``); the current version is specified in
``pyproject.toml`` and is managed with
`bump-my-version <https://github.com/callowayproject/bump-my-version>`__.
Bump the version with::

    RELEASE_VERSION=<new-version> pixi run bump

``pixi run bump-show`` lists the possible next versions and ``pixi run bump-dry``
performs a verbose dry run. The bump configuration does not commit or tag
automatically; make the version-bump commit yourself, and tag it
``v<new-version>``.

Continuous integration
^^^^^^^^^^^^^^^^^^^^^^

The ``.github/workflows/test.yml`` workflow runs the test suite on pushes and
pull requests to ``main`` across an OS × interpreter matrix. Draft pull requests
are skipped.

Publishing
^^^^^^^^^^

The ``.github/workflows/cd.yml`` workflow builds the wheel and sdist and
publishes to PyPI. It is triggered manually with an ``upload`` input: ``0``
builds only, ``1`` publishes to PyPI, ``2`` publishes to TestPyPI. The usual
release sequence is:

1. bump;
2. push;
3. verify the test workflow is green;
4. dispatch ``cd.yml`` — to TestPyPI first if in doubt, then to PyPI;
5. when the wheel is published on PyPI, tag the published commit;
6. (optional) create a release on GitHub.
