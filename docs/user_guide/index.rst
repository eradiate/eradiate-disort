User guide
==========

This document introduces the Eradiate DISORT backend in general terms for users,
covering the most important concepts. The :doc:`../examples/index` section
provides a progressive series of runnable walkthrough tutorials. For a
developer-oriented introduction, see the :doc:`../dev/index`.

Overview
--------

The Eradiate DISORT backend is a fast alternative to Eradiate's default
radiometric kernel based on the Mitsuba 3 renderer when a 1D plane-parallel
geometry is used. Through the nanodisort :cite:p:`LeroyNanoDisort` Python
binding package, it provides an interface to CDISORT
:cite:p:`Buras2011CDISORTCorrection`, a C implementation of the DISORT algorithm
:cite:p:`Stamnes1988DISORT,Stamnes2000DISORTReport`.

This backend trades flexibility for speed. It provides noise-free output
regardless of the simulated scene (surface model, atmospheric profile or aerosol
layers) and can output multiple radiances and fluxes without added computational
burden. The `nanodisort documentation <https://nanodisort.readthedocs.io>`__
further elaborates on the reasons why nanodisort wraps CDISORT rather than
another DISORT implementation, as well as on the updates CDISORT includes
compared to the original Fortran implementation of DISORT.

Using the DISORT backend is recommended if:

- You are simulating radiative transfer in a 1D plane-parallel geometry.
- You don't need to account for polarization (CDISORT simulates only radiative
  intensity).

.. important::

    Being an interface to nanodisort, itself wrapping CDISORT, this backend is
    licensed under the terms of the GNU General Public License, version 3 or
    later (GPL-3.0-or-later). It however interoperates well with Eradiate, on
    which it depends, without this implying additional licensing constraints on
    Eradiate.

Installation
------------

From PyPI
    Use your favourite package manager, *e.g.* ::

        pip install eradiate[kernel,recommended] eradiate-disort

    Note that the Eradiate core dependency must be installed manually.

From sources (developer setup)
    See the :doc:`/dev/index`.

Verifying the installation
    Check the installed version and confirm the extension loads::

        python -c "import eradiate_disort; print(eradiate_disort.__version__)"

Getting started
---------------

Running a simulation requires three steps:

1. build an :class:`~eradiate.experiments.AtmosphereExperiment`;
2. pass it to :class:`~eradiate_disort.DisortBackend`, and
3. inspect the returned :class:`xarray.DataTree`.

Internally, the :meth:`DisortBackend.run() <eradiate_disort.DisortBackend.run>`
method calls
:meth:`DisortBackend.validate() <eradiate_disort.DisortBackend.validate>`
(rejects unsupported configurations),
:meth:`DisortBackend.process() <eradiate_disort.DisortBackend.process>`
(runs the spectral loop), and
:meth:`DisortBackend.postprocess() <eradiate_disort.DisortBackend.postprocess>`
(assembles the output) in sequence.

Before building any experiment, choose a spectral mode with
:func:`eradiate.set_mode`. The ``"ckd"`` (correlated-*k* distribution) is
recommended for production use; ``"mono"`` is available for single-wavelength
calculations.

The following example simulates TOA radiance over a Lambertian surface below a
standard molecular atmosphere at 550 nm, with a solar zenith angle of 30°:

.. code-block:: python

    import numpy as np
    import eradiate
    import eradiate_disort as ed
    from eradiate.experiments import AtmosphereExperiment
    from eradiate.units import unit_registry as ureg

    eradiate.set_mode("ckd")

    exp = AtmosphereExperiment(
        geometry={
            "type": "plane_parallel",
            "toa_altitude": 100.0 * ureg.km,
            "zgrid": np.linspace(0, 100, 101) * ureg.km,  # keep grid coarse for performance
        },
        surface={"type": "lambertian", "reflectance": 0.25},
        atmosphere={"type": "molecular"},
        illumination={"type": "directional", "zenith": 30.0, "azimuth": 0.0},
        measures={
            "type": "disort",
            "id": "measure",  # "measure" is also the default
            "construct": "hplane",  # see DisortMeasure API docs for all constructors
            "azimuth": 0.0,
            "zeniths": np.arange(-75.0, 76.0, 5.0),
            "srf": {"type": "delta", "wavelengths": [550.0]},
        },
    )

    backend = ed.DisortBackend()
    result = backend.run(exp)

    # TOA radiance along the principal plane
    radiance = result["measure/uu"].sel(z=100e3)

The :meth:`DisortBackend.run() <eradiate_disort.DisortBackend.run>` method
returns an :class:`xarray.DataTree` with one subtree per measure, keyed by the
measure's ``id`` field (a default ID is assigned if not specified). Within each
subtree, the main variables are:

- ``uu`` — upwelling spectral radiance [W m⁻² sr⁻¹ nm⁻¹], with dimensions
  ``(w, z, vza, vaa)``.
- ``rfldir`` — direct (beam) downward irradiance [W m⁻² nm⁻¹], with dimensions
  ``(w, z)``.
- ``rfldn`` — diffuse downward irradiance [W m⁻² nm⁻¹], with dimensions
  ``(w, z)``.
- ``flup`` — diffuse upward irradiance [W m⁻² nm⁻¹], with dimensions
  ``(w, z)``.

The ``z`` coordinate is altitude in metres; ``vza`` and ``vaa`` are the viewing
zenith and azimuth angles in degrees.

Supported features
------------------

The backend trades flexibility for speed, so it accepts a deliberately
restricted subset of Eradiate scene configurations. Unsupported configurations
are rejected early by
:meth:`DisortBackend.validate() <eradiate_disort.DisortBackend.validate>`, which
raises a :class:`TypeError` before any solving takes place.

Illumination and surface
^^^^^^^^^^^^^^^^^^^^^^^^^

Only :class:`~eradiate.scenes.illumination.DirectionalIllumination` (a
collimated beam) and a :class:`~eradiate.scenes.bsdfs.LambertianBSDF` (a diffuse,
spectrally varying surface reflectance) are supported. Thermal emission is
disabled; the backend computes solar (shortwave) radiative transfer only.

Atmospheres and phase functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Any of :class:`~eradiate.scenes.atmosphere.MolecularAtmosphere`,
:class:`~eradiate.scenes.atmosphere.ParticleLayer`,
:class:`~eradiate.scenes.atmosphere.HomogeneousAtmosphere` and
:class:`~eradiate.scenes.atmosphere.HeterogeneousAtmosphere` (an arbitrary
stack of the former) are accepted; a scene with no atmosphere is also valid.

Scattering is represented through Legendre moments derived from each component's
phase function. Isotropic, Rayleigh and tabulated particle phase functions are
supported; any other type raises :class:`NotImplementedError`. In a
multi-component (heterogeneous) atmosphere, per-component phase functions are
blended layer by layer, weighted by scattering coefficient.

Measures
^^^^^^^^

The backend accepts only :class:`~eradiate_disort.DisortMeasure` instances. A
single run may contain several measures, but — because CDISORT solves on a single
shared viewing grid (``umu``/``phi``) — **at most one** measure may record the
angular radiance field (*i.e.* declare a ``direction_layout``). Flux quantities
are always computed, so any number of flux-only measures may coexist with the
one radiance-mode measure.

A :class:`~eradiate_disort.DisortMeasure` records output at a set of vertical
levels, specified through either ``z_levels`` (altitudes, snapped to the nearest
grid boundary) or ``utau`` (cumulative optical depths from the TOA); the two are
mutually exclusive, and the default is the TOA/surface pair. Viewing directions
for radiance-mode measures are set through a direction layout, most conveniently
built with the class-method constructors:

.. list-table::

    * - :meth:`~eradiate_disort.DisortMeasure.hplane`
      - Hemispherical plane — zeniths sampled within a principal plane at a
        fixed azimuth.
    * - :meth:`~eradiate_disort.DisortMeasure.aring`
      - Azimuth ring — azimuths sampled at a fixed zenith.
    * - :meth:`~eradiate_disort.DisortMeasure.grid`
      - Full zenith × azimuth grid.

A measure with no direction layout runs CDISORT in flux-only mode (``onlyfl``),
which skips the angular radiance computation entirely.

Spectral modes
^^^^^^^^^^^^^^

Both of Eradiate's spectral modes are supported, selected with
:func:`eradiate.set_mode` before building the experiment: ``"mono"``
(monochromatic) and ``"ckd"`` (correlated-*k* distribution, recommended for
production). The driving measure's spectral response function defines the
spectral grid shared by all measures.

Solver settings
^^^^^^^^^^^^^^^

A few numerical parameters are exposed on :class:`~eradiate_disort.DisortBackend`:

- ``nstr`` (default 16) — the number of streams (angular discretization). Higher
  values improve accuracy at increased computational cost.
- ``nmom`` (default 16) — the number of Legendre moments used to represent phase
  functions and the BRDF.
- ``intensity_correction`` — ``"buras_emde"`` (default) uses the actual phase
  function values and is more accurate for sharply peaked phase functions;
  ``"nakajima_tanaka"`` relies on Legendre moments only and is always available.

Output structure
----------------

:meth:`DisortBackend.run() <eradiate_disort.DisortBackend.run>` returns an
:class:`xarray.DataTree` with one subtree per measure, keyed by the measure's
``id`` (a default ID is assigned when none is given). Index a subtree with its
path, *e.g.* ``result["measure"]`` for the whole subtree or
``result["measure/uu"]`` for a single variable. Each subtree is a self-contained
:class:`xarray.Dataset`, so measures that record different quantities (radiance
vs. flux) or at different locations coexist without conflict.

Variables
^^^^^^^^^

The variables present in a subtree depend on whether the measure records the
full radiance field or fluxes only. Flux quantities are always available; the
angular radiance field ``uu`` is present only for radiance-mode measures.

``uu``
    Spectral radiance [W m⁻² sr⁻¹ nm⁻¹], dimensions ``(w, z, vza, vaa)`` (with
    an extra leading ``g`` dimension in CKD mode before spectral quadrature).

``rfldir``
    Direct-beam downward irradiance [W m⁻² nm⁻¹].

``rfldn``
    Diffuse downward irradiance [W m⁻² nm⁻¹].

``flup``
    Diffuse upward irradiance [W m⁻² nm⁻¹].

``dfdt``
    Flux divergence ``d(net flux)/d(τ)`` [dimensionless].

``uavg``, ``uavgdn``, ``uavgup``, ``uavgso``
    Mean intensities [W m⁻² sr⁻¹ nm⁻¹]: total (direct + diffuse), diffuse
    downward, diffuse upward and direct-beam, respectively.

Flux quantities carry a ``z`` dimension only, so their dimensions are
``(w, z)``.

Coordinates
^^^^^^^^^^^

``w``
    Wavelength [nm]. The driving measure's spectral response function sets the
    spectral grid shared by all measures.

``z``
    Altitude [m], following Eradiate's bottom-to-top convention (see
    `Conventions and gotchas`_). The recorded levels are those requested through
    the measure's ``z_levels`` or ``utau``.

``vza``, ``vaa``
    Viewing zenith and azimuth angles [deg] (radiance-mode measures only).

``sza``, ``saa``
    Scalar solar zenith and azimuth angles [deg], recording the illumination
    geometry of the run.

In CKD mode the per-quadrature-point ``g`` dimension is collapsed by the
post-processing pipeline's quadrature step, so the final output is already
band-integrated and carries no ``g`` coordinate.

Metadata
^^^^^^^^

Variables and coordinates carry CF-style attributes — ``long_name``, ``units``
and, where applicable, ``standard_name`` — so the output is self-describing and
plots or exports inherit sensible labels. Units follow Eradiate's convention
(spectral quantities expressed per nanometre).

Conventions and gotchas
-----------------------

CDISORT and Eradiate make different conventional choices for layer ordering,
azimuth and the optical depth reference point. The backend translates between
the two automatically, so the inputs you provide and the outputs you receive
always follow Eradiate's conventions. The points below clarify what those
conventions are and where the differences may surface.

Layer ordering
    CDISORT numbers layers and phase function moments from the top of the
    atmosphere downwards, whereas Eradiate orders altitude grids from the
    surface upwards. The backend reverses the relevant arrays at the boundary,
    so the atmospheric profile you build and the ``z`` coordinate of the output
    follow Eradiate's bottom-to-top convention: ``z`` increases with altitude,
    and altitude ``0`` is the surface.

Azimuth convention
    Eradiate's ``illumination.azimuth`` denotes the direction the beam
    *originates from* (the source direction), while CDISORT's internal ``phi0``
    denotes the direction the beam *travels towards* — the two differ by 180°.
    The backend applies this offset for you. Specify the illumination azimuth
    using Eradiate's convention, and read the view azimuth (``vaa``) of the
    output in the same convention.

Optical depth reference point
    When recording at specific levels, a :class:`~eradiate_disort.DisortMeasure`
    accepts either ``z_levels`` (altitudes, in Eradiate's bottom-to-top
    convention) or ``utau`` (cumulative optical depth). The two are mutually
    exclusive. Cumulative optical depth is measured **from the top of the
    atmosphere downwards**: ``utau`` is 0 at the TOA and equal to the total
    column optical depth at the surface. This is the DISORT convention and runs
    opposite to the altitude axis, so prefer ``z_levels`` unless you have a
    specific reason to work in optical depth.

FAQ
---

*How should I define the altitude grid?*

    The default altitude grid (for 0 to 120 km, with 100 m layer height) is good
    when working with the Mitsuba backend, whose performance is not impacted by
    the grid size. On the other hand, the DISORT backend's performance is
    impacted significantly by the altitude grid, and it is generally preferrable
    to keep the grid as coarse as possible. The suggested layer height (1 km)
    ensures great performance, and can be refined if necessary.

*What is the best way to specify the sensor's spectral response function?*

    If you process a single wavelength or CKD bin, use a
    :class:`~eradiate.spectral.DeltaSRF`. If you process a single satellite band
    and want the SRF weighting done automatically, use a
    :class:`~eradiate.spectral.BandSRF`. If you do not want SRF weighting or if
    you are processing multiple overlapping bands, use a
    :class:`~eradiate.spectral.UniformSRF`.

    It is not recommended to use a :class:`~eradiate.spectral.DeltaSRF` with a
    sequence of ``wavelengths`` to perform a spectral simulation, as the actual
    CKD bin selection will eventually depend on the molecular absorption
    database binning, which can lead to unexpected behaviour.
