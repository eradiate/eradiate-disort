:hide-toc:
:layout: landing

eradiate-disort documentation
=============================

**Date**: |today| | **Version**: |version|

A DISORT radiative transfer backend for the
`Eradiate radiative transfer model <https://eradiate.eu>`_\ , built on
`nanodisort <https://github.com/eradiate/nanodisort>`_ Python bindings to
CDISORT.

.. grid:: 1 1 2 3
    :gutter: 2
    :padding: 0

    .. grid-item-card:: :iconify:`material-symbols:book-2 height=1.5em` Docs
        :link: user_guide/pipeline_engine
        :link-type: doc

        Read the user guide and design notes.

    .. grid-item-card:: :iconify:`material-symbols:api height=1.5em` API
        :link: api/eradiate_disort
        :link-type: doc

        Browse the API reference.

    .. grid-item-card:: :iconify:`simple-icons:github height=1.5em` GitHub
        :link: https://github.com/eradiate/eradiate-disort/

        Browse the source code.

.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: User Guide

    user_guide/pipeline_engine

.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: API Reference

    api/eradiate_disort
    api/eradiate_disort.pipelines

.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: Design

    design/pipeline_engine
