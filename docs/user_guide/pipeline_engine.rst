Pipeline Engine User Guide
==========================

Overview
--------

The :mod:`eradiate_disort.pipelines` module provides a lightweight,
networkx-based computational pipeline engine. It offers an imperative API for
building and executing directed acyclic graphs (DAGs) of computational tasks,
with built-in support for input injection, validation hooks, and visualization.

Quick Start
-----------

.. code-block:: python

    from eradiate_disort.pipelines import Pipeline

    # Create and populate a pipeline
    pipeline = Pipeline()
    pipeline.add_node("a", lambda: 1)
    pipeline.add_node("b", lambda: 2)
    pipeline.add_node("c", lambda a, b: a + b, dependencies=["a", "b"])

    # Execute
    results = pipeline.execute(outputs=["c"])
    print(results["c"])  # 3

Building Pipelines
------------------

Adding Nodes
^^^^^^^^^^^^

Use ``add_node()`` to register computation steps. Each node has a unique name
and a callable whose parameter names must match the names of its dependencies:

.. code-block:: python

    pipeline = Pipeline()

    pipeline.add_node(
        "raw_data",
        func=load_raw_data,
        description="Load raw data from solver",
    )
    pipeline.add_node(
        "processed",
        func=lambda raw_data: process(raw_data),
        dependencies=["raw_data"],
        description="Apply processing",
    )

``add_node()`` returns ``self``, enabling method chaining:

.. code-block:: python

    pipeline = (
        Pipeline()
        .add_node("a", lambda: 1)
        .add_node("b", lambda a: a + 1, dependencies=["a"])
        .add_node("c", lambda b: b * 2, dependencies=["b"])
    )

Cycles are automatically detected and rejected with a ``ValueError``.

Removing Nodes
^^^^^^^^^^^^^^

.. code-block:: python

    pipeline.remove_node("c")

Nodes with downstream dependents cannot be removed.

Executing Pipelines
-------------------

Basic Execution
^^^^^^^^^^^^^^^

.. code-block:: python

    results = pipeline.execute(outputs=["final"])

If ``outputs`` is omitted, all leaf nodes (nodes with no dependents) are
computed.

Lazy Evaluation
^^^^^^^^^^^^^^^

Only nodes in the dependency chain of requested outputs are executed:

.. code-block:: python

    pipeline.add_node("a", lambda: expensive_computation())
    pipeline.add_node("b", lambda: another_computation())
    pipeline.add_node("c", lambda a: process(a), dependencies=["a"])

    # Only executes "a" and "c"; "b" is skipped
    results = pipeline.execute(outputs=["c"])

Caching
^^^^^^^

Results are cached during execution, so shared dependencies are computed once:

.. code-block:: python

    pipeline.add_node("expensive", expensive_func)
    pipeline.add_node("r1", lambda expensive: f1(expensive), dependencies=["expensive"])
    pipeline.add_node("r2", lambda expensive: f2(expensive), dependencies=["expensive"])

    # "expensive" runs once, reused for both r1 and r2
    results = pipeline.execute(outputs=["r1", "r2"])

The cache is cleared at the start of each ``execute()`` call.

Intermediate Outputs
^^^^^^^^^^^^^^^^^^^^

Any node can be requested as an output, not just leaves:

.. code-block:: python

    results = pipeline.execute(outputs=["raw", "processed", "final"])

Virtual Inputs
--------------

Dependencies that don't correspond to any node are automatically treated as
**virtual inputs**. These must be provided via the ``inputs`` parameter during
execution:

.. code-block:: python

    pipeline = Pipeline()
    pipeline.add_node("b", lambda a: a + 1, dependencies=["a"])

    # "a" is a virtual input
    pipeline.get_virtual_inputs()  # ['a']

    results = pipeline.execute(outputs=["b"], inputs={"a": 10})
    print(results["b"])  # 11

Introspection
^^^^^^^^^^^^^

.. code-block:: python

    # List all virtual inputs
    pipeline.get_virtual_inputs()

    # Get virtual inputs required for specific outputs
    pipeline.get_required_inputs(outputs=["b"])

    # Check if a name is a virtual input
    pipeline.is_virtual_input("a")

Virtual Inputs Becoming Real Nodes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you later add a node with the same name as a virtual input, it becomes a
regular computation node:

.. code-block:: python

    pipeline.add_node("a", lambda: 42)  # "a" is no longer virtual

Node Bypassing
--------------

Provide values for existing nodes via ``inputs`` to skip their computation. This
is useful for injecting cached or mock data:

.. code-block:: python

    # Skip the expensive "raw_data" computation and use cached data
    results = pipeline.execute(
        outputs=["final"],
        inputs={"raw_data": cached_data}
    )

Bypassed nodes and all their upstream dependencies (if not needed by other
nodes) are excluded from execution.

Testing with Bypassing
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def test_postprocessing():
        pipeline = build_full_pipeline()
        mock_data = create_mock_data()

        results = pipeline.execute(
            outputs=["final_result"],
            inputs={"raw_data": mock_data}
        )
        assert validate_result(results["final_result"])

Pre/Post Functions
------------------

Nodes support ``pre_funcs`` and ``post_funcs`` hooks that run before and after
node execution. These can be used for validation, logging, inspection, or any
side effect:

- **Pre-functions** receive the inputs dictionary (``dict[str, Any]``)
- **Post-functions** receive the node output value

.. code-block:: python

    from eradiate_disort.pipelines import validation as pval

    pipeline.add_node(
        "radiance",
        func=compute_radiance,
        dependencies=["raw_data"],
        post_funcs=[
            pval.validate_type(xr.DataArray),
            pval.validate_dataarray_dims(["w", "y", "x"]),
            pval.validate_non_negative(),
            pval.validate_all_finite(),
        ],
    )

Controlling Validation
^^^^^^^^^^^^^^^^^^^^^^

Pre/post functions can be toggled at two levels:

.. code-block:: python

    # Disable globally
    pipeline.set_global_validation(False)

    # Disable per-node
    pipeline.add_node(
        "step",
        func=my_func,
        validate_enabled=False,  # pre/post funcs won't run for this node
    )

Built-in Validators
^^^^^^^^^^^^^^^^^^^

The ``validation`` module provides factory functions for common checks:

.. list-table::
    :header-rows: 1
    :widths: 40 60

    * - Validator
      - Checks
    * - ``validate_type(type)``
      - Output is an instance of ``type``
    * - ``validate_dataarray_dims(dims)``
      - DataArray has required dimensions
    * - ``validate_dataarray_coords(coords)``
      - DataArray has required coordinates
    * - ``validate_shape(shape)``
      - Array shape matches (supports ``None`` wildcards)
    * - ``validate_range(min, max)``
      - All values within range
    * - ``validate_no_nan()``
      - No NaN values
    * - ``validate_no_inf()``
      - No infinite values
    * - ``validate_all_finite()``
      - No NaN or infinite values
    * - ``validate_positive()``
      - All values > 0
    * - ``validate_non_negative()``
      - All values >= 0

Custom validators are plain callables:

.. code-block:: python

    def check_units(value):
        if not hasattr(value, "attrs") or "units" not in value.attrs:
            raise ValueError("Missing units attribute")

    pipeline.add_node("data", func=load, post_funcs=[check_units])

Subgraph Extraction
-------------------

Extract a minimal pipeline containing only the nodes needed for specific
outputs:

.. code-block:: python

    subgraph = pipeline.extract_subgraph(["radiance_srf"])

    # subgraph is a new, independent Pipeline
    results = subgraph.execute(inputs={"raw_data": data})

Virtual inputs required by the subgraph are preserved.

Node Metadata
-------------

Attach arbitrary key-value metadata to nodes:

.. code-block:: python

    pipeline.add_node(
        "radiance_srf",
        func=apply_srf,
        dependencies=["radiance"],
        metadata={"final": "true", "kind": "data"},
    )

Metadata is displayed in visualizations and accessible via ``node.metadata``.

Visualization
-------------

The pipeline can be visualized in several formats.

Jupyter Notebooks
^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Display inline SVG
    pipeline.visualize()

    # With node highlighting and legend
    pipeline.visualize(highlight_nodes=["radiance"], legend=True)

    # Pipelines also auto-display when they are the last expression in a cell
    pipeline

File Export
^^^^^^^^^^^

.. code-block:: python

    # Graphviz DOT format
    pipeline.write_dot("pipeline.dot")

    # PNG image
    pipeline.write_png("pipeline.png")

    # SVG image
    pipeline.write_svg("pipeline.svg")

All export methods accept ``highlight_nodes`` and ``legend`` parameters.

Text Summary
^^^^^^^^^^^^

.. code-block:: python

    pipeline.print_summary()

Outputs a text listing of nodes in execution order with dependencies, metadata,
and pre/post function counts.

Rendering DOT Files
^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    dot -Tpng pipeline.dot -o pipeline.png
    dot -Tsvg pipeline.dot -o pipeline.svg
    dot -Tpdf pipeline.dot -o pipeline.pdf

Pipeline Introspection
----------------------

.. code-block:: python

    # List nodes in topological order
    pipeline.list_nodes()

    # Get a node object
    node = pipeline.get_node("radiance")
    print(node.name, node.description, node.dependencies)

    # Pretty-print a node (requires rich)
    node.pprint()
