# Pipeline Engine Design Note

## Context

The Eradiate radiative transfer model uses computational pipelines for
postprocessing simulation results. These pipelines involve chained operations
(CKD quadrature aggregation, spectral response function application, BRDF
computation, etc.) organized as directed acyclic graphs.

Previously, Eradiate used [Hamilton](https://github.com/dagworks-inc/hamilton)
for pipeline management. While Hamilton is feature-rich, several friction points
motivated exploring an alternative:

1. **Steep learning curve**: Heavy reliance on decorators (`@config.when`,
   `@resolve`, `@parameterize`, `@extract_fields`) requires significant upfront
   investment.
2. **Implicit dependencies**: Function parameter names implicitly define the DAG
   structure, making the graph hard to reason about.
3. **Limited dynamic construction**: Building pipelines conditionally requires
   working around decorator semantics.
4. **Debugging difficulty**: Stacked decorators obscure the execution flow.

## Design Goals

- **Simplicity**: Clear, explicit API with minimal indirection.
- **Flexibility**: Build pipelines programmatically with full Python control.
- **Debuggability**: Straightforward call stack, no decorator layers.
- **Completeness**: Validation hooks, input injection, subgraph extraction,
  visualization.

## Architecture

### Overview

The engine is built on [networkx](https://networkx.org/) and consists of two
core types and a validation utilities module:

```
eradiate_disort.pipelines
├── Pipeline    — DAG manager and executor
├── Node        — Single computation step
└── validation  — Reusable validator factories
```

### Node

An [attrs](https://www.attrs.org/)-decorated class representing a computation
step:

- **`func`**: The callable to execute. Its parameter names must match the names
  of its dependencies.
- **`dependencies`**: Explicit list of upstream node or virtual input names.
- **`pre_funcs` / `post_funcs`**: Hook lists for validation, logging, or
  inspection, controlled by `validate_enabled` and the pipeline's global
  validation flag.
- **`metadata`**: Arbitrary key-value tags (used in visualization and for
  user-defined queries).

### Pipeline

The `Pipeline` class manages a `networkx.DiGraph` and a node registry. Its
internal state:

- `_graph: nx.DiGraph` — DAG structure (includes both computation nodes and
  virtual input placeholders).
- `_nodes: dict[str, Node]` — Maps names to `Node` objects.
- `_virtual_inputs: set[str]` — Tracks dependencies with no backing node.
- `_cache: dict[str, Any]` — Per-execution result cache.
- `validate_globally: bool` — Global toggle for pre/post functions.

## Key Design Decisions

### Imperative API over Decorators

The engine uses method calls (`pipeline.add_node(...)`) rather than decorators
to define nodes. This makes conditional logic, dynamic construction, and
debugging straightforward:

```python
# Plain Python — no framework DSL
if mode == "ckd":
    pipeline.add_node("aggregate", aggregate_ckd, dependencies=["raw"])
else:
    pipeline.add_node("aggregate", lambda raw: raw, dependencies=["raw"])
```

Compared to Hamilton's equivalent:

```python
@config.when(mode="ckd")
def aggregate(raw): ...

@config.when(mode="mono")
def aggregate(raw): ...
```

### Explicit Dependencies

Dependencies are declared as a list of names rather than inferred from function
parameter names. This decouples the function signature from the graph structure
and makes the DAG explicit.

### Virtual Inputs

Dependencies referencing non-existent nodes are automatically classified as
**virtual inputs** and tracked in a separate set. This emerged as a natural
generalization: rather than requiring all source data to be wrapped in
no-argument nodes, external data can be injected at execution time via the
`inputs` parameter.

Virtual inputs are represented in the graph as placeholder nodes (stored with
`node=None` in graph data) so that networkx algorithms (topological sort,
ancestor queries) work uniformly.

A virtual input can later be "promoted" to a real computation node by calling
`add_node()` with the same name.

### Unified `inputs` Parameter

Earlier iterations had a separate `bypass_data` parameter for skipping node
computation. This was unified into a single `inputs` dict that serves two
purposes:

- **Virtual input values**: Provides data for dependencies without backing
  nodes.
- **Node bypasses**: Provides pre-computed values for existing nodes, skipping
  their execution and excluding their upstream dependencies from the execution
  plan.

The `execute()` method distinguishes between the two by checking whether the
key exists in `_nodes` or `_virtual_inputs`.

### Generalized Pre/Post Hooks

Earlier iterations used `pre_validators` / `post_validators` (and a separate
`add_interceptor()` mechanism for side-effect callbacks). These were unified
into `pre_funcs` / `post_funcs`:

- **Pre-functions** receive the gathered inputs dict and can validate, log, or
  transform (though transformation is not encouraged).
- **Post-functions** receive the node output and can validate, log, or capture
  intermediate values.

This subsumes both validation and interception use cases with a single, simpler
mechanism. A global toggle (`validate_globally`) and per-node toggle
(`validate_enabled`) control whether hooks run.

### Visualization Integrated into Pipeline

Rather than a separate visualization module, export methods (`write_dot`,
`write_png`, `write_svg`, `visualize`) live directly on `Pipeline`. This keeps
the API cohesive and supports Jupyter auto-display via `_repr_svg_()`.

Visualization uses [pydot](https://github.com/pydot/pydot) (optional
dependency) with a fixed style vocabulary:

- Computation nodes: filled blue boxes.
- Virtual inputs: filled gold ellipses.
- Highlighted nodes: coral fill.
- Metadata rendered in italics.
- Optional legend subgraph.

## Execution Algorithm

1. **Determine outputs**: Use requested outputs, or default to all leaf nodes.
2. **Classify inputs**: Split `inputs` into node bypasses vs. virtual input
   values; reject unknown keys.
3. **Validate virtual inputs**: Determine which virtual inputs are required
   (considering bypasses that may eliminate upstream paths) and ensure all are
   provided.
4. **Validate connectivity**: Confirm all outputs are reachable from the
   combination of roots (parameter-less nodes), virtual inputs, and bypasses.
5. **Clear cache**: Each `execute()` starts fresh.
6. **Populate cache**: Insert bypass values and virtual input values.
7. **Compute execution order**: Topological sort, filtering to only nodes in
   the dependency chain of requested outputs (excluding bypassed and virtual
   nodes).
8. **Execute nodes**: For each node in order:
   - Gather inputs from cache.
   - Run pre-functions (if validation enabled).
   - Call `node.func(**inputs)`.
   - Run post-functions (if validation enabled).
   - Cache result.
9. **Return results**: Extract requested outputs from cache.

Recursive dependency resolution in `_execute_node` handles edge cases where
topological order alone isn't sufficient (e.g. subgraph boundaries).

## Comparison with Hamilton

| Aspect | Hamilton | Pipeline Engine |
|--------|----------|-----------------|
| API style | Decorator-based | Imperative method calls |
| Dependencies | Implicit (parameter names) | Explicit list |
| Conditional nodes | `@config.when()` | Python `if`/`else` |
| Dynamic construction | Difficult | Natural |
| Validation | Separate add-on | Built-in pre/post hooks |
| Subgraph extraction | Not built-in | `extract_subgraph()` |
| Input injection | Override inputs dict | Unified `inputs` parameter |
| Debugging | Complex (decorator stack) | Standard Python call stack |

## File Organization

```
src/eradiate_disort/pipelines/
├── __init__.py       — Public API: Pipeline, Node, validation
├── core.py           — Pipeline and Node implementation (~1050 lines)
└── validation.py     — Validator factory functions (~325 lines)

tests/pipelines/
├── test_core.py          — Core functionality (46 tests)
├── test_validation.py    — Validator functions (33 tests)
├── test_virtual_inputs.py — Virtual input support (30 tests)
└── test_integration.py   — End-to-end workflows (12 tests)
```

## Dependencies

- **networkx** (required): Topological sort, cycle detection, ancestor queries.
  All graph operations are O(V + E).
- **pydot** (optional): Graphviz DOT generation for visualization.
- **IPython** (optional): Jupyter notebook inline display.

All are lightweight and already available through the Eradiate ecosystem.

## Design Evolution

The implementation went through several iterations:

1. **Initial implementation** (commit `d326641`): Core Pipeline/Node with
   `bypass_data`, `add_interceptor()`, `pre_validators`/`post_validators`,
   separate visualization module.
2. **Virtual inputs** (commit `0f2e421`): Added automatic virtual input
   detection, introspection methods, refactored to attrs.
3. **Visualization consolidation** (commits `585377d`–`e22f3cf`): Integrated
   visualization into Pipeline, added legend support, Jupyter auto-display.
4. **API simplification** (commits `5ef781a`–`2801956`): Renamed `bypass_data`
   to `inputs`, removed `add_interceptor()`, generalized validators to
   `pre_funcs`/`post_funcs`.

## Future Directions

Potential enhancements if needed:

- **Parallel execution**: Compute independent nodes concurrently.
- **Progress callbacks**: For long-running pipelines.
- **Pipeline composition**: Merge multiple pipelines.
- **Serialization**: Save/load pipeline definitions.
- **Integration into Eradiate core**: Migrate from eradiate-disort to the main
  Eradiate package for use in all postprocessing workflows.
