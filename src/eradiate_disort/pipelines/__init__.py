"""Lightweight pipeline engine for eradiate-disort.

This module provides a simple, networkx-based DAG pipeline engine for
building and executing computational workflows. It features an imperative
API, lazy evaluation, input injection, and flexible pre/post hooks.
"""

from . import validation
from .core import Node, Pipeline

__all__ = ["Pipeline", "Node", "validation"]
