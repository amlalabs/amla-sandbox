"""Tool conversion utilities for the Amla sandbox.

This module provides utilities for converting Python functions to tool
definitions and capabilities that can be used in the sandbox.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Sequence, get_type_hints

from ..capabilities import ConstraintSet, MethodCapability
from ..schema import (
    extract_param_description,
    python_type_to_json_schema,
)
from .definition import ToolDefinition


def tool_from_function(func: Callable[..., Any]) -> ToolDefinition:
    """Convert a Python function to a ToolDefinition.

    Extracts name, docstring, and parameter types to create a tool
    definition that can be registered with the sandbox.

    Args:
        func: A Python function with type hints and docstring.

    Returns:
        ToolDefinition with JSON Schema parameters.
    """
    name = func.__name__
    doc = inspect.getdoc(func) or ""

    # Extract first line as description
    description = doc.split("\n")[0] if doc else f"Call {name}"

    # Get type hints
    hints = get_type_hints(func) if hasattr(func, "__annotations__") else {}
    hints.pop("return", None)

    # Get signature for defaults
    sig = inspect.signature(func)

    # Build JSON Schema for parameters
    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue

        # Get type hint
        param_type = hints.get(param_name, Any)
        json_type = python_type_to_json_schema(param_type)

        # Handle both string types and full schema dicts
        if isinstance(json_type, str):
            properties[param_name] = {"type": json_type}
        else:
            properties[param_name] = json_type

        # Extract parameter description from docstring
        param_desc = extract_param_description(doc, param_name)
        if param_desc:
            properties[param_name]["description"] = param_desc

        # Check if required (no default)
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    parameters: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        parameters["required"] = required

    return ToolDefinition(
        name=name,
        description=description,
        parameters=parameters,
    )


def capability_from_function(
    func: Callable[..., Any],
    constraints: ConstraintSet | None = None,
    max_calls: int | None = None,
) -> MethodCapability:
    """Generate a capability from a Python function."""
    name = func.__name__
    return MethodCapability(
        method_pattern=f"mcp:{name}",
        constraints=constraints or ConstraintSet(),
        max_calls=max_calls,
    )


def format_tool_descriptions_js(tools: Sequence[Callable[..., Any]]) -> str:
    """Format tool descriptions for JavaScript usage.

    Creates a markdown-formatted description of available tools
    suitable for inclusion in a system prompt.

    Args:
        tools: List of Python functions to describe.

    Returns:
        Formatted string with tool signatures and descriptions.

    Example::

        def search(query: str) -> list[str]:
            '''Search for information.'''
            ...

        desc = format_tool_descriptions_js([search])
        # ## Available Functions
        #
        # ### `search({query: string})`
        # Search for information.
    """
    lines = ["## Available Functions", ""]

    for func in tools:
        name = func.__name__
        doc = inspect.getdoc(func) or "No description"
        description = doc.split("\n")[0]

        # Get signature
        sig = inspect.signature(func)
        hints = get_type_hints(func) if hasattr(func, "__annotations__") else {}

        # Build JS-style parameter list
        params: list[str] = []
        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue
            param_type = hints.get(param_name, Any)
            js_type = python_type_to_json_schema(param_type)
            # For display, convert dict schemas to readable type name
            if isinstance(js_type, dict):
                js_type_str = js_type.get("type", "object")
            else:
                js_type_str = js_type
            if param.default is inspect.Parameter.empty:
                params.append(f"{param_name}: {js_type_str}")
            else:
                params.append(f"{param_name}?: {js_type_str}")

        params_str = ", ".join(params)
        lines.append(f"### `{name}({{{params_str}}})`")
        lines.append(f"{description}")
        lines.append("")

    return "\n".join(lines)
