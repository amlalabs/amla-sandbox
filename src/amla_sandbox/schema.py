"""JSON Schema generation from Python types.

This module provides utilities for converting Python type hints to JSON Schema,
which is used to generate tool definitions for the sandbox runtime.

The conversion supports:
- Basic types: str, int, float, bool, list, dict
- Optional types: Optional[T] becomes nullable
- List types: List[T] with item schema
- Pydantic models: Uses model_json_schema()
- Dataclasses: Generates schema from fields

Example::

    >>> from amla_sandbox.schema import python_type_to_json_schema
    >>> python_type_to_json_schema(str)
    'string'
    >>> python_type_to_json_schema(list[int])
    {'type': 'array', 'items': {'type': 'integer'}}
"""

from __future__ import annotations

__all__ = [
    "python_type_to_json_schema",
    "dataclass_to_schema",
    "extract_param_description",
]

import dataclasses
from typing import Any, Union, get_args, get_origin


def python_type_to_json_schema(py_type: type) -> str | dict[str, Any]:
    """Convert Python type hint to JSON Schema type.

    Supports:
    - Basic types: str, int, float, bool, list, dict
    - Pydantic models: Uses model_json_schema()
    - Dataclasses: Generates schema from fields
    - Optional[T]: Adds nullable to the type
    - List[T]: Array with item type

    Args:
        py_type: A Python type hint.

    Returns:
        JSON Schema type as a string (e.g., "string") or dict (e.g., {"type": "array"}).

    Example::

        >>> python_type_to_json_schema(str)
        'string'
        >>> python_type_to_json_schema(int)
        'integer'
        >>> python_type_to_json_schema(list[str])
        {'type': 'array', 'items': {'type': 'string'}}
    """
    # Handle None type
    if py_type is type(None):
        return {"type": "null"}

    # Basic types
    if py_type is str:
        return "string"
    if py_type is int:
        return "integer"
    if py_type is float:
        return "number"
    if py_type is bool:
        return "boolean"
    if py_type is list:
        return "array"
    if py_type is dict:
        return "object"

    # Check for generic types (Optional, List, etc.)
    origin = get_origin(py_type)
    args = get_args(py_type)

    # Optional[T] = Union[T, None]
    if origin is Union:
        non_none_args = [a for a in args if a is not type(None)]
        if len(non_none_args) == 1:
            # This is Optional[T]
            inner_schema = python_type_to_json_schema(non_none_args[0])
            if isinstance(inner_schema, str):
                return {"type": inner_schema, "nullable": True}
            else:
                inner_schema["nullable"] = True
                return inner_schema
        # For other Union types, default to string
        return "string"

    # List[T]
    if origin is list:
        if args:
            item_schema = python_type_to_json_schema(args[0])
            if isinstance(item_schema, str):
                return {"type": "array", "items": {"type": item_schema}}
            else:
                return {"type": "array", "items": item_schema}
        return "array"

    # Dict[K, V]
    if origin is dict:
        return "object"

    # Pydantic models
    try:
        from pydantic import BaseModel

        # py_type could be a GenericAlias, so check isinstance(type) first
        if isinstance(py_type, type) and issubclass(py_type, BaseModel):  # pyright: ignore[reportUnnecessaryIsInstance]
            return py_type.model_json_schema()
    except ImportError:
        pass

    # Dataclasses
    # py_type could be a GenericAlias, so check isinstance(type) first
    if dataclasses.is_dataclass(py_type) and isinstance(py_type, type):  # pyright: ignore[reportUnnecessaryIsInstance,reportUnknownArgumentType]
        return dataclass_to_schema(py_type)

    # Default to string for unknown types
    return "string"


def dataclass_to_schema(dc_type: type) -> dict[str, Any]:
    """Convert a dataclass to JSON Schema.

    Args:
        dc_type: A dataclass type.

    Returns:
        JSON Schema object with type, properties, and required fields.

    Example::

        >>> from dataclasses import dataclass
        >>> @dataclass
        ... class Point:
        ...     x: int
        ...     y: int
        >>> dataclass_to_schema(Point)
        {'type': 'object', 'properties': {'x': {'type': 'integer'}, 'y': {'type': 'integer'}}, 'required': ['x', 'y']}
    """
    properties: dict[str, Any] = {}
    required: list[str] = []

    for dc_field in dataclasses.fields(dc_type):
        # dc_field.type can be a type, str (forward ref), or Any
        field_type: Any = dc_field.type
        if isinstance(field_type, str):
            # Forward reference - treat as string
            field_schema: str | dict[str, Any] = "string"
        else:
            # type or generic alias - pass to python_type_to_json_schema
            field_schema = python_type_to_json_schema(field_type)
        if isinstance(field_schema, str):
            properties[dc_field.name] = {"type": field_schema}
        else:
            properties[dc_field.name] = field_schema

        # Check if field has a default
        if (
            dc_field.default is dataclasses.MISSING
            and dc_field.default_factory is dataclasses.MISSING
        ):
            required.append(dc_field.name)

    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required

    return schema


def extract_param_description(docstring: str, param_name: str) -> str | None:
    """Extract parameter description from a Google-style docstring.

    Parses the Args: section of a docstring to find the description
    for a specific parameter.

    Args:
        docstring: The function's docstring.
        param_name: Name of the parameter to find.

    Returns:
        The parameter description, or None if not found.

    Example::

        >>> doc = '''Do something.
        ...
        ... Args:
        ...     name: The person's name.
        ...     age: The person's age in years.
        ... '''
        >>> extract_param_description(doc, "name")
        "The person's name."
        >>> extract_param_description(doc, "missing")
        None
    """
    if not docstring:
        return None

    lines = docstring.split("\n")
    in_args = False

    for line in lines:
        stripped = line.strip()

        if stripped.lower().startswith("args:"):
            in_args = True
            continue

        if in_args:
            # Check for new section
            if stripped.endswith(":") and not stripped.startswith(param_name):
                if ":" not in stripped[:-1]:  # New section header
                    break

            # Look for our parameter
            if stripped.startswith(f"{param_name}:"):
                return stripped[len(param_name) + 1 :].strip()

    return None
