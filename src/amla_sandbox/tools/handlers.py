"""Tool handler creation for the Amla sandbox.

This module provides utilities for creating tool handlers that
route calls from the sandbox to Python functions.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence


def create_tool_handler(
    tools: Sequence[Callable[..., Any]],
) -> Callable[[str, dict[str, Any]], Any]:
    """Create a tool handler that routes calls to Python functions.

    This is the bridge between the sandbox and your Python tools.
    When the agent calls a tool from within the sandbox, this handler
    executes the corresponding Python function.

    Args:
        tools: List of Python functions to expose as tools.

    Returns:
        Handler function for the sandbox.

    Example::

        def add(a: float, b: float) -> float:
            return a + b

        handler = create_tool_handler([add])
        result = handler("add", {"a": 1, "b": 2})  # Returns 3
    """
    tool_map = {func.__name__: func for func in tools}

    def handler(method: str, params: dict[str, Any]) -> Any:
        # Strip mcp: prefix if present (WASM runtime adds it)
        name = method.removeprefix("mcp:")
        if name not in tool_map:
            raise ValueError(f"Unknown tool: {method}")
        func = tool_map[name]
        return func(**params)

    return handler
