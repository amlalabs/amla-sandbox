"""AI SDK-style sandbox tool for Amla sandbox.

This module provides a dead-simple API for creating sandbox-enabled tools
for AI agents, hiding WASM/capability complexity behind sensible defaults.

Quick Start::

    from amla_sandbox import create_sandbox_tool, create_bash_tool

    # Layer 0: Just works - JavaScript sandbox
    sandbox = create_sandbox_tool()

    # Layer 0b: Shell-first (for bash-style usage)
    bash = create_bash_tool()
    bash.run("echo 'hello' | tr 'a-z' 'A-Z'")  # Shell is default

    # Layer 1: With tools
    sandbox = create_sandbox_tool(tools=[get_weather, send_email])

    # Layer 2: With constraints
    sandbox = create_sandbox_tool(
        tools=[transfer_money],
        constraints={"transfer_money": {"amount": "<=1000"}},
        max_calls=50,
    )

    # Use with LangChain/LangGraph
    from langchain.agents import initialize_agent
    agent = initialize_agent(tools=[sandbox], ...)

The developer shouldn't need to understand WASM, PIC, or capability tokens
to get started. They just get a sandbox that's safer than subprocess.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence, cast

from .capabilities import Constraint, ConstraintSet, MethodCapability
from .sandbox import Sandbox
from .tools import capability_from_function, tool_from_function
from .langgraph import SandboxTool

# Default call limit per tool when none specified
DEFAULT_MAX_CALLS = 100


def create_sandbox_tool(
    tools: Sequence[Callable[..., Any]] | None = None,
    *,
    default_language: str,
    constraints: dict[str, dict[str, Any]] | None = None,
    max_calls: int | dict[str, int] | None = None,
    workspace: str | None = None,
) -> SandboxTool:
    """Create a sandbox tool for AI agents.

    This is the simplest way to give an AI agent sandbox access with optional
    tool calling. Complexity is hidden - just pass your functions and go.

    Args:
        tools: Python functions to expose as tools. Each function becomes
            callable from both JavaScript and shell (`tool name.func --arg val`).
        default_language: Default language for run() when not specified.
            Either "javascript" or "shell". Required - be explicit about
            what kind of sandbox you want.
        constraints: Per-tool parameter constraints. Keys are tool names,
            values are dicts mapping param names to constraints like:
            - "<=1000" or ">=0" (numeric bounds)
            - ["USD", "EUR"] (allowed values)
            - "startswith:/api/" (string prefix)
        max_calls: Maximum calls allowed. Can be:
            - int: Same limit for all tools
            - dict: Per-tool limits {"transfer_money": 10}
        workspace: VFS mount point for agent files (default: /data).
            Currently unused but reserved for future VFS mounting.

    Returns:
        LangChain/LangGraph compatible SandboxTool.

    Examples::

        # JavaScript sandbox
        sandbox = create_sandbox_tool(default_language="javascript")
        result = sandbox.run("console.log('hello')")

        # Shell sandbox
        sandbox = create_sandbox_tool(default_language="shell")
        result = sandbox.run("echo 'hello' | tr 'a-z' 'A-Z'")

        # With tools
        sandbox = create_sandbox_tool(
            tools=[get_weather, search_db],
            default_language="javascript",
        )
        result = sandbox.run("const w = await get_weather({city: 'SF'}); console.log(w);")
    """
    tools_list = list(tools) if tools else []
    tool_map = {func.__name__: func for func in tools_list}

    # Convert functions to ToolDefinitions
    tool_defs = [tool_from_function(func) for func in tools_list]

    # Build capabilities with constraints
    caps = _build_capabilities(tools_list, constraints, max_calls)

    # Create handler
    def handler(method: str, params: dict[str, Any]) -> Any:
        # Strip mcp: prefix if present (WASM runtime adds it)
        name = method.removeprefix("mcp:")
        if name not in tool_map:
            raise ValueError(f"Unknown tool: {method}")
        return tool_map[name](**params)

    # Create sandbox
    sandbox = Sandbox(
        tools=tool_defs,
        capabilities=caps,
        tool_handler=handler if tools_list else None,
    )

    return SandboxTool(
        sandbox=sandbox,
        tools=tools_list,
        _tool_map=tool_map,
        default_language=default_language,
    )


def create_bash_tool(
    tools: Sequence[Callable[..., Any]] | None = None,
    *,
    constraints: dict[str, dict[str, Any]] | None = None,
    max_calls: int | dict[str, int] | None = None,
    workspace: str | None = None,
) -> SandboxTool:
    """Create a sandbox tool for AI agents (backward-compatible API).

    This function exists for backward compatibility. For new code, prefer
    create_sandbox_tool() which requires explicit default_language.

    Note: Despite the name, this defaults to JavaScript execution for
    backward compatibility with existing code.

    Args:
        tools: Python functions to expose as tools.
        constraints: Per-tool parameter constraints.
        max_calls: Maximum calls allowed per tool.
        workspace: VFS mount point (reserved for future use).

    Returns:
        LangChain/LangGraph compatible SandboxTool with JavaScript as default.

    Examples::

        # JavaScript is the default (for backward compatibility)
        bash = create_bash_tool()
        result = bash.run("console.log('hello')")  # JavaScript

        # Shell requires explicit language
        result = bash.run("echo 'hello' | tr 'a-z' 'A-Z'", language="shell")

        # With tools
        bash = create_bash_tool(tools=[get_weather])
        result = bash.run("await get_weather({city: 'SF'})")  # JavaScript
    """
    return create_sandbox_tool(
        tools=tools,
        default_language="javascript",
        constraints=constraints,
        max_calls=max_calls,
        workspace=workspace,
    )


def _build_capabilities(
    tools: list[Callable[..., Any]],
    constraints: dict[str, dict[str, Any]] | None,
    max_calls: int | dict[str, int] | None,
) -> list[MethodCapability]:
    """Build capabilities from tools with optional constraints."""
    caps: list[MethodCapability] = []

    for func in tools:
        name = func.__name__

        # Get constraints for this tool
        tool_constraints = constraints.get(name, {}) if constraints else {}
        constraint_set = _parse_constraints(tool_constraints)

        # Get max_calls for this tool
        if max_calls is None:
            limit = DEFAULT_MAX_CALLS
        elif isinstance(max_calls, int):
            limit = max_calls
        else:
            limit = max_calls.get(name, DEFAULT_MAX_CALLS)

        # Create capability
        cap = capability_from_function(
            func,
            constraints=constraint_set if not constraint_set.is_empty() else None,
            max_calls=limit,
        )
        caps.append(cap)

    return caps


def _parse_constraints(spec: dict[str, Any]) -> ConstraintSet:
    """Parse constraint dict into ConstraintSet.

    Supports:
    - "<=100" or ">=0" or "<10" or ">5" or "==42" (numeric)
    - ["a", "b", "c"] (allowed values / enum)
    - "startswith:/api/" (string prefix)
    """
    constraints: list[Constraint] = []

    for param, value in spec.items():
        if isinstance(value, str):
            # Parse string constraint
            constraint = _parse_string_constraint(param, value)
            if constraint:
                constraints.append(constraint)
        elif isinstance(value, list):
            # Enum constraint - cast to list[Any] since spec values can be any type
            constraints.append(Constraint.is_in(param, cast(list[Any], value)))
        elif isinstance(value, (int, float)):
            # Exact value constraint
            constraints.append(Constraint.eq(param, value))

    return ConstraintSet(constraints)


def _parse_string_constraint(param: str, spec: str) -> Constraint | None:
    """Parse a string constraint specification.

    Examples:
    - "<=1000" -> Constraint.le(param, 1000)
    - ">=0" -> Constraint.ge(param, 0)
    - "<100" -> Constraint.lt(param, 100)
    - ">10" -> Constraint.gt(param, 10)
    - "==42" -> Constraint.eq(param, 42)
    - "startswith:/api/" -> Constraint.starts_with(param, "/api/")

    Returns:
        Parsed Constraint, or None if spec is invalid/unrecognized.
    """
    import logging

    logger = logging.getLogger(__name__)

    try:
        # Check for comparison operators
        if spec.startswith("<="):
            return Constraint.le(param, _parse_number(spec[2:]))
        if spec.startswith(">="):
            return Constraint.ge(param, _parse_number(spec[2:]))
        if spec.startswith("=="):
            return Constraint.eq(param, _parse_number(spec[2:]))
        if spec.startswith("<"):
            return Constraint.lt(param, _parse_number(spec[1:]))
        if spec.startswith(">"):
            return Constraint.gt(param, _parse_number(spec[1:]))
    except ValueError as e:
        # Invalid numeric value in constraint - log and skip
        logger.warning(
            "Skipping invalid constraint for '%s': %s (spec: '%s')",
            param,
            e,
            spec,
        )
        return None

    # Check for string constraints
    if spec.startswith("startswith:"):
        prefix = spec.removeprefix("startswith:")
        return Constraint.starts_with(param, prefix)

    # Unknown format - ignore
    return None


def _parse_number(s: str) -> int | float:
    """Parse a numeric string to int or float.

    Args:
        s: String to parse (e.g., "100", "3.14", "-5").

    Returns:
        Parsed number as int or float.

    Raises:
        ValueError: If s is not a valid numeric string.
    """
    s = s.strip()
    try:
        if "." in s:
            return float(s)
        return int(s)
    except ValueError as e:
        raise ValueError(f"Invalid numeric value in constraint: '{s}'") from e
