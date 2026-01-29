"""Tool ingestion from external frameworks.

This module provides converters to import tools from popular agent frameworks
into amla-sandbox's ToolDefinition format.

Supported frameworks:
- LangChain (@tool decorator and BaseTool classes)
- OpenAI (function calling format)
- Anthropic (tool use format)

Example::

    from langchain.tools import tool
    from amla_sandbox import create_sandbox_tool
    from amla_sandbox.tools import from_langchain

    @tool
    def search(query: str) -> str:
        '''Search the web.'''
        return "results..."

    # Ingest LangChain tools
    tools, definitions = from_langchain([search])
    bash = create_sandbox_tool(tools=tools)
"""

from __future__ import annotations

from typing import Any, Callable, Sequence, TypeVar, cast

from .definition import ToolDefinition

T = TypeVar("T")


def from_langchain(
    tools: Sequence[Any],
) -> tuple[list[Callable[..., Any]], list[ToolDefinition]]:
    """Convert LangChain tools to amla-sandbox format.

    Supports both @tool decorated functions and BaseTool subclasses.
    Returns both the callable functions and their ToolDefinitions.

    Args:
        tools: List of LangChain tools (@tool decorated or BaseTool instances).

    Returns:
        Tuple of (callable_functions, tool_definitions).

    Example::

        from langchain.tools import tool

        @tool
        def search(query: str) -> str:
            '''Search the web for information.'''
            return "results"

        @tool
        def calculate(expression: str) -> str:
            '''Evaluate a math expression safely.'''
            # Use a safe math parser here
            return str(result)

        # Convert to amla-sandbox format
        funcs, defs = from_langchain([search, calculate])

        # Use with create_sandbox_tool
        bash = create_sandbox_tool(tools=funcs)

    Note:
        For BaseTool subclasses, the tool's _run method is used as the callable.
        For @tool decorated functions, the underlying function is extracted.
    """
    callables: list[Callable[..., Any]] = []
    definitions: list[ToolDefinition] = []

    for tool in tools:
        func, definition = _convert_langchain_tool(tool)
        callables.append(func)
        definitions.append(definition)

    return callables, definitions


def _convert_langchain_tool(tool: Any) -> tuple[Callable[..., Any], ToolDefinition]:
    """Convert a single LangChain tool."""
    # Check if it's a StructuredTool or similar with a func attribute
    if hasattr(tool, "func") and callable(tool.func):
        # @tool decorated function - extract the underlying function
        func = tool.func
        name = getattr(tool, "name", func.__name__)
        description = getattr(tool, "description", "") or func.__doc__ or ""

        # Get schema from args_schema if available
        if hasattr(tool, "args_schema") and tool.args_schema is not None:
            parameters = _pydantic_to_json_schema(tool.args_schema)
        else:
            # Fall back to extracting from function signature
            from .conversion import tool_from_function

            temp_def = tool_from_function(func)
            parameters = temp_def.parameters

    elif hasattr(tool, "_run") and callable(tool._run):
        # BaseTool subclass
        func = tool._run
        name = getattr(tool, "name", "unknown")
        description = getattr(tool, "description", "") or ""

        # Get schema from args_schema
        parameters: dict[str, Any]
        if hasattr(tool, "args_schema") and tool.args_schema is not None:
            parameters = _pydantic_to_json_schema(tool.args_schema)
        else:
            parameters = {"type": "object", "properties": {}}

    elif callable(tool):
        # Plain function (maybe decorated but we can't detect it)
        func = tool
        name = getattr(tool, "__name__", "unknown")
        description = getattr(tool, "__doc__", "") or ""

        from .conversion import tool_from_function

        temp_def = tool_from_function(func)
        parameters = temp_def.parameters

    else:
        raise TypeError(
            f"Cannot convert tool of type {type(tool).__name__}. "
            "Expected @tool decorated function or BaseTool subclass."
        )

    # Clean up description (take first line)
    description = description.split("\n")[0].strip()

    definition = ToolDefinition(
        name=name,
        description=description,
        parameters=parameters,
    )

    return func, definition


def _pydantic_to_json_schema(model: type) -> dict[str, Any]:
    """Convert a Pydantic model to JSON Schema."""
    # Pydantic v2
    if hasattr(model, "model_json_schema"):
        # model_json_schema returns dict[str, Any] - cast since pyright doesn't know the type
        schema = cast(dict[str, Any], model.model_json_schema())  # pyright: ignore[reportUnknownMemberType]
        # Remove pydantic-specific fields
        schema.pop("title", None)
        schema.pop("$defs", None)
        schema.pop("definitions", None)
        return schema

    # Pydantic v1
    if hasattr(model, "schema"):
        # schema() returns dict[str, Any] - cast since pyright doesn't know the type
        schema_v1 = cast(dict[str, Any], model.schema())  # pyright: ignore[reportUnknownMemberType]
        schema_v1.pop("title", None)
        schema_v1.pop("definitions", None)
        return schema_v1

    return {"type": "object", "properties": {}}


def from_openai_tools(
    tools: Sequence[dict[str, Any]],
    handlers: dict[str, Callable[..., Any]] | None = None,
) -> tuple[list[Callable[..., Any]], list[ToolDefinition]]:
    """Convert OpenAI function calling tools to amla-sandbox format.

    Important: This returns a tuple that must be unpacked. The functions and
    definitions are paired by index - funcs[i] implements defs[i].

    Args:
        tools: List of OpenAI tool definitions in function calling format.
        handlers: Optional dict mapping function names to callable handlers.
            If not provided, placeholder functions are created that raise
            NotImplementedError when called.

    Returns:
        Tuple of (callable_functions, tool_definitions). Use tuple unpacking::

            funcs, defs = from_openai_tools(tools, handlers={...})

        The handler functions will have their __name__ set to match the tool
        definition name, so you can build handler maps with::

            handler_map = {f.__name__: f for f in funcs}

    Example::

        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "City name"}
                        },
                        "required": ["city"]
                    }
                }
            }
        ]

        def my_weather_handler(city: str) -> dict:
            return {"temp": 22, "city": city}

        # Unpack the tuple
        funcs, defs = from_openai_tools(
            openai_tools,
            handlers={"get_weather": my_weather_handler}
        )

        # funcs[0].__name__ == "get_weather" (matches definition)
        # defs[0].name == "get_weather"
    """
    callables: list[Callable[..., Any]] = []
    definitions: list[ToolDefinition] = []
    handlers = handlers or {}

    for tool in tools:
        func, definition = _convert_openai_tool(tool, handlers)
        callables.append(func)
        definitions.append(definition)

    return callables, definitions


def _convert_openai_tool(
    tool: dict[str, Any],
    handlers: dict[str, Callable[..., Any]],
) -> tuple[Callable[..., Any], ToolDefinition]:
    """Convert a single OpenAI tool definition."""
    # Handle both formats:
    # 1. {"type": "function", "function": {...}}
    # 2. {"name": "...", "description": "...", "parameters": {...}}

    if "function" in tool:
        func_def = tool["function"]
    else:
        func_def = tool

    name = func_def["name"]
    description = func_def.get("description", f"Call {name}")
    parameters = func_def.get("parameters", {"type": "object", "properties": {}})

    # Get handler or create placeholder
    if name in handlers:
        func = handlers[name]
        # Set __name__ to match tool definition for consistent handler mapping
        func.__name__ = name
    else:
        # Create a placeholder that raises an error
        def make_placeholder(tool_name: str) -> Callable[..., Any]:
            def placeholder(**_kwargs: Any) -> Any:
                raise NotImplementedError(
                    f"No handler provided for tool '{tool_name}'. "
                    f"Pass handlers={{'{tool_name}': your_function}} to from_openai_tools()."
                )

            placeholder.__name__ = tool_name
            placeholder.__doc__ = description
            return placeholder

        func = make_placeholder(name)

    definition = ToolDefinition(
        name=name,
        description=description,
        parameters=parameters,
    )

    return func, definition


def from_anthropic_tools(
    tools: Sequence[dict[str, Any]],
    handlers: dict[str, Callable[..., Any]] | None = None,
) -> tuple[list[Callable[..., Any]], list[ToolDefinition]]:
    """Convert Anthropic tool definitions to amla-sandbox format.

    Args:
        tools: List of Anthropic tool definitions.
        handlers: Optional dict mapping tool names to callable handlers.

    Returns:
        Tuple of (callable_functions, tool_definitions).

    Example::

        anthropic_tools = [
            {
                "name": "get_weather",
                "description": "Get current weather for a location",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"}
                    },
                    "required": ["city"]
                }
            }
        ]

        funcs, defs = from_anthropic_tools(anthropic_tools, handlers={...})
    """
    callables: list[Callable[..., Any]] = []
    definitions: list[ToolDefinition] = []
    handlers = handlers or {}

    for tool in tools:
        func, definition = _convert_anthropic_tool(tool, handlers)
        callables.append(func)
        definitions.append(definition)

    return callables, definitions


def _convert_anthropic_tool(
    tool: dict[str, Any],
    handlers: dict[str, Callable[..., Any]],
) -> tuple[Callable[..., Any], ToolDefinition]:
    """Convert a single Anthropic tool definition."""
    name = tool["name"]
    description = tool.get("description", f"Call {name}")

    # Anthropic uses input_schema instead of parameters
    parameters = tool.get("input_schema", {"type": "object", "properties": {}})

    # Get handler or create placeholder
    if name in handlers:
        func = handlers[name]
        # Set __name__ to match tool definition for consistent handler mapping
        func.__name__ = name
    else:

        def make_placeholder(tool_name: str) -> Callable[..., Any]:
            def placeholder(**_kwargs: Any) -> Any:
                raise NotImplementedError(
                    f"No handler provided for tool '{tool_name}'. "
                    f"Pass handlers={{'{tool_name}': your_function}} to from_anthropic_tools()."
                )

            placeholder.__name__ = tool_name
            placeholder.__doc__ = description
            return placeholder

        func = make_placeholder(name)

    definition = ToolDefinition(
        name=name,
        description=description,
        parameters=parameters,
    )

    return func, definition


# Convenience aliases
from_openai = from_openai_tools
from_claude = from_anthropic_tools
