"""Tool utilities for the Amla sandbox.

This package provides utilities for defining and converting tools that agents
can use in the sandbox:

- :class:`ToolDefinition` - Describes a tool's name, description, and parameters
- :func:`tool_from_function` - Convert Python functions to ToolDefinitions
- :func:`create_tool_handler` - Create handlers that route calls to functions
- :func:`capability_from_function` - Generate capabilities from functions

Framework ingestion:

- :func:`from_langchain` - Convert LangChain @tool decorated functions
- :func:`from_openai_tools` - Convert OpenAI function calling format
- :func:`from_anthropic_tools` - Convert Anthropic tool format
"""

from .conversion import (
    capability_from_function,
    format_tool_descriptions_js,
    tool_from_function,
)
from .definition import ToolDefinition
from .handlers import create_tool_handler
from .ingest import (
    from_anthropic_tools,
    from_claude,
    from_langchain,
    from_openai,
    from_openai_tools,
)

__all__ = [
    "ToolDefinition",
    "tool_from_function",
    "capability_from_function",
    "format_tool_descriptions_js",
    "create_tool_handler",
    # Framework ingestion
    "from_langchain",
    "from_openai_tools",
    "from_openai",
    "from_anthropic_tools",
    "from_claude",
]
