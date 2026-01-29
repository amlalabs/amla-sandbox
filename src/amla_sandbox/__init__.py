"""Amla Sandbox - Let agents think in code.

Copyright (c) 2025 Amla Labs. MIT License (see LICENSE file).

Quick start (LangGraph)::

    from amla_sandbox import create_sandbox_tool
    from langgraph.prebuilt import create_react_agent

    def get_weather(city: str) -> str:
        return f"Sunny in {city}"

    sandbox = create_sandbox_tool(tools=[get_weather])
    sandbox.run("await get_weather({city: 'SF'})", language="javascript")
    sandbox.run("echo hello | tr 'a-z' 'A-Z'", language="shell")

    # With LangGraph
    agent = create_react_agent(model, [sandbox.as_langchain_tool()])

The idea:
- Agents write JS that runs in WASM (QuickJS)
- Tool results stay in a virtual filesystem
- Shell utilities extract only what the LLM needs

What's provided:
- VFS with async fs APIs
- Shell applets: grep, jq, tr, head, tail, sort, uniq, wc, cut, cat
- QuickJS ES2020 with async/await
- Tool stubs generated from Python functions with capability enforcement
"""

# Audit logging
from .audit import AuditCollector, AuditConfig, AuditEntry

# Capabilities (foundational enforcement layer)
from .capabilities import (
    CallLimitExceededError,
    CapabilityError,
    Constraint,
    ConstraintError,
    ConstraintSet,
    MethodCapability,
    METHOD_CAPABILITY_TYPE,
    MissingParamError,
    Param,
    TypeMismatchError,
    ViolationError,
    method_matches_pattern,
    pattern_is_subset,
)

# Runtime
from .runtime import Runtime, RuntimeConfig, RuntimeError, RuntimeStatus
from .runtime.wasm import (
    AsyncToolHandler,
    SyncToolHandler,
    ToolHandler,
    precompile_module,
)

# Main Sandbox class
from .sandbox import Sandbox

# Tool utilities (extracted to dedicated module)
from .tools import (
    ToolDefinition,
    capability_from_function,
    create_tool_handler,
    format_tool_descriptions_js,
    tool_from_function,
    # Framework ingestion
    from_anthropic_tools,
    from_claude,
    from_langchain,
    from_openai,
    from_openai_tools,
)

# CodeAct integration (JavaScript sandbox for LangGraph)
from .codeact import (
    JS_CODEACT_PROMPT,
    create_amla_codeact,
    create_amla_sandbox,
)

# LangGraph tool-based integration
from .langgraph import ExecutionResult, SandboxTool

# Simple sandbox tool API (AI SDK-style ergonomics)
from .bash_tool import create_sandbox_tool

__all__ = [
    # === High-level API (recommended) ===
    # Simple sandbox tool (AI SDK-style ergonomics)
    "create_sandbox_tool",
    # CodeAct integration (recommended for LangGraph)
    "create_amla_codeact",  # One-liner: CodeAct agent with Amla sandbox
    "create_amla_sandbox",  # Create sandbox fn for custom CodeAct setup
    "JS_CODEACT_PROMPT",  # JavaScript-targeted system prompt
    "format_tool_descriptions_js",  # Format tools for JS prompt
    # LangGraph tool
    "SandboxTool",
    "ExecutionResult",
    # Tool helpers
    "ToolDefinition",
    "tool_from_function",
    "capability_from_function",
    "create_tool_handler",
    # Framework ingestion
    "from_langchain",
    "from_openai_tools",
    "from_openai",
    "from_anthropic_tools",
    "from_claude",
    # === Low-level API ===
    # Main class
    "Sandbox",
    # Runtime
    "Runtime",
    "RuntimeConfig",
    "RuntimeError",
    "RuntimeStatus",
    # Tool handler types
    "ToolHandler",
    "SyncToolHandler",
    "AsyncToolHandler",
    # Capabilities
    "MethodCapability",
    "METHOD_CAPABILITY_TYPE",
    "Constraint",
    "ConstraintSet",
    "Param",
    # Errors
    "CapabilityError",
    "CallLimitExceededError",
    "ConstraintError",
    "MissingParamError",
    "TypeMismatchError",
    "ViolationError",
    # Pattern matching
    "method_matches_pattern",
    "pattern_is_subset",
    # Audit logging
    "AuditCollector",
    "AuditConfig",
    "AuditEntry",
    # Precompilation
    "precompile_module",
]
