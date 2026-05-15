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

# Simple sandbox tool API (AI SDK-style ergonomics)
from .bash_tool import create_sandbox_tool

# Capabilities (foundational enforcement layer)
from .capabilities import (
    TOOL_CALL_CAP_TYPE,
    CallLimitExceededError,
    CapabilityError,
    Constraint,
    ConstraintError,
    ConstraintSet,
    MissingParamError,
    Param,
    ToolCallCap,
    TypeMismatchError,
    ViolationError,
    method_matches_pattern,
    pattern_is_subset,
)

# CodeAct integration (JavaScript sandbox for LangGraph)
from .codeact import (
    JS_CODEACT_PROMPT,
    create_amla_codeact,
    create_amla_sandbox,
)

# LangGraph tool-based integration
from .langgraph import ExecutionResult, SandboxTool

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
    # Framework ingestion
    from_anthropic_tools,
    from_claude,
    from_langchain,
    from_openai,
    from_openai_tools,
    tool_from_function,
)

__all__ = [
    "JS_CODEACT_PROMPT",  # JavaScript-targeted system prompt
    "TOOL_CALL_CAP_TYPE",
    "AsyncToolHandler",
    # Audit logging
    "AuditCollector",
    "AuditConfig",
    "AuditEntry",
    "CallLimitExceededError",
    # Errors
    "CapabilityError",
    "Constraint",
    "ConstraintError",
    "ConstraintSet",
    "ExecutionResult",
    "MissingParamError",
    "Param",
    # Runtime
    "Runtime",
    "RuntimeConfig",
    "RuntimeError",
    "RuntimeStatus",
    # === Low-level API ===
    # Main class
    "Sandbox",
    # LangGraph tool
    "SandboxTool",
    "SyncToolHandler",
    # Capabilities
    "ToolCallCap",
    # Tool helpers
    "ToolDefinition",
    # Tool handler types
    "ToolHandler",
    "TypeMismatchError",
    "ViolationError",
    "capability_from_function",
    # CodeAct integration (recommended for LangGraph)
    "create_amla_codeact",  # One-liner: CodeAct agent with Amla sandbox
    "create_amla_sandbox",  # Create sandbox fn for custom CodeAct setup
    # === High-level API (recommended) ===
    # Simple sandbox tool (AI SDK-style ergonomics)
    "create_sandbox_tool",
    "create_tool_handler",
    "format_tool_descriptions_js",  # Format tools for JS prompt
    "from_anthropic_tools",
    "from_claude",
    # Framework ingestion
    "from_langchain",
    "from_openai",
    "from_openai_tools",
    # Pattern matching
    "method_matches_pattern",
    "pattern_is_subset",
    # Precompilation
    "precompile_module",
    "tool_from_function",
]
