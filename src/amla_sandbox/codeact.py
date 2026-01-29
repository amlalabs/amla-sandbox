"""LangGraph CodeAct integration with Amla sandbox.

This module provides CodeAct-specific functionality for integrating
Amla sandbox with LangGraph's CodeAct agents.

CodeAct agents write code to solve tasks. Amla provides a secure
JavaScript sandbox as an alternative to Python's eval().

Key functions:
- :func:`create_amla_sandbox`: Create a sandbox function for CodeAct
- :func:`create_amla_codeact`: One-liner to create a CodeAct agent
- :func:`format_tool_descriptions_js`: Format tools for JS prompts

Example::

    from amla_sandbox.codeact import create_amla_codeact

    def get_weather(city: str) -> str:
        '''Get current weather for a city.'''
        return f"Sunny in {city}"

    agent = create_amla_codeact(
        "anthropic:claude-sonnet-4-20250514",
        tools=[get_weather],
    )

    result = agent.invoke({"messages": [("user", "Weather in SF?")]})
"""

from __future__ import annotations

__all__ = [
    "JS_CODEACT_PROMPT",
    "SandboxFn",
    "format_tool_descriptions_js",
    "create_amla_sandbox",
    "create_amla_codeact",
]

from typing import Any, Callable, Sequence

from .capabilities import MethodCapability
from .tools import format_tool_descriptions_js

# Avoid circular import - SandboxTool is imported at function level
# from .langgraph import SandboxTool  # Deferred import


# JavaScript-targeted system prompt for CodeAct
JS_CODEACT_PROMPT = """You have access to tools as JavaScript functions in a secure sandbox.

Write JavaScript code to solve the task. The sandbox provides:
- Full ES2020 JavaScript with async/await
- Virtual filesystem for storing intermediate data
- Shell utilities (grep, jq, tr, head, tail, sort, uniq, wc, cut)

{tool_descriptions}

## Examples

### Calling a tool
```javascript
const result = await get_weather({{ city: "San Francisco" }});
console.log(result);
```

### Storing and processing data
```javascript
// Call tool and store result (fs is async-only)
const data = await search({{ query: "AI agents" }});
await fs.writeFile('/tmp/results.json', JSON.stringify(data));

// Process with shell utilities
const {{ stdout: count }} = await shell('cat /tmp/results.json | jq length');
console.log(`Found ${{count.trim()}} results`);
```

### Chaining multiple tools
```javascript
// Get data from multiple sources
const weather = await get_weather({{ city: "NYC" }});
const events = await search_events({{ city: "NYC", date: "today" }});

// Combine and filter (fs methods are async)
await fs.writeFile('/tmp/weather.json', JSON.stringify(weather));
await fs.writeFile('/tmp/events.json', JSON.stringify(events));

// Find outdoor events if weather is good
if (weather.condition === "sunny") {{
    const {{ stdout: outdoor }} = await shell('cat /tmp/events.json | jq "[.[] | select(.outdoor)]"');
    console.log("Outdoor events:", outdoor);
}}
```

Write code to accomplish the user's request. Use console.log() for the final output.
"""


# Type alias for CodeAct sandbox function
SandboxFn = Callable[[str, dict[str, Any]], tuple[str, dict[str, Any]]]
"""Type for CodeAct sandbox function: (code, locals) -> (output, new_vars)"""


def create_amla_sandbox(
    tools: Sequence[Callable[..., Any]],
    *,
    capabilities: Sequence[MethodCapability] | None = None,
) -> SandboxFn:
    """Create an Amla sandbox function for LangGraph CodeAct.

    This creates a sandbox function matching CodeAct's expected interface:
        (code: str, _locals: dict) -> (output: str, new_vars: dict)

    The sandbox executes JavaScript (not Python) in a secure QuickJS/WASM
    environment with capability-based access control.

    Args:
        tools: Python functions to expose in the sandbox.
        capabilities: Optional explicit capabilities for fine-grained control.

    Returns:
        Sandbox function for use with langgraph-codeact.

    Example::

        from amla_sandbox.codeact import create_amla_sandbox, JS_CODEACT_PROMPT
        from langgraph_codeact import create_codeact

        def get_weather(city: str) -> str:
            return f"Sunny in {city}"

        # Create Amla sandbox
        sandbox_fn = create_amla_sandbox([get_weather])

        # Use with CodeAct
        agent = create_codeact(
            model,
            tools=[get_weather],
            sandbox=sandbox_fn,
            prompt=JS_CODEACT_PROMPT,
        )
    """
    # Import here to avoid circular dependency
    from .langgraph import SandboxTool

    # Create the underlying sandbox tool
    sandbox_tool = SandboxTool.from_functions(tools, capabilities=capabilities)

    def sandbox_fn(code: str, _locals: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Execute JavaScript code in Amla sandbox.

        Args:
            code: JavaScript code to execute.
            _locals: Variables from previous turns (tools are already injected).

        Returns:
            Tuple of (stdout output, new variables created).
        """
        # Execute the code
        result = sandbox_tool.execute(code)

        # Format output
        output = result.to_tool_message()

        # For now, we don't persist JS variables between turns
        # (QuickJS doesn't easily expose this)
        # Future: could parse console.log statements for structured output
        new_vars: dict[str, Any] = {}

        return output, new_vars

    return sandbox_fn


def create_amla_codeact(
    model: str | Any,
    tools: Sequence[Callable[..., Any]],
    *,
    capabilities: Sequence[MethodCapability] | None = None,
    prompt: str | None = None,
) -> Any:
    """Create a LangGraph CodeAct agent with Amla sandbox.

    This is the recommended way to create a CodeAct agent with secure
    code execution. The agent writes JavaScript code that runs in the
    Amla sandbox with capability-based access control.

    Args:
        model: Model identifier (e.g., "anthropic:claude-sonnet-4-20250514") or
            LangChain chat model instance.
        tools: Python functions to expose in the sandbox.
        capabilities: Optional explicit capabilities for fine-grained control.
        prompt: Optional custom system prompt. If not provided, uses
            JS_CODEACT_PROMPT with tool descriptions.

    Returns:
        A compiled LangGraph CodeAct agent ready for .invoke() or .stream().

    Example::

        from amla_sandbox import create_amla_codeact

        def get_weather(city: str) -> str:
            '''Get current weather for a city.'''
            return f"Sunny in {city}"

        def search(query: str) -> list[str]:
            '''Search for information.'''
            return [f"Result 1 for {query}", f"Result 2 for {query}"]

        # Create agent - it writes JavaScript, not Python!
        agent = create_amla_codeact(
            "anthropic:claude-sonnet-4-20250514",
            tools=[get_weather, search],
        )

        # Run it
        result = agent.invoke({
            "messages": [("user", "What's the weather in SF and find restaurants")]
        })

    Security Benefits:
        Unlike CodeAct with eval(), Amla provides:
        - Code runs in WASM sandbox, not in your Python process
        - Capability-based access control on tool calls
        - Full audit trail of what was executed
        - No access to Python internals or filesystem
    """
    try:
        from langchain.chat_models import init_chat_model
        from langgraph_codeact import create_codeact  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError(
            "langgraph-codeact is required for create_amla_codeact(). "
            "Install with: pip install langgraph-codeact langchain-anthropic"
        ) from e

    # Initialize model if string
    if isinstance(model, str):
        model = init_chat_model(model)

    # Create sandbox function
    sandbox_fn = create_amla_sandbox(tools, capabilities=capabilities)

    # Build prompt if not provided
    if prompt is None:
        tool_descriptions = format_tool_descriptions_js(tools)
        prompt = JS_CODEACT_PROMPT.format(tool_descriptions=tool_descriptions)

    # Create and return CodeAct agent
    return create_codeact(
        model,
        tools=list(tools),
        eval_fn=sandbox_fn,
        prompt=prompt,
    )
