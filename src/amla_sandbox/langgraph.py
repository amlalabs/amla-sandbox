"""LangGraph CodeAct integration with Amla sandbox.

This module provides seamless integration with LangGraph CodeAct, allowing
agents to write JavaScript code in a secure, capability-controlled sandbox.

Quick Start with CodeAct::

    from amla_sandbox import create_amla_codeact

    # Define tools as simple Python functions
    def get_weather(city: str) -> str:
        '''Get current weather for a city.'''
        return f"Sunny, 72°F in {city}"

    # Create CodeAct agent with Amla sandbox
    agent = create_amla_codeact(
        model="anthropic:claude-sonnet-4-20250514",
        tools=[get_weather],
    )

    # Run it
    result = agent.invoke({"messages": [("user", "What's the weather in SF?")]})

The sandbox provides:
- QuickJS/WASM execution (portable, fast, secure)
- Capability-based tool access control
- Virtual filesystem for intermediate data
- Shell utilities for data processing

For more control, create your own sandbox function::

    from amla_sandbox import create_amla_sandbox

    # Create the sandbox function
    sandbox_fn = create_amla_sandbox(tools=[get_weather])

    # Use with langgraph-codeact directly
    from langgraph_codeact import create_codeact
    agent = create_codeact(model, tools, sandbox_fn, prompt=JS_PROMPT)
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from .capabilities import MethodCapability

# Note: CodeAct functions (create_amla_codeact, create_amla_sandbox, JS_CODEACT_PROMPT)
# are now imported directly from .codeact in __init__.py
from .sandbox import Sandbox
from .tools import (
    capability_from_function,
    tool_from_function,
)


@dataclass
class ExecutionResult:
    """Result of code execution in the sandbox.

    Provides structured access to execution output, making it easy to
    inspect what happened during execution.

    Attributes:
        stdout: Standard output from the execution.
        stderr: Standard error output (warnings, debug info).
        return_value: The final return value, if any.
        error: Error message if execution failed, None otherwise.
        success: True if execution completed without errors.
    """

    stdout: str = ""
    stderr: str = ""
    return_value: Any = None
    error: str | None = None
    success: bool = True

    def __str__(self) -> str:
        """Return stdout for simple string usage."""
        if self.error:
            return f"Error: {self.error}\n{self.stderr}"
        return self.stdout

    def to_tool_message(self) -> str:
        """Format as a tool message for LLM consumption."""
        parts = []
        if self.stdout:
            parts.append(self.stdout)
        if self.stderr:
            parts.append(f"[stderr]: {self.stderr}")
        if self.error:
            parts.append(f"[error]: {self.error}")
        return "\n".join(parts) if parts else "(no output)"


@dataclass
class SandboxTool:
    """A LangGraph-compatible tool that executes code in Amla sandbox.

    This is the main integration point for LangGraph. It wraps the sandbox
    and exposes it as a tool that agents can use to write and execute code.

    The tool accepts JavaScript or shell commands, executes them in the
    sandbox with access to registered tools, and returns structured results.

    Example::

        from amla_sandbox import SandboxTool

        # Create with simple Python functions
        sandbox_tool = SandboxTool.from_functions([
            get_weather,
            search_web,
        ])

        # Use with LangGraph
        from langgraph.prebuilt import create_react_agent
        agent = create_react_agent(model, [sandbox_tool.as_langchain_tool()])

    Attributes:
        sandbox: The underlying Amla Sandbox instance.
        tools: List of Python functions available in the sandbox.
        default_language: Default language for run() when not specified.
    """

    sandbox: Sandbox
    tools: list[Callable[..., Any]] = field(default_factory=lambda: [])
    _tool_map: dict[str, Callable[..., Any]] = field(
        default_factory=lambda: {}, repr=False
    )
    default_language: str = "javascript"

    # LangChain tool interface
    name: str = "sandbox"
    description: str = """Execute code in a secure sandbox.

Set language="javascript" (default) to run JavaScript with:
- async/await for tool calls: `const data = await toolName({param: "value"});`
- Virtual filesystem: `await fs.readFile('/tmp/data.json')`
- Full ES2020 support

Set language="shell" to run shell commands with:
- Utilities: grep, jq, tr, head, tail, sort, uniq, wc, cut, cat
- Pipes: `cat /tmp/data.json | jq '.items[]' | head -5`
"""

    @classmethod
    def from_functions(
        cls,
        tools: Sequence[Callable[..., Any]],
        *,
        capabilities: Sequence[MethodCapability] | None = None,
    ) -> SandboxTool:
        """Create a SandboxTool from Python functions.

        This is the recommended way to create a SandboxTool. Just pass
        your Python functions and everything is wired up automatically.

        Args:
            tools: Python functions to expose in the sandbox.
            capabilities: Optional explicit capabilities. If not provided,
                capabilities are generated automatically to allow all tools.

        Returns:
            Configured SandboxTool ready for use with LangGraph.
        """
        tools_list = list(tools)
        tool_map = {func.__name__: func for func in tools_list}

        # Convert functions to ToolDefinitions
        tool_defs = [tool_from_function(func) for func in tools_list]

        # Generate capabilities if not provided
        if capabilities is None:
            caps = [capability_from_function(func) for func in tools_list]
        else:
            caps = list(capabilities)

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
            tool_handler=handler,
        )

        return cls(
            sandbox=sandbox,
            tools=tools_list,
            _tool_map=tool_map,
        )

    def execute(
        self, code: str, *, stdin: str | bytes | None = None
    ) -> ExecutionResult:
        """Execute JavaScript code in the sandbox.

        Args:
            code: JavaScript code to execute.
            stdin: Optional code to pipe via stdin (bypasses command size limits).
                If provided, code parameter is ignored and stdin is executed.

        Returns:
            ExecutionResult with stdout, stderr, and any errors.
        """
        try:
            output = self.sandbox.execute(code, stdin=stdin)
            return ExecutionResult(
                stdout=output,
                stderr=self.sandbox.last_stderr,
                success=True,
            )
        except Exception as e:
            return ExecutionResult(
                error=str(e),
                stderr=self.sandbox.last_stderr,
                success=False,
            )

    def shell(
        self, command: str, *, stdin: str | bytes | None = None
    ) -> ExecutionResult:
        """Execute a shell command in the sandbox.

        Args:
            command: Shell command to execute.
            stdin: Optional data to pipe via stdin.

        Returns:
            ExecutionResult with stdout, stderr, and any errors.
        """
        try:
            output = self.sandbox.shell(command, stdin=stdin)
            return ExecutionResult(
                stdout=output,
                stderr=self.sandbox.last_stderr,
                success=True,
            )
        except Exception as e:
            return ExecutionResult(
                error=str(e),
                stderr=self.sandbox.last_stderr,
                success=False,
            )

    def run(
        self,
        code: str,
        language: str | None = None,
        *,
        stdin: str | bytes | None = None,
    ) -> str:
        """Execute code and return string result.

        This is the main entry point used by LangChain/LangGraph tools.

        Args:
            code: Code to execute.
            language: Either "javascript" or "shell". If not specified, uses
                the default_language set when creating the tool.
            stdin: Optional data to pipe via stdin. For JavaScript, this bypasses
                the command size limit. For shell, useful for piping to `sh`.

        Returns:
            String output suitable for LLM consumption.
        """
        lang = language if language is not None else self.default_language
        if lang == "shell":
            result = self.shell(code, stdin=stdin)
        else:
            result = self.execute(code, stdin=stdin)
        return result.to_tool_message()

    def as_langchain_tool(self) -> Any:
        """Convert to a LangChain-compatible tool.

        Returns:
            A tool that can be used with LangGraph's create_react_agent.
        """
        try:
            from langchain_core.tools import StructuredTool
            from pydantic import BaseModel, Field

            # Capture default_language for use in closures
            default_lang = self.default_language

            class CodeInput(BaseModel):
                """Input for code execution."""

                code: str = Field(description="Code to execute")
                language: str = Field(
                    default=default_lang,
                    description=f"Language: 'javascript' or 'shell' (default: {default_lang})",
                )

            def _run(code: str, language: str = default_lang) -> str:
                return self.run(code, language)

            # Build a helpful description
            description = self._build_tool_description()

            return StructuredTool(
                name=self.name,
                description=description,
                args_schema=CodeInput,
                func=_run,
            )
        except ImportError as e:
            raise ImportError(
                "langchain_core is required for as_langchain_tool(). "
                "Install with: pip install langchain-core"
            ) from e

    def _build_tool_description(self) -> str:
        """Build a comprehensive description for the LangChain tool.

        This description is what the LLM sees and uses to understand how to
        call the sandbox tool. It includes:
        - Available functions with their signatures and descriptions
        - Both JavaScript and shell execution modes
        - Clear instructions on the code/language parameters
        """
        is_js_default = self.default_language == "javascript"
        js_marker = " (default)" if is_js_default else ""
        shell_marker = " (default)" if not is_js_default else ""

        lines = [
            "Execute code in a secure sandbox. Two modes available:",
            "",
            f"JAVASCRIPT (language='javascript'{js_marker}):",
            "  Call functions with async/await, output with console.log().",
        ]

        if self.tools:
            lines.append("  Available functions:")
            for func in self.tools:
                sig = inspect.signature(func)
                doc = inspect.getdoc(func) or ""
                first_line = doc.split("\n")[0] if doc else ""

                # Build parameter list
                params = []
                for name, param in sig.parameters.items():
                    param_type = "any"
                    if param.annotation != inspect.Parameter.empty:
                        param_type = getattr(
                            param.annotation, "__name__", str(param.annotation)
                        )
                    params.append(f"{name}: {param_type}")

                func_desc = f"    - {func.__name__}({{{', '.join(params)}}})"
                if first_line:
                    func_desc += f" - {first_line}"
                lines.append(func_desc)

            # Show example
            example_func = self.tools[0].__name__
            lines.append(
                f"  Example: const r = await {example_func}({{...}}); console.log(JSON.stringify(r));"
            )
        else:
            lines.append("  Example: console.log('hello');")

        lines.extend(
            [
                "",
                f"SHELL (language='shell'{shell_marker}):",
                "  Run shell commands with: grep, jq, tr, head, tail, sort, uniq, wc, cut, cat, echo",
                "  Example: cat /workspace/data.json | jq '.items[]' | head -5",
            ]
        )

        # Add constraint info if available
        constraints_info = self._get_constraints_summary()
        if constraints_info:
            lines.extend(["", "LIMITS:", constraints_info])

        return "\n".join(lines)

    def _get_constraints_summary(self) -> str:
        """Get a summary of constraints and limits for the sandbox."""
        caps = self.sandbox.get_capabilities()
        limits = []
        for cap in caps:
            name = cap.method_pattern.removeprefix("mcp:")
            if cap.max_calls and cap.max_calls < 100:
                limits.append(f"  - {name}: max {cap.max_calls} calls")
            if not cap.constraints.is_empty():
                limits.append(f"  - {name}: has parameter constraints")
        return "\n".join(limits)

    def get_tool_descriptions(self) -> str:
        """Get formatted descriptions of available tools.

        Useful for including in system prompts so the agent knows
        what tools are available. Includes full docstrings.

        Returns:
            Formatted string describing all tools with their signatures and docs.
        """
        if not self.tools:
            return "No custom tools available. Use shell commands or basic JavaScript."

        lines = ["Available functions in the sandbox:", ""]
        for func in self.tools:
            sig = inspect.signature(func)
            doc = inspect.getdoc(func) or "No description"

            # Build parameter details
            params = []
            for name, param in sig.parameters.items():
                param_type = "any"
                if param.annotation != inspect.Parameter.empty:
                    param_type = getattr(
                        param.annotation, "__name__", str(param.annotation)
                    )
                default = ""
                if param.default != inspect.Parameter.empty:
                    default = f" = {param.default!r}"
                params.append(f"{name}: {param_type}{default}")

            lines.append(f"### {func.__name__}({', '.join(params)})")
            lines.append(doc)
            lines.append("")

        return "\n".join(lines)

    def get_system_prompt(self, *, include_tools: bool = True) -> str:
        """Generate a composable system prompt fragment for LangGraph agents.

        This returns guidance that can be APPENDED to (not replace) the default
        LangGraph/ReAct system prompt. It teaches the LLM how to use the sandbox
        tool effectively.

        Args:
            include_tools: If True, include full tool documentation.
                Set to False if you only want the usage instructions.

        Returns:
            System prompt fragment to append to your agent's prompt.

        Example::

            from langgraph.prebuilt import create_react_agent

            bash = create_bash_tool(tools=[get_weather])

            # Option 1: Use as the full system prompt (simple cases)
            agent = create_react_agent(model, [bash.as_langchain_tool()],
                                       prompt=bash.get_system_prompt())

            # Option 2: Append to existing prompt (recommended for complex agents)
            base_prompt = "You are a helpful assistant..."
            agent = create_react_agent(model, [bash.as_langchain_tool()],
                                       prompt=base_prompt + "\\n\\n" + bash.get_system_prompt())
        """
        sections = []

        # Core sandbox usage instructions
        sections.append("""## Sandbox Tool Usage

You have access to a `sandbox` tool that executes code securely. It accepts two parameters:
- `code`: The code to execute (string)
- `language`: Either "javascript" (default) or "shell"

### JavaScript Mode (language="javascript")
Call functions using async/await and output results with console.log().

**IMPORTANT: Batch multiple operations in a single code block using loops:**
```javascript
// Process multiple items efficiently in ONE tool call
const items = ["item1", "item2", "item3"];
const results = [];
for (const item of items) {
    const data = await getData({id: item});
    const info = await getInfo({id: item});
    results.push({item, data, info});
}
console.log(JSON.stringify(results));
```

Use the virtual filesystem to store intermediate data:
```javascript
await fs.writeFile("/workspace/data.json", JSON.stringify(results));
```

### Shell Mode (language="shell")
Run shell commands with pipes. Available: grep, jq, tr, head, tail, sort, uniq, wc, cut, cat, echo
```shell
cat /workspace/data.json | jq '.items[]' | head -5
```""")

        # Add tool documentation if requested
        if include_tools and self.tools:
            tool_docs = []
            for func in self.tools:
                sig = inspect.signature(func)
                doc = inspect.getdoc(func) or ""
                first_line = doc.split("\n")[0] if doc else "No description"

                params = []
                for name, param in sig.parameters.items():
                    param_type = "any"
                    if param.annotation != inspect.Parameter.empty:
                        param_type = getattr(
                            param.annotation, "__name__", str(param.annotation)
                        )
                    params.append(f"{name}: {param_type}")

                tool_docs.append(
                    f"- `{func.__name__}({{{', '.join(params)}}})` - {first_line}"
                )

            sections.append("\n### Available Functions\n" + "\n".join(tool_docs))

        # Add constraints/limits if any
        constraints = self._get_constraints_summary()
        if constraints:
            sections.append("\n### Limits\n" + constraints)

        return "\n".join(sections)


__all__ = [
    # Main classes
    "ExecutionResult",
    "SandboxTool",
]
