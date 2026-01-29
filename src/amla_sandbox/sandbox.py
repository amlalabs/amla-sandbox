"""Amla Sandbox - Let agents think in code.

The Problem:
    Agents are trained to think by writing code. Claude Code navigates codebases
    with grep, processes data with jq, writes files to track state. But when we
    deploy agents, we strip this away and force them into pure tool-call loops
    where every tool result bloats the conversation context.

The Solution:
    Give agents a scratchpad. The 47KB JSON response lives in the sandbox's
    virtual filesystem. Only the 50-byte extracted ID flows back to the LLM.

Example::

    from amla_sandbox import Sandbox

    # Agent writes code that runs in the sandbox
    sandbox = Sandbox(pca=pca_bytes, tools=my_tools)

    # JavaScript executed in sandbox - tool results stay in VFS
    result = sandbox.execute('''
        // All operations are async - use await
        const txns = await stripe_listTransactions({customer: "cus_123"});
        await fs.writeFile('/workspace/txns.json', JSON.stringify(txns));

        // shell.run() returns stdout, throws on non-zero exit
        const id = await shell.run("grep 'disputed' /workspace/txns.json | head -1");

        // Only the extracted result enters the reasoning loop
        console.log(id);
    ''')
    # result is just "txn_456" - not the full 47KB response

What Agents Get:
    - VFS: In-memory filesystem (async: await fs.readFile, fs.writeFile, etc.)
    - Shell: grep, jq, tr, sort, uniq, head, tail, wc, cut, cat (async)
    - QuickJS: Full ES2020 JavaScript with async/await
    - Tools: Auto-generated async stubs from registered tools
    - Introspection: listTools(), getToolInfo(name)

Architecture::

    ┌─────────────────────────────────────────┐
    │        Python Host (this module)         │
    │  - Drives stepping loop                  │
    │  - Routes host ops to tool handlers      │
    │  - Enforces capabilities                 │
    └────────────────┬────────────────────────┘
                     │ wasmtime
    ┌────────────────▼────────────────────────┐
    │        amla-sandbox.wasm                 │
    │  - QuickJS embedded                      │
    │  - Async scheduler                       │
    │  - VFS, shell commands                   │
    │  - Capability validation                 │
    └─────────────────────────────────────────┘
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Callable, Iterator

from .audit import AuditCollector, AuditConfig, AuditEntry
from .capabilities import MethodCapability
from .runtime import Runtime, RuntimeConfig
from .runtime.wasm import ToolHandler
from .tools import ToolDefinition


@dataclass
class Sandbox:
    """Capability-based sandbox for AI agents.

    Provides agents with a scratchpad environment where they can:
    - Write JavaScript code with async/await
    - Store data in a virtual filesystem
    - Process data with Unix shell utilities
    - Call registered tools (subject to capability enforcement)

    The sandbox is initialized with a PCA (capability token) that defines
    what the agent is allowed to do. All tool calls are validated against
    these capabilities before execution.

    Example::

        from amla_sandbox import Sandbox, ToolDefinition, MethodCapability

        # Create sandbox with tools and capabilities
        sandbox = Sandbox(
            tools=[
                ToolDefinition(
                    name="stripe.listTransactions",
                    description="List Stripe transactions",
                    parameters={"type": "object", "properties": {"limit": {"type": "integer"}}}
                ),
            ],
            capabilities=[MethodCapability(method_pattern="**")],
            tool_handler=lambda method, params: {"transactions": [...]},
        )

        # Execute JavaScript in sandbox (all ops are async)
        result = sandbox.execute('''
            // Tools become global functions: stripe.listTransactions -> stripe_listTransactions
            const data = await stripe_listTransactions({limit: 100});

            // fs operations are async
            await fs.writeFile('/workspace/data.json', JSON.stringify(data));

            // shell.run() returns stdout string, throws on error
            const count = await shell.run('cat /workspace/data.json | wc -l');

            // Discover available tools
            console.log('Tools:', listTools());

            console.log(`Found ${count.trim()} lines`);
        ''')

    Tool Name Transformation:
        Tool names are converted to valid JavaScript identifiers. The following
        characters are replaced with underscores: ``.`` `/`` ``-`` ``:``

        - ``stripe.listTransactions`` → ``stripe_listTransactions()``
        - ``weather/current`` → ``weather_current()``
        - ``my-tool`` → ``my_tool()``

    Attributes:
        pca: Serialized PCA (capability token) for this sandbox.
        tools: List of tool definitions available to the agent.
        capabilities: Capabilities extracted from PCA.
        tool_handler: Handler function for tool calls.
    """

    pca: bytes = b""
    """Serialized PCA (capability token)."""

    tools: list[ToolDefinition] = field(default_factory=lambda: list[ToolDefinition]())
    """Available tools."""

    capabilities: list[MethodCapability] = field(
        default_factory=lambda: list[MethodCapability]()
    )
    """Capabilities for enforcement."""

    trusted_authorities: list[str] = field(default_factory=lambda: list[str]())
    """Trusted authority public keys in 'ed25519:hex' format."""

    tool_handler: ToolHandler | None = None
    """Handler for tool calls: (method, params) -> result"""

    wasm_path: Path | None = None
    """Path to amla-sandbox.wasm. If None, uses bundled."""

    audit_config: AuditConfig | None = None
    """Configuration for audit logging. If None, no audit logging."""

    _runtime: Runtime | None = field(default=None, repr=False)
    _prelude: str | None = field(default=None, repr=False)
    _audit_collector: AuditCollector | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Initialize the sandbox."""
        import json

        # Auto-generate PCA and authority if not provided (development/testing mode)
        pca_bytes = self.pca
        trusted_authorities = list(self.trusted_authorities)

        if not pca_bytes or not trusted_authorities:
            from .auth import EphemeralAuthority

            authority = EphemeralAuthority()
            pca_bytes = authority.create_pca(capabilities=["tool_call:**"]).to_cbor()
            trusted_authorities = [authority.public_key_hex()]

            # Also set default capabilities if not provided
            if not self.capabilities:
                # Use object.__setattr__ to set field during __post_init__
                object.__setattr__(
                    self,
                    "capabilities",
                    [MethodCapability(method_pattern="**")],
                )

        # Convert tools to JSON for WASM runtime
        tools_json = "[]"
        if self.tools:
            tools_json = json.dumps([t.to_dict() for t in self.tools])

        # Create runtime configuration
        config = RuntimeConfig(
            wasm_path=self.wasm_path,
            pca_bytes=pca_bytes,
            trusted_authorities=trusted_authorities,
            capabilities=list(self.capabilities),
            tool_handler=self._handle_tool_call,
            tools_json=tools_json,
        )
        self._runtime = Runtime(config)

        # Set up audit collector if configured
        if self.audit_config is not None:
            self._audit_collector = AuditCollector(self.audit_config)
            self._runtime.set_audit_collector(self._audit_collector)

    def _handle_tool_call(self, method: str, params: dict[str, Any]) -> Any:
        """Handle a tool call from the sandbox.

        This is called by the runtime when agent code makes a tool call.
        Capabilities are enforced before the call is passed to the handler.

        The mcp: prefix is stripped before calling the user's handler for
        consistency - users always see the original tool name they registered.
        """
        if self.tool_handler is None:
            raise RuntimeError(f"No handler for tool call: {method}")
        # Strip mcp: prefix (WASM runtime adds it internally)
        clean_method = method.removeprefix("mcp:")
        return self.tool_handler(clean_method, params)

    def _get_prelude(self) -> str:
        """Get the prelude script, caching it on first access.

        The prelude defines fs and shell wrappers, plus any registered tools
        as global async functions. It's read from /tools/prelude.js in the VFS.

        Returns:
            The prelude JavaScript code, or empty string if not available.
        """
        # Return cached prelude if available
        if self._prelude is not None:
            return self._prelude

        # Try to read prelude from VFS
        if self._runtime is None:
            self._prelude = ""
            return ""

        try:
            self._prelude = self._runtime.execute("cat /tools/prelude.js")
        except RuntimeError as e:
            # Prelude file doesn't exist or can't be read - this is expected
            # for minimal sandbox configurations without tool stubs
            import logging

            logging.getLogger(__name__).debug(
                "Prelude not available (expected for minimal configs): %s", e
            )
            self._prelude = ""

        return self._prelude

    def execute(
        self,
        code: str,
        on_output: Callable[[str], None] | None = None,
        stdin: str | bytes | None = None,
    ) -> str:
        """Execute JavaScript code in the sandbox.

        The code runs in the embedded QuickJS interpreter with:
        - Tools as async global functions (e.g., math_add())
        - fs module for VFS access (async: await fs.readFile())
        - shell() for Unix commands (async: await shell.run())
        - Introspection: listTools(), getToolInfo(name)

        Example::

            result = sandbox.execute('''
                // Tools are async functions: "math.add" -> math_add()
                const sum = await math_add({a: 1, b: 2});

                // fs is async (Sync methods throw helpful errors)
                await fs.writeFile('/workspace/result.json', JSON.stringify(sum));

                // shell.run() returns stdout, throws on non-zero exit
                const content = await shell.run('cat /workspace/result.json');

                // Discover available tools
                console.log('Available:', listTools());

                console.log(content);
            ''')

        Streaming output::

            def stream_handler(chunk: str):
                print(chunk, end="", flush=True)

            sandbox.execute('console.log("hello")', on_output=stream_handler)

        Stdin piping (for large code)::

            # Bypass command size limits by piping code via stdin
            sandbox.execute('', stdin=large_js_code)

        Args:
            code: JavaScript code to execute.
            on_output: Optional callback for streaming output. Called with each
                chunk of stdout as it becomes available.
            stdin: Optional JavaScript code to provide via stdin. Useful for
                bypassing command size limits with large code blocks.

        Returns:
            stdout output from the code execution.

        Raises:
            RuntimeError: If execution fails or sandbox not initialized.
        """
        if self._runtime is None:
            raise RuntimeError("Sandbox not initialized")

        # Get cached prelude (reads from VFS once on first call)
        prelude = self._get_prelude()

        # If stdin provided, use it for the code (bypasses command size limits)
        if stdin is not None:
            stdin_code = stdin if isinstance(stdin, str) else stdin.decode("utf-8")
            # Wrap in async IIFE with .then() to capture return value and .catch() for errors
            wrapped_code = (
                f"(async () => {{\n{stdin_code}\n}})()"
                f".then(r => {{ if (r !== undefined) console.log(typeof r === 'string' ? r : JSON.stringify(r)); }})"
                f".catch(e => {{"
                f"  const msg = (e.name ? e.name + ': ' : '') + (e.message || String(e));"
                f"  console.error(e.stack ? msg + '\\n' + e.stack : msg);"
                f"}});"
            )
            full_code = f"{prelude}\n{wrapped_code}" if prelude else wrapped_code
            # Use `node -` to read from stdin
            return self._runtime.execute("node -", on_output, stdin=full_code)

        # Wrap user code in async IIFE to support top-level await
        # QuickJS script mode doesn't support top-level await natively
        # The .then() handler captures the return value and logs it (if not undefined)
        # The .catch() handler ensures unhandled rejections are reported to stderr
        # Without it, errors like `x.foo.bar` where x is undefined silently fail
        # We format the error with name: message\\nstack for readable output
        wrapped_code = (
            f"(async () => {{\n{code}\n}})()"
            f".then(r => {{ if (r !== undefined) console.log(typeof r === 'string' ? r : JSON.stringify(r)); }})"
            f".catch(e => {{"
            f"  const msg = (e.name ? e.name + ': ' : '') + (e.message || String(e));"
            f"  console.error(e.stack ? msg + '\\n' + e.stack : msg);"
            f"}});"
        )

        # Combine prelude with wrapped user code
        full_code = f"{prelude}\n{wrapped_code}" if prelude else wrapped_code

        # Wrap code in node command
        # The shell's "node -e" applet executes JavaScript
        return self._runtime.execute(f"node -e {_quote_js(full_code)}", on_output)

    async def execute_async(
        self,
        code: str,
        on_output: Callable[[str], None] | None = None,
        stdin: str | bytes | None = None,
    ) -> str:
        """Execute JavaScript code with async tool handler support.

        This is the async version of execute() that properly awaits async
        tool handlers. Use this when your tool_handler is an async function.

        Example::

            async def my_tool_handler(method: str, params: dict) -> Any:
                # Async I/O operations
                response = await httpx.get(f"https://api.example.com/{method}")
                return response.json()

            sandbox = Sandbox(
                tools=[...],
                capabilities=[...],
                tool_handler=my_tool_handler,  # async handler
            )

            # Use execute_async to await the async handler
            result = await sandbox.execute_async('''
                const data = await my_api_call({id: 123});
                console.log(data);
            ''')

        Args:
            code: JavaScript code to execute.
            on_output: Optional callback for streaming output. Called with each
                chunk of stdout as it becomes available.
            stdin: Optional JavaScript code to provide via stdin. Useful for
                bypassing command size limits with large code blocks.

        Returns:
            stdout output from the code execution.

        Raises:
            RuntimeError: If execution fails or sandbox not initialized.
        """
        if self._runtime is None:
            raise RuntimeError("Sandbox not initialized")

        # Get cached prelude (reads from VFS once on first call)
        prelude = self._get_prelude()

        # If stdin provided, use it for the code (bypasses command size limits)
        if stdin is not None:
            stdin_code = stdin if isinstance(stdin, str) else stdin.decode("utf-8")
            wrapped_code = (
                f"(async () => {{\n{stdin_code}\n}})()"
                f".then(r => {{ if (r !== undefined) console.log(typeof r === 'string' ? r : JSON.stringify(r)); }})"
                f".catch(e => {{"
                f"  const msg = (e.name ? e.name + ': ' : '') + (e.message || String(e));"
                f"  console.error(e.stack ? msg + '\\n' + e.stack : msg);"
                f"}});"
            )
            full_code = f"{prelude}\n{wrapped_code}" if prelude else wrapped_code
            return await self._runtime.execute_async(
                "node -", on_output, stdin=full_code
            )

        # Wrap user code in async IIFE to support top-level await
        # The .then() handler captures the return value and logs it (if not undefined)
        # The .catch() handler ensures unhandled rejections are reported to stderr
        wrapped_code = (
            f"(async () => {{\n{code}\n}})()"
            f".then(r => {{ if (r !== undefined) console.log(typeof r === 'string' ? r : JSON.stringify(r)); }})"
            f".catch(e => {{"
            f"  const msg = (e.name ? e.name + ': ' : '') + (e.message || String(e));"
            f"  console.error(e.stack ? msg + '\\n' + e.stack : msg);"
            f"}});"
        )

        # Combine prelude with wrapped user code
        full_code = f"{prelude}\n{wrapped_code}" if prelude else wrapped_code

        # Use async execute to await async tool handlers
        return await self._runtime.execute_async(
            f"node -e {_quote_js(full_code)}", on_output
        )

    def shell(self, command: str, stdin: str | bytes | None = None) -> str:
        """Execute a shell command in the sandbox.

        Provides Unix-like utilities:
        - grep, jq, tr - Text processing
        - sort, uniq, head, tail, wc, cut - Data manipulation
        - cat, ls, mkdir, rm - File operations
        - Pipes and command chaining

        Example::

            # Process JSON with grep and cut
            result = sandbox.shell('cat /tmp/data.json | grep "error" | head -5')

            # Count lines
            count = sandbox.shell('wc -l /tmp/data.json')

            # Pipe long script to sh (bypasses command size limit)
            long_script = '''
            echo "Line 1"
            echo "Line 2"
            # ... more lines
            '''
            result = sandbox.shell('sh', stdin=long_script)

        Args:
            command: Shell command to execute.
            stdin: Optional data to provide on stdin. Useful for piping
                large scripts to `sh` without hitting command size limits.

        Returns:
            Command output.
        """
        if self._runtime is None:
            raise RuntimeError("Sandbox not initialized")
        return self._runtime.execute(command, stdin=stdin)

    def can_call(self, method: str, params: dict[str, Any] | None = None) -> bool:
        """Check if a tool call would be allowed.

        Useful for agents to introspect their permissions before attempting
        a call that might fail.

        Args:
            method: The method to check.
            params: Optional parameters for constraint checking.

        Returns:
            True if some capability allows this call.
        """
        if self._runtime is None:
            return False
        return self._runtime.can_call(method, params)

    @property
    def last_stderr(self) -> str:
        """Get stderr from the last execution.

        Returns:
            Captured stderr output as a string, empty if no runtime.
        """
        if self._runtime is None:
            return ""
        return self._runtime.last_stderr

    def get_capabilities(self) -> list[MethodCapability]:
        """Get all capabilities for this sandbox.

        Returns the capabilities that were provided at initialization,
        extracted from the PCA.

        Returns:
            List of method capabilities.
        """
        return list(self.capabilities)

    def get_remaining_calls(self, capability_key: str) -> int | None:
        """Get remaining calls for a capability.

        Args:
            capability_key: The capability key (e.g., "cap:method:stripe/**").

        Returns:
            Remaining calls, or None if capability has no limit or doesn't exist.

        Example::

            sandbox = Sandbox(
                capabilities=[MethodCapability(method_pattern="api/*", max_calls=10)]
            )
            print(sandbox.get_remaining_calls("cap:method:api/*"))  # 10
            sandbox.execute("await api_call({})")
            print(sandbox.get_remaining_calls("cap:method:api/*"))  # 9
        """
        if self._runtime is None:
            return None
        return self._runtime.get_remaining_calls(capability_key)

    def get_call_counts(self) -> dict[str, int]:
        """Get remaining call counts for all limited capabilities.

        Returns:
            Dict mapping capability key -> remaining calls.
            Only includes capabilities that have max_calls set.

        Example::

            sandbox = Sandbox(capabilities=[
                MethodCapability(method_pattern="read/*", max_calls=100),
                MethodCapability(method_pattern="write/*", max_calls=10),
            ])
            print(sandbox.get_call_counts())
            # {'cap:method:read/*': 100, 'cap:method:write/*': 10}
        """
        if self._runtime is None:
            return {}
        return self._runtime.get_call_counts()

    def get_audit_entries(
        self,
        entry_type: str | None = None,
        since: datetime | None = None,
    ) -> Iterator[AuditEntry]:
        """Get audit log entries with optional filtering.

        Requires audit_config to be set when creating the Sandbox.

        Args:
            entry_type: Filter by entry type (e.g., "tool_call", "stream_chunk").
            since: Only return entries after this timestamp.

        Yields:
            Matching audit entries.

        Example::

            sandbox = Sandbox(
                tools=[...],
                capabilities=[...],
                audit_config=AuditConfig(agent_id="my-agent"),
            )
            sandbox.execute('console.log("hello")')

            # Get all entries
            for entry in sandbox.get_audit_entries():
                print(entry.type, entry.timestamp)

            # Get only stream chunks
            for entry in sandbox.get_audit_entries(entry_type="stream_chunk"):
                print(entry.data)
        """
        if self._audit_collector is None:
            return iter([])
        return self._audit_collector.get_entries(entry_type=entry_type, since=since)

    @property
    def audit_collector(self) -> AuditCollector | None:
        """Get the audit collector for this sandbox.

        Returns:
            The AuditCollector if audit_config was set, otherwise None.
        """
        return self._audit_collector

    def __enter__(self) -> Sandbox:
        """Enter context manager.

        Returns:
            The sandbox instance for use in the with block.

        Example::

            with Sandbox(tools=[...], capabilities=[...]) as sandbox:
                result = sandbox.execute('console.log("hello")')
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context manager and clean up resources.

        The runtime is destroyed when exiting the context manager,
        freeing all WASM resources.

        Args:
            exc_type: Exception type if an exception was raised, None otherwise.
            exc_val: Exception value if an exception was raised, None otherwise.
            exc_tb: Traceback if an exception was raised, None otherwise.
        """
        # Clean up audit collector (flushes file if configured)
        if self._audit_collector is not None:
            self._audit_collector.close()
            self._audit_collector = None

        # Clean up runtime resources
        self._runtime = None


def _quote_js(code: str) -> str:
    """Quote JavaScript code for shell execution."""
    # Escape single quotes and wrap in single quotes
    escaped = code.replace("'", "'\\''")
    return f"'{escaped}'"
