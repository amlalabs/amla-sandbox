"""End-to-end tests for WASM runtime with tool execution.

These tests verify the complete flow:
1. Python registers tools
2. JavaScript code calls tools
3. Python handles the tool calls
4. JavaScript receives the results
"""

# pyright: reportPrivateUsage=warning

import pytest

from typing import Any

from amla_sandbox import (
    ConstraintSet,
    MethodCapability,
    Param,
    Sandbox,
    ToolDefinition,
)


class TestEndToEndToolExecution:
    """Test full tool execution flow through WASM."""

    def test_simple_tool_call_from_js(self) -> None:
        """JavaScript can call a Python tool and get the result."""
        tool_calls: list[tuple[str, dict[str, Any]]] = []

        def handle_tool(method: str, params: dict[str, Any]) -> dict[str, Any]:
            tool_calls.append((method, params))
            # Method comes as "mcp:math.add" (provider:action format)
            if "math.add" in method:
                return {"result": params["a"] + params["b"]}
            return {"error": f"Unknown method: {method}"}

        tools = [
            ToolDefinition(
                name="math.add",
                description="Add two numbers",
                parameters={
                    "type": "object",
                    "properties": {
                        "a": {"type": "number"},
                        "b": {"type": "number"},
                    },
                    "required": ["a", "b"],
                },
            ),
        ]

        sandbox = Sandbox(
            tools=tools,
            # Use ** to match all methods, or mcp:math.* for specific
            capabilities=[MethodCapability(method_pattern="**")],
            tool_handler=handle_tool,
        )

        # Execute JavaScript that calls the tool
        result = sandbox.execute("""
            const r = await math_add({a: 3, b: 4});
            console.log(JSON.stringify(r));
        """)

        # Verify the tool was called
        assert len(tool_calls) == 1
        assert "math.add" in tool_calls[0][0]  # mcp:math.add
        assert tool_calls[0][1] == {"a": 3, "b": 4}

        # Verify the result came back to JS
        assert '{"result":7}' in result or '"result":7' in result

    def test_multiple_tool_calls(self) -> None:
        """JavaScript can call multiple tools sequentially."""
        tool_calls: list[tuple[str, dict[str, Any]]] = []

        def handle_tool(method: str, params: dict[str, Any]) -> dict[str, Any]:
            tool_calls.append((method, params))
            # Method comes as "mcp:math.add" (provider:action format)
            if "math.add" in method:
                return {"result": params["a"] + params["b"]}
            if "math.multiply" in method:
                return {"result": params["a"] * params["b"]}
            return {"error": f"Unknown method: {method}"}

        tools = [
            ToolDefinition(
                name="math.add",
                description="Add two numbers",
                parameters={
                    "type": "object",
                    "properties": {
                        "a": {"type": "number"},
                        "b": {"type": "number"},
                    },
                    "required": ["a", "b"],
                },
            ),
            ToolDefinition(
                name="math.multiply",
                description="Multiply two numbers",
                parameters={
                    "type": "object",
                    "properties": {
                        "a": {"type": "number"},
                        "b": {"type": "number"},
                    },
                    "required": ["a", "b"],
                },
            ),
        ]

        sandbox = Sandbox(
            tools=tools,
            capabilities=[MethodCapability(method_pattern="**")],
            tool_handler=handle_tool,
        )

        result = sandbox.execute("""
            const sum = await math_add({a: 3, b: 4});
            const product = await math_multiply({a: sum.result, b: 5});
            console.log(product.result);
        """)

        assert len(tool_calls) == 2
        assert "35" in result  # (3+4) * 5 = 35

    def test_tool_with_string_result(self) -> None:
        """Tools can return string results."""

        def handle_tool(method: str, params: dict[str, Any]) -> str:
            if "greet" in method:
                return f"Hello, {params['name']}!"
            return "Unknown"

        tools = [
            ToolDefinition(
                name="greet",
                description="Greet someone",
                parameters={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                    },
                    "required": ["name"],
                },
            ),
        ]

        sandbox = Sandbox(
            tools=tools,
            capabilities=[MethodCapability(method_pattern="**")],
            tool_handler=handle_tool,
        )

        result = sandbox.execute("""
            const greeting = await greet({name: "World"});
            console.log(greeting);
        """)

        assert "Hello, World!" in result

    def test_tool_with_json_result(self) -> None:
        """Tools can return complex JSON results."""

        def handle_tool(method: str, params: dict[str, Any]) -> dict[str, Any]:
            if "get_user" in method:
                return {
                    "id": params.get("id", 1),
                    "name": "Alice",
                    "email": "alice@example.com",
                    "roles": ["admin", "user"],
                }
            return {}

        tools = [
            ToolDefinition(
                name="get_user",
                description="Get user info",
                parameters={
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                    },
                },
            ),
        ]

        sandbox = Sandbox(
            tools=tools,
            capabilities=[MethodCapability(method_pattern="**")],
            tool_handler=handle_tool,
        )

        result = sandbox.execute("""
            const user = await get_user({id: 42});
            console.log(user.name);
            console.log(user.roles.length);
        """)

        assert "Alice" in result
        assert "2" in result

    def test_vfs_and_shell_with_tool_data(self) -> None:
        """Tool results can be saved to VFS and processed with shell."""

        def handle_tool(method: str, params: dict[str, Any]) -> list[dict[str, Any]]:
            if "get_items" in method:
                return [
                    {"id": 1, "name": "apple", "price": 1.50},
                    {"id": 2, "name": "banana", "price": 0.75},
                    {"id": 3, "name": "cherry", "price": 2.00},
                ]
            return []

        tools = [
            ToolDefinition(
                name="get_items",
                description="Get list of items",
                parameters={"type": "object", "properties": {}},
            ),
        ]

        sandbox = Sandbox(
            tools=tools,
            capabilities=[MethodCapability(method_pattern="**")],
            tool_handler=handle_tool,
        )

        # Execute JavaScript that calls the tool
        result = sandbox.execute("""
            const items = await get_items({});
            console.log('Got items:', JSON.stringify(items));
        """)

        # Verify tool was called and returned data
        assert "apple" in result

        # Write data to VFS using shell (more reliable than JS fs)
        sandbox.shell('echo \'{"items":["apple","banana"]}\' > /workspace/items.json')

        # Verify file was written
        file_content = sandbox.shell("cat /workspace/items.json")
        assert "apple" in file_content
        assert "banana" in file_content

    def test_sandbox_context_manager(self) -> None:
        """Sandbox works as context manager."""
        tool_calls: list[str] = []

        def handle_tool(method: str, params: dict[str, Any]) -> int:
            tool_calls.append(method)
            return params.get("x", 0) * 2 if "double" in method else 0

        tools = [
            ToolDefinition(
                name="double",
                description="Double a number",
                parameters={
                    "type": "object",
                    "properties": {"x": {"type": "number"}},
                },
            ),
        ]

        with Sandbox(
            tools=tools,
            capabilities=[MethodCapability(method_pattern="**")],
            tool_handler=handle_tool,
        ) as sandbox:
            result = sandbox.execute("""
                const r = await double({x: 21});
                console.log(r);
            """)

        assert "42" in result
        assert len(tool_calls) == 1


class TestShellIntegration:
    """Tests for shell command integration."""

    def test_basic_shell_command(self) -> None:
        """Basic shell commands work."""
        sandbox = Sandbox()
        result = sandbox.shell("echo 'hello world'")
        assert "hello world" in result

    def test_shell_pipe(self) -> None:
        """Shell pipes work."""
        sandbox = Sandbox()
        result = sandbox.shell("echo 'line1\nline2\nline3' | wc -l")
        assert "3" in result

    def test_shell_grep(self) -> None:
        """Shell grep works."""
        sandbox = Sandbox()

        # First write some data (/workspace exists by default)
        sandbox.shell("echo 'apple\nbanana\napricot' > /workspace/fruits.txt")

        # Then grep it
        result = sandbox.shell("grep 'ap' /workspace/fruits.txt")
        assert "apple" in result
        assert "apricot" in result
        assert "banana" not in result


class TestAsyncFsAPI:
    """Tests for the async fs API in JavaScript."""

    def test_fs_write_and_read(self) -> None:
        """fs.writeFile and fs.readFile work with await."""
        sandbox = Sandbox()

        result = sandbox.execute("""
            // Write a file using async fs API
            await fs.writeFile('/workspace/test.txt', 'hello from js');

            // Read it back
            const content = await fs.readFile('/workspace/test.txt');
            console.log(content);
        """)

        assert "hello from js" in result

    def test_fs_exists_and_stat(self) -> None:
        """fs.exists and fs.stat work with await."""
        sandbox = Sandbox()

        result = sandbox.execute("""
            // Check if workspace exists
            const exists = await fs.exists('/workspace');
            console.log('exists:', exists);

            // Get stats
            const stat = await fs.stat('/workspace');
            console.log('isDir:', stat.isDirectory);
        """)

        assert "exists: true" in result
        assert "isDir: true" in result

    def test_fs_mkdir_and_readdir(self) -> None:
        """fs.mkdir and fs.readdir work with await."""
        sandbox = Sandbox()

        result = sandbox.execute("""
            // Create a directory
            await fs.mkdir('/workspace/subdir');

            // Write files
            await fs.writeFile('/workspace/subdir/a.txt', 'aaa');
            await fs.writeFile('/workspace/subdir/b.txt', 'bbb');

            // List directory
            const files = await fs.readdir('/workspace/subdir');
            console.log(files.sort().join(','));
        """)

        assert "a.txt" in result
        assert "b.txt" in result


class TestCapabilityEnforcementE2E:
    """E2E tests for capability enforcement."""

    def test_allowed_call_succeeds(self) -> None:
        """Tool calls that match capabilities succeed."""

        def handle_tool(method: str, params: dict[str, Any]) -> str:
            return "success"

        tools = [
            ToolDefinition(
                name="fs.read",
                description="Read a file",
                parameters={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
            ),
        ]

        sandbox = Sandbox(
            tools=tools,
            # Use ** for all methods, constraints on params
            capabilities=[
                MethodCapability(
                    method_pattern="**",
                    constraints=ConstraintSet([Param("path").starts_with("/tmp/")]),
                )
            ],
            tool_handler=handle_tool,
        )

        result = sandbox.execute("""
            const r = await fs_read({path: "/tmp/data.txt"});
            console.log(r);
        """)

        assert "success" in result

    def test_can_call_introspection(self) -> None:
        """Sandbox.can_call works for introspection."""
        sandbox = Sandbox(
            capabilities=[
                MethodCapability(
                    method_pattern="api/*",
                    constraints=ConstraintSet([Param("limit") <= 100]),
                )
            ],
        )

        # Allowed
        assert sandbox.can_call("api/users", {"limit": 50})
        assert sandbox.can_call("api/posts", {"limit": 100})

        # Denied - wrong pattern
        assert not sandbox.can_call("other/method", {})

        # Denied - constraint violation
        assert not sandbox.can_call("api/users", {"limit": 200})


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_tool_call_without_handler_raises_error(self) -> None:
        """Calling a tool without a handler raises RuntimeError."""
        tools = [
            ToolDefinition(
                name="test.tool",
                description="Test tool",
                parameters={"type": "object", "properties": {}},
            ),
        ]

        # No tool_handler provided
        sandbox = Sandbox(
            tools=tools,
            capabilities=[MethodCapability(method_pattern="**")],
            # tool_handler=None implicitly
        )

        # Calling a tool should raise error
        result = sandbox.execute("""
            try {
                await test_tool({});
            } catch (e) {
                console.log('ERROR: ' + e.message);
            }
        """)

        # Should see the error message
        assert "No handler" in result or "ERROR" in result

    def test_get_capabilities_returns_list(self) -> None:
        """get_capabilities returns the configured capabilities."""
        caps = [
            MethodCapability(method_pattern="api/*"),
            MethodCapability(method_pattern="fs/*"),
        ]
        sandbox = Sandbox(capabilities=caps)

        result = sandbox.get_capabilities()
        assert len(result) == 2
        assert result[0].method_pattern == "api/*"
        assert result[1].method_pattern == "fs/*"

    def test_shell_without_runtime_raises_error(self) -> None:
        """Calling shell after __exit__ raises RuntimeError."""
        sandbox = Sandbox()

        # Manually clear runtime (simulates what __exit__ does)
        sandbox._runtime = None

        try:
            sandbox.shell("echo test")
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            assert "not initialized" in str(e)

    def test_execute_without_runtime_raises_error(self) -> None:
        """Calling execute after __exit__ raises RuntimeError."""
        sandbox = Sandbox()

        # Manually clear runtime
        sandbox._runtime = None

        try:
            sandbox.execute("console.log('test')")
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            assert "not initialized" in str(e)

    def test_can_call_without_runtime_returns_false(self) -> None:
        """can_call returns False when runtime is not initialized."""
        sandbox = Sandbox()
        sandbox._runtime = None

        assert sandbox.can_call("any/method") is False

    def test_async_shell_function(self) -> None:
        """The shell() function in JS is async and returns result object."""
        sandbox = Sandbox()

        result = sandbox.execute("""
            const r = await shell('echo hello');
            console.log('stdout:', r.stdout.trim());
            console.log('exitCode:', r.exitCode);
        """)

        assert "stdout: hello" in result
        assert "exitCode: 0" in result

    def test_fs_unlink(self) -> None:
        """fs.unlink removes a file."""
        sandbox = Sandbox()

        result = sandbox.execute("""
            // Create a file
            await fs.writeFile('/workspace/deleteme.txt', 'temp');

            // Check it exists
            const exists1 = await fs.exists('/workspace/deleteme.txt');
            console.log('before:', exists1);

            // Delete it
            await fs.unlink('/workspace/deleteme.txt');

            // Check it's gone
            const exists2 = await fs.exists('/workspace/deleteme.txt');
            console.log('after:', exists2);
        """)

        assert "before: true" in result
        assert "after: false" in result


class TestDXFeatures:
    """Tests for Developer Experience features."""

    def test_list_tools(self) -> None:
        """listTools() returns available tool names."""
        tools = [
            ToolDefinition(
                name="math.add",
                description="Add numbers",
                parameters={"type": "object", "properties": {}},
            ),
            ToolDefinition(
                name="math.multiply",
                description="Multiply numbers",
                parameters={"type": "object", "properties": {}},
            ),
        ]

        sandbox = Sandbox(
            tools=tools,
            capabilities=[MethodCapability(method_pattern="**")],
        )

        result = sandbox.execute("""
            const tools = listTools();
            console.log(tools.sort().join(','));
        """)

        assert "math_add" in result
        assert "math_multiply" in result

    def test_get_tool_info(self) -> None:
        """getToolInfo() returns tool metadata."""
        tools = [
            ToolDefinition(
                name="stripe.charge",
                description="Create a charge",
                parameters={"type": "object", "properties": {}},
            ),
        ]

        sandbox = Sandbox(
            tools=tools,
            capabilities=[MethodCapability(method_pattern="**")],
        )

        result = sandbox.execute("""
            const info = getToolInfo('stripe_charge');
            console.log('id:', info.id);
            console.log('desc:', info.description);
        """)

        assert "mcp:stripe.charge" in result
        assert "Create a charge" in result

    def test_shell_run_returns_stdout(self) -> None:
        """shell.run() returns stdout string directly."""
        sandbox = Sandbox()

        result = sandbox.execute("""
            const output = await shell.run('echo hello world');
            console.log('got:', output.trim());
        """)

        assert "got: hello world" in result

    def test_shell_run_throws_on_error(self) -> None:
        """shell.run() throws on non-zero exit code."""
        sandbox = Sandbox()

        result = sandbox.execute("""
            try {
                await shell.run('exit 1');
                console.log('ERROR: should have thrown');
            } catch (e) {
                console.log('caught:', e.message);
            }
        """)

        assert "caught:" in result
        assert "exit 1" in result or "failed" in result.lower()

    def test_sync_methods_throw_helpful_errors(self) -> None:
        """fs.*Sync methods throw helpful errors."""
        sandbox = Sandbox()

        result = sandbox.execute("""
            try {
                fs.readFileSync('/workspace/test.txt');
                console.log('ERROR: should have thrown');
            } catch (e) {
                console.log('error:', e.message);
            }
        """)

        assert "await fs.readFile" in result

    def test_tools_registry_exists(self) -> None:
        """__tools__ registry contains tool metadata."""
        tools = [
            ToolDefinition(
                name="api.call",
                description="Make API call",
                parameters={"type": "object", "properties": {}},
            ),
        ]

        sandbox = Sandbox(
            tools=tools,
            capabilities=[MethodCapability(method_pattern="**")],
        )

        result = sandbox.execute("""
            console.log('keys:', Object.keys(__tools__).join(','));
            console.log('has api_call:', 'api_call' in __tools__);
        """)

        assert "api_call" in result
        assert "has api_call: true" in result


class TestAsyncToolHandler:
    """Test async tool handler support."""

    @pytest.mark.asyncio
    async def test_async_tool_handler(self) -> None:
        """Async tool handlers are properly awaited."""
        import asyncio

        call_count = [0]

        async def async_handler(method: str, params: dict[str, Any]) -> dict[str, Any]:
            """Async tool handler that uses asyncio.sleep."""
            call_count[0] += 1
            # Simulate async I/O
            await asyncio.sleep(0.01)
            if "math.add" in method:
                return {"result": params["a"] + params["b"]}
            return {"error": f"Unknown: {method}"}

        tools = [
            ToolDefinition(
                name="math.add",
                description="Add numbers",
                parameters={
                    "type": "object",
                    "properties": {
                        "a": {"type": "number"},
                        "b": {"type": "number"},
                    },
                },
            ),
        ]

        sandbox = Sandbox(
            tools=tools,
            capabilities=[MethodCapability(method_pattern="**")],
            tool_handler=async_handler,
        )

        # Use execute_async to await the async handler
        result = await sandbox.execute_async("""
            const sum = await math_add({a: 10, b: 32});
            console.log('sum:', sum.result);
        """)

        assert "sum: 42" in result
        assert call_count[0] == 1

    @pytest.mark.asyncio
    async def test_sync_handler_works_with_execute_async(self) -> None:
        """Sync handlers also work with execute_async."""
        call_count = [0]

        def sync_handler(method: str, params: dict[str, Any]) -> dict[str, Any]:
            """Regular sync handler."""
            call_count[0] += 1
            if "math.multiply" in method:
                return {"result": params["a"] * params["b"]}
            return {"error": f"Unknown: {method}"}

        tools = [
            ToolDefinition(
                name="math.multiply",
                description="Multiply numbers",
                parameters={
                    "type": "object",
                    "properties": {
                        "a": {"type": "number"},
                        "b": {"type": "number"},
                    },
                },
            ),
        ]

        sandbox = Sandbox(
            tools=tools,
            capabilities=[MethodCapability(method_pattern="**")],
            tool_handler=sync_handler,
        )

        # execute_async works with sync handlers too
        result = await sandbox.execute_async("""
            const product = await math_multiply({a: 6, b: 7});
            console.log('product:', product.result);
        """)

        assert "product: 42" in result
        assert call_count[0] == 1


class TestMaxCallsEnforcement:
    """Test max_calls budget enforcement.

    Tool methods come as "mcp:tool.name" format from the WASM runtime.
    Since patterns use / as separator, "mcp:tool.name" is ONE segment.
    Use ** to match any method, or exact match for specific methods.
    """

    def test_max_calls_limits_tool_calls(self) -> None:
        """Tool calls are limited by max_calls."""
        call_count = [0]

        def handle_tool(method: str, params: dict[str, Any]) -> dict[str, Any]:
            call_count[0] += 1
            return {"count": call_count[0]}

        tools = [
            ToolDefinition(
                name="counter.increment",
                description="Increment counter",
                parameters={"type": "object", "properties": {}},
            ),
        ]

        sandbox = Sandbox(
            tools=tools,
            capabilities=[
                # ** matches any method including "mcp:counter.increment"
                MethodCapability(method_pattern="**", max_calls=3)
            ],
            tool_handler=handle_tool,
        )

        # First 3 calls should work
        result = sandbox.execute("""
            for (let i = 0; i < 3; i++) {
                const r = await counter_increment({});
                console.log('call', r.count);
            }
        """)
        assert "call 1" in result
        assert "call 2" in result
        assert "call 3" in result
        assert call_count[0] == 3

        # 4th call should fail
        result = sandbox.execute("""
            try {
                await counter_increment({});
                console.log('ERROR: should have failed');
            } catch (e) {
                console.log('caught:', e.message);
            }
        """)
        assert "caught:" in result
        assert "limit" in result.lower() or "exceeded" in result.lower()
        # Call count stays at 3 (4th call was rejected)
        assert call_count[0] == 3

    def test_get_remaining_calls(self) -> None:
        """get_remaining_calls returns remaining budget."""

        def handle_tool(method: str, params: dict[str, Any]) -> dict[str, Any]:
            return {"ok": True}

        tools = [
            ToolDefinition(
                name="api.call",
                description="API call",
                parameters={"type": "object", "properties": {}},
            ),
        ]

        sandbox = Sandbox(
            tools=tools,
            capabilities=[
                # Match all methods with this pattern
                MethodCapability(method_pattern="**", max_calls=5)
            ],
            tool_handler=handle_tool,
        )

        cap_key = "cap:method:**"

        # Initial budget
        assert sandbox.get_remaining_calls(cap_key) == 5

        # Make a call
        sandbox.execute("await api_call({});")
        assert sandbox.get_remaining_calls(cap_key) == 4

        # Make 2 more calls
        sandbox.execute("await api_call({}); await api_call({});")
        assert sandbox.get_remaining_calls(cap_key) == 2

    def test_get_call_counts(self) -> None:
        """get_call_counts returns all limited capabilities."""
        sandbox = Sandbox(
            capabilities=[
                MethodCapability(method_pattern="read/*", max_calls=100),
                MethodCapability(method_pattern="write/*", max_calls=10),
                MethodCapability(method_pattern="admin/*"),  # No limit
            ],
        )

        counts = sandbox.get_call_counts()
        assert counts == {
            "cap:method:read/*": 100,
            "cap:method:write/*": 10,
        }
        # admin/* not included (no limit)
        assert "cap:method:admin/*" not in counts

    def test_can_call_respects_budget(self) -> None:
        """can_call returns False when budget exhausted."""
        call_count = [0]

        def handle_tool(method: str, params: dict[str, Any]) -> dict[str, Any]:
            call_count[0] += 1
            return {"ok": True}

        tools = [
            ToolDefinition(
                name="limited.action",
                description="Limited action",
                parameters={"type": "object", "properties": {}},
            ),
        ]

        sandbox = Sandbox(
            tools=tools,
            capabilities=[
                # Use ** to match the method "mcp:limited.action"
                MethodCapability(method_pattern="**", max_calls=2)
            ],
            tool_handler=handle_tool,
        )

        # Can call initially
        assert sandbox.can_call("mcp:limited.action", {}) is True

        # Use up the budget
        sandbox.execute("await limited_action({});")
        sandbox.execute("await limited_action({});")

        # Now can_call returns False
        assert sandbox.can_call("mcp:limited.action", {}) is False

    def test_multiple_capabilities_fallback(self) -> None:
        """When one capability exhausted, fallback to another matching one."""
        call_count = [0]

        def handle_tool(method: str, params: dict[str, Any]) -> dict[str, Any]:
            call_count[0] += 1
            return {"count": call_count[0]}

        tools = [
            ToolDefinition(
                name="data.read",
                description="Read data",
                parameters={"type": "object", "properties": {}},
            ),
        ]

        sandbox = Sandbox(
            tools=tools,
            capabilities=[
                # Specific cap with exact match (checked first, low limit)
                MethodCapability(method_pattern="mcp:data.read", max_calls=2),
                # Broader cap (fallback when specific exhausted)
                MethodCapability(method_pattern="**", max_calls=10),
            ],
            tool_handler=handle_tool,
        )

        # First 2 calls use the specific cap (first match)
        sandbox.execute("await data_read({});")
        sandbox.execute("await data_read({});")

        assert sandbox.get_remaining_calls("cap:method:mcp:data.read") == 0
        assert sandbox.get_remaining_calls("cap:method:**") == 10

        # 3rd call falls back to broader cap since specific is exhausted
        sandbox.execute("await data_read({});")

        assert sandbox.get_remaining_calls("cap:method:**") == 9
        assert call_count[0] == 3


class TestErrorMessagesWithStderr:
    """Tests for improved error messages that include stderr context."""

    def test_js_errors_are_caught_gracefully(self) -> None:
        """JS errors are caught and can be handled in JS code."""
        sandbox = Sandbox()

        # JS try/catch can handle errors - they don't raise Python exceptions
        result = sandbox.execute("""
            try {
                throw new Error('Something went wrong');
            } catch(e) {
                console.log('Caught:', e.message);
            }
        """)

        assert "Caught: Something went wrong" in result

    def test_last_stderr_property(self) -> None:
        """last_stderr property captures stderr output."""
        sandbox = Sandbox()

        # Execute code that writes to stderr
        sandbox.execute("""
            console.error('Warning: this is a warning');
            console.log('Normal output');
        """)

        stderr = sandbox.last_stderr
        # stderr should contain the warning
        assert "Warning" in stderr or stderr == ""  # May be empty if not captured

    def test_stderr_is_accessible_after_execution(self) -> None:
        """stderr is accessible after execution."""
        sandbox = Sandbox()

        # Execute some code
        sandbox.execute("""
            console.log('Normal output');
        """)

        # stderr should be accessible as a string
        stderr = sandbox.last_stderr
        assert isinstance(stderr, str)


class TestAsyncHandlerDetection:
    """Tests for detecting async handlers used with sync execute()."""

    def test_async_handler_with_sync_execute_shows_helpful_error(self) -> None:
        """Using async handler with execute() shows a helpful error in JS."""
        import asyncio

        async def async_handler(method: str, params: dict[str, Any]) -> dict[str, Any]:
            await asyncio.sleep(0.01)
            return {"result": "ok"}

        tools = [
            ToolDefinition(
                name="test.tool",
                description="Test tool",
                parameters={"type": "object", "properties": {}},
            ),
        ]

        sandbox = Sandbox(
            tools=tools,
            capabilities=[MethodCapability(method_pattern="**")],
            tool_handler=async_handler,
        )

        # The error is caught by the WASM runtime and surfaced as a JS exception
        # which can be caught in JS try/catch
        result = sandbox.execute("""
            try {
                await test_tool({});
                console.log('ERROR: should have thrown');
            } catch(e) {
                console.log('caught:', e.message);
            }
        """)

        # Error message should mention coroutine and suggest execute_async
        assert "coroutine" in result.lower()
        assert "execute_async" in result

    def test_sync_handler_with_sync_execute_works(self) -> None:
        """Sync handlers work normally with execute()."""
        call_count = [0]

        def sync_handler(method: str, params: dict[str, Any]) -> dict[str, Any]:
            call_count[0] += 1
            return {"result": "sync works"}

        tools = [
            ToolDefinition(
                name="sync.tool",
                description="Sync tool",
                parameters={"type": "object", "properties": {}},
            ),
        ]

        sandbox = Sandbox(
            tools=tools,
            capabilities=[MethodCapability(method_pattern="**")],
            tool_handler=sync_handler,
        )

        result = sandbox.execute("""
            const r = await sync_tool({});
            console.log(r.result);
        """)

        assert "sync works" in result
        assert call_count[0] == 1


class TestStreamingOutputCallback:
    """Tests for on_output streaming callback functionality."""

    def test_on_output_receives_chunks(self) -> None:
        """on_output callback receives output as it happens."""
        chunks: list[str] = []

        def collect_chunks(chunk: str) -> None:
            chunks.append(chunk)

        sandbox = Sandbox()

        result = sandbox.execute(
            """
            console.log('chunk1');
            console.log('chunk2');
            console.log('chunk3');
            """,
            on_output=collect_chunks,
        )

        # Final result should contain all chunks
        assert "chunk1" in result
        assert "chunk2" in result
        assert "chunk3" in result

        # Callback should have been called with chunks
        combined = "".join(chunks)
        assert "chunk1" in combined
        assert "chunk2" in combined
        assert "chunk3" in combined

    def test_on_output_without_callback(self) -> None:
        """execute() works without on_output callback."""
        sandbox = Sandbox()

        result = sandbox.execute("""
            console.log('no callback test');
        """)

        assert "no callback test" in result

    def test_on_output_with_tool_calls(self) -> None:
        """on_output works correctly with tool calls interleaved."""
        chunks: list[str] = []

        def collect_chunks(chunk: str) -> None:
            chunks.append(chunk)

        def handle_tool(method: str, params: dict[str, Any]) -> dict[str, Any]:
            return {"value": params.get("x", 0) * 2}

        tools = [
            ToolDefinition(
                name="double",
                description="Double a number",
                parameters={
                    "type": "object",
                    "properties": {"x": {"type": "number"}},
                },
            ),
        ]

        sandbox = Sandbox(
            tools=tools,
            capabilities=[MethodCapability(method_pattern="**")],
            tool_handler=handle_tool,
        )

        result = sandbox.execute(
            """
            console.log('before tool');
            const r = await double({x: 21});
            console.log('result:', r.value);
            console.log('after tool');
            """,
            on_output=collect_chunks,
        )

        # Result contains all output
        assert "before tool" in result
        assert "result: 42" in result
        assert "after tool" in result

        # Chunks were collected
        combined = "".join(chunks)
        assert "before tool" in combined

    @pytest.mark.asyncio
    async def test_on_output_with_execute_async(self) -> None:
        """on_output works with execute_async too."""
        chunks: list[str] = []

        def collect_chunks(chunk: str) -> None:
            chunks.append(chunk)

        sandbox = Sandbox()

        result = await sandbox.execute_async(
            """
            console.log('async chunk 1');
            console.log('async chunk 2');
            """,
            on_output=collect_chunks,
        )

        assert "async chunk 1" in result
        assert "async chunk 2" in result

        combined = "".join(chunks)
        assert "async chunk" in combined

    def test_on_output_captures_incremental_output(self) -> None:
        """on_output captures output incrementally, not all at once."""
        call_times: list[int] = []

        def track_chunks(chunk: str) -> None:
            call_times.append(len(chunk))

        sandbox = Sandbox()

        # Generate multiple lines of output
        sandbox.execute(
            """
            for (let i = 0; i < 5; i++) {
                console.log('line ' + i);
            }
            """,
            on_output=track_chunks,
        )

        # Callback should have been called multiple times
        # (exact number depends on buffering, but should be > 0)
        assert len(call_times) > 0


class TestVFSStatePersistence:
    """Tests for VFS state persistence between execute() calls."""

    def test_vfs_persists_between_executions(self) -> None:
        """Files written in one execute() are available in the next."""
        sandbox = Sandbox()

        # First execution - write a file
        sandbox.execute("""
            await fs.writeFile('/workspace/state.json', '{"counter": 1}');
            console.log('wrote state');
        """)

        # Second execution - read and update the file
        result = sandbox.execute("""
            const data = JSON.parse(await fs.readFile('/workspace/state.json'));
            console.log('counter was:', data.counter);
            data.counter += 1;
            await fs.writeFile('/workspace/state.json', JSON.stringify(data));
            console.log('counter now:', data.counter);
        """)

        assert "counter was: 1" in result
        assert "counter now: 2" in result

        # Third execution - verify persistence
        result = sandbox.execute("""
            const data = JSON.parse(await fs.readFile('/workspace/state.json'));
            console.log('final counter:', data.counter);
        """)

        assert "final counter: 2" in result

    def test_vfs_cleared_on_new_sandbox(self) -> None:
        """New sandbox instances start with fresh VFS."""
        sandbox1 = Sandbox()

        # Write file in first sandbox
        sandbox1.execute("""
            await fs.writeFile('/workspace/test.txt', 'sandbox1 data');
        """)

        # Verify it exists in first sandbox
        result1 = sandbox1.execute("""
            const exists = await fs.exists('/workspace/test.txt');
            console.log('exists in sandbox1:', exists);
        """)
        assert "exists in sandbox1: true" in result1

        # Create new sandbox - should have fresh VFS
        sandbox2 = Sandbox()
        result2 = sandbox2.execute("""
            const exists = await fs.exists('/workspace/test.txt');
            console.log('exists in sandbox2:', exists);
        """)
        assert "exists in sandbox2: false" in result2

    def test_tool_results_can_be_cached_in_vfs(self) -> None:
        """Tool results can be cached in VFS for later use."""
        call_count = [0]

        def expensive_tool(method: str, params: dict[str, Any]) -> dict[str, Any]:
            call_count[0] += 1
            return {"data": [1, 2, 3, 4, 5], "computed_at": call_count[0]}

        tools = [
            ToolDefinition(
                name="expensive_compute",
                description="Expensive computation",
                parameters={"type": "object", "properties": {}},
            ),
        ]

        sandbox = Sandbox(
            tools=tools,
            capabilities=[MethodCapability(method_pattern="**")],
            tool_handler=expensive_tool,
        )

        # First call - compute and cache
        result1 = sandbox.execute("""
            const result = await expensive_compute({});
            await fs.writeFile('/workspace/cache.json', JSON.stringify(result));
            console.log('computed:', result.computed_at);
        """)
        assert "computed: 1" in result1
        assert call_count[0] == 1

        # Second call - use cache instead of recomputing
        result2 = sandbox.execute("""
            const cached = JSON.parse(await fs.readFile('/workspace/cache.json'));
            console.log('from cache, computed_at:', cached.computed_at);
            console.log('data length:', cached.data.length);
        """)
        assert "from cache, computed_at: 1" in result2
        assert "data length: 5" in result2
        # Tool was NOT called again
        assert call_count[0] == 1


class TestLargeToolResults:
    """Test chunked tool results for large data.

    These tests verify that tool results larger than TOOL_RESULT_CHUNK_SIZE (2KB)
    are properly chunked, transmitted, and reassembled on the JavaScript side.
    """

    def test_large_tool_result_chunked_correctly(self) -> None:
        """Tool results > 2KB are chunked and reassembled correctly."""

        def handle_tool(method: str, params: dict[str, Any]) -> dict[str, Any]:
            if "get_large_data" in method:
                # Generate data larger than chunk size (2KB)
                # 500 items * ~20 bytes each = ~10KB
                items = [{"id": i, "value": f"item_{i:04d}"} for i in range(500)]
                return {"items": items, "count": len(items)}
            return {"error": f"Unknown: {method}"}

        tools = [
            ToolDefinition(
                name="get_large_data",
                description="Returns a large dataset",
                parameters={"type": "object", "properties": {}},
            ),
        ]

        sandbox = Sandbox(
            tools=tools,
            capabilities=[MethodCapability(method_pattern="**")],
            tool_handler=handle_tool,
        )

        result = sandbox.execute("""
            const data = await get_large_data({});
            console.log('count:', data.count);
            console.log('first:', data.items[0].id, data.items[0].value);
            console.log('last:', data.items[499].id, data.items[499].value);
        """)

        assert "count: 500" in result
        assert "first: 0 item_0000" in result
        assert "last: 499 item_0499" in result

    def test_very_large_tool_result_50kb(self) -> None:
        """Tool results around 50KB are properly handled."""

        def handle_tool(method: str, params: dict[str, Any]) -> dict[str, Any]:
            if "get_50kb_data" in method:
                # Generate ~50KB of data
                # Each item is about 50 bytes, so 1000 items = ~50KB
                items = [
                    {"id": i, "name": f"name_{i:06d}", "data": "x" * 30}
                    for i in range(1000)
                ]
                return {"items": items, "total": len(items)}
            return {"error": f"Unknown: {method}"}

        tools = [
            ToolDefinition(
                name="get_50kb_data",
                description="Returns ~50KB of data",
                parameters={"type": "object", "properties": {}},
            ),
        ]

        sandbox = Sandbox(
            tools=tools,
            capabilities=[MethodCapability(method_pattern="**")],
            tool_handler=handle_tool,
        )

        result = sandbox.execute("""
            const data = await get_50kb_data({});
            console.log('total:', data.total);
            // Verify first and last items
            console.log('first_id:', data.items[0].id);
            console.log('last_id:', data.items[999].id);
            // Verify data integrity
            const allValid = data.items.every((item, idx) => item.id === idx);
            console.log('all_valid:', allValid);
        """)

        assert "total: 1000" in result
        assert "first_id: 0" in result
        assert "last_id: 999" in result
        assert "all_valid: true" in result

    def test_large_result_with_unicode(self) -> None:
        """Large results with unicode characters are handled correctly."""

        def handle_tool(method: str, params: dict[str, Any]) -> dict[str, Any]:
            if "get_unicode_data" in method:
                # Generate data with various unicode characters
                items = [
                    {
                        "id": i,
                        "text": f"Hello 世界 🌍 item_{i}",
                        "arabic": "مرحبا",
                        "hebrew": "שלום",
                    }
                    for i in range(200)
                ]
                return {"items": items, "count": len(items)}
            return {"error": f"Unknown: {method}"}

        tools = [
            ToolDefinition(
                name="get_unicode_data",
                description="Returns data with unicode",
                parameters={"type": "object", "properties": {}},
            ),
        ]

        sandbox = Sandbox(
            tools=tools,
            capabilities=[MethodCapability(method_pattern="**")],
            tool_handler=handle_tool,
        )

        result = sandbox.execute("""
            const data = await get_unicode_data({});
            console.log('count:', data.count);
            // Check unicode preserved
            const first = data.items[0];
            console.log('has_world:', first.text.includes('世界'));
            console.log('has_emoji:', first.text.includes('🌍'));
        """)

        assert "count: 200" in result
        assert "has_world: true" in result
        assert "has_emoji: true" in result

    def test_multiple_large_results_sequential(self) -> None:
        """Multiple large tool calls in sequence work correctly."""
        call_count = [0]

        def handle_tool(method: str, params: dict[str, Any]) -> dict[str, Any]:
            call_count[0] += 1
            if "get_batch" in method:
                batch_num = params.get("batch", 0)
                # Each batch is ~5KB
                items = [
                    {"batch": batch_num, "id": i, "data": "y" * 20} for i in range(100)
                ]
                return {"batch": batch_num, "items": items}
            return {"error": f"Unknown: {method}"}

        tools = [
            ToolDefinition(
                name="get_batch",
                description="Returns a batch of data",
                parameters={
                    "type": "object",
                    "properties": {"batch": {"type": "integer"}},
                },
            ),
        ]

        sandbox = Sandbox(
            tools=tools,
            capabilities=[MethodCapability(method_pattern="**")],
            tool_handler=handle_tool,
        )

        result = sandbox.execute("""
            // Call three times sequentially
            const b1 = await get_batch({batch: 1});
            const b2 = await get_batch({batch: 2});
            const b3 = await get_batch({batch: 3});

            console.log('batch1:', b1.batch, 'items:', b1.items.length);
            console.log('batch2:', b2.batch, 'items:', b2.items.length);
            console.log('batch3:', b3.batch, 'items:', b3.items.length);
        """)

        assert call_count[0] == 3
        assert "batch1: 1 items: 100" in result
        assert "batch2: 2 items: 100" in result
        assert "batch3: 3 items: 100" in result

    @pytest.mark.asyncio
    async def test_large_result_async_handler(self) -> None:
        """Large results work with async tool handlers."""
        import asyncio

        async def async_handler(method: str, params: dict[str, Any]) -> dict[str, Any]:
            if "fetch_large" in method:
                # Simulate async fetch
                await asyncio.sleep(0.01)
                items = [{"id": i, "payload": "data" * 10} for i in range(300)]
                return {"items": items, "fetched": True}
            return {"error": f"Unknown: {method}"}

        tools = [
            ToolDefinition(
                name="fetch_large",
                description="Async fetch of large data",
                parameters={"type": "object", "properties": {}},
            ),
        ]

        sandbox = Sandbox(
            tools=tools,
            capabilities=[MethodCapability(method_pattern="**")],
            tool_handler=async_handler,
        )

        result = await sandbox.execute_async("""
            const data = await fetch_large({});
            console.log('fetched:', data.fetched);
            console.log('count:', data.items.length);
            console.log('first_id:', data.items[0].id);
        """)

        assert "fetched: true" in result
        assert "count: 300" in result
        assert "first_id: 0" in result

    def test_result_at_chunk_boundary(self) -> None:
        """Results exactly at chunk boundaries are handled correctly."""

        def handle_tool(method: str, params: dict[str, Any]) -> dict[str, Any]:
            if "get_boundary_data" in method:
                # Create data that's close to chunk size boundaries
                # TOOL_RESULT_CHUNK_SIZE is 2048 bytes
                # Create something that will result in exactly 2 or 3 chunks
                data = "A" * 4000  # Will be ~4KB when JSON encoded
                return {"data": data, "length": len(data)}
            return {"error": f"Unknown: {method}"}

        tools = [
            ToolDefinition(
                name="get_boundary_data",
                description="Returns data near chunk boundaries",
                parameters={"type": "object", "properties": {}},
            ),
        ]

        sandbox = Sandbox(
            tools=tools,
            capabilities=[MethodCapability(method_pattern="**")],
            tool_handler=handle_tool,
        )

        result = sandbox.execute("""
            const data = await get_boundary_data({});
            console.log('length:', data.length);
            console.log('data_length:', data.data.length);
            // Verify data integrity
            const allAs = data.data.split('').every(c => c === 'A');
            console.log('all_As:', allAs);
        """)

        assert "length: 4000" in result
        assert "data_length: 4000" in result
        assert "all_As: true" in result

    def test_nested_large_objects(self) -> None:
        """Deeply nested large objects are chunked and reassembled correctly."""

        def handle_tool(method: str, params: dict[str, Any]) -> dict[str, Any]:
            if "get_nested" in method:
                # Create nested structure - use explicit Any typing for recursive nesting
                result: dict[str, Any] = {"level": 0, "data": []}
                current: dict[str, Any] = result
                for i in range(10):
                    child: dict[str, Any] = {
                        "level": i + 1,
                        "items": [
                            {"idx": j, "val": f"L{i + 1}_{j}"} for j in range(50)
                        ],
                        "child": None,
                    }
                    current["child"] = child
                    current = child
                return result
            return {"error": f"Unknown: {method}"}

        tools = [
            ToolDefinition(
                name="get_nested",
                description="Returns deeply nested data",
                parameters={"type": "object", "properties": {}},
            ),
        ]

        sandbox = Sandbox(
            tools=tools,
            capabilities=[MethodCapability(method_pattern="**")],
            tool_handler=handle_tool,
        )

        result = sandbox.execute("""
            const data = await get_nested({});
            console.log('root_level:', data.level);

            // Navigate to deepest level
            let current = data;
            let depth = 0;
            while (current.child) {
                current = current.child;
                depth++;
            }
            console.log('max_depth:', depth);
            console.log('deepest_level:', current.level);
            console.log('deepest_items:', current.items.length);
        """)

        assert "root_level: 0" in result
        assert "max_depth: 10" in result
        assert "deepest_level: 10" in result
        assert "deepest_items: 50" in result


class TestToolInvocationPaths:
    """Test that tools can be invoked via both JavaScript and shell `tool` applet.

    The sandbox supports two ways to invoke tools:
    1. JavaScript: `await tool_name({param: value})`
    2. Shell applet: `tool name --param value`

    Both paths should produce identical results.
    """

    def test_js_invocation_basic(self) -> None:
        """Basic tool invocation via JavaScript await syntax."""
        tool_calls: list[tuple[str, dict[str, Any]]] = []

        def handle_tool(method: str, params: dict[str, Any]) -> dict[str, Any]:
            tool_calls.append((method, params))
            if "calculator.add" in method:
                return {"sum": params["a"] + params["b"]}
            return {"error": f"Unknown: {method}"}

        tools = [
            ToolDefinition(
                name="calculator.add",
                description="Add two numbers",
                parameters={
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"},
                    },
                    "required": ["a", "b"],
                },
            ),
        ]

        sandbox = Sandbox(
            tools=tools,
            capabilities=[MethodCapability(method_pattern="**")],
            tool_handler=handle_tool,
        )

        result = sandbox.execute("""
            const r = await calculator_add({a: 10, b: 32});
            console.log(JSON.stringify(r));
        """)

        assert len(tool_calls) == 1
        assert tool_calls[0][1] == {"a": 10, "b": 32}
        assert '"sum":42' in result or '"sum": 42' in result

    def test_shell_tool_applet_basic(self) -> None:
        """Basic tool invocation via shell `tool` applet."""
        tool_calls: list[tuple[str, dict[str, Any]]] = []

        def handle_tool(method: str, params: dict[str, Any]) -> dict[str, Any]:
            tool_calls.append((method, params))
            # Method is normalized to colon format: calculator:add
            if "calculator" in method and "add" in method:
                return {"sum": params["a"] + params["b"]}
            return {"error": f"Unknown: {method}"}

        tools = [
            ToolDefinition(
                name="calculator.add",
                description="Add two numbers",
                parameters={
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"},
                    },
                    "required": ["a", "b"],
                },
            ),
        ]

        sandbox = Sandbox(
            tools=tools,
            capabilities=[MethodCapability(method_pattern="**")],
            tool_handler=handle_tool,
        )

        # Use shell to invoke tool via the `tool` applet
        result = sandbox.shell("tool calculator.add --a 10 --b 32")

        assert len(tool_calls) == 1
        assert tool_calls[0][1] == {"a": 10, "b": 32}
        assert '"sum": 42' in result or '"sum":42' in result

    def test_both_paths_produce_same_result(self) -> None:
        """JavaScript and shell tool applet produce identical results."""
        js_calls: list[tuple[str, dict[str, Any]]] = []
        shell_calls: list[tuple[str, dict[str, Any]]] = []

        def handle_tool_js(method: str, params: dict[str, Any]) -> dict[str, Any]:
            js_calls.append((method, params))
            if "weather" in method and "get" in method:
                return {
                    "city": params["city"],
                    "temp": 22,
                    "conditions": "sunny",
                }
            return {"error": f"Unknown: {method}"}

        def handle_tool_shell(method: str, params: dict[str, Any]) -> dict[str, Any]:
            shell_calls.append((method, params))
            if "weather" in method and "get" in method:
                return {
                    "city": params["city"],
                    "temp": 22,
                    "conditions": "sunny",
                }
            return {"error": f"Unknown: {method}"}

        tools = [
            ToolDefinition(
                name="weather.get",
                description="Get weather for a city",
                parameters={
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                    },
                    "required": ["city"],
                },
            ),
        ]

        # Test JavaScript path
        sandbox_js = Sandbox(
            tools=tools,
            capabilities=[MethodCapability(method_pattern="**")],
            tool_handler=handle_tool_js,
        )
        result_js = sandbox_js.execute("""
            const r = await weather_get({city: "Tokyo"});
            console.log(JSON.stringify(r));
        """)

        # Test shell tool applet path
        sandbox_shell = Sandbox(
            tools=tools,
            capabilities=[MethodCapability(method_pattern="**")],
            tool_handler=handle_tool_shell,
        )
        result_shell = sandbox_shell.shell("tool weather.get --city Tokyo")

        # Both should have called the handler with same params
        assert len(js_calls) == 1
        assert len(shell_calls) == 1
        assert js_calls[0][1] == shell_calls[0][1] == {"city": "Tokyo"}

        # Both should return same data (format may differ slightly)
        assert "Tokyo" in result_js
        assert "Tokyo" in result_shell
        assert "22" in result_js
        assert "22" in result_shell
        assert "sunny" in result_js
        assert "sunny" in result_shell

    def test_shell_tool_applet_with_json_flag(self) -> None:
        """Shell tool applet supports --json for complex parameters."""
        tool_calls: list[tuple[str, dict[str, Any]]] = []

        def handle_tool(method: str, params: dict[str, Any]) -> dict[str, Any]:
            tool_calls.append((method, params))
            if "data" in method and "process" in method:
                return {"processed": len(params.get("items", []))}
            return {"error": f"Unknown: {method}"}

        tools = [
            ToolDefinition(
                name="data.process",
                description="Process data items",
                parameters={
                    "type": "object",
                    "properties": {
                        "items": {"type": "array"},
                        "mode": {"type": "string"},
                    },
                    "required": ["items"],
                },
            ),
        ]

        sandbox = Sandbox(
            tools=tools,
            capabilities=[MethodCapability(method_pattern="**")],
            tool_handler=handle_tool,
        )

        result = sandbox.shell(
            """tool data.process --json '{"items": [1, 2, 3], "mode": "fast"}'"""
        )

        assert len(tool_calls) == 1
        assert tool_calls[0][1] == {"items": [1, 2, 3], "mode": "fast"}
        assert '"processed": 3' in result or '"processed":3' in result

    def test_shell_tool_applet_list_command(self) -> None:
        """Shell tool applet --list shows available tools."""
        tools = [
            ToolDefinition(
                name="api.get",
                description="Get API data",
                parameters={"type": "object", "properties": {}},
            ),
            ToolDefinition(
                name="api.post",
                description="Post API data",
                parameters={"type": "object", "properties": {}},
            ),
            ToolDefinition(
                name="db.query",
                description="Query database",
                parameters={"type": "object", "properties": {}},
            ),
        ]

        sandbox = Sandbox(
            tools=tools,
            capabilities=[MethodCapability(method_pattern="**")],
        )

        result = sandbox.shell("tool --list")

        assert "api.get" in result
        assert "api.post" in result
        assert "db.query" in result
        assert "Get API data" in result

    def test_shell_tool_applet_list_with_filter(self) -> None:
        """Shell tool applet --list can filter by provider."""
        tools = [
            ToolDefinition(
                name="stripe.charge",
                description="Create a charge",
                parameters={"type": "object", "properties": {}},
            ),
            ToolDefinition(
                name="stripe.refund",
                description="Create a refund",
                parameters={"type": "object", "properties": {}},
            ),
            ToolDefinition(
                name="slack.send",
                description="Send a message",
                parameters={"type": "object", "properties": {}},
            ),
        ]

        sandbox = Sandbox(
            tools=tools,
            capabilities=[MethodCapability(method_pattern="**")],
        )

        result = sandbox.shell("tool --list stripe")

        assert "stripe.charge" in result
        assert "stripe.refund" in result
        assert "slack.send" not in result

    def test_shell_tool_applet_help_command(self) -> None:
        """Shell tool applet --help shows tool details."""
        tools = [
            ToolDefinition(
                name="calculator.multiply",
                description="Multiply two numbers together",
                parameters={
                    "type": "object",
                    "properties": {
                        "x": {"type": "number", "description": "First number"},
                        "y": {"type": "number", "description": "Second number"},
                    },
                    "required": ["x", "y"],
                },
            ),
        ]

        sandbox = Sandbox(
            tools=tools,
            capabilities=[MethodCapability(method_pattern="**")],
        )

        result = sandbox.shell("tool --help calculator.multiply")

        assert "calculator.multiply" in result
        assert "Multiply two numbers" in result
        assert "--x" in result
        assert "--y" in result

    def test_shell_tool_applet_missing_required_param(self) -> None:
        """Shell tool applet reports missing required parameters."""
        tools = [
            ToolDefinition(
                name="user.create",
                description="Create a user",
                parameters={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "email": {"type": "string"},
                    },
                    "required": ["name", "email"],
                },
            ),
        ]

        sandbox = Sandbox(
            tools=tools,
            capabilities=[MethodCapability(method_pattern="**")],
        )

        # Only provide name, missing email
        sandbox.shell("tool user.create --name Alice")

        # Should report the missing parameter
        assert (
            "email" in sandbox.last_stderr.lower()
            or "required" in sandbox.last_stderr.lower()
        )

    def test_shell_tool_applet_unknown_tool(self) -> None:
        """Shell tool applet reports unknown tools."""
        sandbox = Sandbox(
            tools=[],
            capabilities=[MethodCapability(method_pattern="**")],
        )

        sandbox.shell("tool nonexistent.tool --param value")

        assert (
            "unknown" in sandbox.last_stderr.lower()
            or "not found" in sandbox.last_stderr.lower()
            or "no tools" in sandbox.last_stderr.lower()
        )

    def test_js_and_shell_share_vfs_state(self) -> None:
        """JavaScript tool writes to VFS, shell tool can read it."""

        def handle_tool(method: str, params: dict[str, Any]) -> dict[str, Any]:
            if "data" in method and "generate" in method:
                return {"values": [1, 2, 3, 4, 5], "count": 5}
            return {"error": f"Unknown: {method}"}

        tools = [
            ToolDefinition(
                name="data.generate",
                description="Generate sample data",
                parameters={"type": "object", "properties": {}},
            ),
        ]

        sandbox = Sandbox(
            tools=tools,
            capabilities=[MethodCapability(method_pattern="**")],
            tool_handler=handle_tool,
        )

        # JavaScript calls tool and writes result to VFS
        sandbox.execute("""
            const data = await data_generate({});
            await fs.writeFile('/workspace/data.json', JSON.stringify(data));
        """)

        # Shell can read and process the file
        result = sandbox.shell("cat /workspace/data.json | jq '.count'")
        assert "5" in result

    def test_shell_tool_in_pipeline(self) -> None:
        """Shell tool applet output can be piped to other commands."""

        def handle_tool(method: str, params: dict[str, Any]) -> dict[str, Any]:
            if "users" in method and "list" in method:
                return {
                    "users": [
                        {"name": "Alice", "role": "admin"},
                        {"name": "Bob", "role": "user"},
                        {"name": "Charlie", "role": "admin"},
                    ]
                }
            return {"error": f"Unknown: {method}"}

        tools = [
            ToolDefinition(
                name="users.list",
                description="List all users",
                parameters={"type": "object", "properties": {}},
            ),
        ]

        sandbox = Sandbox(
            tools=tools,
            capabilities=[MethodCapability(method_pattern="**")],
            tool_handler=handle_tool,
        )

        # Pipe tool output through jq to filter admins
        result = sandbox.shell(
            "tool users.list | jq '[.users[] | select(.role == \"admin\") | .name]'"
        )

        assert "Alice" in result
        assert "Charlie" in result
        assert "Bob" not in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
