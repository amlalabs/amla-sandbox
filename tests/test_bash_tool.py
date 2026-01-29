"""Tests for the bash_tool module - AI SDK-style ergonomics."""

# pyright: reportPrivateUsage=warning

from amla_sandbox import create_sandbox_tool
from amla_sandbox.bash_tool import _parse_constraints, _parse_string_constraint
from amla_sandbox.capabilities import Constraint


# === Sample tools for testing ===


def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


def greet(name: str) -> str:
    """Greet someone."""
    return f"Hello, {name}!"


def transfer_money(amount: float, to_account: str) -> bool:
    """Transfer money to an account."""
    return True


# === Tests for create_sandbox_tool ===


class TestCreateSandboxTool:
    """Tests for the create_sandbox_tool function."""

    def test_no_tools(self) -> None:
        """Test creating sandbox tool with no tools (shell only)."""
        sandbox = create_sandbox_tool()

        assert sandbox.sandbox is not None
        assert len(sandbox.tools) == 0

    def test_with_simple_tools(self) -> None:
        """Test creating sandbox tool with Python functions."""
        sandbox = create_sandbox_tool(tools=[add, greet])

        assert len(sandbox.tools) == 2
        assert sandbox.sandbox is not None
        # Check that tools are registered
        assert "add" in sandbox._tool_map
        assert "greet" in sandbox._tool_map

    def test_with_constraints(self) -> None:
        """Test creating sandbox tool with constraints."""
        sandbox = create_sandbox_tool(
            tools=[transfer_money],
            constraints={"transfer_money": {"amount": "<=1000"}},
        )

        assert len(sandbox.tools) == 1
        # The capability should have constraints
        caps = sandbox.sandbox.capabilities
        assert len(caps) == 1
        assert not caps[0].constraints.is_empty()

    def test_with_max_calls_int(self) -> None:
        """Test creating sandbox tool with global max_calls."""
        sandbox = create_sandbox_tool(
            tools=[add, greet],
            max_calls=50,
        )

        caps = sandbox.sandbox.capabilities
        assert all(c.max_calls == 50 for c in caps)

    def test_with_max_calls_dict(self) -> None:
        """Test creating sandbox tool with per-tool max_calls."""
        sandbox = create_sandbox_tool(
            tools=[add, greet],
            max_calls={"add": 10, "greet": 20},
        )

        caps = sandbox.sandbox.capabilities
        cap_by_name = {c.method_pattern.split(":")[-1]: c for c in caps}
        assert cap_by_name["add"].max_calls == 10
        assert cap_by_name["greet"].max_calls == 20

    def test_default_max_calls(self) -> None:
        """Test that default max_calls is applied."""
        sandbox = create_sandbox_tool(tools=[add])

        caps = sandbox.sandbox.capabilities
        # Default is 100
        assert caps[0].max_calls == 100


# === Tests for constraint parsing ===


class TestConstraintParsing:
    """Tests for constraint dict parsing."""

    def test_parse_le_constraint(self) -> None:
        """Test parsing <= constraint."""
        constraint = _parse_string_constraint("amount", "<=1000")
        assert constraint is not None
        assert isinstance(constraint, Constraint)

    def test_parse_ge_constraint(self) -> None:
        """Test parsing >= constraint."""
        constraint = _parse_string_constraint("amount", ">=0")
        assert constraint is not None

    def test_parse_lt_constraint(self) -> None:
        """Test parsing < constraint."""
        constraint = _parse_string_constraint("count", "<100")
        assert constraint is not None

    def test_parse_gt_constraint(self) -> None:
        """Test parsing > constraint."""
        constraint = _parse_string_constraint("count", ">0")
        assert constraint is not None

    def test_parse_eq_constraint(self) -> None:
        """Test parsing == constraint."""
        constraint = _parse_string_constraint("status", "==42")
        assert constraint is not None

    def test_parse_startswith_constraint(self) -> None:
        """Test parsing startswith: constraint."""
        constraint = _parse_string_constraint("path", "startswith:/api/")
        assert constraint is not None

    def test_parse_list_constraint(self) -> None:
        """Test parsing list (enum) constraint."""
        spec = {"currency": ["USD", "EUR", "GBP"]}
        constraints = _parse_constraints(spec)
        assert not constraints.is_empty()

    def test_parse_numeric_constraint(self) -> None:
        """Test parsing numeric (exact value) constraint."""
        spec = {"status": 200}
        constraints = _parse_constraints(spec)
        assert not constraints.is_empty()

    def test_parse_float_constraint(self) -> None:
        """Test parsing float constraint."""
        constraint = _parse_string_constraint("price", "<=99.99")
        assert constraint is not None

    def test_unknown_format_returns_none(self) -> None:
        """Test that unknown constraint format is ignored."""
        constraint = _parse_string_constraint("foo", "unknown format")
        assert constraint is None


# === Tests for combined usage ===


class TestCombinedUsage:
    """Tests for combined constraints and max_calls."""

    def test_multiple_constraints_per_tool(self) -> None:
        """Test tool with multiple constraints."""
        sandbox = create_sandbox_tool(
            tools=[transfer_money],
            constraints={
                "transfer_money": {
                    "amount": "<=10000",
                    "to_account": "startswith:acct_",
                }
            },
        )

        caps = sandbox.sandbox.capabilities
        assert len(caps) == 1
        # Should have both constraints
        assert not caps[0].constraints.is_empty()

    def test_constraints_and_max_calls(self) -> None:
        """Test tool with both constraints and max_calls."""
        sandbox = create_sandbox_tool(
            tools=[transfer_money],
            constraints={"transfer_money": {"amount": "<=1000"}},
            max_calls={"transfer_money": 5},
        )

        caps = sandbox.sandbox.capabilities
        assert caps[0].max_calls == 5
        assert not caps[0].constraints.is_empty()


# === Tests for Bug Fixes ===


class TestJsRuntimeErrorReporting:
    """Tests for JS runtime error reporting.

    Regression tests for: "JS runtime errors are silent (QuickJS issue)"
    When JS code has a runtime error, it should produce error output to stderr
    instead of silently failing with no output.
    """

    def test_undefined_property_access_reports_error(self) -> None:
        """TypeError from undefined property access should be reported."""
        from amla_sandbox import create_sandbox_tool

        sandbox = create_sandbox_tool()
        result = sandbox.run(
            """
            const x = undefined;
            console.log(x.foo.bar);
        """,
            language="javascript",
        )

        # Error should be reported in output (either stdout with [stderr] prefix or in error)
        assert "TypeError" in result or "undefined" in result.lower(), (
            f"Should report TypeError for undefined property access, got: {result!r}"
        )

    def test_undefined_function_call_reports_error(self) -> None:
        """ReferenceError from calling undefined function should be reported."""
        from amla_sandbox import create_sandbox_tool

        sandbox = create_sandbox_tool()
        result = sandbox.run(
            """
            const r = nonExistentFunction();
            console.log(r);
        """,
            language="javascript",
        )

        assert "ReferenceError" in result or "not defined" in result.lower(), (
            f"Should report ReferenceError for undefined function, got: {result!r}"
        )

    def test_type_error_not_a_function_reports_error(self) -> None:
        """TypeError from calling non-function should be reported."""
        from amla_sandbox import create_sandbox_tool

        sandbox = create_sandbox_tool()
        result = sandbox.run(
            """
            const num = 42;
            num.map(x => x * 2);
        """,
            language="javascript",
        )

        assert "TypeError" in result or "not a function" in result.lower(), (
            f"Should report TypeError for calling non-function, got: {result!r}"
        )

    def test_syntax_error_still_works(self) -> None:
        """Syntax errors should still be reported (they already worked)."""
        from amla_sandbox import create_sandbox_tool

        sandbox = create_sandbox_tool()
        result = sandbox.run("console.log('missing quote)", language="javascript")

        assert "error" in result.lower() or "stderr" in result.lower(), (
            f"Syntax errors should be reported, got: {result!r}"
        )

    def test_caught_errors_work(self) -> None:
        """Try-catch blocks should still work correctly."""
        from amla_sandbox import create_sandbox_tool

        sandbox = create_sandbox_tool()
        result = sandbox.run(
            """
            try {
                throw new Error('Test error');
            } catch (e) {
                console.log('Caught:', e.message);
            }
        """,
            language="javascript",
        )

        assert "Caught: Test error" in result, f"Caught errors should work: {result!r}"


class TestConstraintErrorMessages:
    """Tests for specific constraint error messages.

    Regression tests for: "Constraint error messages are vague"
    Error messages should specify which constraint was violated, not just
    say "No capability authorizes method".
    """

    def test_amount_constraint_violation_shows_details(self) -> None:
        """Amount constraint violation should show the specific constraint."""
        from amla_sandbox import create_sandbox_tool

        def transfer_money(amount: int, currency: str, to: str) -> dict:
            return {"status": "success"}

        sandbox = create_sandbox_tool(
            tools=[transfer_money],
            constraints={
                "transfer_money": {
                    "amount": "<=1000",
                    "currency": ["USD", "EUR"],
                }
            },
        )

        result = sandbox.run(
            """
            try {
                await transfer_money({amount: 5000, currency: 'USD', to: 'alice'});
            } catch (e) {
                console.log('ERROR:', e.message);
            }
        """,
            language="javascript",
        )

        # Should mention amount or the actual value
        assert "amount" in result.lower() or "5000" in result or "1000" in result, (
            f"Error should mention 'amount' constraint, got: {result!r}"
        )

    def test_currency_constraint_violation_shows_details(self) -> None:
        """Currency constraint violation should show the specific constraint."""
        from amla_sandbox import create_sandbox_tool

        def transfer_money(amount: int, currency: str, to: str) -> dict:
            return {"status": "success"}

        sandbox = create_sandbox_tool(
            tools=[transfer_money],
            constraints={
                "transfer_money": {
                    "amount": "<=1000",
                    "currency": ["USD", "EUR"],
                }
            },
        )

        result = sandbox.run(
            """
            try {
                await transfer_money({amount: 100, currency: 'GBP', to: 'bob'});
            } catch (e) {
                console.log('ERROR:', e.message);
            }
        """,
            language="javascript",
        )

        # Should mention currency or the invalid value
        assert (
            "currency" in result.lower()
            or "GBP" in result
            or "USD" in result
            or "EUR" in result
        ), f"Error should mention 'currency' constraint, got: {result!r}"

    def test_valid_call_still_works(self) -> None:
        """Valid calls should succeed (sanity check)."""
        from amla_sandbox import create_sandbox_tool

        def transfer_money(amount: int, currency: str, to: str) -> dict:
            return {"status": "success", "amount": amount}

        sandbox = create_sandbox_tool(
            tools=[transfer_money],
            constraints={
                "transfer_money": {
                    "amount": "<=1000",
                    "currency": ["USD", "EUR"],
                }
            },
        )

        result = sandbox.run(
            """
            const r = await transfer_money({amount: 500, currency: 'USD', to: 'alice'});
            console.log(JSON.stringify(r));
        """,
            language="javascript",
        )

        assert "success" in result, f"Valid call should succeed: {result!r}"


class TestDocumentationAsyncMethods:
    """Tests for documentation showing correct async methods.

    Regression tests for: "Documentation shows sync fs methods that don't exist"
    The JS_CODEACT_PROMPT should use async methods (await fs.writeFile)
    instead of sync methods (fs.writeFileSync) that don't exist.
    """

    def test_no_sync_write_file_in_prompt(self) -> None:
        """JS_CODEACT_PROMPT should not contain writeFileSync."""
        from amla_sandbox import JS_CODEACT_PROMPT

        assert "writeFileSync" not in JS_CODEACT_PROMPT, (
            "JS_CODEACT_PROMPT should not contain sync writeFileSync method"
        )

    def test_no_sync_read_file_in_prompt(self) -> None:
        """JS_CODEACT_PROMPT should not contain readFileSync."""
        from amla_sandbox import JS_CODEACT_PROMPT

        assert "readFileSync" not in JS_CODEACT_PROMPT, (
            "JS_CODEACT_PROMPT should not contain sync readFileSync method"
        )

    def test_async_write_file_in_prompt(self) -> None:
        """JS_CODEACT_PROMPT should contain async fs.writeFile with await."""
        from amla_sandbox import JS_CODEACT_PROMPT

        assert "await fs.writeFile" in JS_CODEACT_PROMPT, (
            "JS_CODEACT_PROMPT should use 'await fs.writeFile()'"
        )

    def test_async_shell_in_prompt(self) -> None:
        """JS_CODEACT_PROMPT should contain async shell with await."""
        from amla_sandbox import JS_CODEACT_PROMPT

        assert "await shell" in JS_CODEACT_PROMPT, (
            "JS_CODEACT_PROMPT should use 'await shell()'"
        )

    def test_sync_methods_fail_at_runtime(self) -> None:
        """Using sync methods should fail at runtime (they don't exist)."""
        from amla_sandbox import create_sandbox_tool

        sandbox = create_sandbox_tool()
        result = sandbox.run(
            """
            try {
                fs.writeFileSync('/workspace/test.txt', 'hello');
                console.log('Sync write succeeded (unexpected)');
            } catch (e) {
                console.log('Sync write failed:', e.message);
            }
        """,
            language="javascript",
        )

        # Sync method should fail or produce error output
        assert (
            "failed" in result.lower()
            or "error" in result.lower()
            or "not a function" in result.lower()
            or "(no output)" in result
            or "undefined" in result.lower()
        ), f"Sync writeFileSync should fail, got: {result!r}"

    def test_async_methods_work(self) -> None:
        """Using async methods should work correctly."""
        from amla_sandbox import create_sandbox_tool

        sandbox = create_sandbox_tool()
        result = sandbox.run(
            """
            await fs.writeFile('/workspace/test.txt', 'hello async');
            const content = await fs.readFile('/workspace/test.txt');
            console.log('Content:', content);
        """,
            language="javascript",
        )

        assert "hello async" in result, f"Async methods should work: {result!r}"
