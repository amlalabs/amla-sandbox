"""Tests for the bash_tool module - AI SDK-style ergonomics."""

# pyright: reportPrivateUsage=warning

from amla_sandbox import create_bash_tool
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


# === Tests for create_bash_tool ===


class TestCreateBashTool:
    """Tests for the create_bash_tool function."""

    def test_no_tools(self) -> None:
        """Test creating bash tool with no tools (shell only)."""
        bash = create_bash_tool()

        assert bash.sandbox is not None
        assert len(bash.tools) == 0

    def test_with_simple_tools(self) -> None:
        """Test creating bash tool with Python functions."""
        bash = create_bash_tool(tools=[add, greet])

        assert len(bash.tools) == 2
        assert bash.sandbox is not None
        # Check that tools are registered
        assert "add" in bash._tool_map
        assert "greet" in bash._tool_map

    def test_with_constraints(self) -> None:
        """Test creating bash tool with constraints."""
        bash = create_bash_tool(
            tools=[transfer_money],
            constraints={"transfer_money": {"amount": "<=1000"}},
        )

        assert len(bash.tools) == 1
        # The capability should have constraints
        caps = bash.sandbox.capabilities
        assert len(caps) == 1
        assert not caps[0].constraints.is_empty()

    def test_with_max_calls_int(self) -> None:
        """Test creating bash tool with global max_calls."""
        bash = create_bash_tool(
            tools=[add, greet],
            max_calls=50,
        )

        caps = bash.sandbox.capabilities
        assert all(c.max_calls == 50 for c in caps)

    def test_with_max_calls_dict(self) -> None:
        """Test creating bash tool with per-tool max_calls."""
        bash = create_bash_tool(
            tools=[add, greet],
            max_calls={"add": 10, "greet": 20},
        )

        caps = bash.sandbox.capabilities
        cap_by_name = {c.method_pattern.split(":")[-1]: c for c in caps}
        assert cap_by_name["add"].max_calls == 10
        assert cap_by_name["greet"].max_calls == 20

    def test_default_max_calls(self) -> None:
        """Test that default max_calls is applied."""
        bash = create_bash_tool(tools=[add])

        caps = bash.sandbox.capabilities
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
        bash = create_bash_tool(
            tools=[transfer_money],
            constraints={
                "transfer_money": {
                    "amount": "<=10000",
                    "to_account": "startswith:acct_",
                }
            },
        )

        caps = bash.sandbox.capabilities
        assert len(caps) == 1
        # Should have both constraints
        assert not caps[0].constraints.is_empty()

    def test_constraints_and_max_calls(self) -> None:
        """Test tool with both constraints and max_calls."""
        bash = create_bash_tool(
            tools=[transfer_money],
            constraints={"transfer_money": {"amount": "<=1000"}},
            max_calls={"transfer_money": 5},
        )

        caps = bash.sandbox.capabilities
        assert caps[0].max_calls == 5
        assert not caps[0].constraints.is_empty()
