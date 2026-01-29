"""Tests for LangGraph integration module."""

from dataclasses import dataclass
from typing import Any, Optional

from amla_sandbox import (
    ExecutionResult,
    SandboxTool,
    capability_from_function,
    create_tool_handler,
    tool_from_function,
)
from amla_sandbox.capabilities import ConstraintSet, Param


# === Test fixtures: sample tools ===


def add(a: float, b: float) -> float:
    """Add two numbers together.

    Args:
        a: First number.
        b: Second number.
    """
    return a + b


def greet(name: str, excited: bool = False) -> str:
    """Greet someone by name.

    Args:
        name: The person's name.
        excited: Whether to use exclamation marks.
    """
    msg = f"Hello, {name}"
    return msg + "!!!" if excited else msg


def get_weather(city: str, units: str = "celsius") -> dict[str, str | float]:
    """Get current weather for a city.

    Args:
        city: The city name.
        units: Temperature units (celsius or fahrenheit).
    """
    return {"city": city, "temp": 22.5, "units": units, "condition": "sunny"}


def transfer_money(amount: float, to_account: str, memo: str = "") -> bool:
    """Transfer money to an account.

    Args:
        amount: Amount to transfer.
        to_account: Destination account ID.
        memo: Optional transfer memo.
    """
    return True


# === Tests for tool_from_function ===


class TestToolFromFunction:
    """Tests for converting Python functions to ToolDefinitions."""

    def test_basic_function(self) -> None:
        """Test converting a simple function."""
        tool = tool_from_function(add)

        assert tool.name == "add"
        assert "Add two numbers" in tool.description
        assert tool.parameters["type"] == "object"
        assert "a" in tool.parameters["properties"]
        assert "b" in tool.parameters["properties"]
        assert tool.parameters["properties"]["a"]["type"] == "number"
        assert set(tool.parameters["required"]) == {"a", "b"}

    def test_function_with_defaults(self) -> None:
        """Test function with optional parameters."""
        tool = tool_from_function(greet)

        assert tool.name == "greet"
        assert "name" in tool.parameters["properties"]
        assert "excited" in tool.parameters["properties"]
        assert tool.parameters["properties"]["name"]["type"] == "string"
        assert tool.parameters["properties"]["excited"]["type"] == "boolean"
        # Only name is required, excited has a default
        assert tool.parameters["required"] == ["name"]

    def test_extracts_param_descriptions(self) -> None:
        """Test that parameter descriptions are extracted from docstrings."""
        tool = tool_from_function(greet)

        assert "description" in tool.parameters["properties"]["name"]
        assert "person's name" in tool.parameters["properties"]["name"]["description"]

    def test_complex_return_type(self) -> None:
        """Test function with complex return type."""
        tool = tool_from_function(get_weather)

        assert tool.name == "get_weather"
        assert "city" in tool.parameters["properties"]
        assert "units" in tool.parameters["properties"]


# === Tests for capability_from_function ===


class TestCapabilityFromFunction:
    """Tests for generating capabilities from functions."""

    def test_basic_capability(self) -> None:
        """Test generating a simple capability."""
        cap = capability_from_function(add)

        assert cap.method_pattern == "mcp:add"
        assert cap.constraints.is_empty()
        assert cap.max_calls is None

    def test_with_constraints(self) -> None:
        """Test capability with parameter constraints."""
        cap = capability_from_function(
            transfer_money,
            constraints=ConstraintSet([Param("amount") <= 1000]),
            max_calls=10,
        )

        assert cap.method_pattern == "mcp:transfer_money"
        assert not cap.constraints.is_empty()
        assert cap.max_calls == 10

        # Test constraint evaluation
        cap.validate_call("mcp:transfer_money", {"amount": 500, "to_account": "acc123"})

    def test_capability_matches_function_name(self) -> None:
        """Test that capability pattern matches function name with mcp: prefix."""
        cap = capability_from_function(get_weather)
        cap.validate_call("mcp:get_weather", {"city": "SF"})


# === Tests for create_tool_handler ===


class TestCreateToolHandler:
    """Tests for creating tool handlers from function lists."""

    def test_routes_to_correct_function(self) -> None:
        """Test that handler routes to correct function."""
        handler = create_tool_handler([add, greet])

        result = handler("add", {"a": 1, "b": 2})
        assert result == 3

        result = handler("greet", {"name": "Alice"})
        assert result == "Hello, Alice"

    def test_raises_on_unknown_tool(self) -> None:
        """Test that unknown tools raise ValueError."""
        handler = create_tool_handler([add])

        try:
            handler("unknown", {})
            assert False, "Should have raised"
        except ValueError as e:
            assert "Unknown tool" in str(e)

    def test_passes_all_parameters(self) -> None:
        """Test that all parameters are passed through."""
        handler = create_tool_handler([greet])

        result = handler("greet", {"name": "Bob", "excited": True})
        assert result == "Hello, Bob!!!"


# === Tests for ExecutionResult ===


class TestExecutionResult:
    """Tests for ExecutionResult dataclass."""

    def test_success_result(self) -> None:
        """Test successful execution result."""
        result = ExecutionResult(stdout="Hello, World!", success=True)

        assert result.success
        assert str(result) == "Hello, World!"
        assert result.to_tool_message() == "Hello, World!"

    def test_error_result(self) -> None:
        """Test error execution result."""
        result = ExecutionResult(
            error="Something went wrong",
            stderr="Debug info here",
            success=False,
        )

        assert not result.success
        assert "Error:" in str(result)
        assert "[error]:" in result.to_tool_message()
        assert "[stderr]:" in result.to_tool_message()

    def test_empty_result(self) -> None:
        """Test empty execution result."""
        result = ExecutionResult()

        assert result.success
        assert result.to_tool_message() == "(no output)"


# === Tests for SandboxTool ===


class TestSandboxTool:
    """Tests for the SandboxTool class."""

    def test_from_functions_creates_tool(self) -> None:
        """Test creating SandboxTool from functions."""
        tool = SandboxTool.from_functions([add, greet])

        assert len(tool.tools) == 2
        assert tool.sandbox is not None

    def test_get_tool_descriptions(self) -> None:
        """Test getting formatted tool descriptions."""
        tool = SandboxTool.from_functions([add, greet])
        desc = tool.get_tool_descriptions()

        assert "add" in desc
        assert "greet" in desc
        assert "Add two numbers" in desc
        assert "Greet someone" in desc

    def test_with_custom_capabilities(self) -> None:
        """Test creating SandboxTool with explicit capabilities."""
        cap = capability_from_function(
            transfer_money,
            constraints=ConstraintSet([Param("amount") <= 100]),
        )

        tool = SandboxTool.from_functions(
            [transfer_money],
            capabilities=[cap],
        )

        # Should have the custom capability (with mcp: prefix from capability_from_function)
        assert len(tool.sandbox.capabilities) == 1
        assert tool.sandbox.capabilities[0].method_pattern == "mcp:transfer_money"


# === Tests for tool_handler integration ===


class TestToolHandlerIntegration:
    """Integration tests for tool handler with sandbox."""

    def test_sandbox_uses_handler(self) -> None:
        """Test that sandbox properly uses tool handler."""
        tool = SandboxTool.from_functions([add, greet])

        # The tool handler should be wired up
        assert tool.sandbox.tool_handler is not None

        # Calling through handler should work
        result = tool.sandbox.tool_handler("add", {"a": 5, "b": 3})
        assert result == 8


# === Tests for Pydantic/Dataclass support ===


@dataclass
class OrderItem:
    """A single item in an order."""

    sku: str
    quantity: int
    price: float = 0.0


@dataclass
class Order:
    """An order with multiple items."""

    order_id: str
    items: list[OrderItem]
    customer_id: Optional[str] = None


class TestComplexTypeSupport:
    """Tests for Pydantic models and dataclasses in tool_from_function."""

    def test_dataclass_parameter(self) -> None:
        """Test function with dataclass parameter generates correct schema."""

        def create_order(item: OrderItem) -> str:
            """Create an order from an item."""
            return f"Order created: {item.sku}"

        tool = tool_from_function(create_order)

        assert tool.name == "create_order"
        # The parameter should have full object schema, not just "string"
        item_schema = tool.parameters["properties"]["item"]
        assert item_schema["type"] == "object"
        assert "properties" in item_schema
        assert "sku" in item_schema["properties"]
        assert "quantity" in item_schema["properties"]
        assert item_schema["properties"]["sku"]["type"] == "string"
        assert item_schema["properties"]["quantity"]["type"] == "integer"

    def test_optional_type(self) -> None:
        """Test function with Optional parameter."""

        def find_user(user_id: str, email: Optional[str] = None) -> dict[str, Any]:
            """Find a user by ID or email."""
            return {"id": user_id, "email": email}

        tool = tool_from_function(find_user)

        # user_id should be required
        assert "user_id" in tool.parameters["required"]
        # email should be optional and nullable
        email_schema = tool.parameters["properties"]["email"]
        # Optional adds nullable: true
        assert email_schema.get("nullable") is True

    def test_list_of_basic_type(self) -> None:
        """Test function with List[str] parameter."""

        def process_tags(tags: list[str]) -> str:
            """Process a list of tags."""
            return ", ".join(tags)

        tool = tool_from_function(process_tags)

        tags_schema = tool.parameters["properties"]["tags"]
        assert tags_schema["type"] == "array"
        assert tags_schema["items"]["type"] == "string"

    def test_pydantic_model_if_available(self) -> None:
        """Test function with Pydantic model (skipped if Pydantic not installed)."""
        try:
            from pydantic import BaseModel

            class UserModel(BaseModel):
                name: str
                age: int

            def create_user(user: UserModel) -> str:
                """Create a user."""
                return f"Created {user.name}"

            tool = tool_from_function(create_user)

            user_schema = tool.parameters["properties"]["user"]
            # Pydantic generates full JSON Schema
            assert user_schema["type"] == "object"
            assert "properties" in user_schema
            assert "name" in user_schema["properties"]
            assert "age" in user_schema["properties"]

        except ImportError:
            # Pydantic not installed, skip test
            pass
