#!/usr/bin/env python3
"""Testing Sandbox Code Tutorial - Unit and Integration Testing

This tutorial covers:
  1. Unit testing tool handlers
  2. Testing capability enforcement
  3. Testing JavaScript execution
  4. Integration testing patterns
  5. Mocking and test fixtures
  6. Test organization best practices

Prerequisites:
    uv pip install "git+https://github.com/amlalabs/amla-sandbox[dev]"

Run:
    python testing_sandbox_code.py

Time: ~15 minutes
"""

from typing import Any

from amla_sandbox import create_sandbox_tool

# =============================================================================
# Part 1: Introduction to Testing
# =============================================================================


def part1_introduction() -> None:
    """Overview of testing strategies."""
    print("\n" + "=" * 60)
    print("Part 1: Introduction to Testing")
    print("=" * 60)

    print("""
WHAT TO TEST?

1. TOOL HANDLERS
   - Business logic works correctly
   - Error handling behaves properly
   - Edge cases are handled

2. CAPABILITIES
   - Allowed calls succeed
   - Denied calls fail appropriately
   - Constraints enforce limits

3. JAVASCRIPT EXECUTION
   - Tool calls work from JS
   - Data flows correctly
   - Errors propagate properly

4. INTEGRATION
   - End-to-end workflows
   - Multi-tool interactions
   - State persistence

TESTING PRINCIPLES:
   - Test at the right level (unit vs integration)
   - Keep tests fast and isolated
   - Use fixtures for common setup
   - Test both success and failure paths
    """)


# =============================================================================
# Part 2: Unit Testing Tool Handlers
# =============================================================================


def part2_unit_testing_handlers() -> None:
    """Unit test tool handlers in isolation."""
    print("\n" + "=" * 60)
    print("Part 2: Unit Testing Tool Handlers")
    print("=" * 60)

    # Example handler to test
    def calculate_discount(
        price: float, discount_percent: float, min_price: float = 0.0
    ) -> dict[str, Any]:
        """Calculate discounted price.

        Args:
            price: Original price
            discount_percent: Discount percentage (0-100)
            min_price: Minimum price floor
        """
        if price < 0:
            return {"error": "Price cannot be negative"}
        if not 0 <= discount_percent <= 100:
            return {"error": "Discount must be 0-100"}

        discounted = price * (1 - discount_percent / 100)
        final_price = max(discounted, min_price)

        return {
            "original_price": price,
            "discount_percent": discount_percent,
            "final_price": round(final_price, 2),
            "savings": round(price - final_price, 2),
        }

    # Unit tests for the handler
    print("Unit tests for calculate_discount handler:")

    def test_basic_discount() -> None:
        result = calculate_discount(100.0, 20.0)
        assert result["final_price"] == 80.0
        assert result["savings"] == 20.0
        print("  ✓ test_basic_discount")

    def test_zero_discount() -> None:
        result = calculate_discount(100.0, 0.0)
        assert result["final_price"] == 100.0
        assert result["savings"] == 0.0
        print("  ✓ test_zero_discount")

    def test_full_discount() -> None:
        result = calculate_discount(100.0, 100.0)
        assert result["final_price"] == 0.0
        assert result["savings"] == 100.0
        print("  ✓ test_full_discount")

    def test_min_price_floor() -> None:
        result = calculate_discount(100.0, 90.0, min_price=20.0)
        assert result["final_price"] == 20.0  # 10% would be 10, but min is 20
        print("  ✓ test_min_price_floor")

    def test_negative_price_error() -> None:
        result = calculate_discount(-50.0, 10.0)
        assert "error" in result
        print("  ✓ test_negative_price_error")

    def test_invalid_discount_error() -> None:
        result = calculate_discount(100.0, 150.0)
        assert "error" in result
        print("  ✓ test_invalid_discount_error")

    # Run all tests
    test_basic_discount()
    test_zero_discount()
    test_full_discount()
    test_min_price_floor()
    test_negative_price_error()
    test_invalid_discount_error()

    print("\nAll handler unit tests passed!")


# =============================================================================
# Part 3: Testing Capability Enforcement
# =============================================================================


def part3_capability_testing() -> None:
    """Test capability enforcement."""
    print("\n" + "=" * 60)
    print("Part 3: Testing Capability Enforcement")
    print("=" * 60)

    # Tool functions
    def account_read(account_id: str) -> dict[str, Any]:
        """Read account info."""
        return {"account_id": account_id, "status": "ok"}

    def account_write(account_id: str, amount: int) -> dict[str, Any]:
        """Write to account."""
        return {"account_id": account_id, "amount": amount, "status": "ok"}

    # Create with constraints and max_calls
    sandbox = create_sandbox_tool(
        tools=[account_read, account_write],
        constraints={"account_write": {"amount": "<=1000"}},
        max_calls={"account_write": 5},
    )

    print("Capability enforcement tests:")

    def test_read_allowed() -> None:
        result = sandbox.run(
            'await account_read({account_id: "a1"}); console.log("ok");',
            language="javascript",
        )
        assert "ok" in result
        print("  ✓ account_read is allowed")

    def test_write_within_limits() -> None:
        result = sandbox.run(
            'await account_write({account_id: "a1", amount: 500}); console.log("ok");',
            language="javascript",
        )
        assert "ok" in result
        print("  ✓ account_write with amount=500 is allowed")

    def test_write_exceeds_constraint() -> None:
        result = sandbox.run(
            """
            try {
                await account_write({account_id: "a1", amount: 2000});
                console.log("ALLOWED");
            } catch (e) {
                console.log("DENIED");
            }
        """,
            language="javascript",
        )
        assert "DENIED" in result
        print("  ✓ account_write with amount=2000 is denied (constraint)")

    def test_max_calls_enforcement() -> None:
        # Create a fresh sandbox to test max_calls
        sandbox2 = create_sandbox_tool(
            tools=[account_write],
            max_calls={"account_write": 3},
        )

        # Make calls up to limit
        for i in range(3):
            sandbox2.run(
                f'await account_write({{account_id: "a1", amount: {i * 100}}});',
                language="javascript",
            )

        # Next call should be denied
        result = sandbox2.run(
            """
            try {
                await account_write({account_id: "a1", amount: 100});
                console.log("ALLOWED");
            } catch (e) {
                console.log("DENIED");
            }
        """,
            language="javascript",
        )
        assert "DENIED" in result
        print("  ✓ account_write denied after max_calls exceeded")

    test_read_allowed()
    test_write_within_limits()
    test_write_exceeds_constraint()
    test_max_calls_enforcement()

    print("\nAll capability tests passed!")


# =============================================================================
# Part 4: Testing JavaScript Execution
# =============================================================================


def part4_js_execution_testing() -> None:
    """Test JavaScript execution and tool integration."""
    print("\n" + "=" * 60)
    print("Part 4: Testing JavaScript Execution")
    print("=" * 60)

    # Create tools with predictable responses
    def get_user(user_id: str) -> dict[str, Any]:
        users = {
            "u1": {"id": "u1", "name": "Alice", "balance": 100},
            "u2": {"id": "u2", "name": "Bob", "balance": 50},
        }
        return users.get(user_id, {"error": "User not found"})

    def transfer(from_id: str, to_id: str, amount: int) -> dict[str, Any]:
        if amount <= 0:
            return {"error": "Amount must be positive"}
        return {"from": from_id, "to": to_id, "amount": amount, "status": "completed"}

    sandbox = create_sandbox_tool(tools=[get_user, transfer])

    print("JavaScript execution tests:")

    def test_tool_call_from_js() -> None:
        result = sandbox.run(
            """
            const user = await get_user({user_id: "u1"});
            console.log(user.name);
        """,
            language="javascript",
        )
        assert "Alice" in result
        print("  ✓ Tool call from JS returns correct data")

    def test_error_handling_in_js() -> None:
        result = sandbox.run(
            """
            const user = await get_user({user_id: "invalid"});
            console.log(user.error ? "ERROR" : user.name);
        """,
            language="javascript",
        )
        assert "ERROR" in result
        print("  ✓ Error responses handled correctly in JS")

    def test_multi_step_workflow() -> None:
        result = sandbox.run(
            """
            const alice = await get_user({user_id: "u1"});
            const bob = await get_user({user_id: "u2"});

            if (alice.balance >= 25) {
                const transfer_result = await transfer({
                    from_id: alice.id,
                    to_id: bob.id,
                    amount: 25
                });
                console.log(transfer_result.status);
            }
        """,
            language="javascript",
        )
        assert "completed" in result
        print("  ✓ Multi-step workflow executes correctly")

    def test_vfs_persistence() -> None:
        # Write in first call (use /workspace which exists)
        sandbox.run(
            "await fs.writeFile('/workspace/data.txt', 'hello');",
            language="javascript",
        )

        # Read in second call
        result = sandbox.run(
            """
            const content = await fs.readFile('/workspace/data.txt');
            console.log(content);
        """,
            language="javascript",
        )
        assert "hello" in result
        print("  ✓ VFS persists between calls")

    test_tool_call_from_js()
    test_error_handling_in_js()
    test_multi_step_workflow()
    test_vfs_persistence()

    print("\nAll JS execution tests passed!")


# =============================================================================
# Part 5: Integration Testing
# =============================================================================


def part5_integration_testing() -> None:
    """End-to-end integration testing."""
    print("\n" + "=" * 60)
    print("Part 5: Integration Testing")
    print("=" * 60)

    # Simulate a complete e-commerce workflow
    orders_db: list[dict[str, Any]] = []
    inventory: dict[str, int] = {"widget": 10, "gadget": 5}

    def check_inventory(product: str) -> dict[str, Any]:
        stock = inventory.get(product, 0)
        return {"product": product, "in_stock": stock}

    def create_order(product: str, quantity: int) -> dict[str, Any]:
        stock = inventory.get(product, 0)
        if stock < quantity:
            return {"error": "Insufficient stock"}

        inventory[product] = stock - quantity
        order = {
            "order_id": f"ord_{len(orders_db) + 1}",
            "product": product,
            "quantity": quantity,
            "status": "created",
        }
        orders_db.append(order)
        return order

    def get_order(order_id: str) -> dict[str, Any]:
        for order in orders_db:
            if order["order_id"] == order_id:
                return order
        return {"error": "Order not found"}

    sandbox = create_sandbox_tool(
        tools=[check_inventory, create_order, get_order], max_calls=50
    )

    print("Integration test: E-commerce workflow")

    def test_complete_order_flow() -> None:
        # Check inventory
        result = sandbox.run(
            """
            const stock = await check_inventory({product: "widget"});
            console.log("Stock:", stock.in_stock);
        """,
            language="javascript",
        )
        assert "10" in result

        # Create order
        result = sandbox.run(
            """
            const order = await create_order({product: "widget", quantity: 3});
            console.log(order.order_id);
            await fs.writeFile('/workspace/last_order.txt', order.order_id);
        """,
            language="javascript",
        )
        assert "ord_1" in result

        # Verify inventory reduced
        result = sandbox.run(
            """
            const stock = await check_inventory({product: "widget"});
            console.log(stock.in_stock);
        """,
            language="javascript",
        )
        assert "7" in result  # 10 - 3 = 7

        # Retrieve order
        result = sandbox.run(
            """
            const orderId = await fs.readFile('/workspace/last_order.txt');
            const order = await get_order({order_id: orderId});
            console.log(order.status);
        """,
            language="javascript",
        )
        assert "created" in result

        print("  ✓ Complete order flow works correctly")

    def test_insufficient_stock() -> None:
        result = sandbox.run(
            """
            const order = await create_order({product: "gadget", quantity: 100});
            console.log(order.error || "success");
        """,
            language="javascript",
        )
        assert "Insufficient" in result
        print("  ✓ Insufficient stock returns error")

    test_complete_order_flow()
    test_insufficient_stock()

    print("\nAll integration tests passed!")


# =============================================================================
# Part 6: Test Fixtures and Helpers
# =============================================================================


def part6_fixtures() -> None:
    """Using fixtures and test helpers."""
    print("\n" + "=" * 60)
    print("Part 6: Test Fixtures and Helpers")
    print("=" * 60)

    print("""
TEST FIXTURE PATTERNS:

1. FACTORY FUNCTIONS
   Create pre-configured sandboxes for different test scenarios

2. MOCK TOOLS
   Create tools with predictable responses for testing

3. CALL TRACKING
   Wrap tools to track calls and responses

4. SNAPSHOT TESTING
   Compare outputs against known-good snapshots
    """)

    # Track calls for testing
    calls: list[tuple[str, dict[str, Any]]] = []
    responses: dict[str, dict[str, Any]] = {"test_method": {"value": 42}}

    def test_method(key: str) -> dict[str, Any]:
        """A test method."""
        call = ("test_method", {"key": key})
        calls.append(call)
        return responses.get("test_method", {"mock": True})

    # Factory function
    def create_test_bash(tools_list: list[Any] | None = None) -> Any:
        """Factory for creating test sandboxes."""
        return create_sandbox_tool(tools=tools_list or [test_method])

    print("Using test fixtures:")

    def test_with_factory() -> None:
        sandbox = create_test_bash()
        result = sandbox.run("console.log('test');", language="javascript")
        assert "test" in result
        print("  ✓ Factory creates working sandbox")

    def test_with_call_tracking() -> None:
        calls.clear()
        sandbox = create_test_bash()
        sandbox.run('await test_method({key: "value"});', language="javascript")

        assert len(calls) == 1
        assert calls[0][0] == "test_method"
        assert calls[0][1]["key"] == "value"
        print("  ✓ Call tracking works correctly")

    test_with_factory()
    test_with_call_tracking()

    print("\nFixture patterns demonstrated!")


# =============================================================================
# Part 7: Test Organization
# =============================================================================


def part7_organization() -> None:
    """Best practices for test organization."""
    print("\n" + "=" * 60)
    print("Part 7: Test Organization Best Practices")
    print("=" * 60)

    print("""
RECOMMENDED TEST STRUCTURE:

tests/
├── conftest.py           # Shared fixtures
├── unit/
│   ├── test_handlers.py  # Handler unit tests
│   └── test_utils.py     # Utility function tests
├── integration/
│   ├── test_workflows.py # End-to-end workflows
│   └── test_sandbox.py   # Sandbox integration
└── fixtures/
    ├── tools.py          # Tool fixtures
    └── data.py           # Test data

PYTEST EXAMPLE:

# conftest.py
import pytest
from amla_sandbox import create_sandbox_tool

@pytest.fixture
def bash():
    '''Fresh sandbox for each test.'''
    return create_sandbox_tool()

@pytest.fixture
def bash_with_tools():
    '''Sandbox with custom tools.'''
    def my_tool(param: str) -> dict:
        return {"param": param}
    return create_sandbox_tool(tools=[my_tool])

# test_handlers.py
def test_discount_handler(bash):
    result = sandbox.run('''
        const d = await calculate_discount({price: 100, percent: 20});
        console.log(d.final_price);
    ''')
    assert '80' in result

RUNNING TESTS:

  pytest tests/                    # All tests
  pytest tests/unit/               # Unit tests only
  pytest tests/ -k 'discount'      # Tests matching pattern
  pytest tests/ -v                 # Verbose output
  pytest tests/ --tb=short         # Short tracebacks
    """)


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Run all tutorial parts."""
    print("=" * 60)
    print("Testing Sandbox Code Tutorial")
    print("Unit and Integration Testing")
    print("=" * 60)

    part1_introduction()
    part2_unit_testing_handlers()
    part3_capability_testing()
    part4_js_execution_testing()
    part5_integration_testing()
    part6_fixtures()
    part7_organization()

    print("\n" + "=" * 60)
    print("Tutorial Complete!")
    print("=" * 60)
    print("""
Key takeaways:

1. UNIT TEST HANDLERS
   - Test handlers in isolation
   - Cover success and error paths
   - Test edge cases

2. TEST CAPABILITIES
   - Use can_call() to verify authorization
   - Test constraint enforcement
   - Verify max_calls limits

3. TEST JS EXECUTION
   - Verify tool calls from JavaScript
   - Test error propagation
   - Check VFS persistence

4. INTEGRATION TESTS
   - Test complete workflows
   - Verify state changes
   - Test multi-step operations

5. USE FIXTURES
   - Factory functions for sandboxes
   - Mock handlers for tracking
   - Shared setup in conftest.py

You now have a complete testing strategy for your sandbox-based agents!
    """)


if __name__ == "__main__":
    main()
