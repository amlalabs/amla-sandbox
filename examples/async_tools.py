#!/usr/bin/env python3
"""Async Tool Handlers Tutorial

This tutorial covers:
  1. Adding tools to the sandbox
  2. Sync vs async tool handlers
  3. Calling external APIs from tools
  4. Error handling in tool handlers
  5. Tool call patterns and best practices

Prerequisites:
    pip install amla-sandbox

Run:
    python async_tool_handlers.py

Time: ~15 minutes
"""

import asyncio
import time
from typing import Any

from amla_sandbox import (
    Sandbox,
    MethodCapability,
    ToolDefinition,
    create_sandbox_tool,
    tool_from_function,
)

# =============================================================================
# Part 1: Basic Tool Creation
# =============================================================================


def part1_basic_tools() -> None:
    """Create tools from Python functions."""
    print("\n" + "=" * 60)
    print("Part 1: Basic Tool Creation")
    print("=" * 60)

    # The simplest way: pass functions directly to create_sandbox_tool
    def get_current_time() -> dict[str, str]:
        """Get the current time."""
        return {"time": time.strftime("%Y-%m-%d %H:%M:%S")}

    def add_numbers(a: float, b: float) -> dict[str, float]:
        """Add two numbers together.

        Args:
            a: First number
            b: Second number
        """
        return {"a": a, "b": b, "sum": a + b}

    def greet(name: str, greeting: str = "Hello") -> dict[str, str]:
        """Generate a greeting.

        Args:
            name: The person to greet
            greeting: The greeting to use (default: Hello)
        """
        return {"message": f"{greeting}, {name}!"}

    # Create bash tool with these functions
    sandbox = create_sandbox_tool(tools=[get_current_time, add_numbers, greet])

    # Call tools from JavaScript
    print("\nCalling tools from JavaScript:")

    output = sandbox.run(
        """
        const time = await get_current_time({});
        console.log("Current time:", time.time);
    """,
        language="javascript",
    )
    print(f"  {output}")

    output = sandbox.run(
        """
        const result = await add_numbers({a: 10, b: 32});
        console.log(`${result.a} + ${result.b} = ${result.sum}`);
    """,
        language="javascript",
    )
    print(f"  {output}")

    output = sandbox.run(
        """
        const greeting = await greet({name: "World", greeting: "Howdy"});
        console.log(greeting.message);
    """,
        language="javascript",
    )
    print(f"  {output}")

    # Tools can also be called via shell syntax
    print("\nCalling tools via shell syntax:")
    output = sandbox.run("tool greet --name Alice", language="shell")
    print(f"  {output}")


# =============================================================================
# Part 2: Tool Definition with JSON Schema
# =============================================================================


def part2_tool_definitions() -> None:
    """Create tools with explicit JSON Schema definitions."""
    print("\n" + "=" * 60)
    print("Part 2: Tool Definition with JSON Schema")
    print("=" * 60)

    # For more control, define tools with explicit schemas
    tools = [
        ToolDefinition(
            name="weather.get",
            description="Get weather for a city",
            parameters={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "units": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "default": "celsius",
                    },
                },
                "required": ["city"],
            },
        ),
        ToolDefinition(
            name="weather.forecast",
            description="Get weather forecast",
            parameters={
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "days": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 7,
                        "default": 3,
                    },
                },
                "required": ["city"],
            },
        ),
    ]

    # Tool handler function receives (method, params)
    def handle_tool(method: str, params: dict[str, Any]) -> dict[str, Any]:
        print(f"    [Handler] {method}({params})")

        if method == "weather.get":
            return {
                "city": params["city"],
                "temperature": 22,
                "units": params.get("units", "celsius"),
                "condition": "sunny",
            }

        if method == "weather.forecast":
            days = params.get("days", 3)
            return {
                "city": params["city"],
                "forecast": [
                    {"day": i + 1, "high": 20 + i, "low": 15 + i} for i in range(days)
                ],
            }

        raise ValueError(f"Unknown method: {method}")

    # Create sandbox with explicit tools and handler
    sandbox = Sandbox(
        tools=tools,
        capabilities=[MethodCapability(method_pattern="**")],  # Allow all
        tool_handler=handle_tool,
    )

    print("\nExplicit tool definitions:")

    # Tool names with dots become underscored in JS
    result = sandbox.execute("""
        const weather = await weather_get({city: "London", units: "celsius"});
        console.log("Weather:", JSON.stringify(weather));
    """)
    print(f"  {result}")

    result = sandbox.execute("""
        const forecast = await weather_forecast({city: "Paris", days: 5});
        console.log("Forecast:", JSON.stringify(forecast));
    """)
    print(f"  {result}")


# =============================================================================
# Part 3: Async Tool Handlers
# =============================================================================


def part3_async_handlers() -> None:
    """Use async tool handlers for I/O operations."""
    print("\n" + "=" * 60)
    print("Part 3: Async Tool Handlers")
    print("=" * 60)

    print("""
When your tools need to perform I/O operations (API calls, database queries),
you should use async handlers. This allows the Python runtime to handle
multiple concurrent operations efficiently.
    """)

    async def fetch_user(user_id: str) -> dict[str, Any]:
        """Simulate an async database query."""
        await asyncio.sleep(0.1)  # Simulate I/O latency
        return {
            "id": user_id,
            "name": f"User_{user_id}",
            "email": f"user{user_id}@example.com",
        }

    async def call_api(endpoint: str) -> dict[str, Any]:
        """Simulate an async API call."""
        await asyncio.sleep(0.1)  # Simulate network latency
        return {
            "endpoint": endpoint,
            "data": {"status": "success", "message": f"Response from {endpoint}"},
        }

    # Async tool handler
    async def async_handler(method: str, params: dict[str, Any]) -> dict[str, Any]:
        print(f"    [Async Handler] {method}({params})")

        if method == "fetch_user":
            return await fetch_user(params["user_id"])

        if method == "call_api":
            return await call_api(params["endpoint"])

        raise ValueError(f"Unknown method: {method}")

    tools = [
        ToolDefinition(
            name="fetch_user",
            description="Fetch a user from the database",
            parameters={
                "type": "object",
                "properties": {"user_id": {"type": "string"}},
                "required": ["user_id"],
            },
        ),
        ToolDefinition(
            name="call_api",
            description="Call an external API",
            parameters={
                "type": "object",
                "properties": {"endpoint": {"type": "string"}},
                "required": ["endpoint"],
            },
        ),
    ]

    # For async handlers, we need to run in an async context
    async def run_async_demo():
        sandbox = Sandbox(
            tools=tools,
            capabilities=[MethodCapability(method_pattern="**")],
            tool_handler=async_handler,  # Pass async function
        )

        print("\nAsync tool calls:")

        # Await the execution when using async handler
        result = await sandbox.execute_async("""
            const user = await fetch_user({user_id: "123"});
            console.log("User:", JSON.stringify(user));
        """)
        print(f"  {result}")

        result = await sandbox.execute_async("""
            const response = await call_api({endpoint: "/api/v1/status"});
            console.log("API response:", JSON.stringify(response));
        """)
        print(f"  {result}")

        # Multiple concurrent tool calls
        result = await sandbox.execute_async("""
            // These will be handled concurrently by the async handler
            const [user1, user2, api] = await Promise.all([
                fetch_user({user_id: "100"}),
                fetch_user({user_id: "101"}),
                call_api({endpoint: "/api/batch"})
            ]);
            console.log("Got users:", user1.name, "and", user2.name);
            console.log("API:", api.data.status);
        """)
        print(f"  {result}")

    asyncio.run(run_async_demo())


# =============================================================================
# Part 4: Error Handling in Tool Handlers
# =============================================================================


def part4_error_handling() -> None:
    """Handle errors gracefully in tool handlers."""
    print("\n" + "=" * 60)
    print("Part 4: Error Handling in Tool Handlers")
    print("=" * 60)

    def risky_handler(method: str, params: dict[str, Any]) -> dict[str, Any]:
        """A handler that can fail."""
        if method == "divide":
            a, b = params["a"], params["b"]
            if b == 0:
                raise ValueError("Cannot divide by zero")
            return {"result": a / b}

        if method == "fetch_data":
            source = params["source"]
            if source == "bad_source":
                raise ConnectionError(f"Failed to connect to {source}")
            return {"data": f"Data from {source}"}

        if method == "validate":
            value = params["value"]
            if value < 0:
                raise ValueError("Value must be non-negative")
            return {"valid": True, "value": value}

        return {"error": f"Unknown method: {method}"}

    tools = [
        ToolDefinition(
            name="divide",
            description="Divide two numbers",
            parameters={
                "type": "object",
                "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                "required": ["a", "b"],
            },
        ),
        ToolDefinition(
            name="fetch_data",
            description="Fetch data from a source",
            parameters={
                "type": "object",
                "properties": {"source": {"type": "string"}},
                "required": ["source"],
            },
        ),
        ToolDefinition(
            name="validate",
            description="Validate a value",
            parameters={
                "type": "object",
                "properties": {"value": {"type": "number"}},
                "required": ["value"],
            },
        ),
    ]

    sandbox = Sandbox(
        tools=tools,
        capabilities=[MethodCapability(method_pattern="**")],
        tool_handler=risky_handler,
    )

    print("\nHandling errors in JavaScript:")

    # Use try/catch in JavaScript to handle errors gracefully
    result = sandbox.execute("""
        try {
            const result = await divide({a: 10, b: 0});
            console.log("Result:", result);
        } catch (error) {
            console.log("Caught error:", error.message);
        }
    """)
    print(f"  {result}")

    # Successful call for comparison
    result = sandbox.execute("""
        const result = await divide({a: 10, b: 2});
        console.log("10 / 2 =", result.result);
    """)
    print(f"  {result}")

    # Pattern: Return error status instead of throwing
    print("\nPattern: Graceful error handling with fallback:")
    result = sandbox.execute("""
        async function safeFetch(source) {
            try {
                const data = await fetch_data({source});
                return {success: true, data: data.data};
            } catch (error) {
                return {success: false, error: error.message};
            }
        }

        const result1 = await safeFetch("good_source");
        console.log("Good source:", result1);

        const result2 = await safeFetch("bad_source");
        console.log("Bad source:", result2);
    """)
    print(f"  {result}")


# =============================================================================
# Part 5: Tool Call Patterns
# =============================================================================


def part5_patterns() -> None:
    """Common patterns for tool usage."""
    print("\n" + "=" * 60)
    print("Part 5: Tool Call Patterns")
    print("=" * 60)

    # Set up tools for demo
    database = {
        "users": [
            {"id": "1", "name": "Alice", "email": "alice@example.com"},
            {"id": "2", "name": "Bob", "email": "bob@example.com"},
            {"id": "3", "name": "Charlie", "email": "charlie@example.com"},
        ],
        "orders": [
            {"id": "o1", "user_id": "1", "amount": 100},
            {"id": "o2", "user_id": "1", "amount": 200},
            {"id": "o3", "user_id": "2", "amount": 150},
        ],
    }

    def db_handler(method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "db.query":
            table = params["table"]
            filters = params.get("filters", {})
            data = database.get(table, [])
            for key, value in filters.items():
                data = [row for row in data if row.get(key) == value]
            return {"rows": data}

        if method == "db.count":
            table = params["table"]
            return {"count": len(database.get(table, []))}

        return {"error": f"Unknown: {method}"}

    tools = [
        ToolDefinition(
            name="db.query",
            description="Query a database table",
            parameters={
                "type": "object",
                "properties": {
                    "table": {"type": "string"},
                    "filters": {"type": "object"},
                },
                "required": ["table"],
            },
        ),
        ToolDefinition(
            name="db.count",
            description="Count rows in a table",
            parameters={
                "type": "object",
                "properties": {"table": {"type": "string"}},
                "required": ["table"],
            },
        ),
    ]

    sandbox = Sandbox(
        tools=tools,
        capabilities=[MethodCapability(method_pattern="**")],
        tool_handler=db_handler,
    )

    # Pattern 1: Sequential queries
    print("\nPattern 1: Sequential Queries")
    result = sandbox.execute("""
        // Get a user, then get their orders
        const users = await db_query({table: "users", filters: {name: "Alice"}});
        const user = users.rows[0];
        console.log("Found user:", user.name);

        const orders = await db_query({table: "orders", filters: {user_id: user.id}});
        console.log("Orders:", orders.rows.length);
    """)
    print(f"  {result}")

    # Pattern 2: Parallel queries
    print("\nPattern 2: Parallel Queries")
    result = sandbox.execute("""
        // Fetch multiple things in parallel
        const [userCount, orderCount] = await Promise.all([
            db_count({table: "users"}),
            db_count({table: "orders"})
        ]);
        console.log("Users:", userCount.count, "Orders:", orderCount.count);
    """)
    print(f"  {result}")

    # Pattern 3: Aggregation in JS
    print("\nPattern 3: Aggregation")
    result = sandbox.execute("""
        // Fetch all data, aggregate in JavaScript
        const users = (await db_query({table: "users"})).rows;
        const orders = (await db_query({table: "orders"})).rows;

        // Join and aggregate
        const userTotals = users.map(user => {
            const userOrders = orders.filter(o => o.user_id === user.id);
            const total = userOrders.reduce((sum, o) => sum + o.amount, 0);
            return {name: user.name, orderCount: userOrders.length, total};
        });

        console.log("User totals:");
        userTotals.forEach(u => console.log(`  ${u.name}: ${u.orderCount} orders, $${u.total}`));
    """)
    print(f"  {result}")

    # Pattern 4: Results to VFS for later
    print("\nPattern 4: Cache Results in VFS")
    sandbox.execute("""
        const allUsers = (await db_query({table: "users"})).rows;
        await fs.writeFile('/cache/users.json', JSON.stringify(allUsers));
        console.log("Cached", allUsers.length, "users to VFS");
    """)

    result = sandbox.execute("""
        // Later: read from cache instead of querying again
        const cachedUsers = JSON.parse(await fs.readFile('/cache/users.json'));
        console.log("Loaded from cache:", cachedUsers.map(u => u.name).join(", "));
    """)
    print(f"  {result}")


# =============================================================================
# Part 6: Using tool_from_function
# =============================================================================


def part6_tool_from_function() -> None:
    """Use the tool_from_function helper for cleaner code."""
    print("\n" + "=" * 60)
    print("Part 6: Using tool_from_function")
    print("=" * 60)

    # Define typed functions - docstrings become tool descriptions!
    def search_products(
        query: str, category: str = "all", max_results: int = 10
    ) -> dict[str, Any]:
        """Search for products in the catalog.

        Args:
            query: Search terms
            category: Product category to filter by
            max_results: Maximum number of results to return
        """
        # Simulate search
        results = [
            {"id": i, "name": f"Product matching '{query}'", "category": category}
            for i in range(min(3, max_results))
        ]
        return {"query": query, "results": results}

    def get_product_details(product_id: int) -> dict[str, Any]:
        """Get detailed information about a product.

        Args:
            product_id: The unique product identifier
        """
        return {
            "id": product_id,
            "name": f"Product {product_id}",
            "price": 29.99,
            "stock": 42,
            "description": "A great product",
        }

    # Convert functions to ToolDefinitions
    # This automatically extracts name, description, and parameter schema!
    search_tool = tool_from_function(search_products)
    details_tool = tool_from_function(get_product_details)

    print("Auto-generated tool definitions:")
    print(f"  {search_tool.name}: {search_tool.description}")
    print(f"  {details_tool.name}: {details_tool.description}")

    # Use with create_sandbox_tool for the simplest API
    sandbox = create_sandbox_tool(tools=[search_products, get_product_details])

    print("\nUsing auto-generated tools:")
    result = sandbox.run(
        """
        const results = await search_products({query: "laptop", category: "electronics"});
        console.log("Found", results.results.length, "products");

        const details = await get_product_details({product_id: 42});
        console.log("Product:", details.name, "- $" + details.price);
    """,
        language="javascript",
    )
    print(f"  {result}")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Run all tutorial parts."""
    print("=" * 60)
    print("Async Tool Handlers Tutorial")
    print("=" * 60)

    part1_basic_tools()
    part2_tool_definitions()
    part3_async_handlers()
    part4_error_handling()
    part5_patterns()
    part6_tool_from_function()

    print("\n" + "=" * 60)
    print("Tutorial Complete!")
    print("=" * 60)
    print("""
Key takeaways:

1. create_sandbox_tool(tools=[...]) is the easiest way
2. Python functions become JavaScript-callable tools
3. Docstrings become tool descriptions
4. Use async handlers for I/O operations
5. Handle errors gracefully with try/catch
6. Combine sequential and parallel tool calls

Tool handler signature:
  def handler(method: str, params: dict) -> Any

Async handler signature:
  async def handler(method: str, params: dict) -> Any

Tool naming convention:
  Python: my_tool or my.tool
  JavaScript: my_tool()

Next tutorial: 05_multi_agent_delegation.py
    Learn capability attenuation and agent-to-agent delegation.
    """)


if __name__ == "__main__":
    main()
