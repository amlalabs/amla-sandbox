#!/usr/bin/env python3
"""LangGraph Integration Tutorial - Building AI Agents

This tutorial covers:
  1. Setting up LangGraph with amla-sandbox
  2. Using create_react_agent for simple agents
  3. Building custom agent graphs
  4. Tool execution patterns
  5. State management across agent turns
  6. Production deployment considerations

Prerequisites:
    pip install "amla-sandbox[langgraph]"
    pip install langchain-anthropic  # or langchain-openai

Run:
    python langgraph_integration.py

Time: ~25 minutes
"""

import os
from typing import Any

from amla_sandbox import (
    create_sandbox_tool,
    Sandbox,
    MethodCapability,
    ConstraintSet,
    Param,
    ToolDefinition,
)

# =============================================================================
# Part 1: Introduction to LangGraph + amla-sandbox
# =============================================================================


def part1_introduction() -> None:
    """Overview of LangGraph integration."""
    print("\n" + "=" * 60)
    print("Part 1: Introduction to LangGraph + amla-sandbox")
    print("=" * 60)

    print("""
WHY LANGGRAPH + Amla Sandbox?

LangGraph: Orchestration framework for LLM agents
  - Defines agent behavior as a graph
  - Manages state between agent turns
  - Provides pre-built patterns (ReAct, etc.)

amla-sandbox: Secure code execution
  - WASM-based sandboxing
  - Capability-controlled tool access
  - Code mode efficiency (fewer LLM calls)

Together, they provide:
  1. ORCHESTRATION - LangGraph manages the agent loop
  2. SECURITY - amla-sandbox enforces capability boundaries
  3. EFFICIENCY - Code mode reduces token usage by 98%*

*Source: https://www.anthropic.com/engineering/code-execution-with-mcp

Integration patterns:
  a) Simple: create_sandbox_tool() + as_langchain_tool()
  b) Advanced: SandboxTool.from_functions(...) with constraints
  c) CodeAct: Agent writes code to accomplish tasks
    """)


# =============================================================================
# Part 2: Simple Integration with create_sandbox_tool
# =============================================================================


def part2_simple_integration() -> None:
    """The simplest way to use LangGraph with amla-sandbox."""
    print("\n" + "=" * 60)
    print("Part 2: Simple Integration with create_sandbox_tool")
    print("=" * 60)

    # Define some tools
    def get_weather(city: str) -> dict[str, Any]:
        """Get the current weather for a city.

        Args:
            city: The city name (e.g., "San Francisco", "Tokyo")
        """
        weather_data = {
            "San Francisco": {"temp": 18, "condition": "foggy"},
            "Tokyo": {"temp": 25, "condition": "sunny"},
            "London": {"temp": 12, "condition": "rainy"},
            "New York": {"temp": 22, "condition": "cloudy"},
        }
        data = weather_data.get(city, {"temp": 20, "condition": "unknown"})
        return {"city": city, **data}

    def search_products(query: str, max_results: int = 5) -> dict[str, Any]:
        """Search for products in the catalog.

        Args:
            query: Search terms
            max_results: Maximum number of results to return
        """
        products = [
            {"id": "p1", "name": f"Widget matching '{query}'", "price": 29.99},
            {"id": "p2", "name": f"Gadget for '{query}'", "price": 49.99},
            {"id": "p3", "name": f"Premium '{query}' item", "price": 99.99},
        ]
        return {"query": query, "results": products[:max_results]}

    # Create bash tool with Python functions
    sandbox = create_sandbox_tool(
        tools=[get_weather, search_products],
        max_calls=50,  # Rate limit
    )

    # Get the LangChain-compatible tool
    langchain_tool = sandbox.as_langchain_tool()

    print("Tool created and ready for LangGraph:")
    print(f"  Name: {langchain_tool.name}")
    print(f"  Description: {langchain_tool.description[:60]}...")

    # Test the tool directly
    print("\nTesting tool execution:")
    result = sandbox.run(
        """
        const weather = await get_weather({city: "Tokyo"});
        console.log("Weather in Tokyo:", JSON.stringify(weather));

        const products = await search_products({query: "laptop", max_results: 2});
        console.log("Found products:", products.results.length);
    """,
        language="javascript",
    )
    print(f"  {result}")

    # Show how it would be used with LangGraph
    print("\n" + "-" * 40)
    print("Usage with LangGraph (code example):")
    print("""
    from langchain_anthropic import ChatAnthropic
    from langgraph.prebuilt import create_react_agent

    model = ChatAnthropic(model="claude-3-5-sonnet-20241022")
    agent = create_react_agent(model, [sandbox.as_langchain_tool()])

    result = agent.invoke({
        "messages": [("user", "What's the weather in Tokyo?")]
    })
    """)


# =============================================================================
# Part 3: Using create_react_agent
# =============================================================================


def part3_react_agent() -> None:
    """Build a ReAct agent with LangGraph."""
    print("\n" + "=" * 60)
    print("Part 3: Using create_react_agent")
    print("=" * 60)

    # Check for API key
    has_api_key = bool(
        os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")
    )

    # Define tools for a customer service agent
    def lookup_order(order_id: str) -> dict[str, Any]:
        """Look up order details by order ID.

        Args:
            order_id: The order identifier (e.g., "ORD-12345")
        """
        orders = {
            "ORD-12345": {
                "status": "shipped",
                "items": ["Widget x2"],
                "tracking": "1Z999AA10123456784",
            },
            "ORD-67890": {
                "status": "processing",
                "items": ["Gadget x1", "Cable x3"],
                "tracking": None,
            },
        }
        order = orders.get(order_id)
        if order:
            return {"order_id": order_id, **order}
        return {"error": f"Order {order_id} not found"}

    def check_inventory(product_id: str) -> dict[str, Any]:
        """Check inventory levels for a product.

        Args:
            product_id: The product identifier
        """
        inventory = {
            "WIDGET-001": {"in_stock": 150, "reorder_point": 50},
            "GADGET-002": {"in_stock": 3, "reorder_point": 20},
        }
        return inventory.get(product_id, {"error": f"Product {product_id} not found"})

    def create_support_ticket(
        customer_id: str, issue: str, priority: str = "normal"
    ) -> dict[str, Any]:
        """Create a customer support ticket.

        Args:
            customer_id: The customer identifier
            issue: Description of the issue
            priority: Priority level (low, normal, high, urgent)
        """
        return {
            "ticket_id": f"TKT-{hash(issue) % 10000:04d}",
            "customer_id": customer_id,
            "issue": issue,
            "priority": priority,
            "status": "open",
        }

    # Create bash tool with constraints
    sandbox = create_sandbox_tool(
        tools=[lookup_order, check_inventory, create_support_ticket],
        constraints={
            "create_support_ticket": {
                "priority": ["low", "normal", "high"],  # No "urgent" without approval
            }
        },
        max_calls=20,
    )

    print("Customer Service Agent Configuration:")
    print("-" * 40)
    print("Tools available:")
    print("  - lookup_order: Find order status and tracking")
    print("  - check_inventory: Check product stock levels")
    print("  - create_support_ticket: Open support tickets")
    print("\nConstraints:")
    print("  - create_support_ticket.priority: [low, normal, high]")
    print("  - max_calls: 20 per session")

    # Test the tools
    print("\nTesting tool execution:")
    result = sandbox.run(
        """
        // Look up an order
        const order = await lookup_order({order_id: "ORD-12345"});
        console.log("Order status:", order.status);
        console.log("Tracking:", order.tracking);

        // Check inventory
        const inv = await check_inventory({product_id: "GADGET-002"});
        console.log("\\nGadget stock:", inv.in_stock);
        if (inv.in_stock < inv.reorder_point) {
            console.log("WARNING: Below reorder point!");
        }

        // Create a ticket
        const ticket = await create_support_ticket({
            customer_id: "CUST-789",
            issue: "Order delayed",
            priority: "high"
        });
        console.log("\\nTicket created:", ticket.ticket_id);
    """,
        language="javascript",
    )
    print(result)

    if has_api_key:
        print("\n" + "-" * 40)
        print("Running with real LLM...")
        try:
            from langchain_anthropic import ChatAnthropic  # type: ignore[import-untyped]
            from langgraph.prebuilt import create_react_agent  # type: ignore[import-untyped]

            model = ChatAnthropic(
                model=os.environ.get(  # type: ignore[call-arg]
                    "ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"
                )
            )
            agent = create_react_agent(model, [sandbox.as_langchain_tool()])  # type: ignore[reportUnknownVariableType]

            result = agent.invoke(
                {"messages": [("user", "Check the status of order ORD-12345")]}
            )

            # Print the final response
            for msg in result["messages"]:
                if hasattr(msg, "content") and msg.content:
                    role = msg.__class__.__name__
                    print(f"  {role}: {msg.content[:150]}...")

        except ImportError:
            print(
                "  LangGraph not installed. Run: pip install 'amla-sandbox[langgraph]'"
            )
    else:
        print("\n(Set ANTHROPIC_API_KEY to run with real LLM)")


# =============================================================================
# Part 4: Advanced Sandbox Configuration
# =============================================================================


def part4_advanced_config() -> None:
    """Configure sandbox with fine-grained capabilities."""
    print("\n" + "=" * 60)
    print("Part 4: Advanced Sandbox Configuration")
    print("=" * 60)

    # Define a banking API with strict constraints
    tools = [
        ToolDefinition(
            name="banking.balance",
            description="Get account balance",
            parameters={
                "type": "object",
                "properties": {"account_id": {"type": "string"}},
                "required": ["account_id"],
            },
        ),
        ToolDefinition(
            name="banking.transfer",
            description="Transfer funds between accounts",
            parameters={
                "type": "object",
                "properties": {
                    "from_account": {"type": "string"},
                    "to_account": {"type": "string"},
                    "amount": {"type": "number"},
                    "currency": {"type": "string"},
                },
                "required": ["from_account", "to_account", "amount"],
            },
        ),
        ToolDefinition(
            name="banking.statement",
            description="Get account statement",
            parameters={
                "type": "object",
                "properties": {
                    "account_id": {"type": "string"},
                    "days": {"type": "integer"},
                },
                "required": ["account_id"],
            },
        ),
    ]

    def handler(method: str, params: dict[str, Any]) -> dict[str, Any]:
        print(f"    [Banking API] {method}({params})")
        if method == "banking.balance":
            return {
                "account_id": params["account_id"],
                "balance": 5000.00,
                "currency": "USD",
            }
        if method == "banking.transfer":
            return {"transfer_id": "TXN-001", "status": "completed", **params}
        if method == "banking.statement":
            return {
                "account_id": params["account_id"],
                "transactions": [
                    {"date": "2024-01-15", "amount": -50, "description": "Coffee Shop"},
                    {"date": "2024-01-14", "amount": 1000, "description": "Deposit"},
                ],
            }
        return {"error": f"Unknown method: {method}"}

    # Create sandbox with strict capabilities
    sandbox = Sandbox(
        tools=tools,
        capabilities=[
            # Read-only balance check - unlimited
            MethodCapability(method_pattern="banking.balance"),
            # Transfers - constrained
            MethodCapability(
                method_pattern="banking.transfer",
                constraints=ConstraintSet(
                    [
                        Param("amount") > 0,
                        Param("amount") <= 1000,  # Max $1000 per transfer
                        Param("currency").is_in(
                            ["USD", "EUR"]
                        ),  # Only certain currencies
                    ]
                ),
                max_calls=3,  # Max 3 transfers per session
            ),
            # Statements - limited time range
            MethodCapability(
                method_pattern="banking.statement",
                constraints=ConstraintSet(
                    [
                        Param("days") <= 30,  # Max 30 days of history
                    ]
                ),
                max_calls=5,
            ),
        ],
        tool_handler=handler,
    )

    print("Banking Agent Capabilities:")
    print("-" * 40)
    for cap in sandbox.get_capabilities():
        print(f"  {cap.method_pattern}")
        if cap.max_calls:
            print(f"    max_calls: {cap.max_calls}")
        if not cap.constraints.is_empty():
            print("    has_constraints: yes")
    print()

    # Test authorization
    print("Authorization tests:")
    tests = [
        ("banking.balance", {"account_id": "ACC-123"}, True),
        (
            "banking.transfer",
            {"from_account": "A", "to_account": "B", "amount": 500, "currency": "USD"},
            True,
        ),
        (
            "banking.transfer",
            {"from_account": "A", "to_account": "B", "amount": 5000, "currency": "USD"},
            False,
        ),  # Too much
        (
            "banking.transfer",
            {"from_account": "A", "to_account": "B", "amount": 500, "currency": "BTC"},
            False,
        ),  # Wrong currency
        ("banking.statement", {"account_id": "ACC-123", "days": 30}, True),
        (
            "banking.statement",
            {"account_id": "ACC-123", "days": 90},
            False,
        ),  # Too many days
    ]

    for method, params, expected in tests:
        can = sandbox.can_call(method, params)
        status = "PASS" if can == expected else "FAIL"
        result = "allowed" if can else "denied"
        print(f"  [{status}] {method}: {result}")

    # Execute with constraints
    print("\nExecuting authorized operations:")
    result = sandbox.execute("""
        // Check balance - always allowed
        const balance = await banking_balance({account_id: "ACC-123"});
        console.log("Balance:", balance.balance, balance.currency);

        // Transfer within limits
        const transfer = await banking_transfer({
            from_account: "ACC-123",
            to_account: "ACC-456",
            amount: 500,
            currency: "USD"
        });
        console.log("Transfer:", transfer.status);
    """)
    print(f"  {result}")


# =============================================================================
# Part 5: State Management Across Turns
# =============================================================================


def part5_state_management() -> None:
    """Manage state across multiple agent turns."""
    print("\n" + "=" * 60)
    print("Part 5: State Management Across Agent Turns")
    print("=" * 60)

    print("""
VFS for Agent State:
  - /state/   - Persistent state between turns
  - /cache/   - Cached API responses
  - /memory/  - Conversation memory
  - /work/    - Working files for current task

The VFS persists across run() calls on the SAME bash instance.
This enables multi-turn conversations with stateful agents.
    """)

    def get_user_profile(user_id: str) -> dict[str, Any]:
        """Get user profile information."""
        return {
            "id": user_id,
            "name": "Alice Smith",
            "preferences": {"theme": "dark", "language": "en"},
        }

    def update_preferences(user_id: str, preferences: dict[str, Any]) -> dict[str, Any]:
        """Update user preferences."""
        return {"user_id": user_id, "updated": preferences, "status": "success"}

    sandbox = create_sandbox_tool(tools=[get_user_profile, update_preferences])

    print("Multi-turn agent conversation:")
    print("-" * 40)

    # Turn 1: Initialize state
    print("\nTurn 1: User asks about their profile")
    sandbox.run(
        """
        // Initialize state directory
        await fs.mkdir('/state', { recursive: true });

        // Fetch and cache user profile
        const profile = await get_user_profile({user_id: "USER-001"});

        // Store in state for future turns
        await fs.writeFile('/state/user.json', JSON.stringify(profile));

        console.log("Loaded profile for:", profile.name);
    """,
        language="javascript",
    )
    print("  [Agent stored user profile in /state/user.json]")

    # Turn 2: Reference previous state
    print("\nTurn 2: User asks to change theme")
    result = sandbox.run(
        """
        // Load state from previous turn
        const profile = JSON.parse(await fs.readFile('/state/user.json'));

        // Current preference
        console.log("Current theme:", profile.preferences.theme);

        // Update preference
        profile.preferences.theme = "light";
        const result = await update_preferences({
            user_id: profile.id,
            preferences: profile.preferences
        });

        // Save updated state
        await fs.writeFile('/state/user.json', JSON.stringify(profile));

        console.log("Updated theme to:", profile.preferences.theme);
    """,
        language="javascript",
    )
    print(f"  {result}")

    # Turn 3: Show accumulated state
    print("\nTurn 3: Verify state persisted")
    result = sandbox.run(
        """
        const profile = JSON.parse(await fs.readFile('/state/user.json'));
        console.log("User:", profile.name);
        console.log("Theme:", profile.preferences.theme);
        console.log("State persisted correctly!");
    """,
        language="javascript",
    )
    print(f"  {result}")


# =============================================================================
# Part 6: Code Mode Efficiency
# =============================================================================


def part6_code_mode() -> None:
    """Demonstrate code mode efficiency benefits."""
    print("\n" + "=" * 60)
    print("Part 6: Code Mode Efficiency")
    print("=" * 60)

    print("""
CODE MODE vs TOOL MODE

Tool Mode (traditional):
  LLM -> tool_call -> LLM -> tool_call -> LLM -> ... (10 round trips)
  Cost: High (many LLM calls, context grows)

Code Mode (amla-sandbox):
  LLM -> script (does all 10 things) -> result
  Cost: Low (1 LLM call, minimal context)

Example task: "Find all users who made orders > $100 and email them"

Tool Mode: 10+ LLM calls
  1. list_users() -> LLM analyzes
  2. get_orders(user_1) -> LLM analyzes
  3. get_orders(user_2) -> LLM analyzes
  ... (for each user)
  N. send_email(eligible_users) -> LLM confirms

Code Mode: 1 LLM call that writes:
  const users = await list_users();
  const eligible = [];
  for (const user of users) {
    const orders = await get_orders({user_id: user.id});
    if (orders.some(o => o.amount > 100)) {
      eligible.push(user);
    }
  }
  await send_bulk_email({users: eligible});
    """)

    # Demonstrate the difference
    def list_users() -> dict[str, Any]:
        return {
            "users": [
                {"id": "u1", "email": "alice@example.com"},
                {"id": "u2", "email": "bob@example.com"},
                {"id": "u3", "email": "charlie@example.com"},
            ]
        }

    def get_orders(user_id: str) -> dict[str, Any]:
        orders = {
            "u1": [{"id": "o1", "amount": 150}, {"id": "o2", "amount": 50}],
            "u2": [{"id": "o3", "amount": 80}],
            "u3": [{"id": "o4", "amount": 200}, {"id": "o5", "amount": 300}],
        }
        return {"orders": orders.get(user_id, [])}

    call_count = [0]

    def send_bulk_email(user_ids: list[str], message: str) -> dict[str, Any]:
        call_count[0] += 1
        return {"sent_to": len(user_ids), "status": "success"}

    sandbox = create_sandbox_tool(tools=[list_users, get_orders, send_bulk_email])

    print("\nCode mode execution:")
    result = sandbox.run(
        """
        // One script does everything
        const {users} = await list_users({});

        const eligible = [];
        for (const user of users) {
            const {orders} = await get_orders({user_id: user.id});
            const hasLargeOrder = orders.some(o => o.amount > 100);
            if (hasLargeOrder) {
                eligible.push(user.id);
            }
        }

        console.log("Eligible users:", eligible.length, "of", users.length);

        if (eligible.length > 0) {
            const result = await send_bulk_email({
                user_ids: eligible,
                message: "Thank you for your order!"
            });
            console.log("Emails sent:", result.sent_to);
        }
    """,
        language="javascript",
    )
    print(f"  {result}")
    print("\n  This would take 10+ LLM calls in tool mode!")
    print("  With code mode: 1 LLM call generates the script above.")


# =============================================================================
# Part 7: Production Deployment
# =============================================================================


def part7_production() -> None:
    """Production deployment considerations."""
    print("\n" + "=" * 60)
    print("Part 7: Production Deployment Considerations")
    print("=" * 60)

    print("""
PRODUCTION CHECKLIST

1. CAPABILITY BOUNDARIES
   - Define explicit capabilities per tenant/user
   - Use constraints to limit parameter values
   - Set appropriate max_calls budgets

2. ERROR HANDLING
   - Handle capability violations gracefully
   - Implement retry logic for transient failures
   - Log all errors for debugging

3. MONITORING
   - Track tool call frequency and latency
   - Monitor capability violations
   - Alert on anomalies

4. RESOURCE MANAGEMENT
   - Reuse bash instances when possible (VFS persists)
   - Clean up state between users
   - Set execution timeouts

5. SECURITY
   - Never expose raw tool handlers to untrusted input
   - Validate all parameters in handlers
   - Redact sensitive data from responses

6. TESTING
   - Unit test tool handlers
   - Integration test with sandbox
   - Load test capability limits
    """)

    # Example: Production-ready agent setup
    print("Example production setup:")
    print("-" * 40)

    def create_production_agent(
        tenant_id: str, tier: str = "standard"
    ) -> tuple[Any, dict[str, int]]:
        """Create a production-ready agent for a tenant."""

        # Define tier-based rate limits
        rate_limits = {
            "free": {"search": 10, "create": 5},
            "standard": {"search": 100, "create": 50},
            "enterprise": {"search": 1000, "create": 500},
        }
        limits = rate_limits.get(tier, rate_limits["standard"])

        def search(query: str) -> dict[str, Any]:
            return {"query": query, "results": []}

        def create_item(name: str, data: dict[str, Any]) -> dict[str, Any]:
            return {"id": f"item_{hash(name) % 10000}", "name": name}

        sandbox = create_sandbox_tool(
            tools=[search, create_item],
            max_calls=limits,
        )

        return sandbox, limits

    # Create agents for different tiers
    for tier in ["free", "standard", "enterprise"]:
        _bash, limits = create_production_agent(f"tenant_{tier}", tier)
        print(f"\n  {tier.upper()} tier:")
        print(f"    search: {limits.get('search', 'unlimited')} calls")
        print(f"    create: {limits.get('create', 'unlimited')} calls")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Run all tutorial parts."""
    print("=" * 60)
    print("LangGraph Integration Tutorial")
    print("Building AI Agents with amla-sandbox")
    print("=" * 60)

    part1_introduction()
    part2_simple_integration()
    part3_react_agent()
    part4_advanced_config()
    part5_state_management()
    part6_code_mode()
    part7_production()

    print("\n" + "=" * 60)
    print("Tutorial Complete!")
    print("=" * 60)
    print("""
Key takeaways:

1. create_sandbox_tool() + as_langchain_tool() for simple integration
2. Use Sandbox for fine-grained capability control
3. VFS enables stateful multi-turn conversations
4. Code mode is 10-100x more efficient than tool mode
5. Always set appropriate rate limits and constraints

Integration summary:
  Simple:   sandbox = create_sandbox_tool(tools=[...])
            agent = create_react_agent(model, [sandbox.as_langchain_tool()])

  Advanced: sandbox_tool = SandboxTool.from_functions([...], capabilities=[...])
            agent = create_react_agent(model, [sandbox_tool.as_langchain_tool()])

Next tutorial: 12_constraint_dsl.py
    Deep dive into the constraint system.
    """)


if __name__ == "__main__":
    main()
