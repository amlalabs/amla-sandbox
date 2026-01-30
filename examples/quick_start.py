#!/usr/bin/env python3
"""Sandbox Tool: The simplest way to give agents code execution.

This example demonstrates `create_sandbox_tool()` - a dead-simple API inspired by
Vercel's AI SDK that hides WASM/capability complexity behind sensible defaults.

Progressive disclosure layers:
  - Layer 0: Shell or JavaScript only (no tools)
  - Layer 1: With Python functions as tools
  - Layer 2: With constraints and call limits

Requirements:
    uv pip install "git+https://github.com/amlalabs/amla-sandbox"

Usage:
    python quick_start.py
"""

from typing import Any

from amla_sandbox import create_sandbox_tool


# =============================================================================
# Sample tools for demonstration
# =============================================================================


def get_weather(city: str) -> dict[str, str | float]:
    """Get current weather for a city.

    Args:
        city: The city name (e.g., "San Francisco").
    """
    weather_data = {
        "San Francisco": {"temp": 18.5, "condition": "foggy"},
        "New York": {"temp": 22.0, "condition": "sunny"},
        "Tokyo": {"temp": 25.0, "condition": "cloudy"},
    }
    return {
        "city": city,
        **weather_data.get(city, {"temp": 20.0, "condition": "unknown"}),
    }


def transfer_money(
    amount: float, to_account: str, currency: str = "USD"
) -> dict[str, str]:
    """Transfer money to an account.

    Args:
        amount: Amount to transfer.
        to_account: Destination account ID.
        currency: Currency code (USD, EUR, GBP).
    """
    return {
        "status": "completed",
        "amount": str(amount),
        "to": to_account,
        "currency": currency,
        "reference": "TXN-12345",
    }


def search_database(query: str, limit: int = 10) -> list[dict[str, str]]:
    """Search the database.

    Args:
        query: Search query string.
        limit: Maximum results to return.
    """
    return [
        {"id": "1", "title": f"Result for '{query}'", "score": "0.95"},
        {"id": "2", "title": f"Another match for '{query}'", "score": "0.87"},
    ][:limit]


# =============================================================================
# Layer 0: Shell Only
# =============================================================================


def demo_layer_0() -> None:
    """Shell-only sandbox - no external tools."""
    print("\n" + "=" * 60)
    print("Layer 0: Shell Only")
    print("=" * 60)

    # Create a sandbox
    sandbox = create_sandbox_tool()

    # Execute shell commands
    output = sandbox.run(
        "echo 'Hello from sandbox!' | tr 'a-z' 'A-Z'", language="shell"
    )
    print(f"Shell output: {output}")

    # Use built-in shell utilities for data processing
    output = sandbox.run(
        'echo \'[{"name": "Alice", "score": 95}, {"name": "Bob", "score": 87}]\' '
        "| jq 'sort_by(.score) | reverse | .[0].name'",
        language="shell",
    )
    print(f"JSON processing: {output}")


# =============================================================================
# Layer 1: With Tools
# =============================================================================


def demo_layer_1() -> None:
    """Sandbox with Python functions as tools."""
    print("\n" + "=" * 60)
    print("Layer 1: With Tools")
    print("=" * 60)

    # Create sandbox with Python functions
    sandbox = create_sandbox_tool(
        tools=[get_weather, search_database],
    )

    # Tools can be called from JavaScript
    output = sandbox.run(
        """
        const weather = await get_weather({city: "San Francisco"});
        console.log("Weather:", JSON.stringify(weather));
    """,
        language="javascript",
    )
    print(f"JS tool call: {output}")

    # Tools can also be invoked via shell syntax!
    # This uses the `tool` command built into the sandbox shell
    output = sandbox.run("tool get_weather --city Tokyo", language="shell")
    print(f"Shell tool call: {output}")

    # Combine shell and tools
    output = sandbox.run(
        'tool search_database --query "important docs" --limit 5 | jq ".[0].title"',
        language="shell",
    )
    print(f"Combined: {output}")


# =============================================================================
# Layer 2: With Constraints
# =============================================================================


def demo_layer_2() -> None:
    """Sandbox with constraints and call limits."""
    print("\n" + "=" * 60)
    print("Layer 2: With Constraints")
    print("=" * 60)

    # Create sandbox with security constraints
    sandbox = create_sandbox_tool(
        tools=[transfer_money, get_weather],
        constraints={
            # Limit transfers to $1000 max, only USD/EUR
            "transfer_money": {
                "amount": "<=1000",
                "currency": ["USD", "EUR"],
            },
            # No constraints on weather
        },
        max_calls={
            "transfer_money": 5,  # Max 5 transfers per session
            "get_weather": 100,  # Weather is cheap, allow more
        },
    )

    # Valid transfer - within constraints
    output = sandbox.run(
        """
        const result = await transfer_money({
            amount: 500,
            to_account: "acct_12345",
            currency: "USD"
        });
        console.log("Transfer:", JSON.stringify(result));
    """,
        language="javascript",
    )
    print(f"Valid transfer: {output}")

    # Another valid transfer with EUR
    output = sandbox.run(
        """
        const result = await transfer_money({
            amount: 999,
            to_account: "acct_67890",
            currency: "EUR"
        });
        console.log("EUR Transfer:", JSON.stringify(result));
    """,
        language="javascript",
    )
    print(f"EUR transfer: {output}")

    # Check max_calls is set
    caps = sandbox.sandbox.capabilities
    transfer_cap = next((c for c in caps if "transfer_money" in c.method_pattern), None)
    if transfer_cap:
        print(f"Transfer max_calls: {transfer_cap.max_calls}")
        print(f"Transfer has constraints: {not transfer_cap.constraints.is_empty()}")


# =============================================================================
# LangGraph Integration Example
# =============================================================================


def demo_langgraph_integration() -> None:
    """Show how to use with LangGraph (if available)."""
    print("\n" + "=" * 60)
    print("LangGraph Integration")
    print("=" * 60)

    import os

    try:
        from langchain_anthropic import ChatAnthropic
        from langgraph.prebuilt import create_react_agent
    except ImportError:
        print(
            "LangGraph not installed. Install with: uv pip install langgraph langchain-anthropic"
        )
        _print_langgraph_example()
        return

    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set. Showing example code only.")
        _print_langgraph_example()
        return

    # Create sandbox tool
    sandbox = create_sandbox_tool(
        tools=[get_weather, search_database],
        max_calls=50,
    )

    # Create agent
    model = ChatAnthropic(
        model=os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")  # type: ignore[call-arg]
    )
    agent: Any = create_react_agent(model, [sandbox.as_langchain_tool()])

    # Run
    result = agent.invoke({"messages": [("user", "What's the weather in Tokyo?")]})

    for msg in result["messages"]:
        if hasattr(msg, "content") and msg.content:
            print(f"{msg.__class__.__name__}: {msg.content[:200]}")


def _print_langgraph_example() -> None:
    """Print example code for LangGraph integration."""
    print("""
Example code:

    from langchain_anthropic import ChatAnthropic
    from langgraph.prebuilt import create_react_agent
    from amla_sandbox import create_sandbox_tool

    # Create sandbox with your functions
    sandbox = create_sandbox_tool(
        tools=[get_weather, search_database],
        max_calls=50,
    )

    # Use with LangGraph
    model = ChatAnthropic(model="claude-3-5-sonnet-20241022")
    agent = create_react_agent(model, [sandbox.as_langchain_tool()])

    result = agent.invoke({
        "messages": [("user", "What's the weather in Tokyo?")]
    })
    """)


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Run all demos."""
    print("=" * 60)
    print("create_sandbox_tool() - AI SDK-Style Ergonomics")
    print("=" * 60)
    print("""
The simplest way to give agents code execution:

    from amla_sandbox import create_sandbox_tool

    # Shell sandbox
    sandbox = create_sandbox_tool()
    sandbox.run("echo hello | tr 'a-z' 'A-Z'", language="shell")

    # JavaScript sandbox with tools
    sandbox = create_sandbox_tool(
        tools=[my_func1, my_func2],
    )

    # With constraints
    sandbox = create_sandbox_tool(
        tools=[transfer_money],
        constraints={"transfer_money": {"amount": "<=1000"}},
        max_calls=10,
    )
    """)

    demo_layer_0()
    demo_layer_1()
    demo_layer_2()
    demo_langgraph_integration()

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
