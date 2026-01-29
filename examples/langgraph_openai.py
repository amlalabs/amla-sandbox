#!/usr/bin/env python3
"""LangGraph + OpenAI: Real LLM Agent with Sandboxed Tools

This example demonstrates a complete LangGraph agent using OpenAI's GPT models
with amla-sandbox for secure tool execution.

Prerequisites:
    pip install "amla-sandbox[langgraph]"
    export OPENAI_API_KEY=your-key-here

Run:
    python langgraph_openai.py              # Run all examples
    python langgraph_openai.py --example 1  # Run only example 1
    python langgraph_openai.py --list       # List available examples

This example requires a valid OpenAI API key and will make actual API calls.
"""

import argparse
import os
import sys
from typing import Any

from dotenv import load_dotenv

# Load .env file (searches current dir and parents)
load_dotenv()


# =============================================================================
# Tools for the Agent
# =============================================================================


def get_weather(city: str) -> dict[str, Any]:
    """Get the current weather for a city.

    Args:
        city: The city name (e.g., "San Francisco", "Tokyo", "London")
    """
    weather_data = {
        "san francisco": {"temp": 18, "condition": "foggy", "humidity": 75},
        "tokyo": {"temp": 25, "condition": "sunny", "humidity": 60},
        "london": {"temp": 12, "condition": "rainy", "humidity": 85},
        "new york": {"temp": 22, "condition": "partly cloudy", "humidity": 55},
        "paris": {"temp": 16, "condition": "cloudy", "humidity": 70},
    }
    city_lower = city.lower()
    data = weather_data.get(
        city_lower, {"temp": 20, "condition": "unknown", "humidity": 50}
    )
    return {
        "city": city,
        "temperature_celsius": data["temp"],
        "condition": data["condition"],
        "humidity_percent": data["humidity"],
    }


def search_products(query: str, max_results: int = 3) -> dict[str, Any]:
    """Search for products in the catalog.

    Args:
        query: Search terms (e.g., "electronics", "sports")
        max_results: Maximum number of results to return (default: 3)
    """
    all_products: list[dict[str, str | float]] = [
        {
            "id": "p001",
            "name": "Wireless Headphones",
            "price": 79.99,
            "category": "electronics",
        },
        {
            "id": "p002",
            "name": "Bluetooth Speaker",
            "price": 49.99,
            "category": "electronics",
        },
        {"id": "p003", "name": "USB-C Hub", "price": 39.99, "category": "electronics"},
        {"id": "p004", "name": "Running Shoes", "price": 89.99, "category": "sports"},
        {"id": "p005", "name": "Yoga Mat", "price": 29.99, "category": "sports"},
        {"id": "p006", "name": "Water Bottle", "price": 19.99, "category": "sports"},
    ]

    query_lower = query.lower()
    matching = [
        p
        for p in all_products
        if query_lower in str(p["name"]).lower() or query_lower in str(p["category"])
    ]

    return {
        "query": query,
        "total_found": len(matching),
        "results": matching[:max_results],
    }


def get_stock_price(symbol: str) -> dict[str, Any]:
    """Get the current stock price for a ticker symbol.

    Args:
        symbol: Stock ticker symbol (e.g., "AAPL", "GOOGL", "MSFT")
    """
    stocks = {
        "AAPL": {"price": 178.50, "change": 2.30},
        "GOOGL": {"price": 141.25, "change": -0.75},
        "MSFT": {"price": 378.90, "change": 4.20},
    }
    symbol_upper = symbol.upper()
    data = stocks.get(symbol_upper)
    if data:
        return {"symbol": symbol_upper, **data}
    return {"symbol": symbol_upper, "error": "Symbol not found"}


# =============================================================================
# Examples
# =============================================================================


def example_1_basic_sandbox() -> None:
    """Example 1: Basic code generation with sandbox tool."""
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent
    from amla_sandbox import create_bash_tool

    print("\n" + "=" * 60)
    print("Example 1: Code Generation (Recommended with GPT-5)")
    print("=" * 60)

    # Create sandbox with tools - they run in a secure WASM sandbox
    bash = create_bash_tool(tools=[get_weather], max_calls=10)

    model = ChatOpenAI(
        model=os.environ.get("OPENAI_MODEL", "gpt-5"),
        temperature=0,
    )

    # Use code generation - LLM writes JavaScript to call tools
    agent: Any = create_react_agent(
        model, [bash.as_langchain_tool()], prompt=bash.get_system_prompt()
    )

    print("\nSystem prompt teaches LLM to write JavaScript:")
    print(bash.get_system_prompt()[:300] + "...")

    print("\n" + "-" * 40)
    print("Query: Get the weather in Tokyo")
    print("-" * 40)

    result = agent.invoke(
        {"messages": [("user", "Get the weather in Tokyo")]},
        config={"recursion_limit": 10},
    )

    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.get("name") == "sandbox":
                    print(
                        f"\n[Generated JS]: {tc.get('args', {}).get('code', '')[:200]}..."
                    )
        if msg.__class__.__name__ == "AIMessage" and getattr(msg, "content", ""):
            print(f"\nAssistant: {msg.content}")

    print("\n" + "-" * 40)
    print("Example 1 completed!")


def example_2_multiple_tools() -> None:
    """Example 2: Multiple tools with code generation."""
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent
    from amla_sandbox import create_bash_tool

    print("\n" + "=" * 60)
    print("Example 2: Multiple Tools with Code Generation")
    print("=" * 60)

    bash = create_bash_tool(tools=[get_weather, search_products], max_calls=20)

    model = ChatOpenAI(
        model=os.environ.get("OPENAI_MODEL", "gpt-5"),
        temperature=0,
    )

    # Code generation with multiple tools available
    agent: Any = create_react_agent(
        model, [bash.as_langchain_tool()], prompt=bash.get_system_prompt()
    )

    print("\nAvailable tools in sandbox:")
    print("  - get_weather(city): Get weather for a city")
    print("  - search_products(query, category?): Search products")

    print("\n" + "-" * 40)
    print("Query: What's the weather in Paris and find me outdoor products?")
    print("-" * 40)

    result = agent.invoke(
        {
            "messages": [
                ("user", "What's the weather in Paris and find me outdoor products?")
            ]
        },
        config={"recursion_limit": 15},
    )

    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.get("name") == "sandbox":
                    print(
                        f"\n[Generated JS]: {tc.get('args', {}).get('code', '')[:300]}..."
                    )
        if msg.__class__.__name__ == "AIMessage" and getattr(msg, "content", ""):
            print(f"\nAssistant: {msg.content}")

    print("\n" + "-" * 40)
    print("Example 2 completed!")


def example_3_shell_mode() -> None:
    """Example 3: Using shell mode with language='shell'."""
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent
    from amla_sandbox import create_bash_tool

    print("\n" + "=" * 60)
    print("Example 3: Shell Mode Execution")
    print("=" * 60)

    bash = create_bash_tool(tools=[get_weather], max_calls=10)

    # First, let's create some data with JavaScript
    print("\nCreating test data with JavaScript...")
    result = bash.run("""
        const cities = ["Tokyo", "Paris", "London"];
        const data = [];
        for (const city of cities) {
            const w = await get_weather({city});
            data.push(w);
        }
        await fs.writeFile("/workspace/weather.json", JSON.stringify(data, null, 2));
        console.log("Wrote weather data to /workspace/weather.json");
    """)
    print(f"  {result}")

    # Now use shell mode to process it
    print("\nProcessing with shell (language='shell')...")
    result = bash.run(
        'cat /workspace/weather.json | jq ".[].city" | tr -d \'"\'', language="shell"
    )
    print(f"  Cities: {result}")

    result = bash.run(
        'cat /workspace/weather.json | jq ".[] | .temperature_celsius" | sort -n | tail -1',
        language="shell",
    )
    print(f"  Hottest temp: {result.strip()}C")

    model = ChatOpenAI(
        model=os.environ.get("OPENAI_MODEL", "gpt-5"),
        temperature=0,
    )

    # The system prompt now includes shell instructions
    agent: Any = create_react_agent(
        model, [bash.as_langchain_tool()], prompt=bash.get_system_prompt()
    )

    print("\n" + "-" * 40)
    print("Query: Use shell to count the weather entries")
    print("-" * 40)

    result = agent.invoke(
        {
            "messages": [
                (
                    "user",
                    "Use the shell (language='shell') to count how many entries "
                    "are in /workspace/weather.json using jq",
                )
            ]
        },
        config={"recursion_limit": 10},
    )

    for msg in result["messages"]:
        if msg.__class__.__name__ == "AIMessage" and getattr(msg, "content", ""):
            print(f"\nAssistant: {msg.content}")

    print("\n" + "-" * 40)
    print("Example 3 completed!")


def example_4_with_constraints() -> None:
    """Example 4: Sandbox with constraints and rate limits."""
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent
    from amla_sandbox import create_bash_tool

    print("\n" + "=" * 60)
    print("Example 4: Constrained Sandbox")
    print("=" * 60)

    # Create sandbox with constraints - enforced at runtime
    bash = create_bash_tool(
        tools=[search_products, get_stock_price],
        constraints={
            "search_products": {"max_results": "<=5"},
        },
        max_calls={
            "search_products": 3,  # Max 3 searches
            "get_stock_price": 5,  # Max 5 stock lookups
        },
    )

    print("\nConstraints configured:")
    print("  - search_products.max_results <= 5")
    print("  - search_products: max 3 calls")
    print("  - get_stock_price: max 5 calls")

    model = ChatOpenAI(
        model=os.environ.get("OPENAI_MODEL", "gpt-5"),
        temperature=0,
    )

    # Use code generation - constraints enforced in WASM sandbox
    agent: Any = create_react_agent(
        model, [bash.as_langchain_tool()], prompt=bash.get_system_prompt()
    )

    print("\n" + "-" * 40)
    print("Query: Search for sports products")
    print("-" * 40)

    result = agent.invoke(
        {"messages": [("user", "Search for sports products")]},
        config={"recursion_limit": 15},
    )

    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.get("name") == "sandbox":
                    print(
                        f"\n[Generated JS]: {tc.get('args', {}).get('code', '')[:300]}..."
                    )
        if msg.__class__.__name__ == "AIMessage" and getattr(msg, "content", ""):
            print(f"\nAssistant: {msg.content}")

    print("\n" + "-" * 40)
    print("Example 4 completed!")


def example_5_composable_prompt() -> None:
    """Example 5: Custom persona with composable system prompt."""
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent
    from amla_sandbox import create_bash_tool

    print("\n" + "=" * 60)
    print("Example 5: Custom Persona with Code Generation")
    print("=" * 60)

    bash = create_bash_tool(tools=[get_weather, search_products], max_calls=20)

    model = ChatOpenAI(
        model=os.environ.get("OPENAI_MODEL", "gpt-5"),
        temperature=0,
    )

    # Custom persona + sandbox instructions combined
    persona = """You are a helpful shopping assistant. You help users find products
based on weather conditions. Always be friendly and concise.

When checking weather, consider:
- Sunny/warm -> recommend sports and outdoor gear
- Rainy/cold -> recommend electronics and indoor items
"""
    # Compose: persona + sandbox usage instructions
    full_prompt = persona + "\n" + bash.get_system_prompt()

    print("Composable prompt (persona + sandbox instructions):")
    print("-" * 40)
    print(full_prompt[:400] + "...")

    # Use code generation with combined prompt
    agent: Any = create_react_agent(
        model, [bash.as_langchain_tool()], prompt=full_prompt
    )

    print("\n" + "-" * 40)
    print("Query: Weather-based product recommendation")
    print("-" * 40)

    result = agent.invoke(
        {
            "messages": [
                (
                    "user",
                    "Check the weather in Tokyo and recommend products based on the conditions",
                )
            ]
        },
        config={"recursion_limit": 20},
    )

    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.get("name") == "sandbox":
                    print(
                        f"\n[Generated JS]: {tc.get('args', {}).get('code', '')[:300]}..."
                    )
        if msg.__class__.__name__ == "AIMessage" and getattr(msg, "content", ""):
            print(f"\nAssistant: {msg.content}")

    print("\n" + "-" * 40)
    print("Example 5 completed!")


def example_6_separate_tools() -> None:
    """Example 6: Separate JS and Shell tools (cleaner for LLM)."""
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent
    from amla_sandbox import create_sandbox_tool

    print("\n" + "=" * 60)
    print("Example 6: Separate JS and Shell Tools")
    print("=" * 60)

    # Create sandbox with explicit default language
    sandbox = create_sandbox_tool(
        tools=[get_weather, search_products],
        default_language="javascript",
        max_calls=20,
    )

    # Get separate tools instead of one combined tool
    tools = sandbox.as_langchain_tools()
    print(f"\nCreated {len(tools)} separate tools:")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description[:60]}...")

    model = ChatOpenAI(
        model=os.environ.get("OPENAI_MODEL", "gpt-5"),
        temperature=0,
    )

    # Use the system prompt designed for separate tools
    agent: Any = create_react_agent(
        model, tools, prompt=sandbox.get_system_prompt_for_separate_tools()
    )

    print("\nSystem prompt (for separate tools):")
    print("-" * 40)
    print(sandbox.get_system_prompt_for_separate_tools()[:400] + "...")

    print("\n" + "-" * 40)
    print("Query: Get weather in Tokyo and Paris, find hottest city")
    print("-" * 40)

    result = agent.invoke(
        {
            "messages": [
                (
                    "user",
                    "Get the weather in Tokyo and Paris, then tell me which city is hotter. "
                    "Use JavaScript to fetch the data and shell to compare.",
                )
            ]
        },
        config={"recursion_limit": 15},
    )

    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_name = tc.get("name", "")
                args = tc.get("args", {})
                if tool_name == "sandbox_js":
                    print(f"\n[JS Code]: {args.get('code', '')[:200]}...")
                elif tool_name == "sandbox_shell":
                    print(f"\n[Shell Command]: {args.get('command', '')}")
        if msg.__class__.__name__ == "AIMessage" and getattr(msg, "content", ""):
            print(f"\nAssistant: {msg.content}")

    print("\n" + "-" * 40)
    print("Example 6 completed!")


# =============================================================================
# Main
# =============================================================================


EXAMPLES = {
    1: ("Basic Code Generation", example_1_basic_sandbox),
    2: ("Multiple Tools", example_2_multiple_tools),
    3: ("Shell Mode", example_3_shell_mode),
    4: ("Constrained Sandbox", example_4_with_constraints),
    5: ("Custom Persona", example_5_composable_prompt),
    6: ("Separate JS/Shell Tools", example_6_separate_tools),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LangGraph + OpenAI examples with amla-sandbox"
    )
    parser.add_argument(
        "--example", "-e", type=int, help="Run a specific example (1-6)"
    )
    parser.add_argument(
        "--list", "-l", action="store_true", help="List available examples"
    )
    args = parser.parse_args()

    if args.list:
        print("Available examples:")
        for num, (name, _) in EXAMPLES.items():
            print(f"  {num}. {name}")
        return

    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set")
        print("\nSet it with:")
        print("  export OPENAI_API_KEY=your-key-here")
        print("\nOr create a .env file with:")
        print("  OPENAI_API_KEY=your-key-here")
        sys.exit(1)

    print("=" * 60)
    print("LangGraph + OpenAI: Real LLM Agent Examples")
    print("=" * 60)
    print(f"\nUsing model: {os.environ.get('OPENAI_MODEL', 'gpt-5')}")

    try:
        if args.example:
            if args.example not in EXAMPLES:
                print(f"Error: Unknown example {args.example}")
                print("Use --list to see available examples")
                sys.exit(1)
            name, func = EXAMPLES[args.example]
            print(f"\nRunning example {args.example}: {name}")
            func()
        else:
            # Run all examples
            for num, (name, func) in EXAMPLES.items():
                print(f"\n{'=' * 60}")
                print(f"Running example {num}: {name}")
                func()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
    print("""
Key Takeaways:

1. create_bash_tool() / create_sandbox_tool() creates a sandbox with your functions
2. as_langchain_tool() returns a single "sandbox" tool with language parameter
3. as_langchain_tools() returns separate "sandbox_js" and "sandbox_shell" tools
4. get_system_prompt() / get_system_prompt_for_separate_tools() teaches the LLM

Usage patterns:

  # Single tool with language parameter (simpler)
  bash = create_bash_tool(tools=[func1, func2])
  agent = create_react_agent(model, [bash.as_langchain_tool()],
                             prompt=bash.get_system_prompt())

  # Separate JS and shell tools (cleaner for LLM)
  sandbox = create_sandbox_tool(tools=[func1], default_language="javascript")
  tools = sandbox.as_langchain_tools()  # Returns [sandbox_js, sandbox_shell]
  agent = create_react_agent(model, tools,
                             prompt=sandbox.get_system_prompt_for_separate_tools())

  # Shell mode for Unix pipelines
  bash = create_bash_tool(tools=[func1])
  result = bash.run("cat data.json | jq '.items'", language="shell")
    """)


if __name__ == "__main__":
    main()
