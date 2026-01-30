#!/usr/bin/env python3
"""Framework Ingestion: Import tools from LangChain, OpenAI, and Anthropic formats.

This example demonstrates how to ingest tool definitions from popular agent
frameworks into amla-sandbox's format, enabling a seamless migration path.

Supported frameworks:
- LangChain (@tool decorated functions, BaseTool subclasses)
- OpenAI (function calling format)
- Anthropic (tool use format)

Requirements:
    uv pip install "git+https://github.com/amlalabs/amla-sandbox"
    uv pip install langchain  # Optional, for LangChain examples

Usage:
    python 11_framework_ingestion.py
"""

from typing import Any

from amla_sandbox import create_sandbox_tool
from amla_sandbox.tools import (
    from_anthropic_tools,
    from_langchain,
    from_openai_tools,
)


# =============================================================================
# Example 1: Ingest OpenAI Function Calling Format
# =============================================================================


def openai_example() -> None:
    """Demonstrate ingesting OpenAI-style tool definitions."""
    print("\n" + "=" * 60)
    print("Example 1: OpenAI Function Calling Format")
    print("=" * 60)

    # OpenAI function calling format (as returned by tool schemas)
    openai_tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "The city name",
                        }
                    },
                    "required": ["city"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_database",
                "description": "Search records in the database",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer", "default": 10},
                    },
                    "required": ["query"],
                },
            },
        },
    ]

    # Actual implementations for the tools
    def get_weather(city: str) -> dict[str, Any]:
        return {"city": city, "temp": 72, "condition": "sunny"}

    def search_database(query: str, limit: int = 10) -> list[dict[str, str]]:
        return [{"id": "1", "result": f"Match for {query}"}][:limit]

    # Ingest with handlers
    funcs, defs = from_openai_tools(
        openai_tools,
        handlers={
            "get_weather": get_weather,
            "search_database": search_database,
        },
    )

    print(f"Ingested {len(defs)} tools:")
    for d in defs:
        print(f"  - {d.name}: {d.description}")

    # Use with create_sandbox_tool
    sandbox = create_sandbox_tool(tools=funcs)
    result = sandbox.run(
        'const w = await get_weather({city: "SF"}); console.log(w);',
        language="javascript",
    )
    print(f"Execution result: {result}")


# =============================================================================
# Example 2: Ingest Anthropic Tool Format
# =============================================================================


def anthropic_example() -> None:
    """Demonstrate ingesting Anthropic-style tool definitions."""
    print("\n" + "=" * 60)
    print("Example 2: Anthropic Tool Format")
    print("=" * 60)

    # Anthropic tool format (as used with Claude)
    anthropic_tools = [
        {
            "name": "add_numbers",
            "description": "Add two numbers together",
            "input_schema": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"},
                },
                "required": ["a", "b"],
            },
        },
        {
            "name": "send_email",
            "description": "Send an email message",
            "input_schema": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Recipient email"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["to", "subject", "body"],
            },
        },
    ]

    # Implementations
    def add_numbers(a: float, b: float) -> dict[str, Any]:
        return {"a": a, "b": b, "result": a + b}

    def send_email(to: str, subject: str, body: str) -> dict[str, str]:
        return {
            "status": "sent",
            "to": to,
            "subject": subject,
            "body_len": str(len(body)),
        }

    # Ingest with handlers
    funcs, defs = from_anthropic_tools(
        anthropic_tools,
        handlers={
            "add_numbers": add_numbers,
            "send_email": send_email,
        },
    )

    print(f"Ingested {len(defs)} tools:")
    for d in defs:
        print(f"  - {d.name}: {d.description}")

    # Use with create_sandbox_tool (with constraints!)
    sandbox = create_sandbox_tool(
        tools=funcs,
        constraints={
            "send_email": {
                "to": "startswith:@company.com",  # Only internal emails
            }
        },
    )

    result = sandbox.run(
        "const r = await add_numbers({a: 5, b: 3}); console.log(r);",
        language="javascript",
    )
    print(f"Addition result: {result}")


# =============================================================================
# Example 3: Ingest LangChain Tools
# =============================================================================


def langchain_example() -> None:
    """Demonstrate ingesting LangChain-decorated tools."""
    print("\n" + "=" * 60)
    print("Example 3: LangChain @tool Decorated Functions")
    print("=" * 60)

    try:
        from langchain_core.tools import tool  # type: ignore[import-not-found]
    except ImportError:
        print("LangChain not installed. Skipping this example.")
        print("Install with: pip install langchain-core")
        return

    # LangChain @tool decorated functions
    @tool
    def translate(text: str, to_language: str = "Spanish") -> str:
        """Translate text to another language."""
        # Mock translation
        translations = {
            "Hello": {"Spanish": "Hola", "French": "Bonjour"},
            "Goodbye": {"Spanish": "Adiós", "French": "Au revoir"},
        }
        return translations.get(text, {}).get(to_language, f"[{text} in {to_language}]")

    @tool
    def summarize(text: str, max_words: int = 50) -> str:
        """Summarize a text to a maximum number of words."""
        words = text.split()[:max_words]
        return " ".join(words) + ("..." if len(text.split()) > max_words else "")

    # Ingest from LangChain
    funcs, defs = from_langchain([translate, summarize])

    print(f"Ingested {len(defs)} tools:")
    for d in defs:
        print(f"  - {d.name}: {d.description}")
        print(f"    Parameters: {list(d.parameters.get('properties', {}).keys())}")

    # Use with create_sandbox_tool
    sandbox = create_sandbox_tool(tools=funcs)
    result = sandbox.run(
        'const t = await translate({text: "Hello", to_language: "Spanish"}); console.log(t);',
        language="javascript",
    )
    print(f"Translation result: {result}")


# =============================================================================
# Example 4: Mixed Framework Ingestion
# =============================================================================


def mixed_example() -> None:
    """Demonstrate mixing tools from different frameworks."""
    print("\n" + "=" * 60)
    print("Example 4: Mixed Framework Tools")
    print("=" * 60)

    # OpenAI-style tool
    openai_tools = [
        {
            "type": "function",
            "function": {
                "name": "fetch_data",
                "description": "Fetch data from API",
                "parameters": {
                    "type": "object",
                    "properties": {"endpoint": {"type": "string"}},
                    "required": ["endpoint"],
                },
            },
        }
    ]

    # Anthropic-style tool
    anthropic_tools = [
        {
            "name": "store_result",
            "description": "Store a result to the database",
            "input_schema": {
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "value": {"type": "string"},
                },
                "required": ["key", "value"],
            },
        }
    ]

    # Implementations
    def fetch_data(endpoint: str) -> dict[str, str]:
        return {"data": f"Response from {endpoint}"}

    def store_result(key: str, value: str) -> dict[str, str]:
        return {"stored": key, "value": value}

    # Ingest from both formats
    openai_funcs, openai_defs = from_openai_tools(
        openai_tools, handlers={"fetch_data": fetch_data}
    )
    anthropic_funcs, anthropic_defs = from_anthropic_tools(
        anthropic_tools, handlers={"store_result": store_result}
    )

    # Combine all tools
    all_funcs = openai_funcs + anthropic_funcs
    all_defs = openai_defs + anthropic_defs

    print(f"Combined {len(all_defs)} tools from different frameworks:")
    for d in all_defs:
        print(f"  - {d.name}: {d.description}")

    # Create unified bash tool
    sandbox = create_sandbox_tool(tools=all_funcs)

    # Use both tools together
    script = """
    const data = await fetch_data({endpoint: "/api/users"});
    await store_result({key: "users", value: JSON.stringify(data)});
    console.log("Data fetched and stored!");
    """
    result = sandbox.run(script, language="javascript")
    print(f"Combined execution: {result}")


# =============================================================================
# Main
# =============================================================================


if __name__ == "__main__":
    print("Framework Ingestion Examples")
    print("============================")
    print("Demonstrating how to import tools from external frameworks.")

    openai_example()
    anthropic_example()
    langchain_example()
    mixed_example()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
