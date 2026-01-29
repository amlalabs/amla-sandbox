"""Tests for tool ingestion from external frameworks."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from amla_sandbox.tools import (
    ToolDefinition,
    from_anthropic_tools,
    from_claude,
    from_langchain,
    from_openai,
    from_openai_tools,
)


# === Tests for from_openai_tools ===


class TestFromOpenAITools:
    """Tests for OpenAI function calling format ingestion."""

    def test_basic_tool_conversion(self) -> None:
        """Test converting a simple OpenAI tool definition."""
        tools: list[dict[str, Any]] = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "City name"}
                        },
                        "required": ["city"],
                    },
                },
            }
        ]

        funcs, defs = from_openai_tools(tools)

        assert len(funcs) == 1
        assert len(defs) == 1
        assert defs[0].name == "get_weather"
        assert defs[0].description == "Get current weather for a city"
        assert "city" in defs[0].parameters["properties"]

    def test_multiple_tools(self) -> None:
        """Test converting multiple OpenAI tool definitions."""
        tools: list[dict[str, Any]] = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the web",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]

        funcs, defs = from_openai_tools(tools)

        assert len(funcs) == 2
        assert len(defs) == 2
        names = {d.name for d in defs}
        assert names == {"get_weather", "search"}

    def test_with_handlers(self) -> None:
        """Test providing actual handler functions."""

        def get_weather(city: str) -> dict[str, Any]:
            return {"city": city, "temp": 72}

        tools: list[dict[str, Any]] = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        funcs, _ = from_openai_tools(tools, handlers={"get_weather": get_weather})

        assert len(funcs) == 1
        assert funcs[0] is get_weather
        # Handler should work
        result = funcs[0](city="SF")
        assert result == {"city": "SF", "temp": 72}

    def test_placeholder_without_handler(self) -> None:
        """Test that placeholder function raises when no handler provided."""
        tools: list[dict[str, Any]] = [
            {
                "type": "function",
                "function": {
                    "name": "missing_handler",
                    "description": "No handler for this",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        funcs, _ = from_openai_tools(tools)

        with pytest.raises(NotImplementedError, match="No handler provided"):
            funcs[0]()

    def test_flat_format(self) -> None:
        """Test the simplified format without 'function' wrapper."""
        tools: list[dict[str, Any]] = [
            {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            }
        ]

        _, defs = from_openai_tools(tools)

        assert len(defs) == 1
        assert defs[0].name == "get_weather"

    def test_from_openai_alias(self) -> None:
        """Test that from_openai is an alias for from_openai_tools."""
        assert from_openai is from_openai_tools


# === Tests for from_anthropic_tools ===


class TestFromAnthropicTools:
    """Tests for Anthropic tool format ingestion."""

    def test_basic_tool_conversion(self) -> None:
        """Test converting a simple Anthropic tool definition."""
        tools: list[dict[str, Any]] = [
            {
                "name": "get_weather",
                "description": "Get current weather for a location",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"}
                    },
                    "required": ["city"],
                },
            }
        ]

        funcs, defs = from_anthropic_tools(tools)

        assert len(funcs) == 1
        assert len(defs) == 1
        assert defs[0].name == "get_weather"
        assert defs[0].description == "Get current weather for a location"
        assert "city" in defs[0].parameters["properties"]

    def test_multiple_tools(self) -> None:
        """Test converting multiple Anthropic tool definitions."""
        tools: list[dict[str, Any]] = [
            {
                "name": "get_weather",
                "description": "Get weather",
                "input_schema": {"type": "object", "properties": {}},
            },
            {
                "name": "search",
                "description": "Search the web",
                "input_schema": {"type": "object", "properties": {}},
            },
        ]

        funcs, defs = from_anthropic_tools(tools)

        assert len(funcs) == 2
        assert len(defs) == 2
        names = {d.name for d in defs}
        assert names == {"get_weather", "search"}

    def test_with_handlers(self) -> None:
        """Test providing actual handler functions."""

        def search(query: str) -> str:
            return f"Results for: {query}"

        tools: list[dict[str, Any]] = [
            {
                "name": "search",
                "description": "Search",
                "input_schema": {"type": "object", "properties": {}},
            }
        ]

        funcs, _ = from_anthropic_tools(tools, handlers={"search": search})

        assert funcs[0] is search
        assert funcs[0](query="test") == "Results for: test"

    def test_placeholder_without_handler(self) -> None:
        """Test that placeholder raises when no handler provided."""
        tools: list[dict[str, Any]] = [
            {
                "name": "missing_handler",
                "description": "No handler",
                "input_schema": {"type": "object", "properties": {}},
            }
        ]

        funcs, _ = from_anthropic_tools(tools)

        with pytest.raises(NotImplementedError, match="No handler provided"):
            funcs[0]()

    def test_from_claude_alias(self) -> None:
        """Test that from_claude is an alias for from_anthropic_tools."""
        assert from_claude is from_anthropic_tools


# === Tests for from_langchain ===


class TestFromLangChain:
    """Tests for LangChain tool ingestion."""

    def test_plain_callable(self) -> None:
        """Test converting a plain Python function."""

        def my_tool(query: str) -> str:
            """Search for something."""
            return f"Found: {query}"

        funcs, defs = from_langchain([my_tool])

        assert len(funcs) == 1
        assert len(defs) == 1
        assert funcs[0] is my_tool
        assert defs[0].name == "my_tool"
        assert "Search for something" in defs[0].description

    def test_structured_tool_mock(self) -> None:
        """Test converting a LangChain StructuredTool (mocked)."""

        # Mock a StructuredTool-like object
        def underlying_func(x: int) -> int:
            """Double a number."""
            return x * 2

        mock_tool = MagicMock()
        mock_tool.func = underlying_func
        mock_tool.name = "doubler"
        mock_tool.description = "Doubles a number"
        mock_tool.args_schema = None

        funcs, defs = from_langchain([mock_tool])

        assert len(funcs) == 1
        assert len(defs) == 1
        assert funcs[0] is underlying_func
        assert defs[0].name == "doubler"
        assert defs[0].description == "Doubles a number"

    def test_base_tool_mock(self) -> None:
        """Test converting a LangChain BaseTool (mocked)."""

        # Mock a BaseTool-like object
        def run_impl(query: str) -> str:
            return f"Result: {query}"

        mock_tool = MagicMock()
        mock_tool.func = None  # No func attribute
        del mock_tool.func  # Remove it entirely
        mock_tool._run = run_impl
        mock_tool.name = "my_base_tool"
        mock_tool.description = "A base tool implementation"
        mock_tool.args_schema = None

        funcs, defs = from_langchain([mock_tool])

        assert len(funcs) == 1
        assert funcs[0] is run_impl
        assert defs[0].name == "my_base_tool"

    def test_invalid_tool_type(self) -> None:
        """Test that invalid tool types raise TypeError."""

        class NotATool:
            pass

        with pytest.raises(TypeError, match="Cannot convert tool"):
            from_langchain([NotATool()])

    def test_multiple_tools(self) -> None:
        """Test converting multiple tools."""

        def tool_a(x: int) -> int:
            """Tool A."""
            return x

        def tool_b(s: str) -> str:
            """Tool B."""
            return s

        funcs, defs = from_langchain([tool_a, tool_b])

        assert len(funcs) == 2
        assert len(defs) == 2
        names = {d.name for d in defs}
        assert names == {"tool_a", "tool_b"}


# === Tests for ToolDefinition structure ===


class TestToolDefinitionFromIngest:
    """Test that ingested tools produce valid ToolDefinitions."""

    def test_openai_tool_definition_structure(self) -> None:
        """Test that OpenAI tools produce proper ToolDefinition."""
        tools: list[dict[str, Any]] = [
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Perform calculation",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string"},
                            "precision": {"type": "integer"},
                        },
                        "required": ["expression"],
                    },
                },
            }
        ]

        _, defs = from_openai_tools(tools)

        assert isinstance(defs[0], ToolDefinition)
        assert defs[0].name == "calculate"
        assert defs[0].description == "Perform calculation"
        assert defs[0].parameters["type"] == "object"
        assert "expression" in defs[0].parameters["properties"]
        assert "precision" in defs[0].parameters["properties"]
        assert defs[0].parameters["required"] == ["expression"]

    def test_anthropic_tool_definition_structure(self) -> None:
        """Test that Anthropic tools produce proper ToolDefinition."""
        tools: list[dict[str, Any]] = [
            {
                "name": "database_query",
                "description": "Query the database",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "sql": {"type": "string", "description": "SQL query"},
                        "limit": {"type": "integer", "default": 100},
                    },
                    "required": ["sql"],
                },
            }
        ]

        _, defs = from_anthropic_tools(tools)

        assert isinstance(defs[0], ToolDefinition)
        assert defs[0].name == "database_query"
        assert defs[0].description == "Query the database"
        assert defs[0].parameters["type"] == "object"
        assert "sql" in defs[0].parameters["properties"]
