#!/usr/bin/env python3
"""Sandbox with tool handlers example.

This example shows how to:
1. Define tools with JSON Schema
2. Create a sandbox with capabilities
3. Check authorization with can_call()
4. Execute JavaScript code that calls tools
5. Handle tool calls from the sandbox
"""

from typing import Any

from amla_sandbox import (
    Sandbox,
    MethodCapability,
    ConstraintSet,
    Param,
    ToolDefinition,
)

# =============================================================================
# Define Tools (what the agent can call)
# =============================================================================

TOOLS = [
    ToolDefinition(
        name="weather/current",
        description="Get current weather for a location",
        parameters={
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    ),
    ToolDefinition(
        name="weather/forecast",
        description="Get weather forecast for a location",
        parameters={
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "days": {"type": "integer", "minimum": 1, "maximum": 14},
            },
            "required": ["city"],
        },
    ),
    ToolDefinition(
        name="notes/create",
        description="Create a new note",
        parameters={
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "content": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["title", "content"],
        },
    ),
    ToolDefinition(
        name="notes/delete",
        description="Delete a note",
        parameters={
            "type": "object",
            "properties": {
                "id": {"type": "string"},
            },
            "required": ["id"],
        },
    ),
]

# =============================================================================
# Define Capabilities (what the agent is ALLOWED to call)
# =============================================================================

CAPABILITIES = [
    # Allow current weather (no constraints)
    MethodCapability(
        method_pattern="weather/current",
    ),
    # Allow forecast with day limit
    MethodCapability(
        method_pattern="weather/forecast",
        constraints=ConstraintSet(
            [
                Param("days") <= 7,  # Max 7-day forecast
            ]
        ),
    ),
    # Allow creating notes, but not deleting
    MethodCapability(
        method_pattern="notes/create",
        max_calls=10,  # Max 10 notes per session
    ),
    # Note: notes/delete is NOT granted - agent cannot delete notes
]

# =============================================================================
# Tool Handler (executes tool calls)
# =============================================================================


def handle_tool_call(method: str, params: dict[str, Any]) -> Any:
    """Handle tool calls from the sandbox.

    In a real application, this would call actual APIs.
    """
    print(f"  [Tool Call] {method}({params})")

    if method == "weather/current":
        return {
            "city": params["city"],
            "temperature": 22,
            "conditions": "sunny",
        }

    if method == "weather/forecast":
        days = params.get("days", 3)
        return {
            "city": params["city"],
            "forecast": [{"day": i, "temp": 20 + i} for i in range(days)],
        }

    if method == "notes/create":
        return {
            "id": "note_12345",
            "title": params["title"],
            "created": True,
        }

    return {"error": f"Unknown method: {method}"}


# =============================================================================
# Create Sandbox and Test
# =============================================================================


def main() -> None:
    """Run the sandbox capability demo."""
    sandbox = Sandbox(
        tools=TOOLS,
        capabilities=CAPABILITIES,
        tool_handler=handle_tool_call,
    )

    print("=== Sandbox Capability Check ===")
    print()

    # Check what the agent can do
    test_calls = [
        ("weather/current", {"city": "London"}),
        ("weather/forecast", {"city": "Paris", "days": 5}),
        ("weather/forecast", {"city": "Tokyo", "days": 14}),  # Exceeds limit
        ("notes/create", {"title": "Test", "content": "Hello"}),
        ("notes/delete", {"id": "note_123"}),  # Not allowed
    ]

    for method, params in test_calls:
        allowed = sandbox.can_call(method, params)
        status = "✓ allowed" if allowed else "✗ denied"
        print(f"  {status}: {method}({params})")

    print()
    print("=== Available Capabilities ===")
    for cap in sandbox.get_capabilities():
        print(f"  - {cap.method_pattern} (max_calls={cap.max_calls})")

    # =========================================================================
    # Execute JavaScript Code (actual tool calls)
    # =========================================================================

    print()
    print("=== Executing JavaScript Code ===")
    print()

    # Execute JavaScript that calls allowed tools
    # Tool names use underscores: "weather/current" -> weather_current()
    result = sandbox.execute(
        """
        // Get current weather (allowed)
        const weather = await weather_current({city: "London"});
        console.log("Weather:", JSON.stringify(weather));

        // Get 5-day forecast (allowed, within limit)
        const forecast = await weather_forecast({city: "Paris", days: 5});
        console.log("Forecast days:", forecast.forecast.length);

        // Create a note (allowed)
        const note = await notes_create({
            title: "Meeting Notes",
            content: "Discussed weather API integration"
        });
        console.log("Created note:", note.id);
        """
    )
    print("Output:", result)

    # Show remaining call budget
    print()
    print("=== Call Budget Status ===")
    counts = sandbox.get_call_counts()
    for cap_key, remaining in counts.items():
        print(f"  {cap_key}: {remaining} calls remaining")


if __name__ == "__main__":
    main()
