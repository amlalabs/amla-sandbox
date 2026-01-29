"""Tool definition for the Amla sandbox.

This module provides the ToolDefinition class that describes tools
available to agents in the sandbox.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolDefinition:
    """Definition of a tool available to the sandbox.

    This is the format for registering external tools that the agent
    can call from within the sandbox. Tools are converted to JavaScript
    function stubs that the agent can invoke.

    Attributes:
        name: Tool name (method name for JSON-RPC calls).
        description: Human-readable description of what the tool does.
        parameters: JSON Schema for tool parameters.
    """

    name: str
    """Tool name (method name for JSON-RPC calls)."""

    description: str
    """Human-readable description of what the tool does."""

    parameters: dict[str, Any] = field(default_factory=lambda: {})
    """JSON Schema for tool parameters."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to MCP-compatible dictionary for WASM runtime.

        Uses inputSchema key to match MCP tool format expected by Rust.

        Returns:
            Dictionary with name, description, and inputSchema.
        """
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.parameters,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolDefinition":
        """Create from dictionary.

        Handles both inputSchema (MCP format) and parameters keys.

        Args:
            data: Dictionary with name, description, and parameters/inputSchema.

        Returns:
            ToolDefinition instance.
        """
        params = data.get("inputSchema") or data.get("parameters", {})
        return cls(
            name=data["name"],
            description=data["description"],
            parameters=params,
        )
