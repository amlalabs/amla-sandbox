#!/usr/bin/env python3
"""Audit Logging Example.

This example demonstrates the audit logging infrastructure:
1. AuditConfig - configuring what/where to log
2. AuditCollector - collecting and filtering entries
3. AuditEntry - structured log entries with agent context
4. File output - writing JSONL logs
5. Turn correlation - tracking multi-turn agent execution
6. Custom enrichment - adding deployment/environment info

The audit system is designed to capture metadata about sandbox operations
(tool calls, output streams, command lifecycle). The Python infrastructure
handles collection, enrichment, and output. When the Rust runtime is
instrumented, events will automatically flow through this system.
"""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from amla_sandbox import (
    MethodCapability,
    Sandbox,
    ToolDefinition,
)
from amla_sandbox.audit import AuditCollector, AuditConfig, AuditEntry


# =============================================================================
# Example 1: Basic Audit Configuration
# =============================================================================


def example_basic_config() -> None:
    """Configure audit logging with agent context."""
    print("=" * 60)
    print("Example 1: Basic Audit Configuration")
    print("=" * 60)
    print()

    # Create audit config with agent context
    config = AuditConfig(
        agent_id="my-agent-001",
        trace_id="trace-abc123",
    )

    print("AuditConfig created:")
    print(f"  agent_id: {config.agent_id}")
    print(f"  trace_id: {config.trace_id}")
    print(f"  output_path: {config.output_path} (None = in-memory only)")
    print(f"  capture_binary: {config.capture_binary}")
    print()

    # Create collector with config
    collector = AuditCollector(config)
    print(f"AuditCollector created: {len(collector)} entries")
    print(f"  turn_id: {collector.turn_id}")
    print()


# =============================================================================
# Example 2: Working with AuditEntry
# =============================================================================


def example_audit_entries() -> None:
    """Create and serialize audit entries."""
    print("=" * 60)
    print("Example 2: Working with AuditEntry")
    print("=" * 60)
    print()

    # Create sample entries (as would be parsed from WASM buffer)
    now = datetime.now(timezone.utc)

    entry = AuditEntry(
        type="tool_call",
        session_id="session-12345",
        timestamp=now,
        data={
            "op_id": 1,
            "tool": "calculator.add",
            "params_hash": "a1b2c3d4",
        },
        agent_id="demo-agent",
        trace_id="trace-demo",
        turn_id=0,
    )

    print("AuditEntry created:")
    print(f"  type: {entry.type}")
    print(f"  session_id: {entry.session_id}")
    print(f"  timestamp: {entry.timestamp}")
    print(f"  agent_id: {entry.agent_id}")
    print(f"  turn_id: {entry.turn_id}")
    print()

    # Convert to dict
    print("As dictionary:")
    d = entry.to_dict()
    for key in ["type", "session_id", "agent_id", "turn_id", "tool"]:
        if key in d:
            print(f"  {key}: {d[key]}")
    print()

    # Convert to JSONL
    print("As JSONL:")
    jsonl = entry.to_jsonl()
    print(f"  {jsonl[:80]}...")
    print()


# =============================================================================
# Example 3: Collecting and Filtering Entries
# =============================================================================


def example_filtering() -> None:
    """Filter audit entries by type and time."""
    print("=" * 60)
    print("Example 3: Collecting and Filtering Entries")
    print("=" * 60)
    print()

    config = AuditConfig(agent_id="filter-demo")
    collector = AuditCollector(config)

    # Create sample entries of different types
    base_time = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    entries = [
        AuditEntry(
            type="command_create",
            session_id="sess",
            timestamp=base_time,
            data={"command": "echo hello"},
        ),
        AuditEntry(
            type="tool_call",
            session_id="sess",
            timestamp=base_time,
            data={"tool": "calculator.add"},
        ),
        AuditEntry(
            type="stream_chunk",
            session_id="sess",
            timestamp=base_time,
            data={"stream": "stdout", "preview": "hello"},
        ),
        AuditEntry(
            type="tool_call",
            session_id="sess",
            timestamp=base_time,
            data={"tool": "calculator.multiply"},
        ),
        AuditEntry(
            type="command_exit",
            session_id="sess",
            timestamp=base_time,
            data={"exit_code": 0},
        ),
    ]

    # Add entries to collector
    collector.add_entries(entries, write_to_file=False)
    print(f"Total entries: {len(collector)}")
    print()

    # Filter by type
    print("Filter by type:")
    for entry_type in ["tool_call", "stream_chunk", "command_create"]:
        matching = list(collector.get_entries(entry_type=entry_type))
        print(f"  {entry_type}: {len(matching)} entries")
    print()

    # Get all entries and show types
    print("All entry types:")
    type_counts: dict[str, int] = {}
    for entry in collector.entries:
        type_counts[entry.type] = type_counts.get(entry.type, 0) + 1
    for t, count in sorted(type_counts.items()):
        print(f"  {t}: {count}")
    print()


# =============================================================================
# Example 4: Turn-Based Correlation
# =============================================================================


def example_turn_correlation() -> None:
    """Track entries across agent turns."""
    print("=" * 60)
    print("Example 4: Turn-Based Correlation")
    print("=" * 60)
    print()

    config = AuditConfig(agent_id="multi-turn-agent")
    collector = AuditCollector(config)

    now = datetime.now(timezone.utc)

    # Simulate multi-turn execution
    print("Simulating multi-turn agent:")
    print()

    for turn in range(3):
        print(f"  Turn {collector.turn_id}:")

        # Add entries for this turn
        for action in ["think", "tool_call", "respond"]:
            entry = AuditEntry(
                type=action,
                session_id="sess",
                timestamp=now,
                data={"action": f"Turn {turn} - {action}"},
                agent_id=config.agent_id,
                turn_id=collector.turn_id,
            )
            collector.add_entry(entry, write_to_file=False)
            print(f"    - {action}")

        # Move to next turn
        collector.new_turn()

    print()
    print(f"Final turn_id: {collector.turn_id}")
    print(f"Total entries: {len(collector)}")
    print()

    # Group entries by turn
    print("Entries per turn:")
    by_turn: dict[int, int] = {}
    for entry in collector.entries:
        tid = entry.turn_id or 0
        by_turn[tid] = by_turn.get(tid, 0) + 1

    for tid in sorted(by_turn):
        print(f"  Turn {tid}: {by_turn[tid]} entries")
    print()


# =============================================================================
# Example 5: File Output (JSONL)
# =============================================================================


def example_file_output() -> None:
    """Write audit logs to JSONL file."""
    print("=" * 60)
    print("Example 5: File Output (JSONL)")
    print("=" * 60)
    print()

    # Create temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        output_path = Path(f.name)

    print(f"Output file: {output_path}")
    print()

    # Create collector with file output
    config = AuditConfig(
        output_path=output_path,
        agent_id="file-demo",
        trace_id="trace-file",
    )

    with AuditCollector(config) as collector:
        # Manually write some entries (simulating runtime events)
        now = datetime.now(timezone.utc)

        for i in range(3):
            entry = AuditEntry(
                type="demo_event",
                session_id=f"session-{i}",
                timestamp=now,
                data={"event_number": i, "message": f"Event {i}"},
                agent_id=config.agent_id,
                trace_id=config.trace_id,
                turn_id=collector.turn_id,
            )
            collector.add_entry(entry, write_to_file=True)

    # Read and display file
    print("File contents:")
    content = output_path.read_text()
    lines = content.strip().split("\n")
    for i, line in enumerate(lines):
        parsed = json.loads(line)
        print(f"  {i + 1}. type={parsed['type']}, agent_id={parsed['agent_id']}")

    print()
    print(f"Written {len(lines)} entries to {output_path.name}")

    # Cleanup
    output_path.unlink()
    print()


# =============================================================================
# Example 6: Custom Enrichment
# =============================================================================


def example_custom_enrichment() -> None:
    """Add custom fields to audit entries."""
    print("=" * 60)
    print("Example 6: Custom Enrichment")
    print("=" * 60)
    print()

    # Define custom enricher
    def add_deployment_context(data: dict[str, Any]) -> dict[str, Any]:
        """Add deployment info to every entry."""
        data["environment"] = "production"
        data["region"] = "us-west-2"
        data["pod"] = "agent-worker-7b8c9d"
        return data

    config = AuditConfig(
        agent_id="enriched-agent",
        custom_enricher=add_deployment_context,
    )

    _collector = AuditCollector(config)  # Would be used by runtime
    print("Custom enricher configured")
    print()

    # Create entry and apply enricher
    data = {"original_field": "original_value"}

    # Enricher is applied when draining from runtime
    # For demo, apply manually
    enriched_data = add_deployment_context(data.copy())

    entry = AuditEntry(
        type="enriched_event",
        session_id="sess",
        timestamp=datetime.now(timezone.utc),
        data=enriched_data,
        agent_id=config.agent_id,
    )

    print("Entry with enrichment:")
    for key in ["original_field", "environment", "region", "pod"]:
        print(f"  {key}: {entry.data.get(key)}")
    print()


# =============================================================================
# Example 7: Sandbox Integration
# =============================================================================


def example_sandbox_integration() -> None:
    """Integrate audit logging with Sandbox."""
    print("=" * 60)
    print("Example 7: Sandbox Integration")
    print("=" * 60)
    print()

    # Define a simple tool
    tools = [
        ToolDefinition(
            name="greet",
            description="Say hello",
            parameters={"type": "object", "properties": {"name": {"type": "string"}}},
        )
    ]

    def handler(method: str, params: dict[str, Any]) -> dict[str, Any]:
        return {"message": f"Hello, {params.get('name', 'World')}!"}

    # Create sandbox with audit config
    config = AuditConfig(
        agent_id="sandbox-demo",
        trace_id="trace-sandbox",
    )

    print("Creating Sandbox with audit_config...")
    with Sandbox(
        tools=tools,
        capabilities=[MethodCapability(method_pattern="**")],
        tool_handler=handler,
        audit_config=config,
    ) as sandbox:
        collector = sandbox.audit_collector
        print(f"  audit_collector: {collector is not None}")
        print(f"  audit_collector.turn_id: {collector.turn_id if collector else 'N/A'}")
        print()

        # Execute some code
        result = sandbox.execute('console.log("Hello from sandbox!");')
        print(f"Execution result: {result.strip()}")
        print()

        # Check audit entries
        # Note: Entries will be empty until Rust instrumentation is added
        collector = sandbox.audit_collector
        if collector:
            print(f"Audit entries collected: {len(collector)}")
            if len(collector) == 0:
                print("  (Rust runtime instrumentation pending)")

    print()
    print("Sandbox closed, audit collector cleaned up")
    print()


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Run all audit logging examples."""
    print()
    print("AUDIT LOGGING EXAMPLES")
    print("=" * 60)
    print()

    example_basic_config()
    example_audit_entries()
    example_filtering()
    example_turn_correlation()
    example_file_output()
    example_custom_enrichment()
    example_sandbox_integration()

    print("=" * 60)
    print("Summary: Audit Logging Features")
    print("=" * 60)
    print("""
1. AUDIT CONFIG
   - agent_id, trace_id for correlation
   - output_path for JSONL file output
   - custom_enricher for adding fields

2. AUDIT ENTRY
   - Structured log entry with type, timestamp, data
   - Enriched with agent context (agent_id, trace_id, turn_id)
   - Serializable to dict and JSONL

3. AUDIT COLLECTOR
   - Collects entries from runtime buffer
   - Filter by type, timestamp
   - Track turns with new_turn()
   - Context manager for cleanup

4. SANDBOX INTEGRATION
   - Pass audit_config to Sandbox
   - Access via sandbox.audit_collector
   - Automatic cleanup on exit
""")


if __name__ == "__main__":
    main()
