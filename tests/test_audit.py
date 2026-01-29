"""Tests for audit logging functionality."""

# pyright: reportPrivateUsage=warning

from __future__ import annotations

from typing import Any

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path


from amla_sandbox import MethodCapability, Sandbox
from amla_sandbox.audit import AuditCollector, AuditConfig, AuditEntry


class TestAuditConfig:
    """Tests for AuditConfig."""

    def test_default_config(self) -> None:
        config = AuditConfig()

        assert config.output_path is None
        assert config.agent_id is None
        assert config.trace_id is None
        assert config.capture_binary is False
        assert config.binary_dir is None
        assert config.custom_enricher is None

    def test_config_with_options(self) -> None:
        config = AuditConfig(
            output_path=Path("/tmp/audit.jsonl"),
            agent_id="agent-123",
            trace_id="trace-456",
            capture_binary=True,
            binary_dir=Path("/tmp/binaries"),
        )

        assert config.output_path == Path("/tmp/audit.jsonl")
        assert config.agent_id == "agent-123"
        assert config.trace_id == "trace-456"
        assert config.capture_binary is True
        assert config.binary_dir == Path("/tmp/binaries")

    def test_custom_enricher(self) -> None:
        def my_enricher(data: dict[str, Any]) -> dict[str, Any]:
            data["custom_field"] = "custom_value"
            return data

        config = AuditConfig(custom_enricher=my_enricher)
        assert config.custom_enricher is not None


class TestAuditEntry:
    """Tests for AuditEntry."""

    def test_basic_entry(self) -> None:
        ts = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        entry = AuditEntry(
            type="test_event",
            session_id="session-abc",
            timestamp=ts,
            data={"key": "value"},
        )

        assert entry.type == "test_event"
        assert entry.session_id == "session-abc"
        assert entry.timestamp == ts
        assert entry.data == {"key": "value"}
        assert entry.agent_id is None
        assert entry.trace_id is None
        assert entry.turn_id is None
        assert entry.binary_path is None

    def test_entry_with_enrichment(self) -> None:
        ts = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        entry = AuditEntry(
            type="test_event",
            session_id="session-abc",
            timestamp=ts,
            data={"key": "value"},
            agent_id="agent-123",
            trace_id="trace-456",
            turn_id=5,
        )

        assert entry.agent_id == "agent-123"
        assert entry.trace_id == "trace-456"
        assert entry.turn_id == 5

    def test_to_dict(self) -> None:
        ts = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        entry = AuditEntry(
            type="test_event",
            session_id="session-abc",
            timestamp=ts,
            data={"extra": "data"},
            agent_id="agent-123",
            trace_id="trace-456",
            turn_id=3,
        )

        d = entry.to_dict()

        assert d["type"] == "test_event"
        assert d["session_id"] == "session-abc"
        assert d["timestamp"] == "2025-01-01T12:00:00+00:00"
        assert d["extra"] == "data"  # Merged from data dict
        assert d["agent_id"] == "agent-123"
        assert d["trace_id"] == "trace-456"
        assert d["turn_id"] == 3

    def test_to_dict_minimal(self) -> None:
        ts = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        entry = AuditEntry(
            type="test_event",
            session_id="session-abc",
            timestamp=ts,
            data={},
        )

        d = entry.to_dict()

        assert d["type"] == "test_event"
        assert "agent_id" not in d
        assert "trace_id" not in d
        assert "turn_id" not in d
        assert "binary_path" not in d

    def test_to_jsonl(self) -> None:
        ts = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        entry = AuditEntry(
            type="test_event",
            session_id="session-abc",
            timestamp=ts,
            data={"key": "value"},
        )

        jsonl = entry.to_jsonl()

        # Should be valid JSON
        parsed = json.loads(jsonl)
        assert parsed["type"] == "test_event"
        assert parsed["session_id"] == "session-abc"
        assert parsed["key"] == "value"

        # Should be single line
        assert "\n" not in jsonl


class TestAuditCollector:
    """Tests for AuditCollector."""

    def test_default_collector(self) -> None:
        collector = AuditCollector()

        assert len(collector) == 0
        assert collector.turn_id == 0
        assert list(collector.get_entries()) == []

    def test_collector_with_config(self) -> None:
        config = AuditConfig(agent_id="agent-123")
        collector = AuditCollector(config)

        assert collector._config.agent_id == "agent-123"

    def test_new_turn(self) -> None:
        collector = AuditCollector()

        assert collector.turn_id == 0
        collector.new_turn()
        assert collector.turn_id == 1
        collector.new_turn()
        assert collector.turn_id == 2

    def test_clear(self) -> None:
        collector = AuditCollector()

        # Manually add an entry for testing
        ts = datetime.now(timezone.utc)
        entry = AuditEntry(type="test", session_id="test", timestamp=ts, data={})
        collector._entries.append(entry)

        assert len(collector) == 1
        collector.clear()
        assert len(collector) == 0

    def test_entries_property(self) -> None:
        collector = AuditCollector()

        ts = datetime.now(timezone.utc)
        entry1 = AuditEntry(type="a", session_id="test", timestamp=ts, data={})
        entry2 = AuditEntry(type="b", session_id="test", timestamp=ts, data={})
        collector._entries.extend([entry1, entry2])

        entries = collector.entries
        assert len(entries) == 2
        assert entries[0].type == "a"
        assert entries[1].type == "b"

    def test_get_entries_filter_by_type(self) -> None:
        collector = AuditCollector()

        ts = datetime.now(timezone.utc)
        collector._entries.extend(
            [
                AuditEntry(type="tool_call", session_id="test", timestamp=ts, data={}),
                AuditEntry(
                    type="stream_chunk", session_id="test", timestamp=ts, data={}
                ),
                AuditEntry(type="tool_call", session_id="test", timestamp=ts, data={}),
            ]
        )

        tool_calls = list(collector.get_entries(entry_type="tool_call"))
        assert len(tool_calls) == 2

        stream_chunks = list(collector.get_entries(entry_type="stream_chunk"))
        assert len(stream_chunks) == 1

    def test_get_entries_filter_by_since(self) -> None:
        collector = AuditCollector()

        ts1 = datetime(2025, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        ts2 = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        ts3 = datetime(2025, 1, 1, 14, 0, 0, tzinfo=timezone.utc)

        collector._entries.extend(
            [
                AuditEntry(type="a", session_id="test", timestamp=ts1, data={}),
                AuditEntry(type="b", session_id="test", timestamp=ts2, data={}),
                AuditEntry(type="c", session_id="test", timestamp=ts3, data={}),
            ]
        )

        # Filter entries after 11:00
        since = datetime(2025, 1, 1, 11, 0, 0, tzinfo=timezone.utc)
        entries = list(collector.get_entries(since=since))
        assert len(entries) == 2
        assert entries[0].type == "b"
        assert entries[1].type == "c"

    def test_context_manager(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            output_path = Path(f.name)

        config = AuditConfig(output_path=output_path)

        with AuditCollector(config) as collector:
            # Add entry directly for testing
            ts = datetime.now(timezone.utc)
            entry = AuditEntry(type="test", session_id="test", timestamp=ts, data={})
            collector._entries.append(entry)
            collector._file.write(entry.to_jsonl() + "\n")
            collector._file.flush()

        # File should be closed after context manager exits
        assert collector._file is None

        # Verify file was written
        content = output_path.read_text()
        assert "test" in content

        # Cleanup
        output_path.unlink()

    def test_capture_binary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            binary_dir = Path(tmpdir)
            config = AuditConfig(capture_binary=True, binary_dir=binary_dir)
            collector = AuditCollector(config)

            # Capture some binary data
            data = b"Hello, World!"
            path = collector.capture_binary("abc123", data)

            assert path is not None
            assert path.name == "abc123.bin"
            assert path.read_bytes() == data

    def test_capture_binary_no_dir(self) -> None:
        config = AuditConfig(capture_binary=False)
        collector = AuditCollector(config)

        path = collector.capture_binary("abc123", b"data")
        assert path is None

    def test_file_output(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            output_path = Path(f.name)

        config = AuditConfig(output_path=output_path, agent_id="test-agent")
        collector = AuditCollector(config)

        # Manually add and write entries
        ts = datetime.now(timezone.utc)
        entry = AuditEntry(
            type="test_event",
            session_id="session-123",
            timestamp=ts,
            data={"key": "value"},
            agent_id=config.agent_id,
        )
        collector._entries.append(entry)
        collector._file.write(entry.to_jsonl() + "\n")
        collector._file.flush()

        collector.close()

        # Verify file content
        content = output_path.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 1

        parsed = json.loads(lines[0])
        assert parsed["type"] == "test_event"
        assert parsed["session_id"] == "session-123"
        assert parsed["agent_id"] == "test-agent"
        assert parsed["key"] == "value"

        # Cleanup
        output_path.unlink()


class TestSandboxAuditIntegration:
    """Tests for Sandbox integration with audit logging."""

    def test_sandbox_with_audit_config(self) -> None:
        """Test that Sandbox correctly initializes with audit config."""
        config = AuditConfig(agent_id="test-agent", trace_id="trace-123")

        sandbox = Sandbox(
            tools=[],
            capabilities=[MethodCapability(method_pattern="**")],
            audit_config=config,
        )

        # Audit collector should be created
        assert sandbox.audit_collector is not None
        assert sandbox._audit_collector is not None
        assert sandbox._audit_collector._config.agent_id == "test-agent"
        assert sandbox._audit_collector._config.trace_id == "trace-123"

    def test_sandbox_without_audit_config(self) -> None:
        """Test that Sandbox works without audit config."""
        sandbox = Sandbox(
            tools=[],
            capabilities=[MethodCapability(method_pattern="**")],
        )

        assert sandbox.audit_collector is None
        assert sandbox._audit_collector is None

        # get_audit_entries should return empty iterator
        entries = list(sandbox.get_audit_entries())
        assert entries == []

    def test_sandbox_audit_context_manager_cleanup(self) -> None:
        """Test that audit collector is closed when exiting context manager."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            output_path = Path(f.name)

        config = AuditConfig(output_path=output_path, agent_id="test-agent")

        with Sandbox(
            tools=[],
            capabilities=[MethodCapability(method_pattern="**")],
            audit_config=config,
        ) as sandbox:
            collector = sandbox.audit_collector
            assert collector is not None
            assert collector._file is not None

        # After exiting context, collector should be closed
        assert sandbox._audit_collector is None

        # Cleanup
        output_path.unlink()

    def test_sandbox_get_audit_entries(self) -> None:
        """Test get_audit_entries method on Sandbox."""
        config = AuditConfig(agent_id="test-agent")

        with Sandbox(
            tools=[],
            capabilities=[MethodCapability(method_pattern="**")],
            audit_config=config,
        ) as sandbox:
            collector = sandbox.audit_collector
            assert collector is not None

            # Manually add some entries for testing
            ts = datetime.now(timezone.utc)
            collector._entries.extend(
                [
                    AuditEntry(
                        type="tool_call",
                        session_id="test",
                        timestamp=ts,
                        data={},
                    ),
                    AuditEntry(
                        type="stream_chunk",
                        session_id="test",
                        timestamp=ts,
                        data={},
                    ),
                ]
            )

            # Test getting all entries
            all_entries = list(sandbox.get_audit_entries())
            assert len(all_entries) == 2

            # Test filtering by type
            tool_calls = list(sandbox.get_audit_entries(entry_type="tool_call"))
            assert len(tool_calls) == 1
            assert tool_calls[0].type == "tool_call"
