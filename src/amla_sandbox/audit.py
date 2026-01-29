"""Audit log collection and enrichment for amla-sandbox.

This module provides tools for collecting and enriching audit logs from the
WASM runtime. Logs are structured JSONL entries that capture:

- Host operation requests and responses
- Stream chunks (stdout, stderr, stdin)
- Command lifecycle events
- Tool calls with capability enforcement

Example usage::

    from amla_sandbox import Sandbox
    from amla_sandbox.audit import AuditConfig, AuditCollector

    # Create collector with agent context
    config = AuditConfig(
        output_path=Path("audit.jsonl"),
        agent_id="agent-123",
        trace_id="trace-456",
    )
    collector = AuditCollector(config)

    # Create sandbox with audit collector
    sandbox = Sandbox(tools=[...], audit_collector=collector)

    # Execute code (logs are collected automatically)
    result = sandbox.execute("ls -la")

    # Get audit entries
    for entry in collector.get_entries():
        print(entry)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Callable, Iterator

if TYPE_CHECKING:
    from .runtime.wasm import Runtime


@dataclass
class AuditConfig:
    """Configuration for audit logging.

    Attributes:
        output_path: Path to write JSONL logs. If None, logs are kept in memory only.
        agent_id: Agent identifier for correlation across sessions.
        trace_id: Distributed trace ID for integration with tracing systems.
        capture_binary: Whether to capture full binary data at host level.
        binary_dir: Directory for binary data files ({content_hash}.bin).
        custom_enricher: Optional function to add custom fields to each entry.
    """

    # Output
    output_path: Path | None = None

    # Context enrichment (added to each entry from Python)
    agent_id: str | None = None
    trace_id: str | None = None

    # Optional binary data capture
    capture_binary: bool = False
    binary_dir: Path | None = None

    # Custom enrichment
    custom_enricher: Callable[[dict[str, Any]], dict[str, Any]] | None = None


@dataclass
class AuditEntry:
    """A single audit log entry with agent context.

    Entries are created by draining the WASM runtime's audit buffer and
    enriching with Python-side context.

    Attributes:
        type: Entry type (e.g., "host_op_request", "stream_chunk", "command_create").
        session_id: Session identifier from the runtime.
        timestamp: When the event occurred (from runtime).
        data: Full metadata from the Rust runtime.
        agent_id: Agent identifier (from Python context).
        trace_id: Trace ID (from Python context).
        turn_id: Agent turn number (from Python context).
        binary_path: Path to captured binary data file (if capture_binary=True).
    """

    type: str
    session_id: str
    timestamp: datetime
    data: dict[str, Any]

    # Python enrichment
    agent_id: str | None = None
    trace_id: str | None = None
    turn_id: int | None = None

    # Optional captured binary
    binary_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "type": self.type,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
            **self.data,
        }
        if self.agent_id:
            result["agent_id"] = self.agent_id
        if self.trace_id:
            result["trace_id"] = self.trace_id
        if self.turn_id is not None:
            result["turn_id"] = self.turn_id
        if self.binary_path:
            result["binary_path"] = str(self.binary_path)
        return result

    def to_jsonl(self) -> str:
        """Serialize to JSONL format (single line)."""
        return json.dumps(self.to_dict())


class AuditCollector:
    """Collects and enriches audit logs from WASM runtime.

    The collector drains JSONL entries from the runtime's ring buffer,
    parses them, enriches with Python-side context (agent_id, trace_id, turn_id),
    and optionally writes to a file.

    Usage::

        collector = AuditCollector(AuditConfig(
            output_path=Path("audit.jsonl"),
            agent_id="my-agent",
        ))

        # In the stepping loop:
        entries = collector.drain_from_runtime(runtime)

        # At end of agent turn:
        collector.new_turn()

        # Query entries:
        for entry in collector.get_entries(entry_type="tool_call"):
            print(entry)
    """

    def __init__(self, config: AuditConfig | None = None) -> None:
        """Initialize the audit collector.

        Args:
            config: Configuration for the collector. If None, uses defaults.
        """
        self._config = config or AuditConfig()
        self._entries: list[AuditEntry] = []
        self._file: IO[str] | None = None
        self._turn_id = 0

        if self._config.output_path:
            self._file = open(self._config.output_path, "a")

        if self._config.binary_dir:
            self._config.binary_dir.mkdir(parents=True, exist_ok=True)

    def drain_from_runtime(self, runtime: Runtime) -> list[AuditEntry]:
        """Drain audit logs from WASM runtime and enrich with context.

        This should be called after each runtime step to collect logs
        before the buffer fills up.

        Args:
            runtime: The WASM runtime to drain from.

        Returns:
            List of enriched audit entries.
        """
        # Get raw JSONL from WASM
        raw_jsonl = runtime._drain_audit_buffer()  # pyright: ignore[reportPrivateUsage]
        if not raw_jsonl:
            return []

        entries: list[AuditEntry] = []
        for line in raw_jsonl.strip().split("\n"):
            if not line:
                continue

            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Parse timestamp
            timestamp_str = raw.get("timestamp", "")
            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                timestamp = datetime.now()

            entry = AuditEntry(
                type=raw.get("type", "unknown"),
                session_id=raw.get("session_id", ""),
                timestamp=timestamp,
                data=raw,
                agent_id=self._config.agent_id,
                trace_id=self._config.trace_id,
                turn_id=self._turn_id,
            )

            # Apply custom enricher
            if self._config.custom_enricher:
                entry.data = self._config.custom_enricher(entry.data)

            entries.append(entry)
            self._entries.append(entry)

            # Write to file if configured
            if self._file:
                self._file.write(entry.to_jsonl() + "\n")
                self._file.flush()

        return entries

    def new_turn(self) -> None:
        """Mark the start of a new agent turn.

        This increments the turn_id for correlation of entries within
        an agent's turn-based execution.
        """
        self._turn_id += 1

    def get_entries(
        self,
        entry_type: str | None = None,
        since: datetime | None = None,
    ) -> Iterator[AuditEntry]:
        """Get collected entries with optional filtering.

        Args:
            entry_type: Filter by entry type (e.g., "tool_call", "stream_chunk").
            since: Only return entries after this timestamp.

        Yields:
            Matching audit entries.
        """
        for entry in self._entries:
            if entry_type and entry.type != entry_type:
                continue
            if since and entry.timestamp < since:
                continue
            yield entry

    def capture_binary(self, content_hash: str, data: bytes) -> Path | None:
        """Save binary data to the binary_dir.

        This is called by the runtime when capture_binary=True to save
        full binary content for correlation with metadata entries.

        Args:
            content_hash: BLAKE3 hash of the content (for filename).
            data: Binary data to save.

        Returns:
            Path to the saved file, or None if binary_dir is not configured.
        """
        if not self._config.binary_dir:
            return None

        path = self._config.binary_dir / f"{content_hash}.bin"
        path.write_bytes(data)
        return path

    def clear(self) -> None:
        """Clear all collected entries."""
        self._entries.clear()

    def close(self) -> None:
        """Close any open file handles."""
        if self._file:
            self._file.close()
            self._file = None

    def add_entry(self, entry: AuditEntry, *, write_to_file: bool = True) -> None:
        """Add an entry to the collector.

        This is useful for programmatically creating audit entries outside
        of the normal runtime drain flow.

        Args:
            entry: The audit entry to add.
            write_to_file: If True and output_path is configured, write to file.
        """
        self._entries.append(entry)
        if write_to_file and self._file:
            self._file.write(entry.to_jsonl() + "\n")
            self._file.flush()

    def add_entries(
        self, entries: list[AuditEntry], *, write_to_file: bool = True
    ) -> None:
        """Add multiple entries to the collector.

        Args:
            entries: The audit entries to add.
            write_to_file: If True and output_path is configured, write to file.
        """
        for entry in entries:
            self.add_entry(entry, write_to_file=write_to_file)

    @property
    def entries(self) -> list[AuditEntry]:
        """Get all collected entries."""
        return self._entries

    @property
    def turn_id(self) -> int:
        """Get the current turn ID."""
        return self._turn_id

    def __enter__(self) -> AuditCollector:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - closes file handles."""
        self.close()

    def __len__(self) -> int:
        """Get number of collected entries."""
        return len(self._entries)
