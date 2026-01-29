"""WASM runtime wrapper using wasmtime.

This module provides the low-level interface to the amla-sandbox WASM module.
It implements the stepping protocol and host operation routing.
"""

from __future__ import annotations

import base64
import inspect
import json
import logging
import threading as _threading
import time
from collections.abc import Awaitable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Protocol, Union

from ..capabilities import CallLimitExceededError, CapabilityError, MethodCapability

if TYPE_CHECKING:
    import wasmtime

    from ..audit import AuditEntry

_logger = logging.getLogger(__name__)


class AuditCollectorProtocol(Protocol):
    """Protocol for audit collectors that can drain entries from the runtime.

    This enables duck typing - any class with a drain_from_runtime method
    can be used as an audit collector.
    """

    def drain_from_runtime(self, runtime: "Runtime") -> list["AuditEntry"]:
        """Drain audit entries from the runtime."""
        ...


class RuntimeError(Exception):
    """Error from the WASM runtime."""

    pass


class RuntimeStatus(Enum):
    """Status of the runtime after a step."""

    RUNNING = "running"
    """Runtime is executing (can make progress)."""

    ALL_BLOCKED = "all_blocked"
    """All commands are blocked on host operations."""

    ALL_DONE = "all_done"
    """All commands have completed."""

    ERROR = "error"
    """Runtime encountered an error."""

    PANIC = "panic"
    """Runtime panicked and was killed."""


SyncToolHandler = Callable[[str, dict[str, Any]], Any]
"""Type for synchronous tool handlers: (method, params) -> result"""

AsyncToolHandler = Callable[[str, dict[str, Any]], Awaitable[Any]]
"""Type for async tool handlers: (method, params) -> awaitable result"""

ToolHandler = Union[SyncToolHandler, AsyncToolHandler]
"""Type for tool call handlers: sync or async (method, params) -> result"""


@dataclass
class RuntimeConfig:
    """Configuration for the WASM runtime."""

    wasm_path: Path | None = None
    """Path to amla-sandbox.wasm. If None, uses bundled WASM."""

    pca_bytes: bytes = b""
    """Serialized PCA (capability token) for this runtime instance."""

    trusted_authorities: list[str] = field(default_factory=lambda: list[str]())
    """List of trusted authority public keys in 'ed25519:hex' format.

    PCAs must be signed by one of these authorities to be accepted.
    For testing, use EphemeralAuthority to generate ephemeral keys.
    """

    capabilities: list[MethodCapability] = field(
        default_factory=lambda: list[MethodCapability]()
    )
    """Capabilities extracted from PCA for enforcement."""

    tool_handler: ToolHandler | None = None
    """Handler for tool_call operations."""

    tools_json: str = "[]"
    """JSON array of MCP tool definitions for stub generation."""

    max_steps: int = 10000
    """Maximum steps before timeout."""

    # Buffer size configuration (advanced tuning)
    output_buffer_size: int = 8192
    """Output buffer size in bytes for WASM calls. Default: 8KB."""

    tool_result_chunk_size: int = 2048
    """Chunk size for large tool results in bytes. Default: 2KB.

    Large tool results are split into chunks to fit within the output buffer.
    This value should be less than output_buffer_size / 3 to account for
    base64 encoding overhead and JSON envelope.
    """

    max_tool_result_size: int = 10 * 1024 * 1024
    """Maximum accumulated tool result size in bytes. Default: 10MB.

    Tool results larger than this will cause an error.
    """


# Default buffer sizes (used when config not available at module level)
OUTPUT_BUFFER_SIZE = 8192
TOOL_RESULT_CHUNK_SIZE = 2048
MAX_TOOL_RESULT_SIZE = 10 * 1024 * 1024

# Memory layout for host data (must not conflict with WASM stack/heap)
# IMPORTANT: These values are baked into the WASM binary and CANNOT be changed
# without recompiling the Rust code. They define the memory regions used for
# communication between Python and WASM.
OUTPUT_BUFFER_SIZE = 8192

# Chunk size for large tool results (2KB raw = ~2.7KB base64 + envelope)
# This is conservative to ensure chunks fit within OUTPUT_BUFFER_SIZE
TOOL_RESULT_CHUNK_SIZE = 2048

# Maximum accumulated tool result size (10MB)
MAX_TOOL_RESULT_SIZE = 10 * 1024 * 1024

# Memory layout for host data (must not conflict with WASM stack/heap)
# Using larger buffers to accommodate prelude + user code (can be ~3KB+)
CMD_PTR = 1024  # Command string starts at 1KB
OUT_PTR = 8192  # Output buffer at 8KB (gives 7KB for commands)
SUBMIT_PTR = 16384  # Submit buffer at 16KB
AUDIT_PTR = 24576  # Audit drain buffer at 24KB


def _create_tool_result_responses(
    op_id: int,
    runtime_id: int,
    result: Any,
    chunk_size: int = TOOL_RESULT_CHUNK_SIZE,
    max_size: int = MAX_TOOL_RESULT_SIZE,
) -> list[dict[str, Any]]:
    """Create tool result response(s), chunking if necessary.

    For small results, returns a single tool_result response.
    For large results, returns multiple tool_result_chunk responses.

    Args:
        op_id: The host operation ID.
        runtime_id: The runtime ID.
        result: The tool result value (will be JSON-serialized).
        chunk_size: Maximum raw bytes per chunk (default 2KB).
        max_size: Maximum total result size in bytes (default 10MB).

    Returns:
        List of response dicts. Usually 1 element, multiple for large results.
    """
    # Serialize the result to JSON bytes
    try:
        result_bytes = json.dumps(result).encode("utf-8")
    except (TypeError, ValueError) as e:
        # Result isn't JSON-serializable, return error
        return [
            {
                "id": op_id,
                "runtime_id": runtime_id,
                "result": {
                    "type": "error",
                    "code": "internal",
                    "message": f"Tool result not JSON-serializable: {e}",
                },
            }
        ]

    # Check if chunking is needed
    if len(result_bytes) <= chunk_size:
        # Small result - use atomic tool_result
        return [
            {
                "id": op_id,
                "runtime_id": runtime_id,
                "result": {"type": "tool_result", "result": result},
            }
        ]

    # Large result - split into chunks
    if len(result_bytes) > max_size:
        # Too large even for chunking
        return [
            {
                "id": op_id,
                "runtime_id": runtime_id,
                "result": {
                    "type": "tool_result_error",
                    "message": f"Tool result too large: {len(result_bytes)} bytes (max {max_size})",
                },
            }
        ]

    responses: list[dict[str, Any]] = []
    for i in range(0, len(result_bytes), chunk_size):
        chunk_data = result_bytes[i : i + chunk_size]
        is_final = (i + chunk_size) >= len(result_bytes)
        responses.append(
            {
                "id": op_id,
                "runtime_id": runtime_id,
                "result": {
                    "type": "tool_result_chunk",
                    "data": base64.b64encode(chunk_data).decode("ascii"),
                    "eof": is_final,
                },
            }
        )

    return responses


# =============================================================================
# Module Cache - Compile WASM once, reuse across all Runtime instances
# =============================================================================
# WASM compilation takes ~260ms. By caching the compiled module and engine,
# subsequent Runtime creations only need to create a new Store (~0.01ms).
#
# We also support disk-based caching of precompiled modules. Precompiled
# modules deserialize in ~0.5ms vs ~260ms for fresh compilation. The cache
# key includes the WASM hash, wasmtime version, and platform to ensure
# compatibility.
#
# This is thread-safe: multiple threads can safely get the cached module.

_module_cache_lock = _threading.Lock()
_cached_engine: Any = None
_cached_module: Any = None
_cached_wasm_path: str | None = None


def get_cache_dir() -> Path:
    """Get the cache directory for precompiled WASM modules.

    Uses ~/.cache/amla-sandbox on Unix, or appropriate cache dir on Windows.
    """
    import os
    import sys

    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))

    cache_dir = base / "amla-sandbox"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _get_cache_key(wasm_path: Path) -> str:
    """Compute cache key for a WASM file.

    The key includes:
    - SHA256 hash of the WASM file (first 16 chars)
    - wasmtime version
    - platform (e.g., linux-x86_64)
    """
    import hashlib
    import platform
    import sys

    # Hash the WASM file content
    with open(wasm_path, "rb") as f:
        wasm_hash = hashlib.sha256(f.read()).hexdigest()[:16]

    # Get wasmtime version

    # wasmtime doesn't expose __version__, so use the package metadata
    try:
        from importlib.metadata import version

        wt_version = version("wasmtime")
    except Exception:
        wt_version = "unknown"

    # Get platform info
    plat = f"{sys.platform}-{platform.machine()}"

    return f"{wasm_hash}-wt{wt_version}-{plat}"


def get_precompiled_path(wasm_path: Path) -> Path:
    """Get the path to the precompiled .cwasm file for a WASM file."""
    cache_key = _get_cache_key(wasm_path)
    return get_cache_dir() / f"{cache_key}.cwasm"


def default_wasm_path() -> Path:
    """Get the path to the bundled WASM module."""
    from .._wasm import get_wasm_path

    return get_wasm_path()


def precompile_module(wasm_path: Path | None = None) -> Path:
    """Precompile the WASM module and save to disk cache.

    This can be called explicitly to warm the cache, or it happens
    automatically on first use. Subsequent loads will deserialize
    the precompiled module (~0.5ms) instead of compiling (~260ms).

    Args:
        wasm_path: Path to WASM file. If None, uses the bundled module.

    Returns:
        Path to the precompiled .cwasm file.
    """
    import wasmtime

    if wasm_path is None:
        wasm_path = default_wasm_path()

    cwasm_path = get_precompiled_path(wasm_path)

    # Compile and serialize
    engine = wasmtime.Engine()
    module = wasmtime.Module.from_file(engine, str(wasm_path))  # pyright: ignore[reportUnknownMemberType]
    compiled_bytes = module.serialize()

    # Write atomically (write to temp, then rename)
    import tempfile

    cwasm_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=cwasm_path.parent, delete=False, suffix=".tmp"
    ) as f:
        f.write(compiled_bytes)
        temp_path = Path(f.name)

    temp_path.rename(cwasm_path)
    return cwasm_path


def _get_cached_module(wasm_path: Path) -> tuple[Any, Any]:
    """Get or create cached wasmtime Engine and Module.

    Thread-safe. Checks for precompiled module on disk first (~0.5ms),
    falls back to fresh compilation (~260ms) if not found.

    Returns:
        Tuple of (engine, module) for wasmtime.
    """
    global _cached_engine, _cached_module, _cached_wasm_path

    path_str = str(wasm_path)

    # Fast path: already cached in memory for this path
    if _cached_module is not None and _cached_wasm_path == path_str:
        return _cached_engine, _cached_module

    with _module_cache_lock:
        # Double-check after acquiring lock
        if _cached_module is not None and _cached_wasm_path == path_str:
            return _cached_engine, _cached_module

        # Import wasmtime here to avoid import at module level
        import wasmtime

        _cached_engine = wasmtime.Engine()

        # Try to load precompiled module from disk cache
        cwasm_path = get_precompiled_path(wasm_path)
        if cwasm_path.exists():
            try:
                _cached_module = wasmtime.Module.deserialize_file(
                    _cached_engine, str(cwasm_path)
                )
                _cached_wasm_path = path_str
                return _cached_engine, _cached_module
            except Exception as e:
                # Precompiled module is invalid (e.g., wasmtime version mismatch)
                # Delete stale cache and fall through to fresh compilation
                _logger.debug(
                    "Failed to load precompiled module from %s: %s. Will recompile.",
                    cwasm_path,
                    e,
                )
                try:
                    cwasm_path.unlink(missing_ok=True)
                except OSError:
                    pass  # Can't delete - that's fine

        # Compile from WASM source
        _cached_module = wasmtime.Module.from_file(_cached_engine, path_str)  # pyright: ignore[reportUnknownMemberType]
        _cached_wasm_path = path_str

        # Save precompiled module to disk cache (async, non-blocking)
        try:
            compiled_bytes = _cached_module.serialize()
            cwasm_path.parent.mkdir(parents=True, exist_ok=True)

            import tempfile

            with tempfile.NamedTemporaryFile(
                dir=cwasm_path.parent, delete=False, suffix=".tmp"
            ) as f:
                f.write(compiled_bytes)
                temp_path = Path(f.name)
            temp_path.rename(cwasm_path)
        except Exception as e:
            # Cache write failed - not critical, continue without caching
            _logger.debug(
                "Failed to cache compiled module to %s: %s",
                cwasm_path,
                e,
            )

        return _cached_engine, _cached_module


class Runtime:
    """WASM runtime instance.

    Each runtime is bound to a specific PCA and enforces its capabilities.
    The runtime lifecycle matches the PCA lifecycle.

    Example::

        from amla_sandbox.runtime import Runtime, RuntimeConfig
        from amla_sandbox.capabilities import MethodCapability

        # For testing - easiest approach
        runtime = Runtime.for_testing(
            capabilities=["tool_call:**"],
            tool_handler=my_tool_handler,
        )

        # For production - explicit PCA and trusted authorities
        from amla_sandbox.auth import EphemeralAuthority

        authority = EphemeralAuthority()  # or load from secure storage
        pca = authority.create_pca(capabilities=["tool_call:stripe/**"])

        config = RuntimeConfig(
            pca_bytes=pca.to_cbor(),
            trusted_authorities=[authority.public_key_hex()],
            capabilities=[MethodCapability(method_pattern="stripe/**")],
            tool_handler=my_tool_handler,
        )
        runtime = Runtime(config)

        # Execute a command
        result = runtime.execute("echo 'hello world'")
    """

    @classmethod
    def for_testing(
        cls,
        capabilities: list[str] | None = None,
        tool_handler: ToolHandler | None = None,
        tools_json: str = "[]",
        wasm_path: Path | None = None,
        max_steps: int = 10000,
    ) -> "Runtime":
        """Create a runtime for testing with an ephemeral authority.

        This is the easiest way to create a runtime for tests. It generates
        an ephemeral Ed25519 keypair, creates a signed PCA with the specified
        capabilities, and configures trusted authorities automatically.

        Args:
            capabilities: List of capability patterns like "tool_call:**".
                Defaults to ["tool_call:**"] (allow all tool calls).
            tool_handler: Handler for tool_call operations.
            tools_json: JSON array of MCP tool definitions.
            wasm_path: Custom path to WASM file.
            max_steps: Maximum steps before timeout.

        Returns:
            Configured Runtime ready for testing.

        Example::

            # Allow all tools
            runtime = Runtime.for_testing(tool_handler=my_handler)

            # Restrict to specific tools
            runtime = Runtime.for_testing(
                capabilities=["tool_call:stripe/**"],
                tool_handler=my_handler,
            )
        """
        from ..auth import EphemeralAuthority

        # Create ephemeral authority
        authority = EphemeralAuthority()

        # Create PCA with specified capabilities
        if capabilities is None:
            capabilities = ["tool_call:**"]
        pca = authority.create_pca(capabilities=capabilities)

        # Convert capability patterns to MethodCapability objects
        method_caps: list[MethodCapability] = []
        for cap in capabilities:
            if cap.startswith("tool_call:"):
                pattern = cap.removeprefix("tool_call:")
            else:
                pattern = cap
            method_caps.append(MethodCapability(method_pattern=pattern))

        # Create config
        config = RuntimeConfig(
            wasm_path=wasm_path,
            pca_bytes=pca.to_cbor(),
            trusted_authorities=[authority.public_key_hex()],
            capabilities=method_caps,
            tool_handler=tool_handler,
            tools_json=tools_json,
            max_steps=max_steps,
        )

        return cls(config)

    def __init__(self, config: RuntimeConfig) -> None:
        """Initialize the runtime with configuration.

        Args:
            config: Runtime configuration including PCA and capabilities.

        Raises:
            RuntimeError: If WASM runtime cannot be loaded.
        """
        self._config = config
        self._capabilities = list(config.capabilities)
        self._tool_handler = config.tool_handler
        self._output_chunks: list[bytes] = []
        self._stderr_chunks: list[bytes] = []
        self._audit_collector: AuditCollectorProtocol | None = None

        # Call counters for max_calls enforcement
        # Maps capability key -> remaining calls (only for caps with max_calls)
        self._call_counts: dict[str, int] = {}
        for cap in self._capabilities:
            if cap.max_calls is not None:
                self._call_counts[cap.key()] = cap.max_calls

        # wasmtime objects (initialized in _load_wasm, always set before use)
        # Typed as optional since they start as None, but _load_wasm() is called
        # in __init__ so they're always set before any other method runs.
        # Use the store/instance properties for type-safe non-optional access.
        self.__store: wasmtime.Store | None = None
        self.__instance: wasmtime.Instance | None = None
        self._runtime_id: int = 0

        # Command execution timing (populated after execute())
        self._last_exit_code: int = 0
        self._last_elapsed_ns: int | None = None
        self._last_user_time_ns: int | None = None

        # Stdin buffer for piping data to commands
        self._stdin_data: bytes = b""
        self._stdin_pos: int = 0

        # Thread lock for WASM access - wasmtime Store is NOT thread-safe
        # LangGraph/LangChain may call tools from ThreadPoolExecutor
        self._lock = _threading.Lock()

        # Load WASM immediately - no lazy loading, fatal if fails
        self._load_wasm()

    @property
    def _store(self) -> wasmtime.Store:
        """Get the wasmtime Store (asserts initialized)."""
        assert self.__store is not None, "Runtime not initialized"
        return self.__store

    @property
    def _instance(self) -> wasmtime.Instance:
        """Get the wasmtime Instance (asserts initialized)."""
        assert self.__instance is not None, "Runtime not initialized"
        return self.__instance

    def _load_wasm(self) -> None:
        """Load the WASM module via wasmtime.

        Uses a minimal WASI shim that only provides what the sandbox needs:
        - clock_time_get: For timestamps in scheduling
        - random_get: For generating IDs
        - fd_write (stderr): For panic messages
        - Everything else returns ERRNO_NOSYS

        The WASM module is compiled once and cached globally. Subsequent
        Runtime instances reuse the cached module, reducing creation time
        from ~260ms to ~5ms.

        Raises:
            RuntimeError: If WASM cannot be loaded or runtime creation fails.
        """
        try:
            import wasmtime
        except ImportError as e:
            raise RuntimeError(
                "wasmtime is required but not installed.\n"
                "Install with: pip install wasmtime"
            ) from e

        # Find WASM file
        wasm_path = self._config.wasm_path
        if wasm_path is None:
            from .._wasm import get_wasm_path

            wasm_path = get_wasm_path()

        if not wasm_path.exists():
            raise RuntimeError(
                f"WASM file not found: {wasm_path}\n"
                "Build amla-sandbox with: cargo build --release --target wasm32-wasip1"
            )

        # Get cached engine and module (compiles on first call, ~260ms)
        # Subsequent calls return cached, making Runtime creation ~50x faster
        engine, module = _get_cached_module(wasm_path)

        # Create new store for this instance (stores are not shared)
        self.__store = wasmtime.Store(engine)

        # Create linker with minimal WASI shim
        linker = wasmtime.Linker(engine)
        self._define_minimal_wasi(linker, engine)

        # Instantiate
        self.__instance = linker.instantiate(self.__store, module)

        # Create runtime from PCA
        self._create_runtime()

    def _define_minimal_wasi(self, linker: Any, engine: Any) -> None:
        """Define minimal WASI imports for the sandbox.

        The amla runtime is a SANDBOXED environment. It should NOT have access to:
        - Real filesystem (it has an in-memory VFS)
        - Real environment variables
        - Real stdin/stdout (virtualized through host ops)

        The only WASI syscalls we provide are:
        - clock_time_get: For timestamps in scheduling
        - random_get: For generating IDs
        - fd_write (stderr only): For panic messages

        Note: In wasmtime-py 40+, Func callbacks receive just the WASM args (no Caller).
        We access memory via the instance reference stored after instantiation.
        """
        import os
        import struct

        import wasmtime

        # WASI error codes
        ERRNO_SUCCESS = 0
        ERRNO_BADF = 8
        ERRNO_NOSYS = 52

        # Random counter for deterministic PRNG
        random_counter = [0]

        # Memory reference (set after instantiation by _load_wasm)
        memory_ref: list[Any] = [None]

        def get_memory() -> Any:
            """Get memory from instance (set after linker.instantiate)."""
            if memory_ref[0] is None:
                memory_ref[0] = self._instance.exports(self._store)["memory"]
            return memory_ref[0]

        # Store memory_ref so we can set it after instantiation
        self._memory_ref = memory_ref

        # Helper to create typed functions
        def make_func(params: list[Any], results: list[Any], fn: Any) -> Any:
            return wasmtime.Func(
                self._store,
                wasmtime.FuncType(params, results),
                fn,
            )

        # clock_time_get(clock_id: i32, precision: i64, out: i32) -> i32
        def clock_time_get_impl(_clock_id: int, _precision: int, out_ptr: int) -> int:
            memory = get_memory()
            nanos = int(time.time() * 1_000_000_000)
            data = struct.pack("<Q", nanos)
            memory.write(self._store, data, out_ptr)
            return ERRNO_SUCCESS

        clock_time_get = make_func(
            [wasmtime.ValType.i32(), wasmtime.ValType.i64(), wasmtime.ValType.i32()],
            [wasmtime.ValType.i32()],
            clock_time_get_impl,
        )

        # random_get(buf: i32, buf_len: i32) -> i32
        def random_get_impl(buf_ptr: int, buf_len: int) -> int:
            memory = get_memory()
            data = bytearray(buf_len)
            for i in range(buf_len):
                random_counter[0] = (
                    random_counter[0] * 1103515245 + 12345
                ) & 0xFFFFFFFF
                data[i] = random_counter[0] & 0xFF
            memory.write(self._store, bytes(data), buf_ptr)
            return ERRNO_SUCCESS

        random_get = make_func(
            [wasmtime.ValType.i32(), wasmtime.ValType.i32()],
            [wasmtime.ValType.i32()],
            random_get_impl,
        )

        # fd_write(fd: i32, iovs: i32, iovs_len: i32, nwritten: i32) -> i32
        def fd_write_impl(
            fd: int, iovs_ptr: int, iovs_len: int, nwritten_ptr: int
        ) -> int:
            if fd != 2:  # Only stderr
                return ERRNO_BADF

            memory = get_memory()
            total_written = 0

            for i in range(iovs_len):
                iov_ptr = iovs_ptr + i * 8
                iov_data = memory.read(self._store, iov_ptr, iov_ptr + 8)
                ptr = struct.unpack("<I", bytes(iov_data[0:4]))[0]
                length = struct.unpack("<I", bytes(iov_data[4:8]))[0]
                text_data = memory.read(self._store, ptr, ptr + length)
                os.write(2, bytes(text_data))
                total_written += length

            memory.write(self._store, struct.pack("<I", total_written), nwritten_ptr)
            return ERRNO_SUCCESS

        fd_write = make_func(
            [wasmtime.ValType.i32()] * 4,
            [wasmtime.ValType.i32()],
            fd_write_impl,
        )

        # environ_sizes_get(count: i32, size: i32) -> i32
        def environ_sizes_get_impl(count_ptr: int, size_ptr: int) -> int:
            memory = get_memory()
            memory.write(self._store, struct.pack("<I", 0), count_ptr)
            memory.write(self._store, struct.pack("<I", 0), size_ptr)
            return ERRNO_SUCCESS

        environ_sizes_get = make_func(
            [wasmtime.ValType.i32()] * 2,
            [wasmtime.ValType.i32()],
            environ_sizes_get_impl,
        )

        # environ_get(environ: i32, environ_buf: i32) -> i32
        def environ_get_impl(_e: int, _eb: int) -> int:
            return ERRNO_SUCCESS

        environ_get = make_func(
            [wasmtime.ValType.i32()] * 2,
            [wasmtime.ValType.i32()],
            environ_get_impl,
        )

        # fd_prestat_get(fd: i32, buf: i32) -> i32
        def fd_prestat_get_impl(_fd: int, _buf: int) -> int:
            return ERRNO_BADF

        fd_prestat_get = make_func(
            [wasmtime.ValType.i32()] * 2,
            [wasmtime.ValType.i32()],
            fd_prestat_get_impl,
        )

        # fd_prestat_dir_name(fd: i32, path: i32, path_len: i32) -> i32
        def fd_prestat_dir_name_impl(_fd: int, _p: int, _pl: int) -> int:
            return ERRNO_BADF

        fd_prestat_dir_name = make_func(
            [wasmtime.ValType.i32()] * 3,
            [wasmtime.ValType.i32()],
            fd_prestat_dir_name_impl,
        )

        # sched_yield() -> i32
        def sched_yield_impl() -> int:
            return ERRNO_SUCCESS

        sched_yield = make_func([], [wasmtime.ValType.i32()], sched_yield_impl)

        # proc_exit(code: i32) -> void
        def proc_exit_impl(code: int) -> None:
            raise RuntimeError(f"WASM called proc_exit({code})")

        proc_exit = make_func([wasmtime.ValType.i32()], [], proc_exit_impl)

        # Stub that returns ERRNO_NOSYS with custom signature
        def make_nosys_func(params: list[Any]) -> Any:
            def impl(*_args: Any) -> int:
                return ERRNO_NOSYS

            return make_func(params, [wasmtime.ValType.i32()], impl)

        # Define all WASI imports
        wasi = "wasi_snapshot_preview1"
        linker.define(self._store, wasi, "clock_time_get", clock_time_get)
        linker.define(self._store, wasi, "random_get", random_get)
        linker.define(self._store, wasi, "fd_write", fd_write)
        linker.define(self._store, wasi, "environ_sizes_get", environ_sizes_get)
        linker.define(self._store, wasi, "environ_get", environ_get)
        linker.define(self._store, wasi, "fd_prestat_get", fd_prestat_get)
        linker.define(self._store, wasi, "fd_prestat_dir_name", fd_prestat_dir_name)
        linker.define(self._store, wasi, "sched_yield", sched_yield)
        linker.define(self._store, wasi, "proc_exit", proc_exit)

        # Stub out remaining WASI functions with correct signatures
        i32 = wasmtime.ValType.i32()
        i64 = wasmtime.ValType.i64()

        # fd_read(fd: i32, iovs: i32, iovs_len: i32, nread: i32) -> i32
        linker.define(self._store, wasi, "fd_read", make_nosys_func([i32] * 4))
        # fd_close(fd: i32) -> i32
        linker.define(self._store, wasi, "fd_close", make_nosys_func([i32]))
        # fd_seek(fd: i32, offset: i64, whence: i32, newoffset: i32) -> i32
        linker.define(
            self._store, wasi, "fd_seek", make_nosys_func([i32, i64, i32, i32])
        )
        # fd_tell(fd: i32, offset: i32) -> i32
        linker.define(self._store, wasi, "fd_tell", make_nosys_func([i32] * 2))
        # fd_filestat_get(fd: i32, buf: i32) -> i32
        linker.define(self._store, wasi, "fd_filestat_get", make_nosys_func([i32] * 2))
        # fd_fdstat_get(fd: i32, buf: i32) -> i32
        linker.define(self._store, wasi, "fd_fdstat_get", make_nosys_func([i32] * 2))
        # fd_fdstat_set_flags(fd: i32, flags: i32) -> i32
        linker.define(
            self._store, wasi, "fd_fdstat_set_flags", make_nosys_func([i32] * 2)
        )
        # path_open(fd, dirflags, path, path_len, oflags, fs_rights_base: i64,
        #           fs_rights_inheriting: i64, fdflags, fd_out) -> i32
        linker.define(
            self._store,
            wasi,
            "path_open",
            make_nosys_func([i32, i32, i32, i32, i32, i64, i64, i32, i32]),
        )
        # path_filestat_get(fd, flags, path, path_len, buf) -> i32
        linker.define(
            self._store, wasi, "path_filestat_get", make_nosys_func([i32] * 5)
        )

    def _create_runtime(self) -> None:
        """Create the runtime from PCA bytes.

        Raises:
            RuntimeError: If runtime creation fails.
        """
        # Get memory and exports
        memory = self._instance.exports(self._store)["memory"]

        # Set trusted authorities first
        self._set_trusted_authorities(memory)

        # Check if PCA is provided
        if not self._config.pca_bytes:
            raise RuntimeError(
                "No PCA provided. Use Runtime.for_testing() for tests, or provide "
                "pca_bytes from an EphemeralAuthority."
            )

        # Get runtime creation function
        runtime_new_fn = self._instance.exports(self._store).get(
            "runtime_new_with_tools"
        )

        # If runtime_new_with_tools not available, try runtime_new
        if runtime_new_fn is None:
            runtime_new_fn = self._instance.exports(self._store)["runtime_new"]
            use_tools = False
        else:
            use_tools = True

        # Write PCA to WASM memory
        pca_ptr = self._alloc_and_write(memory, self._config.pca_bytes)
        pca_len = len(self._config.pca_bytes)

        if use_tools:
            # Write tools JSON to memory
            tools_bytes = self._config.tools_json.encode("utf-8")
            tools_ptr = self._alloc_and_write(memory, tools_bytes, pca_ptr + pca_len)
            tools_len = len(tools_bytes)

            self._runtime_id = runtime_new_fn(
                self._store, pca_ptr, pca_len, tools_ptr, tools_len
            )
        else:
            self._runtime_id = runtime_new_fn(self._store, pca_ptr, pca_len)

        if self._runtime_id == 0:
            # Get detailed error from WASM
            error_msg = self._get_last_error(memory)
            if error_msg:
                raise RuntimeError(f"Failed to create runtime: {error_msg}")
            else:
                raise RuntimeError(
                    "Failed to create runtime from PCA. "
                    "Ensure the PCA is signed by a trusted authority."
                )

    def _set_trusted_authorities(self, memory: Any) -> None:
        """Set trusted authorities in WASM runtime.

        Args:
            memory: WASM memory object.

        Raises:
            RuntimeError: If no trusted authorities are configured.
        """
        if not self._config.trusted_authorities:
            raise RuntimeError(
                "No trusted authorities configured. Either:\n"
                "1. Use Runtime.for_testing() which sets up authorities automatically\n"
                "2. Provide trusted_authorities in RuntimeConfig"
            )

        # Get set_trusted_authorities export
        set_auth_fn = self._instance.exports(self._store).get("set_trusted_authorities")
        if set_auth_fn is None:
            raise RuntimeError(
                "set_trusted_authorities export not found. "
                "Ensure you're using a compatible WASM build."
            )

        # Encode authorities as JSON array
        authorities_json = json.dumps(self._config.trusted_authorities)
        auth_bytes = authorities_json.encode("utf-8")
        auth_ptr = self._alloc_and_write(memory, auth_bytes)

        # Call WASM function
        count = set_auth_fn(self._store, auth_ptr, len(auth_bytes))
        if count == 0:
            raise RuntimeError(
                f"Failed to set trusted authorities. "
                f"Check public key format: {self._config.trusted_authorities}"
            )

    def _get_last_error(self, memory: Any) -> str:
        """Get the last error message from the WASM runtime.

        Args:
            memory: WASM memory object.

        Returns:
            Error message string, or empty string if no error.
        """
        get_error_fn = self._instance.exports(self._store).get("get_last_error")
        if get_error_fn is None:
            return ""

        # Allocate buffer for error message
        error_buffer_ptr = OUT_PTR  # Reuse output buffer area
        error_buffer_len = 1024

        # Call WASM function
        error_len = get_error_fn(self._store, error_buffer_ptr, error_buffer_len)
        if error_len == 0:
            return ""

        # Read error message from memory
        mem_data = memory.read(self._store, error_buffer_ptr, error_len)
        return mem_data.decode("utf-8", errors="replace")

    def set_audit_collector(self, collector: Any) -> None:
        """Set the audit collector for this runtime.

        The collector will receive audit log entries after each step.
        This is typically called by the Sandbox during initialization.

        Args:
            collector: An AuditCollector instance (or any object with
                drain_from_runtime method).
        """
        self._audit_collector = collector

    @property
    def last_stderr(self) -> str:
        """Get stderr from the last execution.

        Returns:
            Captured stderr output as a string.
        """
        return b"".join(self._stderr_chunks).decode("utf-8", errors="replace")

    @property
    def last_exit_code(self) -> int:
        """Get exit code from the last execution.

        Returns:
            Exit code (0 for success, non-zero for failure).
        """
        return self._last_exit_code

    @property
    def last_elapsed_ns(self) -> int | None:
        """Get wall-clock elapsed time from the last execution in nanoseconds.

        This is the total time from command creation to exit, including
        time spent waiting for host operations (sys time).

        Returns:
            Elapsed nanoseconds, or None if timing not available.
        """
        return self._last_elapsed_ns

    @property
    def last_user_time_ns(self) -> int | None:
        """Get user time from the last execution in nanoseconds.

        User time is the time spent executing inside the WASM runtime,
        not including time waiting for host operations like sleep or tool calls.

        Returns:
            User time in nanoseconds, or None if timing not available.
        """
        return self._last_user_time_ns

    @property
    def last_sys_time_ns(self) -> int | None:
        """Get sys time from the last execution in nanoseconds.

        Sys time is the time spent waiting for host operations like sleep,
        tool calls, and I/O. It's calculated as elapsed_ns - user_time_ns.

        Returns:
            Sys time in nanoseconds, or None if timing not available.
        """
        if self._last_elapsed_ns is not None and self._last_user_time_ns is not None:
            return self._last_elapsed_ns - self._last_user_time_ns
        return None

    def _format_stderr(self) -> str:
        """Format stderr for inclusion in error messages.

        Returns:
            Stripped stderr if non-empty, empty string otherwise.
        """
        stderr = b"".join(self._stderr_chunks).decode("utf-8", errors="replace")
        return stderr.strip()

    def _alloc_and_write(self, memory: Any, data: bytes, offset: int = 0) -> int:
        """Write data to WASM linear memory.

        Uses a simple bump allocator in the low memory region (1KB-16KB) that
        the WASM runtime expects for host-provided data. This matches the
        memory layout used by the Node.js test harness.

        Args:
            memory: WASM memory object.
            data: Data to write.
            offset: Hint for allocation offset (used for sequential allocations).

        Returns:
            Pointer to written data in WASM memory.
        """
        # Use low memory region (1KB-16KB) for host data
        # The WASM runtime expects data in this region, not at 64KB+
        # which conflicts with stack/heap
        if offset == 0:
            base_offset = 1024  # Start at 1KB like Node.js harness
        else:
            # Sequential allocation after previous data
            base_offset = offset

        mem_len = memory.data_len(self._store)
        if base_offset + len(data) > mem_len:
            # Grow memory if needed
            pages_needed = ((base_offset + len(data) - mem_len) // 65536) + 1
            memory.grow(self._store, pages_needed)

        # Write data using wasmtime's Memory.write()
        memory.write(self._store, data, base_offset)

        return base_offset

    def _read_memory(self, memory: Any, ptr: int, length: int) -> bytes:
        """Read data from WASM linear memory.

        Args:
            memory: WASM memory object.
            ptr: Pointer in WASM memory.
            length: Number of bytes to read.

        Returns:
            Bytes read from memory.
        """
        # Use wasmtime's Memory.read()
        return bytes(memory.read(self._store, ptr, ptr + length))

    def _drain_audit_buffer(self, max_bytes: int = 8192) -> str:
        """Drain audit log buffer from WASM runtime.

        This reads available audit log entries from the runtime's ring buffer
        and returns them as a JSONL string. The entries are removed from the
        buffer after reading.

        Args:
            max_bytes: Maximum bytes to read in one call.

        Returns:
            JSONL string of audit entries (newline-separated).
            Empty string if no entries available or exports not found.
        """
        if self._runtime_id == 0:
            return ""

        # Check if audit exports exist
        exports = self._instance.exports(self._store)
        audit_available_fn = exports.get("audit_available")
        audit_drain_fn = exports.get("audit_drain")

        if audit_available_fn is None or audit_drain_fn is None:
            return ""

        # Check how many bytes are available
        available = audit_available_fn(self._store, self._runtime_id)
        if available == 0:
            return ""

        # Drain audit logs using fixed AUDIT_PTR region
        memory = exports["memory"]
        to_read = min(available, max_bytes)

        n = audit_drain_fn(self._store, self._runtime_id, AUDIT_PTR, to_read)
        if n == 0:
            return ""

        data = self._read_memory(memory, AUDIT_PTR, n)
        return data.decode("utf-8", errors="replace")

    def _validate_tool_call(
        self, method: str, params: dict[str, Any], consume: bool = True
    ) -> MethodCapability:
        """Validate a tool call against capabilities.

        Finds a capability that authorizes the call and optionally consumes
        one call from its budget (if max_calls is set).

        Args:
            method: The method being called.
            params: The call parameters.
            consume: If True, decrement the call counter for the matching capability.

        Returns:
            The capability that authorized the call.

        Raises:
            CapabilityError: If no capability authorizes this call.
            CallLimitExceededError: If all matching capabilities are exhausted.
        """
        exhausted_caps: list[MethodCapability] = []
        # Track constraint violations for pattern-matching capabilities
        # This allows us to report specific constraint failures rather than
        # the generic "no capability authorizes" message
        constraint_violations: list[tuple[str, str]] = []  # (pattern, error_msg)

        for cap in self._capabilities:
            try:
                cap.validate_call(method, params)
                # Pattern and constraints match - check call limit
                cap_key = cap.key()

                if cap_key in self._call_counts:
                    remaining = self._call_counts[cap_key]
                    if remaining <= 0:
                        # This capability is exhausted, try next
                        exhausted_caps.append(cap)
                        continue
                    if consume:
                        self._call_counts[cap_key] = remaining - 1

                return cap  # Found a capability that allows this call
            except CapabilityError as e:
                # Check if pattern matched but constraints failed
                error_msg = str(e)
                if "does not match pattern" not in error_msg:
                    # Pattern matched but constraints failed - record the violation
                    constraint_violations.append((cap.method_pattern, error_msg))
                continue

        # Check if we had matching caps but they were all exhausted
        if exhausted_caps:
            # Report the first exhausted capability
            cap = exhausted_caps[0]
            raise CallLimitExceededError(cap.key(), cap.max_calls or 0)

        # If any pattern matched but constraints failed, report the specific violations
        if constraint_violations:
            if len(constraint_violations) == 1:
                pattern, error = constraint_violations[0]
                raise CapabilityError(
                    f"Method '{method}' matched pattern '{pattern}' but failed constraint check: {error}"
                )
            else:
                violations_str = "; ".join(
                    f"'{p}': {e}" for p, e in constraint_violations
                )
                raise CapabilityError(
                    f"Method '{method}' matched patterns but failed constraint checks: {violations_str}"
                )

        # No capability matched at all
        raise CapabilityError(
            f"No capability authorizes method '{method}'. "
            f"Available patterns: {[c.method_pattern for c in self._capabilities]}"
        )

    def _error_response(
        self, op_id: int, runtime_id: int, code: str, message: str
    ) -> dict[str, Any]:
        """Create an error response dict."""
        return {
            "id": op_id,
            "runtime_id": runtime_id,
            "result": {"type": "error", "code": code, "message": message},
        }

    def _ok_response(
        self, op_id: int, runtime_id: int, result_type: str, **kwargs: Any
    ) -> dict[str, Any]:
        """Create a success response dict."""
        return {
            "id": op_id,
            "runtime_id": runtime_id,
            "result": {"type": result_type, **kwargs},
        }

    def _handle_non_tool_op(
        self, op_type: str, op_id: int, runtime_id: int, request: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Handle non-tool host operations.

        Returns response dict, or None if this op_type is not handled here.
        """
        if op_type == "wake_at":
            current_nanos = int(time.time() * 1_000_000_000)
            return self._ok_response(
                op_id, runtime_id, "woke_at", current_time_nanos=current_nanos
            )

        if op_type == "output":
            stream = request.get("stream", 1)
            data_b64 = request.get("data", "")
            data = base64.b64decode(data_b64) if data_b64 else b""
            if stream == 1:  # stdout
                self._output_chunks.append(data)
                on_output = getattr(self, "_on_output", None)
                if on_output is not None:
                    on_output(data.decode("utf-8", errors="replace"))
            elif stream == 2:  # stderr
                self._stderr_chunks.append(data)
            return self._ok_response(op_id, runtime_id, "output_ack")

        if op_type == "command_exit":
            self._last_exit_code = request.get("code", 0)
            self._last_elapsed_ns = request.get("elapsed_ns")
            self._last_user_time_ns = request.get("user_time_ns")
            return self._ok_response(op_id, runtime_id, "exit_ack")

        if op_type == "read_stdin":
            if self._stdin_pos >= len(self._stdin_data):
                return self._ok_response(
                    op_id, runtime_id, "stdin_data", data="", eof=True
                )
            remaining = self._stdin_data[self._stdin_pos :]
            self._stdin_pos = len(self._stdin_data)
            return self._ok_response(
                op_id,
                runtime_id,
                "stdin_data",
                data=base64.b64encode(remaining).decode("ascii"),
                eof=True,
            )

        if op_type == "get_timestamp":
            current_nanos = int(time.time() * 1_000_000_000)
            return self._ok_response(
                op_id, runtime_id, "timestamp", nanos=current_nanos
            )

        if op_type == "vfs_read":
            return self._error_response(
                op_id, runtime_id, "unsupported", "VFS read not supported from host"
            )

        if op_type == "delegate":
            return self._error_response(
                op_id, runtime_id, "unsupported", "Delegation not implemented"
            )

        return None  # Not handled - caller should handle this op_type

    def _handle_host_op(
        self, op_id: int, runtime_id: int, request: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle a host operation from the runtime (sync version).

        Args:
            op_id: Operation ID for correlation.
            runtime_id: Runtime that issued the operation.
            request: The host operation request.

        Returns:
            Response to send back to the runtime.
        """
        op_type = request.get("type", "")

        # Handle non-tool operations via shared helper
        result = self._handle_non_tool_op(op_type, op_id, runtime_id, request)
        if result is not None:
            return result

        if op_type == "tool_call":
            tool = request.get("tool", "")
            params = request.get("params", {})

            # Enforce capabilities
            try:
                self._validate_tool_call(tool, params)
            except CapabilityError as e:
                return self._error_response(
                    op_id, runtime_id, "permission_denied", str(e)
                )

            if self._tool_handler is None:
                return self._error_response(
                    op_id, runtime_id, "unsupported", "No tool handler configured"
                )

            try:
                result = self._tool_handler(tool, params)

                # Detect unawaited coroutine (async handler with sync execute)
                if inspect.iscoroutine(result):
                    result.close()
                    raise RuntimeError(
                        f"Tool handler returned a coroutine for '{tool}'. "
                        "Use execute_async() instead of execute() for async handlers."
                    )

                return self._ok_response(
                    op_id, runtime_id, "tool_result", result=result
                )
            except Exception as e:
                params_str = str(params)[:200] + (
                    "..." if len(str(params)) > 200 else ""
                )
                error_msg = f"Tool '{tool}' failed: {e}\nParameters: {params_str}"
                return self._error_response(op_id, runtime_id, "internal", error_msg)

        # Unknown operation
        return self._error_response(
            op_id, runtime_id, "unsupported", f"Unknown operation: {op_type}"
        )

    def execute(
        self,
        command: str,
        on_output: Callable[[str], None] | None = None,
        stdin: str | bytes | None = None,
    ) -> str:
        """Execute a shell command in the sandbox.

        This runs the command through the embedded QuickJS interpreter
        with access to registered tools (subject to capability enforcement).

        This method is thread-safe and uses a lock to serialize WASM access.
        LangGraph/LangChain may call tools from ThreadPoolExecutor threads.

        Args:
            command: Shell command to execute.
            on_output: Optional callback for streaming output. Called with each
                chunk of stdout as it becomes available.
            stdin: Optional data to provide on stdin. Use this to pipe large
                scripts to `sh` without hitting command size limits.

        Returns:
            Command output as string.

        Raises:
            RuntimeError: If execution fails.
            CapabilityError: If a tool call is not authorized.

        Example::

            # Stream output
            def stream_handler(chunk: str):
                print(chunk, end="", flush=True)
            result = runtime.execute("echo hello", on_output=stream_handler)

            # Pipe script to sh (bypasses command size limit)
            long_script = "echo line1\\necho line2\\n..."
            result = runtime.execute("sh", stdin=long_script)
        """
        if self._runtime_id == 0:
            raise RuntimeError("Runtime not initialized")

        # Acquire lock for thread-safe WASM access (wasmtime Store is NOT thread-safe)
        with self._lock:
            return self._execute_impl(command, on_output, stdin)

    def _execute_impl(
        self,
        command: str,
        on_output: Callable[[str], None] | None = None,
        stdin: str | bytes | None = None,
    ) -> str:
        """Internal execute implementation (lock must be held by caller)."""
        # Clear output buffers
        self._output_chunks.clear()
        self._stderr_chunks.clear()

        # Set up stdin buffer
        if stdin is None:
            self._stdin_data = b""
        elif isinstance(stdin, str):
            self._stdin_data = stdin.encode("utf-8")
        else:
            self._stdin_data = stdin
        self._stdin_pos = 0

        # Store streaming callback for use in _handle_host_op
        self._on_output = on_output

        # Get WASM exports
        memory = self._instance.exports(self._store)["memory"]
        cmd_create_fn = self._instance.exports(self._store)["cmd_create"]
        runtime_step_fn = self._instance.exports(self._store)["runtime_step"]
        submit_fn = self._instance.exports(self._store)["submit"]

        # Create command using fixed memory layout
        cmd_bytes = command.encode("utf-8")
        if len(cmd_bytes) > OUT_PTR - CMD_PTR:
            raise RuntimeError(
                f"Command too long: {len(cmd_bytes)} bytes (max {OUT_PTR - CMD_PTR})"
            )
        memory.write(self._store, cmd_bytes, CMD_PTR)
        cmd_handle = cmd_create_fn(
            self._store, self._runtime_id, CMD_PTR, len(cmd_bytes)
        )

        if cmd_handle == 0:
            raise RuntimeError(f"Failed to create command: {command}")

        # Stepping loop
        steps = 0
        while steps < self._config.max_steps:
            steps += 1

            # Step the runtime
            n = runtime_step_fn(
                self._store, self._runtime_id, OUT_PTR, OUTPUT_BUFFER_SIZE
            )

            # Drain audit buffer after each step (if collector configured)
            if self._audit_collector is not None:
                self._audit_collector.drain_from_runtime(self)

            if n == 0:
                raise RuntimeError("runtime_step returned 0 (runtime not found)")

            # Parse response
            response_bytes = self._read_memory(memory, OUT_PTR, n)
            response = json.loads(response_bytes.decode("utf-8"))

            status = response.get("status", "running")

            # Handle status
            if isinstance(status, dict):
                # Status with message (error or panic)
                stderr = self._format_stderr()
                if "error" in status:
                    err: Any = status["error"]  # pyright: ignore[reportUnknownVariableType]
                    msg = f"Runtime error: {err.get('message', status) if isinstance(err, dict) else status}"  # pyright: ignore[reportUnknownMemberType]
                    if stderr:
                        msg += f"\n\nStderr output:\n{stderr}"
                    raise RuntimeError(msg)
                if "panic" in status:
                    pan: Any = status["panic"]  # pyright: ignore[reportUnknownVariableType]
                    msg = f"Runtime panic: {pan.get('message', status) if isinstance(pan, dict) else status}"  # pyright: ignore[reportUnknownMemberType]
                    if stderr:
                        msg += f"\n\nStderr output:\n{stderr}"
                    raise RuntimeError(msg)
            elif status == "all_done":
                break
            elif status == "error":
                stderr = self._format_stderr()
                msg = "Runtime encountered an error"
                if stderr:
                    msg += f"\n\nStderr output:\n{stderr}"
                raise RuntimeError(msg)
            elif status == "panic":
                stderr = self._format_stderr()
                msg = "Runtime panicked"
                if stderr:
                    msg += f"\n\nStderr output:\n{stderr}"
                raise RuntimeError(msg)

            # Process host operations
            host_ops = response.get("host_ops", [])
            if not host_ops:
                # No host ops and not done - this shouldn't happen
                if status == "all_blocked":
                    raise RuntimeError("Runtime blocked with no pending host ops")
                continue

            # Handle each host op and collect results
            # Tool results may be chunked into multiple responses
            all_results: list[dict[str, Any]] = []
            for op in host_ops:
                op_id = op.get("id", 0)
                runtime_id = op.get("runtime_id", self._runtime_id)
                request = op.get("request", {})
                result = self._handle_host_op(op_id, runtime_id, request)

                # Check if this is a tool result that needs chunking
                if result.get("result", {}).get(
                    "type"
                ) == "tool_result" and "result" in result.get("result", {}):
                    tool_result = result["result"]["result"]
                    chunked = _create_tool_result_responses(
                        op_id,
                        runtime_id,
                        tool_result,
                        chunk_size=self._config.tool_result_chunk_size,
                        max_size=self._config.max_tool_result_size,
                    )
                    all_results.extend(chunked)
                else:
                    all_results.append(result)

            # Submit results, chunking if necessary to stay within buffer
            # For large tool results, submit one chunk at a time
            for result in all_results:
                results_json = json.dumps([result]).encode("utf-8")
                if len(results_json) > OUTPUT_BUFFER_SIZE:
                    raise RuntimeError(
                        f"Single result too large: {len(results_json)} bytes"
                    )
                memory.write(self._store, results_json, SUBMIT_PTR)
                submitted = submit_fn(self._store, SUBMIT_PTR, len(results_json))

                # For tool_result_chunk with eof=false, Rust returns 0 because
                # the chunk is accumulated but not yet completed. This is expected.
                # Only the final chunk (eof=true) counts as "processed".
                inner = result.get("result", {})
                is_non_final_chunk = inner.get(
                    "type"
                ) == "tool_result_chunk" and not inner.get("eof", True)
                if submitted == 0 and not is_non_final_chunk:
                    raise RuntimeError("Failed to submit host op results")

        if steps >= self._config.max_steps:
            raise RuntimeError(
                f"Execution exceeded maximum steps ({self._config.max_steps})"
            )

        # Combine output chunks
        output = b"".join(self._output_chunks).decode("utf-8", errors="replace")
        return output

    async def execute_async(
        self,
        command: str,
        on_output: Callable[[str], None] | None = None,
        stdin: str | bytes | None = None,
    ) -> str:
        """Execute a shell command in the sandbox with async tool handler support.

        This is the async version of execute() that properly awaits async tool
        handlers. Use this when your tool_handler is an async function.

        The WASM stepping loop itself is fast (microseconds per step). The async
        support is for tool handlers that perform I/O operations.

        This method is thread-safe and uses a lock to serialize WASM access.

        Args:
            command: Shell command to execute.
            on_output: Optional callback for streaming output. Called with each
                chunk of stdout as it becomes available.
            stdin: Optional data to provide on stdin. Useful for piping
                large scripts to `sh` or `node` without hitting command size limits.

        Returns:
            Command output as string.

        Raises:
            RuntimeError: If execution fails.
            CapabilityError: If a tool call is not authorized.
        """
        if self._runtime_id == 0:
            raise RuntimeError("Runtime not initialized")

        # Acquire lock for thread-safe WASM access
        # Note: Using regular lock is fine here since WASM calls are fast.
        # The lock only serializes WASM stepping, not the async tool handlers.
        with self._lock:
            return await self._execute_async_impl(command, on_output, stdin)

    async def _execute_async_impl(
        self,
        command: str,
        on_output: Callable[[str], None] | None = None,
        stdin: str | bytes | None = None,
    ) -> str:
        """Internal async execute implementation (lock must be held by caller)."""
        # Clear output buffers
        self._output_chunks.clear()
        self._stderr_chunks.clear()

        # Set up stdin buffer
        if stdin is None:
            self._stdin_data = b""
        elif isinstance(stdin, str):
            self._stdin_data = stdin.encode("utf-8")
        else:
            self._stdin_data = stdin
        self._stdin_pos = 0

        # Store streaming callback for use in _handle_host_op_async
        self._on_output = on_output

        # Get WASM exports
        memory = self._instance.exports(self._store)["memory"]
        cmd_create_fn = self._instance.exports(self._store)["cmd_create"]
        runtime_step_fn = self._instance.exports(self._store)["runtime_step"]
        submit_fn = self._instance.exports(self._store)["submit"]

        # Create command using fixed memory layout
        cmd_bytes = command.encode("utf-8")
        if len(cmd_bytes) > OUT_PTR - CMD_PTR:
            raise RuntimeError(
                f"Command too long: {len(cmd_bytes)} bytes (max {OUT_PTR - CMD_PTR})"
            )
        memory.write(self._store, cmd_bytes, CMD_PTR)
        cmd_handle = cmd_create_fn(
            self._store, self._runtime_id, CMD_PTR, len(cmd_bytes)
        )

        if cmd_handle == 0:
            raise RuntimeError(f"Failed to create command: {command}")

        # Stepping loop
        steps = 0
        while steps < self._config.max_steps:
            steps += 1

            # Step the runtime
            n = runtime_step_fn(
                self._store, self._runtime_id, OUT_PTR, OUTPUT_BUFFER_SIZE
            )

            # Drain audit buffer after each step (if collector configured)
            if self._audit_collector is not None:
                self._audit_collector.drain_from_runtime(self)

            if n == 0:
                raise RuntimeError("runtime_step returned 0 (runtime not found)")

            # Parse response
            response_bytes = self._read_memory(memory, OUT_PTR, n)
            response = json.loads(response_bytes.decode("utf-8"))

            status = response.get("status", "running")

            # Handle status
            if isinstance(status, dict):
                # Status with message (error or panic)
                stderr = self._format_stderr()
                if "error" in status:
                    err: Any = status["error"]  # pyright: ignore[reportUnknownVariableType]
                    msg = f"Runtime error: {err.get('message', status) if isinstance(err, dict) else status}"  # pyright: ignore[reportUnknownMemberType]
                    if stderr:
                        msg += f"\n\nStderr output:\n{stderr}"
                    raise RuntimeError(msg)
                if "panic" in status:
                    pan: Any = status["panic"]  # pyright: ignore[reportUnknownVariableType]
                    msg = f"Runtime panic: {pan.get('message', status) if isinstance(pan, dict) else status}"  # pyright: ignore[reportUnknownMemberType]
                    if stderr:
                        msg += f"\n\nStderr output:\n{stderr}"
                    raise RuntimeError(msg)
            elif status == "all_done":
                break
            elif status == "error":
                stderr = self._format_stderr()
                msg = "Runtime encountered an error"
                if stderr:
                    msg += f"\n\nStderr output:\n{stderr}"
                raise RuntimeError(msg)
            elif status == "panic":
                stderr = self._format_stderr()
                msg = "Runtime panicked"
                if stderr:
                    msg += f"\n\nStderr output:\n{stderr}"
                raise RuntimeError(msg)

            # Process host operations
            host_ops = response.get("host_ops", [])
            if not host_ops:
                # No host ops and not done - this shouldn't happen
                if status == "all_blocked":
                    raise RuntimeError("Runtime blocked with no pending host ops")
                continue

            # Handle each host op (with async support)
            # Tool results may be chunked into multiple responses
            all_results: list[dict[str, Any]] = []
            for op in host_ops:
                op_id = op.get("id", 0)
                runtime_id = op.get("runtime_id", self._runtime_id)
                request = op.get("request", {})
                result = await self._handle_host_op_async(op_id, runtime_id, request)

                # Check if this is a tool result that needs chunking
                if result.get("result", {}).get(
                    "type"
                ) == "tool_result" and "result" in result.get("result", {}):
                    tool_result = result["result"]["result"]
                    chunked = _create_tool_result_responses(
                        op_id,
                        runtime_id,
                        tool_result,
                        chunk_size=self._config.tool_result_chunk_size,
                        max_size=self._config.max_tool_result_size,
                    )
                    all_results.extend(chunked)
                else:
                    all_results.append(result)

            # Submit results, chunking if necessary to stay within buffer
            # For large tool results, submit one chunk at a time
            for result in all_results:
                results_json = json.dumps([result]).encode("utf-8")
                if len(results_json) > OUTPUT_BUFFER_SIZE:
                    raise RuntimeError(
                        f"Single result too large: {len(results_json)} bytes"
                    )
                memory.write(self._store, results_json, SUBMIT_PTR)
                submitted = submit_fn(self._store, SUBMIT_PTR, len(results_json))

                # For tool_result_chunk with eof=false, Rust returns 0 because
                # the chunk is accumulated but not yet completed. This is expected.
                # Only the final chunk (eof=true) counts as "processed".
                inner = result.get("result", {})
                is_non_final_chunk = inner.get(
                    "type"
                ) == "tool_result_chunk" and not inner.get("eof", True)
                if submitted == 0 and not is_non_final_chunk:
                    raise RuntimeError("Failed to submit host op results")

        if steps >= self._config.max_steps:
            raise RuntimeError(
                f"Execution exceeded maximum steps ({self._config.max_steps})"
            )

        # Combine output chunks
        output = b"".join(self._output_chunks).decode("utf-8", errors="replace")
        return output

    async def _handle_host_op_async(
        self, op_id: int, runtime_id: int, request: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle a host operation with async tool handler support.

        This is the async version of _handle_host_op that awaits async tool handlers.

        Args:
            op_id: Operation ID for correlation.
            runtime_id: Runtime that issued the operation.
            request: The host operation request.

        Returns:
            Response to send back to the runtime.
        """
        op_type = request.get("type", "")

        # Handle non-tool operations via shared helper
        result = self._handle_non_tool_op(op_type, op_id, runtime_id, request)
        if result is not None:
            return result

        if op_type == "tool_call":
            tool = request.get("tool", "")
            params = request.get("params", {})

            # Enforce capabilities
            try:
                self._validate_tool_call(tool, params)
            except CapabilityError as e:
                return self._error_response(
                    op_id, runtime_id, "permission_denied", str(e)
                )

            if self._tool_handler is None:
                return self._error_response(
                    op_id, runtime_id, "unsupported", "No tool handler configured"
                )

            try:
                # Call the handler - it might be async or return a coroutine
                result = self._tool_handler(tool, params)

                # If result is awaitable (coroutine, Task, Future, etc.), await it
                if inspect.isawaitable(result):
                    result = await result

                return self._ok_response(
                    op_id, runtime_id, "tool_result", result=result
                )
            except Exception as e:
                params_str = str(params)[:200] + (
                    "..." if len(str(params)) > 200 else ""
                )
                error_msg = f"Tool '{tool}' failed: {e}\nParameters: {params_str}"
                return self._error_response(op_id, runtime_id, "internal", error_msg)

        # Unknown operation
        return self._error_response(
            op_id, runtime_id, "unsupported", f"Unknown host operation: {op_type}"
        )

    def can_call(self, method: str, params: dict[str, Any] | None = None) -> bool:
        """Check if a method call would be allowed.

        This checks both capability matching AND remaining call budget.
        It does NOT consume a call - use this for introspection.

        Args:
            method: The method to check.
            params: Optional parameters (for constraint checking).

        Returns:
            True if some capability would allow this call (with remaining budget).
        """
        if params is None:
            params = {}

        try:
            self._validate_tool_call(method, params, consume=False)
            return True
        except (CapabilityError, CallLimitExceededError):
            return False

    def get_capabilities(self) -> list[MethodCapability]:
        """Get all capabilities for this runtime.

        Returns:
            List of method capabilities.
        """
        return list(self._capabilities)

    def get_remaining_calls(self, capability_key: str) -> int | None:
        """Get remaining calls for a capability.

        Args:
            capability_key: The capability key (e.g., "cap:method:stripe/**").

        Returns:
            Remaining calls, or None if capability has no limit or doesn't exist.

        Example::

            runtime = Runtime(config)
            remaining = runtime.get_remaining_calls("cap:method:stripe/charges/*")
            print(f"Can make {remaining} more Stripe charges")
        """
        return self._call_counts.get(capability_key)

    def get_call_counts(self) -> dict[str, int]:
        """Get remaining call counts for all limited capabilities.

        Returns:
            Dict mapping capability key -> remaining calls.
            Only includes capabilities that have max_calls set.

        Example::

            runtime = Runtime(config)
            for key, remaining in runtime.get_call_counts().items():
                print(f"{key}: {remaining} calls remaining")
        """
        return dict(self._call_counts)

    def register_tools(self, tools: list[dict[str, Any]]) -> None:
        """Register available tools with the runtime.

        Tools are registered via host operation and converted to JS interfaces
        for the embedded QuickJS interpreter.

        Args:
            tools: List of tool definitions with name, description, and schema.
        """
        # Tools are passed at runtime creation via runtime_new_with_tools
        # This method is kept for API compatibility but is a no-op
        pass

    def __del__(self) -> None:
        """Clean up runtime resources."""
        if (
            self._runtime_id != 0
            and self._store is not None
            and self._instance is not None
        ):
            try:
                destroy_fn = self._instance.exports(self._store).get("runtime_destroy")
                if destroy_fn is not None:
                    destroy_fn(self._store, self._runtime_id)
            except Exception:
                pass  # Ignore errors during cleanup
