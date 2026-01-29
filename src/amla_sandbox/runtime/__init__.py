"""WASM runtime integration via wasmtime.

This module provides the core runtime that:
- Loads and executes amla-sandbox.wasm via wasmtime
- Implements the stepping protocol (runtime_step/submit)
- Handles host operations (tool calls, I/O, timestamps)
- Manages capability enforcement

Architecture::

    ┌─────────────────────────────────────────┐
    │        Python Host (this module)         │
    │  - Drives stepping loop                  │
    │  - Routes host ops to handlers           │
    │  - Enforces capabilities                 │
    └────────────────┬────────────────────────┘
                     │ wasmtime
    ┌────────────────▼────────────────────────┐
    │        amla-sandbox.wasm                 │
    │  - QuickJS embedded                      │
    │  - Coroutine scheduler                   │
    │  - VFS, shell commands                   │
    └─────────────────────────────────────────┘
"""

from .wasm import Runtime, RuntimeConfig, RuntimeError, RuntimeStatus

__all__ = [
    "Runtime",
    "RuntimeConfig",
    "RuntimeError",
    "RuntimeStatus",
]
