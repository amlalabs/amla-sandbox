#!/usr/bin/env python3
"""Call budget enforcement example.

This example demonstrates:
1. max_calls limits - cap how many times a tool can be called
2. Budget tracking - introspect remaining calls
3. Budget exhaustion - graceful handling when budget runs out
4. Multi-capability fallback - broader capability as backup

Use cases:
- Rate limiting expensive API calls (e.g., max 100 GPT-4 calls)
- Preventing runaway loops (e.g., max 1000 tool invocations)
- Cost control (e.g., max $50 worth of API calls)
"""

from amla_sandbox import (
    MethodCapability,
    Sandbox,
    ToolDefinition,
)

# =============================================================================
# Setup: Tools and Mock Handler
# =============================================================================

# Track actual calls for demonstration
call_log: list[str] = []


def tool_handler(method: str, params: dict[str, object]) -> dict[str, object]:
    """Mock tool handler that logs calls."""
    call_log.append(method)
    return {"status": "ok", "call_number": len(call_log)}


# Define a simple counter tool
counter_tool = ToolDefinition(
    name="counter.increment",
    description="Increment a counter",
    parameters={"type": "object", "properties": {"amount": {"type": "integer"}}},
)

api_tool = ToolDefinition(
    name="api.call",
    description="Make an API call",
    parameters={"type": "object", "properties": {"endpoint": {"type": "string"}}},
)


# =============================================================================
# Example 1: Basic max_calls Enforcement
# =============================================================================

print("=" * 60)
print("Example 1: Basic Call Budget")
print("=" * 60)
print()

# Create capability with 3-call budget
limited_cap = MethodCapability(
    method_pattern="**",  # Match all methods
    max_calls=3,
)

print(f"Capability: {limited_cap.key()}")
print(f"Budget: {limited_cap.max_calls} calls")
print()

# Create sandbox with limited capability
sandbox = Sandbox(
    tools=[counter_tool],
    capabilities=[limited_cap],
    tool_handler=tool_handler,
)

# Check initial budget
print("Initial budget:")
print(f"  Remaining calls: {sandbox.get_remaining_calls(limited_cap.key())}")
print()

# Make calls until budget exhausted
print("Making calls:")
for i in range(1, 5):
    remaining = sandbox.get_remaining_calls(limited_cap.key())
    can_call = sandbox.can_call("mcp:counter.increment")

    print(f"  Call #{i}: remaining={remaining}, can_call={can_call}")

    if can_call:
        try:
            result = sandbox.execute(
                f"const r = await counter_increment({{amount: {i}}}); console.log(JSON.stringify(r));"
            )
            print(f"    Result: {result.strip()}")
        except Exception as e:
            print(f"    Error: {e}")
    else:
        print("    Skipped (budget exhausted)")

print()
print(f"Final remaining: {sandbox.get_remaining_calls(limited_cap.key())}")
print(f"Total calls made: {len(call_log)}")
print()


# =============================================================================
# Example 2: Budget Introspection
# =============================================================================

print("=" * 60)
print("Example 2: Budget Introspection")
print("=" * 60)
print()

call_log.clear()

# Multiple capabilities with different budgets
read_cap = MethodCapability(
    method_pattern="data/read",
    max_calls=100,
)

write_cap = MethodCapability(
    method_pattern="data/write",
    max_calls=10,
)

unlimited_cap = MethodCapability(
    method_pattern="logs/**",
    # No max_calls = unlimited
)

sandbox = Sandbox(
    tools=[],
    capabilities=[read_cap, write_cap, unlimited_cap],
    tool_handler=tool_handler,
)

print("All budgets:")
budgets = sandbox.get_call_counts()
for key, remaining in budgets.items():
    print(f"  {key}: {remaining} calls remaining")

print()
print("Note: 'logs/**' not shown (unlimited)")
print()

# Check specific capability
print("Checking specific capabilities:")
print(f"  data/read: {sandbox.get_remaining_calls(read_cap.key())} remaining")
print(f"  data/write: {sandbox.get_remaining_calls(write_cap.key())} remaining")
print(
    f"  logs/**: {sandbox.get_remaining_calls(unlimited_cap.key())} (None = unlimited)"
)
print()


# =============================================================================
# Example 3: Graceful Budget Exhaustion
# =============================================================================

print("=" * 60)
print("Example 3: Handling Budget Exhaustion")
print("=" * 60)
print()

call_log.clear()

# Capability that runs out quickly
tiny_cap = MethodCapability(
    method_pattern="**",
    max_calls=2,
)

sandbox = Sandbox(
    tools=[counter_tool],
    capabilities=[tiny_cap],
    tool_handler=tool_handler,
)


def safe_call(sandbox: Sandbox, code: str, description: str, method: str) -> None:
    """Make a call with proper pre-flight budget check.

    Note: sandbox.execute() doesn't raise CallLimitExceededError directly.
    Instead, the error is communicated to the JavaScript code via promise
    rejection. Use can_call() to pre-check budget before executing.
    """
    remaining = sandbox.get_remaining_calls(tiny_cap.key())
    print(f"  {description} (budget: {remaining})")

    # Pre-flight check - this is the proper pattern for budget enforcement
    if not sandbox.can_call(method):
        if remaining == 0:
            print(f"    Budget exhausted for: {tiny_cap.key()}")
            print(f"    Max was: {tiny_cap.max_calls} calls")
        else:
            print("    Call not authorized")
        return

    result = sandbox.execute(code)
    print(f"    Success: {result.strip()}")


safe_call(
    sandbox,
    "console.log(await counter_increment({amount: 1}));",
    "First call",
    "mcp:counter.increment",
)
safe_call(
    sandbox,
    "console.log(await counter_increment({amount: 2}));",
    "Second call",
    "mcp:counter.increment",
)
safe_call(
    sandbox,
    "console.log(await counter_increment({amount: 3}));",
    "Third call",
    "mcp:counter.increment",
)
print()


# =============================================================================
# Example 4: Multi-Capability Fallback Strategy
# =============================================================================

print("=" * 60)
print("Example 4: Fallback to Broader Capability")
print("=" * 60)
print()

call_log.clear()

# Scenario: Specific capability runs out, falls back to broader one
specific_cap = MethodCapability(
    method_pattern="mcp:counter.increment",  # Exact match
    max_calls=2,
)

broad_cap = MethodCapability(
    method_pattern="**",  # Matches everything
    max_calls=100,
)

sandbox = Sandbox(
    tools=[counter_tool],
    capabilities=[specific_cap, broad_cap],  # Order matters for selection
    tool_handler=tool_handler,
)

print("Strategy: Use specific cap first, fall back to broad cap")
print()
print(
    f"Specific ({specific_cap.key()}): {sandbox.get_remaining_calls(specific_cap.key())} calls"
)
print(
    f"Broad ({broad_cap.key()}): {sandbox.get_remaining_calls(broad_cap.key())} calls"
)
print()

# The runtime will use the first matching capability
# When specific is exhausted, it falls back to broad
print("Making 4 calls:")
for i in range(1, 5):
    specific_remaining = sandbox.get_remaining_calls(specific_cap.key())
    broad_remaining = sandbox.get_remaining_calls(broad_cap.key())
    print(f"  Call #{i}: specific={specific_remaining}, broad={broad_remaining}")

    # Note: We don't need to catch CallLimitExceededError here because
    # the broad capability will be used as a fallback when specific is exhausted
    result = sandbox.execute(f"console.log(await counter_increment({{amount: {i}}}));")
    print(f"    OK: {result.strip()}")

print()
print("Final budgets:")
print(f"  Specific: {sandbox.get_remaining_calls(specific_cap.key())} (exhausted)")
print(f"  Broad: {sandbox.get_remaining_calls(broad_cap.key())} (still has budget)")
print()


# =============================================================================
# Example 5: Pre-flight Budget Check
# =============================================================================

print("=" * 60)
print("Example 5: Pre-flight Budget Check")
print("=" * 60)
print()

call_log.clear()

cap = MethodCapability(method_pattern="**", max_calls=5)
sandbox = Sandbox(
    tools=[counter_tool],
    capabilities=[cap],
    tool_handler=tool_handler,
)


def execute_with_preflight(sandbox: Sandbox, method: str, code: str) -> str | None:
    """Check budget before executing - good DX pattern."""
    # Pre-flight check
    if not sandbox.can_call(method):
        remaining = sandbox.get_remaining_calls(cap.key())
        if remaining == 0:
            print(f"  Skipping {method}: budget exhausted")
        else:
            print(f"  Skipping {method}: not authorized")
        return None

    # Execute with confidence
    return sandbox.execute(code)


# Simulate an agent loop that respects budgets
print("Agent loop with budget awareness:")
operations = [
    ("mcp:counter.increment", "await counter_increment({amount: 1})"),
    ("mcp:counter.increment", "await counter_increment({amount: 2})"),
    ("mcp:counter.increment", "await counter_increment({amount: 3})"),
    ("mcp:counter.increment", "await counter_increment({amount: 4})"),
    ("mcp:counter.increment", "await counter_increment({amount: 5})"),
    ("mcp:counter.increment", "await counter_increment({amount: 6})"),
    ("mcp:counter.increment", "await counter_increment({amount: 7})"),
]

for method, code in operations:
    result = execute_with_preflight(sandbox, method, f"console.log({code});")
    if result:
        print(f"  {method}: {result.strip()}")

print()
print(f"Completed {len(call_log)} of {len(operations)} operations")
print()


# =============================================================================
# Summary
# =============================================================================

print("=" * 60)
print("Summary: Call Budget Best Practices")
print("=" * 60)
print("""
1. Set max_calls on capabilities to limit expensive operations
2. Use get_remaining_calls() to check budget before loops
3. Use can_call() for quick pre-flight authorization + budget check
4. Layer capabilities: specific (low budget) + broad (high budget)
5. Agents can introspect budgets and plan accordingly

Note: sandbox.execute() doesn't raise Python exceptions for budget
exhaustion. Budget enforcement happens inside the WASM runtime and
errors are communicated to JavaScript via promise rejection.
Use can_call() to pre-check before executing.
""")
