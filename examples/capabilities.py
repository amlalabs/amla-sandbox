#!/usr/bin/env python3
"""Basic capability enforcement example.

This example shows how to:
1. Define capabilities with patterns and constraints
2. Validate tool calls against capabilities
3. Handle authorization failures
"""

from amla_sandbox import (
    MethodCapability,
    ConstraintSet,
    Param,
    CapabilityError,
)

# =============================================================================
# Define Capabilities
# =============================================================================

# Capability: Allow reading any Stripe data, but limit results
read_stripe = MethodCapability(
    method_pattern="stripe/*/list",
    constraints=ConstraintSet(
        [
            Param("limit") <= 100,  # Max 100 records per call
        ]
    ),
    max_calls=1000,  # Budget: 1000 calls total
)

# Capability: Allow creating charges with limits
create_charges = MethodCapability(
    method_pattern="stripe/charges/create",
    constraints=ConstraintSet(
        [
            Param("amount") >= 50,  # Min $0.50
            Param("amount") <= 10000,  # Max $100.00
            Param("currency").is_in(["usd", "eur"]),
        ]
    ),
    max_calls=100,
)

# =============================================================================
# Validate Tool Calls
# =============================================================================


def check_call(cap: MethodCapability, method: str, params: dict[str, object]) -> None:
    """Try a tool call and report result."""
    try:
        cap.validate_call(method, params)
        print(f"  ✓ {method}({params})")
    except CapabilityError as e:
        print(f"  ✗ {method}({params})")
        print(f"    Error: {e}")


print("=== Read-Only Stripe Capability ===")
print(f"Pattern: {read_stripe.method_pattern}")
print()

# These should pass
check_call(read_stripe, "stripe/charges/list", {"limit": 50})
check_call(read_stripe, "stripe/customers/list", {"limit": 10})

# These should fail
check_call(read_stripe, "stripe/charges/list", {"limit": 200})  # Exceeds limit
check_call(read_stripe, "stripe/charges/create", {"amount": 100})  # Wrong pattern

print()
print("=== Create Charges Capability ===")
print(f"Pattern: {create_charges.method_pattern}")
print()

# These should pass
check_call(
    create_charges,
    "stripe/charges/create",
    {
        "amount": 5000,
        "currency": "usd",
    },
)

# These should fail
check_call(
    create_charges,
    "stripe/charges/create",
    {
        "amount": 10,  # Too low
        "currency": "usd",
    },
)
check_call(
    create_charges,
    "stripe/charges/create",
    {
        "amount": 50000,  # Too high
        "currency": "usd",
    },
)
check_call(
    create_charges,
    "stripe/charges/create",
    {
        "amount": 1000,
        "currency": "gbp",  # Not allowed
    },
)

# =============================================================================
# Capability Introspection
# =============================================================================

print()
print("=== Capability Introspection ===")
print(f"Key: {create_charges.key()}")
print(f"Max calls: {create_charges.max_calls}")
print(f"Constraints: {len(create_charges.constraints)} rules")

# Serialize to dict (for storage/transmission)
data = create_charges.to_dict()
print(f"Serialized: {data}")

# Restore from dict
restored = MethodCapability.from_dict(data)
print(f"Restored pattern: {restored.method_pattern}")
