#!/usr/bin/env python3
"""Constraint DSL examples.

This example shows all available constraint types and how to combine them.
"""

from amla_sandbox import (
    MethodCapability,
    ConstraintSet,
    Param,
    CapabilityError,
)
from amla_sandbox.capabilities import Constraint

# =============================================================================
# Comparison Constraints
# =============================================================================

print("=== Comparison Constraints ===")

# Using Param DSL (ergonomic)
amount_constraints = ConstraintSet(
    [
        Param("amount") >= 100,  # Greater than or equal
        Param("amount") <= 10000,  # Less than or equal
        Param("price") > 0,  # Greater than
        Param("count") < 1000,  # Less than
        Param("status") == "active",  # Equals
        Param("type") != "deprecated",  # Not equals
    ]
)

# Using Constraint class (explicit)
explicit_constraints = ConstraintSet(
    [
        Constraint.ge("amount", 100),
        Constraint.le("amount", 10000),
        Constraint.gt("price", 0),
        Constraint.lt("count", 1000),
        Constraint.eq("status", "active"),
        Constraint.ne("type", "deprecated"),
    ]
)

# Test
params = {
    "amount": 500,
    "price": 10,
    "count": 50,
    "status": "active",
    "type": "standard",
}
amount_constraints.evaluate(params)
print(f"  ✓ Passed: {params}")

# =============================================================================
# Set Membership Constraints
# =============================================================================

print()
print("=== Set Membership Constraints ===")

membership = ConstraintSet(
    [
        Param("currency").is_in(["usd", "eur", "gbp"]),
        Param("method").not_in(["DELETE", "DROP", "TRUNCATE"]),
    ]
)

membership.evaluate({"currency": "usd", "method": "SELECT"})
print("  ✓ currency=usd, method=SELECT")

try:
    membership.evaluate({"currency": "jpy", "method": "SELECT"})
except Exception as e:
    print(f"  ✗ currency=jpy: {e}")

# =============================================================================
# String Constraints
# =============================================================================

print()
print("=== String Constraints ===")

string_constraints = ConstraintSet(
    [
        Param("path").starts_with("/api/v2/"),
        Param("file").ends_with(".json"),
        Param("query").contains("SELECT"),
    ]
)

string_constraints.evaluate(
    {
        "path": "/api/v2/users",
        "file": "config.json",
        "query": "SELECT * FROM users",
    }
)
print("  ✓ All string constraints passed")

# =============================================================================
# Existence Constraints
# =============================================================================

print()
print("=== Existence Constraints ===")

existence = ConstraintSet(
    [
        Param("user_id").exists(),
        Param("deprecated_field").not_exists(),
    ]
)

existence.evaluate({"user_id": "u_123", "other": "value"})
print("  ✓ user_id exists, deprecated_field absent")

# =============================================================================
# Nested Path Constraints
# =============================================================================

print()
print("=== Nested Path Constraints ===")

nested = ConstraintSet(
    [
        Param("user/role") == "admin",
        Param("config/features/beta").exists(),
        Param("headers/authorization").starts_with("Bearer "),
    ]
)

nested.evaluate(
    {
        "user": {"role": "admin", "name": "Alice"},
        "config": {"features": {"beta": True}},
        "headers": {"authorization": "Bearer token123"},
    }
)
print("  ✓ All nested paths validated")

# =============================================================================
# Composite Constraints (AND/OR)
# =============================================================================

print()
print("=== Composite Constraints ===")

# OR: Either condition satisfies
either_admin_or_owner = Constraint.or_(
    [
        Constraint.eq("role", "admin"),
        Constraint.eq("role", "owner"),
    ]
)

cap_with_or = MethodCapability(
    method_pattern="admin/*",
    constraints=ConstraintSet([either_admin_or_owner]),
)

cap_with_or.validate_call("admin/dashboard", {"role": "admin"})
print("  ✓ role=admin passes OR constraint")

cap_with_or.validate_call("admin/dashboard", {"role": "owner"})
print("  ✓ role=owner passes OR constraint")

# AND: Both conditions must be satisfied
high_value_usd = Constraint.and_(
    [
        Constraint.ge("amount", 10000),
        Constraint.eq("currency", "usd"),
    ]
)

cap_with_and = MethodCapability(
    method_pattern="payment/large",
    constraints=ConstraintSet([high_value_usd]),
)

cap_with_and.validate_call("payment/large", {"amount": 50000, "currency": "usd"})
print("  ✓ amount>=10000 AND currency=usd passes")

try:
    cap_with_and.validate_call("payment/large", {"amount": 50000, "currency": "eur"})
except CapabilityError:
    print("  ✗ amount>=10000 but currency=eur fails AND constraint")

# =============================================================================
# Complex Real-World Example
# =============================================================================

print()
print("=== Real-World Example: Payment Processing ===")

# Payments under $100: any currency
# Payments $100+: USD only, require customer_id
# All payments require currency field
payment_cap = MethodCapability(
    method_pattern="payments/process",
    constraints=ConstraintSet(
        [
            Param("customer_id").exists(),
            Param("currency").exists(),  # Currency is always required
            Constraint.or_(
                [
                    # Small payments: any currency
                    Constraint.lt("amount", 10000),
                    # Large payments: USD only
                    Constraint.and_(
                        [
                            Constraint.ge("amount", 10000),
                            Constraint.eq("currency", "usd"),
                        ]
                    ),
                ]
            ),
        ]
    ),
)

# Small EUR payment - OK
payment_cap.validate_call(
    "payments/process",
    {
        "customer_id": "cus_123",
        "amount": 500,
        "currency": "eur",
    },
)
print("  ✓ Small EUR payment allowed")

# Large USD payment - OK
payment_cap.validate_call(
    "payments/process",
    {
        "customer_id": "cus_456",
        "amount": 50000,
        "currency": "usd",
    },
)
print("  ✓ Large USD payment allowed")

# Large EUR payment - DENIED
try:
    payment_cap.validate_call(
        "payments/process",
        {
            "customer_id": "cus_789",
            "amount": 50000,
            "currency": "eur",
        },
    )
except CapabilityError:
    print("  ✗ Large EUR payment denied (must be USD)")
