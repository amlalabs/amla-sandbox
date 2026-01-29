"""Integration tests with real tools and capabilities.

These tests verify the full flow of:
1. Tool definition and registration
2. Capability parsing and enforcement
3. Tool call validation against capabilities
"""

# pyright: reportPrivateUsage=warning

import pytest

from amla_sandbox import (
    MethodCapability,
    ConstraintSet,
    Param,
    CapabilityError,
    ToolDefinition,
)
from amla_sandbox.capabilities import Constraint
from amla_sandbox.runtime import Runtime


# =============================================================================
# Real Tool Definitions
# =============================================================================

# Stripe-like payment API
STRIPE_TOOLS = [
    ToolDefinition(
        name="stripe/charges/create",
        description="Create a new charge",
        parameters={
            "type": "object",
            "properties": {
                "amount": {"type": "integer", "description": "Amount in cents"},
                "currency": {"type": "string", "enum": ["usd", "eur", "gbp"]},
                "customer": {"type": "string", "description": "Customer ID"},
                "description": {"type": "string"},
            },
            "required": ["amount", "currency"],
        },
    ),
    ToolDefinition(
        name="stripe/charges/list",
        description="List charges",
        parameters={
            "type": "object",
            "properties": {
                "customer": {"type": "string"},
                "limit": {"type": "integer", "default": 10},
            },
        },
    ),
    ToolDefinition(
        name="stripe/charges/refund",
        description="Refund a charge",
        parameters={
            "type": "object",
            "properties": {
                "charge": {"type": "string", "description": "Charge ID"},
                "amount": {"type": "integer", "description": "Refund amount in cents"},
            },
            "required": ["charge"],
        },
    ),
    ToolDefinition(
        name="stripe/customers/create",
        description="Create a new customer",
        parameters={
            "type": "object",
            "properties": {
                "email": {"type": "string"},
                "name": {"type": "string"},
            },
            "required": ["email"],
        },
    ),
    ToolDefinition(
        name="stripe/customers/delete",
        description="Delete a customer",
        parameters={
            "type": "object",
            "properties": {
                "customer": {"type": "string", "description": "Customer ID"},
            },
            "required": ["customer"],
        },
    ),
]

# Filesystem-like API
FILESYSTEM_TOOLS = [
    ToolDefinition(
        name="fs/read",
        description="Read a file",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path"},
                "encoding": {"type": "string", "default": "utf-8"},
            },
            "required": ["path"],
        },
    ),
    ToolDefinition(
        name="fs/write",
        description="Write a file",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path"},
                "content": {"type": "string", "description": "File content"},
                "mode": {"type": "string", "enum": ["overwrite", "append"]},
            },
            "required": ["path", "content"],
        },
    ),
    ToolDefinition(
        name="fs/delete",
        description="Delete a file",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path"},
            },
            "required": ["path"],
        },
    ),
    ToolDefinition(
        name="fs/list",
        description="List directory contents",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path"},
                "recursive": {"type": "boolean", "default": False},
            },
            "required": ["path"],
        },
    ),
]

# Database-like API
DATABASE_TOOLS = [
    ToolDefinition(
        name="db/query",
        description="Execute a SQL query",
        parameters={
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "SQL query"},
                "database": {"type": "string", "description": "Database name"},
                "params": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["sql", "database"],
        },
    ),
    ToolDefinition(
        name="db/insert",
        description="Insert a record",
        parameters={
            "type": "object",
            "properties": {
                "table": {"type": "string"},
                "database": {"type": "string"},
                "data": {"type": "object"},
            },
            "required": ["table", "database", "data"],
        },
    ),
]


# =============================================================================
# Capability Definitions
# =============================================================================


def make_readonly_stripe_cap() -> MethodCapability:
    """Capability that only allows reading Stripe data, not mutations."""
    return MethodCapability(
        method_pattern="stripe/*/list",
        constraints=ConstraintSet(
            [
                Param("limit") <= 100,  # Max 100 records per call
            ]
        ),
        max_calls=1000,
    )


def make_limited_charge_cap() -> MethodCapability:
    """Capability for creating charges with limits."""
    return MethodCapability(
        method_pattern="stripe/charges/create",
        constraints=ConstraintSet(
            [
                Param("amount") <= 10000,  # Max $100.00
                Param("amount") >= 50,  # Min $0.50
                Param("currency").is_in(["usd", "eur"]),  # Only USD and EUR
            ]
        ),
        max_calls=100,
    )


def make_sandboxed_fs_cap() -> MethodCapability:
    """Capability for filesystem access limited to /tmp."""
    return MethodCapability(
        method_pattern="fs/*",
        constraints=ConstraintSet(
            [
                Param("path").starts_with("/tmp/"),
            ]
        ),
    )


def make_readonly_fs_cap() -> MethodCapability:
    """Capability for read-only filesystem access."""
    return MethodCapability(
        method_pattern="fs/read",
        constraints=ConstraintSet(
            [
                Param("path").starts_with("/data/"),
            ]
        ),
    )


def make_readonly_db_cap() -> MethodCapability:
    """Capability for read-only database access."""
    return MethodCapability(
        method_pattern="db/query",
        constraints=ConstraintSet(
            [
                Param("sql").starts_with("SELECT"),
                Param("database").is_in(["analytics", "reporting"]),
            ]
        ),
    )


# =============================================================================
# Integration Tests
# =============================================================================


class TestToolRegistration:
    """Tests for tool definition and registration."""

    def test_tool_serialization(self) -> None:
        """Tools can be serialized to dict and back."""
        for tool in STRIPE_TOOLS + FILESYSTEM_TOOLS:
            d = tool.to_dict()
            restored = ToolDefinition.from_dict(d)
            assert restored.name == tool.name
            assert restored.description == tool.description
            assert restored.parameters == tool.parameters

    def test_tool_schemas_are_valid(self) -> None:
        """All tool schemas are valid JSON Schema objects."""
        for tool in STRIPE_TOOLS + FILESYSTEM_TOOLS + DATABASE_TOOLS:
            params = tool.parameters
            assert params.get("type") == "object"
            assert "properties" in params


class TestCapabilityEnforcement:
    """Tests for capability enforcement on tool calls."""

    def test_readonly_stripe_allows_list(self) -> None:
        """Read-only Stripe cap allows list operations."""
        cap = make_readonly_stripe_cap()

        # Allowed
        cap.validate_call("stripe/charges/list", {"limit": 50})
        cap.validate_call("stripe/customers/list", {"limit": 10})

    def test_readonly_stripe_denies_mutations(self) -> None:
        """Read-only Stripe cap denies create/delete operations."""
        cap = make_readonly_stripe_cap()

        # Denied - pattern doesn't match
        with pytest.raises(CapabilityError):
            cap.validate_call("stripe/charges/create", {"amount": 100})
        with pytest.raises(CapabilityError):
            cap.validate_call("stripe/customers/delete", {"customer": "cus_123"})

    def test_readonly_stripe_enforces_limit(self) -> None:
        """Read-only Stripe cap enforces max limit constraint."""
        cap = make_readonly_stripe_cap()

        # Allowed - within limit
        cap.validate_call("stripe/charges/list", {"limit": 100})

        # Denied - exceeds limit
        with pytest.raises(CapabilityError):
            cap.validate_call("stripe/charges/list", {"limit": 101})

    def test_limited_charge_amount_bounds(self) -> None:
        """Limited charge cap enforces amount bounds."""
        cap = make_limited_charge_cap()

        # Allowed
        cap.validate_call(
            "stripe/charges/create",
            {
                "amount": 5000,
                "currency": "usd",
            },
        )

        # Denied - too low
        with pytest.raises(CapabilityError):
            cap.validate_call(
                "stripe/charges/create",
                {
                    "amount": 10,
                    "currency": "usd",
                },
            )

        # Denied - too high
        with pytest.raises(CapabilityError):
            cap.validate_call(
                "stripe/charges/create",
                {
                    "amount": 50000,
                    "currency": "usd",
                },
            )

    def test_limited_charge_currency_restriction(self) -> None:
        """Limited charge cap restricts allowed currencies."""
        cap = make_limited_charge_cap()

        # Allowed
        cap.validate_call("stripe/charges/create", {"amount": 1000, "currency": "usd"})
        cap.validate_call("stripe/charges/create", {"amount": 1000, "currency": "eur"})

        # Denied - GBP not allowed
        with pytest.raises(CapabilityError):
            cap.validate_call(
                "stripe/charges/create",
                {
                    "amount": 1000,
                    "currency": "gbp",
                },
            )

    def test_sandboxed_fs_path_restriction(self) -> None:
        """Sandboxed FS cap restricts paths to /tmp/."""
        cap = make_sandboxed_fs_cap()

        # Allowed - /tmp paths
        cap.validate_call("fs/read", {"path": "/tmp/data.json"})
        cap.validate_call("fs/write", {"path": "/tmp/output.txt", "content": "data"})
        cap.validate_call("fs/delete", {"path": "/tmp/old.log"})
        cap.validate_call("fs/list", {"path": "/tmp/subdir/"})

        # Denied - outside /tmp
        with pytest.raises(CapabilityError):
            cap.validate_call("fs/read", {"path": "/etc/passwd"})
        with pytest.raises(CapabilityError):
            cap.validate_call(
                "fs/write", {"path": "/home/user/.ssh/config", "content": "x"}
            )
        with pytest.raises(CapabilityError):
            cap.validate_call("fs/delete", {"path": "/var/log/syslog"})

    def test_readonly_fs_only_read(self) -> None:
        """Read-only FS cap only allows read operations."""
        cap = make_readonly_fs_cap()

        # Allowed
        cap.validate_call("fs/read", {"path": "/data/report.csv"})

        # Denied - wrong operation
        with pytest.raises(CapabilityError):
            cap.validate_call("fs/write", {"path": "/data/report.csv", "content": "x"})
        with pytest.raises(CapabilityError):
            cap.validate_call("fs/delete", {"path": "/data/report.csv"})

    def test_readonly_db_sql_restriction(self) -> None:
        """Read-only DB cap only allows SELECT queries."""
        cap = make_readonly_db_cap()

        # Allowed
        cap.validate_call(
            "db/query",
            {
                "sql": "SELECT * FROM users",
                "database": "analytics",
            },
        )

        # Denied - not SELECT
        with pytest.raises(CapabilityError):
            cap.validate_call(
                "db/query",
                {
                    "sql": "DELETE FROM users WHERE id = 1",
                    "database": "analytics",
                },
            )
        with pytest.raises(CapabilityError):
            cap.validate_call(
                "db/query",
                {
                    "sql": "INSERT INTO users VALUES (1, 'test')",
                    "database": "analytics",
                },
            )

    def test_readonly_db_database_restriction(self) -> None:
        """Read-only DB cap restricts allowed databases."""
        cap = make_readonly_db_cap()

        # Allowed databases
        cap.validate_call(
            "db/query",
            {
                "sql": "SELECT 1",
                "database": "analytics",
            },
        )
        cap.validate_call(
            "db/query",
            {
                "sql": "SELECT 1",
                "database": "reporting",
            },
        )

        # Denied - production database
        with pytest.raises(CapabilityError):
            cap.validate_call(
                "db/query",
                {
                    "sql": "SELECT * FROM users",
                    "database": "production",
                },
            )


class TestMultipleCapabilities:
    """Tests for enforcement with multiple capabilities."""

    def test_any_matching_cap_allows(self) -> None:
        """A call is allowed if ANY capability authorizes it."""
        caps = [
            make_readonly_stripe_cap(),
            make_limited_charge_cap(),
        ]

        # Try each cap - at least one should match
        allowed = False
        for cap in caps:
            try:
                cap.validate_call(
                    "stripe/charges/create",
                    {
                        "amount": 5000,
                        "currency": "usd",
                    },
                )
                allowed = True
                break
            except CapabilityError:
                continue

        assert allowed, "Expected at least one capability to allow the call"

    def test_no_matching_cap_denies(self) -> None:
        """A call is denied if NO capability authorizes it."""
        caps = [
            make_readonly_stripe_cap(),  # Only allows list
            make_sandboxed_fs_cap(),  # Only allows fs/*
        ]

        # This should be denied by all caps
        for cap in caps:
            with pytest.raises(CapabilityError):
                cap.validate_call(
                    "stripe/charges/create",
                    {
                        "amount": 5000,
                        "currency": "usd",
                    },
                )


class TestCapabilitySubsumption:
    """Tests for capability subsumption (attenuation validation)."""

    def test_narrower_pattern_is_subset(self) -> None:
        """Narrower pattern is valid attenuation."""
        parent = MethodCapability(method_pattern="stripe/**")
        child = MethodCapability(method_pattern="stripe/charges/*")

        assert child.is_subset_of(parent)

    def test_broader_pattern_not_subset(self) -> None:
        """Broader pattern is not valid attenuation."""
        parent = MethodCapability(method_pattern="stripe/charges/*")
        child = MethodCapability(method_pattern="stripe/**")

        assert not child.is_subset_of(parent)

    def test_stricter_constraints_is_subset(self) -> None:
        """Stricter constraints are valid attenuation."""
        parent = MethodCapability(
            method_pattern="stripe/**",
            constraints=ConstraintSet([Param("amount") <= 10000]),
        )
        child = MethodCapability(
            method_pattern="stripe/charges/*",
            constraints=ConstraintSet([Param("amount") <= 5000]),
        )

        assert child.is_subset_of(parent)

    def test_looser_constraints_not_subset(self) -> None:
        """Looser constraints are not valid attenuation."""
        parent = MethodCapability(
            method_pattern="stripe/**",
            constraints=ConstraintSet([Param("amount") <= 5000]),
        )
        child = MethodCapability(
            method_pattern="stripe/charges/*",
            constraints=ConstraintSet([Param("amount") <= 10000]),
        )

        assert not child.is_subset_of(parent)

    def test_lower_max_calls_is_subset(self) -> None:
        """Lower max_calls is valid attenuation."""
        parent = MethodCapability(method_pattern="stripe/**", max_calls=100)
        child = MethodCapability(method_pattern="stripe/charges/*", max_calls=50)

        assert child.is_subset_of(parent)

    def test_higher_max_calls_not_subset(self) -> None:
        """Higher max_calls is not valid attenuation."""
        parent = MethodCapability(method_pattern="stripe/**", max_calls=50)
        child = MethodCapability(method_pattern="stripe/charges/*", max_calls=100)

        assert not child.is_subset_of(parent)


class TestRuntimeCapabilityEnforcement:
    """Tests for Runtime-level capability enforcement."""

    def test_runtime_validates_tool_calls(self) -> None:
        """Runtime validates tool calls against capabilities."""
        # Use for_testing() which sets up ephemeral keypairs and PCA
        runtime = Runtime.for_testing(
            capabilities=["tool_call:**"],  # Allow all for this test
        )
        # Override capabilities for more specific testing
        runtime._capabilities = [
            make_limited_charge_cap(),
            make_readonly_stripe_cap(),
        ]

        # Allowed by limited_charge_cap
        assert runtime.can_call(
            "stripe/charges/create",
            {
                "amount": 5000,
                "currency": "usd",
            },
        )

        # Allowed by readonly_stripe_cap
        assert runtime.can_call("stripe/charges/list", {"limit": 50})

        # Denied by all caps
        assert not runtime.can_call("stripe/customers/delete", {"customer": "cus_123"})

    def test_runtime_returns_all_capabilities(self) -> None:
        """Runtime exposes all capabilities for introspection."""
        caps = [
            make_limited_charge_cap(),
            make_readonly_stripe_cap(),
            make_sandboxed_fs_cap(),
        ]
        # Use for_testing() and override capabilities
        runtime = Runtime.for_testing(capabilities=["tool_call:**"])
        runtime._capabilities = caps

        returned = runtime.get_capabilities()
        assert len(returned) == 3

        # Check patterns are preserved
        patterns = {c.method_pattern for c in returned}
        assert "stripe/charges/create" in patterns
        assert "stripe/*/list" in patterns
        assert "fs/*" in patterns


class TestComplexConstraints:
    """Tests for complex constraint combinations."""

    def test_nested_path_constraints(self) -> None:
        """Constraints can access nested object paths."""
        cap = MethodCapability(
            method_pattern="api/request",
            constraints=ConstraintSet(
                [
                    Constraint.eq("headers/content-type", "application/json"),
                    Constraint.starts_with("body/user/email", "admin@"),
                ]
            ),
        )

        # Allowed
        cap.validate_call(
            "api/request",
            {
                "headers": {"content-type": "application/json"},
                "body": {"user": {"email": "admin@example.com"}},
            },
        )

        # Denied - wrong content-type
        with pytest.raises(CapabilityError):
            cap.validate_call(
                "api/request",
                {
                    "headers": {"content-type": "text/plain"},
                    "body": {"user": {"email": "admin@example.com"}},
                },
            )

        # Denied - wrong email prefix
        with pytest.raises(CapabilityError):
            cap.validate_call(
                "api/request",
                {
                    "headers": {"content-type": "application/json"},
                    "body": {"user": {"email": "user@example.com"}},
                },
            )

    def test_combined_and_or_constraints(self) -> None:
        """Complex AND/OR constraint combinations."""
        # Must be: (amount < 1000) OR (amount >= 1000 AND currency = "usd")
        cap = MethodCapability(
            method_pattern="payment/process",
            constraints=ConstraintSet(
                [
                    Constraint.or_(
                        [
                            Constraint.lt("amount", 1000),
                            Constraint.and_(
                                [
                                    Constraint.ge("amount", 1000),
                                    Constraint.eq("currency", "usd"),
                                ]
                            ),
                        ]
                    ),
                ]
            ),
        )

        # Allowed - small amount, any currency
        cap.validate_call("payment/process", {"amount": 500, "currency": "eur"})

        # Allowed - large amount, USD
        cap.validate_call("payment/process", {"amount": 5000, "currency": "usd"})

        # Denied - large amount, non-USD
        with pytest.raises(CapabilityError):
            cap.validate_call("payment/process", {"amount": 5000, "currency": "eur"})


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_capabilities_denies_all(self) -> None:
        """Runtime with no capabilities denies all tool calls."""
        # Use for_testing() but then clear capabilities
        runtime = Runtime.for_testing(capabilities=["tool_call:**"])
        runtime._capabilities = []  # Clear capabilities

        assert not runtime.can_call("any/method", {})
        assert not runtime.can_call("stripe/charges/create", {"amount": 100})

    def test_wildcard_capability_allows_all(self) -> None:
        """Capability with ** pattern allows all methods."""
        cap = MethodCapability(method_pattern="**")

        cap.validate_call("anything", {})
        cap.validate_call("stripe/charges/create", {"amount": 999999})
        cap.validate_call("deeply/nested/path/method", {"x": 1})

    def test_empty_params_with_no_constraints(self) -> None:
        """Empty params work when no constraints require them."""
        cap = MethodCapability(method_pattern="simple/*")

        cap.validate_call("simple/method", {})

    def test_missing_required_param_fails(self) -> None:
        """Missing param that constraint needs raises CapabilityError."""
        cap = MethodCapability(
            method_pattern="api/*",
            constraints=ConstraintSet([Param("required_field") == "value"]),
        )

        # MissingParamError is wrapped in CapabilityError by validate_call
        with pytest.raises(CapabilityError, match="Missing parameter"):
            cap.validate_call("api/method", {})
