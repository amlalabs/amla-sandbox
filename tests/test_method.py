"""Tests for MethodCapability."""

import pytest

from amla_sandbox.capabilities import (
    CallLimitExceededError,
    CapabilityError,
    ConstraintSet,
    MethodCapability,
    METHOD_CAPABILITY_TYPE,
    Param,
)
from amla_sandbox.capabilities.constraints import Constraint


class TestMethodCapability:
    """Tests for MethodCapability."""

    def test_basic_creation(self) -> None:
        cap = MethodCapability(method_pattern="stripe/charges/*")

        assert cap.method_pattern == "stripe/charges/*"
        assert cap.constraints.is_empty()
        assert cap.max_calls is None
        assert cap.input_schema is None

    def test_with_all_options(self) -> None:
        cap = MethodCapability(
            method_pattern="stripe/charges/*",
            constraints=ConstraintSet([Param("amount") <= 1000]),
            max_calls=100,
            input_schema={"type": "object"},
        )

        assert cap.method_pattern == "stripe/charges/*"
        assert len(cap.constraints) == 1
        assert cap.max_calls == 100
        assert cap.input_schema == {"type": "object"}

    def test_key(self) -> None:
        cap = MethodCapability(method_pattern="stripe/charges/*")
        assert cap.key() == "cap:method:stripe/charges/*"

        cap2 = MethodCapability(method_pattern="**")
        assert cap2.key() == "cap:method:**"


class TestMethodCapabilityValidateCall:
    """Tests for validate_call method."""

    def test_pattern_match(self) -> None:
        cap = MethodCapability(method_pattern="stripe/charges/*")

        # Matches
        cap.validate_call("stripe/charges/create", {})
        cap.validate_call("stripe/charges/refund", {})

        # Doesn't match
        with pytest.raises(CapabilityError):
            cap.validate_call("stripe/customers/create", {})
        with pytest.raises(CapabilityError):
            cap.validate_call("github/repos/list", {})

    def test_with_constraints(self) -> None:
        cap = MethodCapability(
            method_pattern="stripe/charges/*",
            constraints=ConstraintSet(
                [
                    Param("amount") >= 100,
                    Param("amount") <= 10000,
                ]
            ),
        )

        # Valid
        cap.validate_call("stripe/charges/create", {"amount": 500})

        # Below minimum
        with pytest.raises(CapabilityError):
            cap.validate_call("stripe/charges/create", {"amount": 50})

        # Above maximum
        with pytest.raises(CapabilityError):
            cap.validate_call("stripe/charges/create", {"amount": 50000})


class TestMethodCapabilityIsSubsetOf:
    """Tests for is_subset_of method."""

    def test_pattern_subset(self) -> None:
        parent = MethodCapability(method_pattern="stripe/**")

        # Valid: narrower pattern
        child1 = MethodCapability(method_pattern="stripe/charges/*")
        assert child1.is_subset_of(parent)

        child2 = MethodCapability(method_pattern="stripe/charges/create")
        assert child2.is_subset_of(parent)

        # Invalid: different prefix
        child3 = MethodCapability(method_pattern="github/**")
        assert not child3.is_subset_of(parent)

        # Invalid: broader pattern
        child4 = MethodCapability(method_pattern="**")
        assert not child4.is_subset_of(parent)

    def test_max_calls_subset(self) -> None:
        parent = MethodCapability(method_pattern="stripe/**", max_calls=100)

        # Valid: lower limit
        child1 = MethodCapability(method_pattern="stripe/charges/*", max_calls=50)
        assert child1.is_subset_of(parent)

        # Valid: same limit
        child2 = MethodCapability(method_pattern="stripe/charges/*", max_calls=100)
        assert child2.is_subset_of(parent)

        # Invalid: higher limit
        child3 = MethodCapability(method_pattern="stripe/charges/*", max_calls=200)
        assert not child3.is_subset_of(parent)

        # Invalid: no limit (unlimited)
        child4 = MethodCapability(method_pattern="stripe/charges/*")
        assert not child4.is_subset_of(parent)

    def test_constraints_subset(self) -> None:
        parent = MethodCapability(
            method_pattern="stripe/**",
            constraints=ConstraintSet(
                [
                    Constraint.le("amount", 10000),
                ]
            ),
        )

        # Valid: stricter constraint
        child1 = MethodCapability(
            method_pattern="stripe/charges/*",
            constraints=ConstraintSet(
                [
                    Constraint.le("amount", 5000),
                ]
            ),
        )
        assert child1.is_subset_of(parent)

        # Valid: adds extra constraint on different param
        child2 = MethodCapability(
            method_pattern="stripe/charges/*",
            constraints=ConstraintSet(
                [
                    Constraint.le("amount", 10000),
                    Constraint.is_in("currency", ["USD", "EUR"]),
                ]
            ),
        )
        assert child2.is_subset_of(parent)

        # Invalid: looser constraint
        child3 = MethodCapability(
            method_pattern="stripe/charges/*",
            constraints=ConstraintSet(
                [
                    Constraint.le("amount", 20000),
                ]
            ),
        )
        assert not child3.is_subset_of(parent)


class TestMethodCapabilitySerialization:
    """Tests for serialization/deserialization."""

    def test_to_dict_minimal(self) -> None:
        cap = MethodCapability(method_pattern="stripe/charges/*")
        d = cap.to_dict()

        assert d == {"method_pattern": "stripe/charges/*"}

    def test_to_dict_full(self) -> None:
        cap = MethodCapability(
            method_pattern="stripe/charges/*",
            constraints=ConstraintSet([Param("amount") <= 1000]),
            max_calls=100,
            input_schema={"type": "object"},
        )
        d = cap.to_dict()

        assert d["method_pattern"] == "stripe/charges/*"
        assert "constraints" in d
        assert d["max_calls"] == 100
        assert d["input_schema"] == {"type": "object"}

    def test_from_dict_minimal(self) -> None:
        cap = MethodCapability.from_dict({"method_pattern": "stripe/charges/*"})

        assert cap.method_pattern == "stripe/charges/*"
        assert cap.constraints.is_empty()
        assert cap.max_calls is None

    def test_from_dict_full(self) -> None:
        cap = MethodCapability.from_dict(
            {
                "method_pattern": "stripe/charges/*",
                "constraints": [
                    {"type": "le", "param": "amount", "value": 1000},
                ],
                "max_calls": 100,
                "input_schema": {"type": "object"},
            }
        )

        assert cap.method_pattern == "stripe/charges/*"
        assert len(cap.constraints) == 1
        assert cap.max_calls == 100
        assert cap.input_schema == {"type": "object"}

    def test_roundtrip(self) -> None:
        original = MethodCapability(
            method_pattern="stripe/charges/*",
            constraints=ConstraintSet(
                [
                    Param("amount") <= 1000,
                    Param("currency").is_in(["USD", "EUR"]),
                ]
            ),
            max_calls=100,
        )

        d = original.to_dict()
        restored = MethodCapability.from_dict(d)

        assert restored.method_pattern == original.method_pattern
        assert len(restored.constraints) == len(original.constraints)
        assert restored.max_calls == original.max_calls


class TestMethodCapabilityType:
    """Tests for METHOD_CAPABILITY_TYPE constant."""

    def test_constant_value(self) -> None:
        assert METHOD_CAPABILITY_TYPE == "method"


class TestCallLimitExceededError:
    """Tests for CallLimitExceededError."""

    def test_error_message(self) -> None:
        """Error message includes capability key and max_calls."""
        error = CallLimitExceededError("cap:method:stripe/charges/*", 100)

        assert "cap:method:stripe/charges/*" in str(error)
        assert "100" in str(error)
        assert "exceeded" in str(error).lower()

    def test_attributes(self) -> None:
        """Error has capability_key and max_calls attributes."""
        error = CallLimitExceededError("cap:method:api/**", 50)

        assert error.capability_key == "cap:method:api/**"
        assert error.max_calls == 50

    def test_is_capability_error(self) -> None:
        """CallLimitExceededError is a subclass of CapabilityError."""
        error = CallLimitExceededError("cap:method:test", 10)

        assert isinstance(error, CapabilityError)
        assert isinstance(error, Exception)

    def test_can_be_caught_as_capability_error(self) -> None:
        """CallLimitExceededError can be caught as CapabilityError."""
        with pytest.raises(CapabilityError):
            raise CallLimitExceededError("cap:method:test", 10)

    def test_zero_max_calls(self) -> None:
        """Error works with zero max_calls (edge case)."""
        error = CallLimitExceededError("cap:method:limited", 0)

        assert error.max_calls == 0
        assert "0" in str(error)
