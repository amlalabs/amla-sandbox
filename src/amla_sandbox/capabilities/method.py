"""Method capability for JSON-RPC method protection.

This module provides MethodCapability for protecting JSON-RPC method calls
with glob patterns, parameter constraints, and call count limits.

Example::

    >>> from amla_sandbox.capabilities import MethodCapability
    >>> from amla_sandbox.capabilities import Param, ConstraintSet
    >>>
    >>> # Create a capability for Stripe charges
    >>> cap = MethodCapability(
    ...     method_pattern="stripe/charges/*",
    ...     constraints=ConstraintSet([
    ...         Param("amount") <= 10000,
    ...         Param("currency").is_in(["USD", "EUR"]),
    ...     ]),
    ...     max_calls=100,
    ... )
    >>>
    >>> # Validate a call
    >>> cap.validate_call("stripe/charges/create", {"amount": 500, "currency": "USD"})
    >>>
    >>> # Invalid amount raises error
    >>> cap.validate_call("stripe/charges/create", {"amount": 50000, "currency": "USD"})
    Traceback (most recent call last):
    ...
    amla_sandbox.capabilities.CapabilityError: ...
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .constraints import ConstraintError, ConstraintSet
from .patterns import method_matches_pattern, pattern_is_subset


class CapabilityError(Exception):
    """Error during capability validation."""

    pass


class CallLimitExceededError(CapabilityError):
    """Raised when a capability's max_calls limit has been reached.

    Attributes:
        capability_key: The key of the capability that was exhausted.
        max_calls: The maximum calls that were allowed.
    """

    def __init__(self, capability_key: str, max_calls: int) -> None:
        self.capability_key = capability_key
        self.max_calls = max_calls
        super().__init__(
            f"Call limit exceeded for '{capability_key}': max {max_calls} calls allowed"
        )


METHOD_CAPABILITY_TYPE = "method"


@dataclass
class MethodCapability:
    """Capability protecting JSON-RPC method calls.

    A method capability grants permission to call methods matching a glob pattern,
    subject to parameter constraints and optional call count limits.

    Attenuation:
        A child capability is a valid attenuation of a parent if:
        1. Child method_pattern is a subset of parent (matches fewer methods)
        2. Child inherits all parent constraints and may add more (more restrictive)
        3. Child max_calls <= parent max_calls (if parent has a limit)
        4. Child input_schema is compatible with parent (if parent has a schema)

    Key Format:
        The capability key is derived from the method pattern:
        ``cap:method:{pattern}`` (e.g., ``cap:method:stripe/charges/*``)

    Attributes:
        method_pattern: Glob pattern for method names (e.g., "stripe/charges/*")
        constraints: Parameter constraints (all must pass)
        max_calls: Maximum calls allowed (None = unlimited)
        input_schema: Optional JSON Schema for parameter validation
    """

    method_pattern: str
    constraints: ConstraintSet = field(default_factory=ConstraintSet)
    max_calls: int | None = None
    input_schema: dict[str, Any] | None = None

    def key(self) -> str:
        """Get the capability key derived from the method pattern.

        Keys are formatted as ``cap:method:{pattern}``.

        Returns:
            The capability key string.
        """
        return f"cap:method:{self.method_pattern}"

    def validate_call(self, method: str, params: dict[str, Any]) -> None:
        """Validate a method call against this capability.

        Checks:
        1. Method name matches the pattern
        2. Parameters satisfy all constraints

        Note: max_calls is not checked here - that's tracked externally by the CTA.

        Args:
            method: The method name being called
            params: The parameters for the call

        Raises:
            CapabilityError: If the method doesn't match or constraints are violated
        """
        # Check method pattern
        if not method_matches_pattern(method, self.method_pattern):
            raise CapabilityError(
                f"method '{method}' does not match pattern '{self.method_pattern}'"
            )

        # Check constraints
        try:
            self.constraints.evaluate(params)
        except ConstraintError as e:
            raise CapabilityError(str(e)) from e

    def is_subset_of(self, parent: MethodCapability) -> bool:
        """Check if this capability is a valid attenuation of a parent.

        A child is a valid attenuation if it grants equal or fewer permissions:
        1. Child pattern is subset of parent pattern
        2. Child constraints are at least as restrictive as parent
        3. Child max_calls <= parent max_calls (if parent has limit)

        Args:
            parent: The parent capability to check against

        Returns:
            True if this is a valid attenuation of parent

        Example::

            >>> parent = MethodCapability(method_pattern="stripe/**", max_calls=100)
            >>>
            >>> # Valid: narrower pattern, lower limit
            >>> child = MethodCapability(method_pattern="stripe/charges/*", max_calls=50)
            >>> child.is_subset_of(parent)
            True
            >>>
            >>> # Invalid: broader pattern
            >>> invalid = MethodCapability(method_pattern="**")
            >>> invalid.is_subset_of(parent)
            False
        """
        # 1. Pattern must be subset
        if not pattern_is_subset(self.method_pattern, parent.method_pattern):
            return False

        # 2. Constraints: parent must subsume child
        if not parent.constraints.subsumes(self.constraints):
            return False

        # 3. max_calls: if parent has limit, child must have equal or lower limit
        if parent.max_calls is not None:
            if self.max_calls is None:
                # Parent has limit but child doesn't - privilege escalation!
                return False
            if self.max_calls > parent.max_calls:
                return False
        # If parent has no limit, any child limit is fine

        # TODO: input_schema compatibility check
        # For now, if parent has schema, child must have compatible subset
        # This is complex to implement correctly, so we'll be permissive

        return True

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dictionary for serialization.

        Returns:
            Dictionary representation of this capability.
        """
        result: dict[str, Any] = {
            "method_pattern": self.method_pattern,
        }

        if not self.constraints.is_empty():
            # Convert constraints to serializable format
            result["constraints"] = [
                _constraint_to_dict(c) for c in self.constraints.constraints
            ]

        if self.max_calls is not None:
            result["max_calls"] = self.max_calls

        if self.input_schema is not None:
            result["input_schema"] = self.input_schema

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MethodCapability:
        """Create from a dictionary.

        Args:
            data: Dictionary representation

        Returns:
            New MethodCapability instance
        """
        constraints_data = data.get("constraints", [])
        constraints = ConstraintSet([_dict_to_constraint(c) for c in constraints_data])

        return cls(
            method_pattern=data["method_pattern"],
            constraints=constraints,
            max_calls=data.get("max_calls"),
            input_schema=data.get("input_schema"),
        )


def _constraint_to_dict(c: Any) -> dict[str, Any]:
    """Convert a Constraint to a serializable dictionary."""
    from .constraints import Constraint

    if not isinstance(c, Constraint):
        raise TypeError(f"Expected Constraint, got {type(c)}")

    result: dict[str, Any] = {"type": c.type}

    if c.param:
        result["param"] = c.param
    if c.value is not None:
        result["value"] = c.value
    if c.values:
        result["values"] = c.values
    if c.prefix:
        result["prefix"] = c.prefix
    if c.suffix:
        result["suffix"] = c.suffix
    if c.substring:
        result["substring"] = c.substring
    if c.constraints:
        result["constraints"] = [_constraint_to_dict(cc) for cc in c.constraints]

    return result


def _dict_to_constraint(data: dict[str, Any]) -> Any:
    """Convert a dictionary to a Constraint."""
    from .constraints import Constraint

    constraint_type = data["type"]

    return Constraint(
        type=constraint_type,
        param=data.get("param", ""),
        value=data.get("value"),
        values=data.get("values", []),
        prefix=data.get("prefix", ""),
        suffix=data.get("suffix", ""),
        substring=data.get("substring", ""),
        constraints=[_dict_to_constraint(c) for c in data.get("constraints", [])],
    )
