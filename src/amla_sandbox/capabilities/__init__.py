"""Capability enforcement for the Amla sandbox.

This module provides:
- :class:`ToolCallCap` for protecting tool calls
- :class:`ConstraintSet` and :class:`Constraint` for parameter constraints
- :class:`Param` for ergonomic constraint building
- Pattern matching utilities for glob patterns

Example::

    >>> from amla_sandbox.capabilities import ToolCallCap, ConstraintSet, Param
    >>>
    >>> # Create a capability for Stripe charges
    >>> cap = ToolCallCap(
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
"""

from .constraints import (
    Constraint,
    ConstraintError,
    ConstraintSet,
    MissingParamError,
    Param,
    TypeMismatchError,
    ViolationError,
)
from .patterns import method_matches_pattern, pattern_is_subset
from .tool_call import (
    TOOL_CALL_CAP_TYPE,
    CallLimitExceededError,
    CapabilityError,
    ToolCallCap,
)

__all__ = [
    "TOOL_CALL_CAP_TYPE",
    "CallLimitExceededError",
    # Errors
    "CapabilityError",
    # Constraints
    "Constraint",
    "ConstraintError",
    "ConstraintSet",
    "MissingParamError",
    "Param",
    # Main types
    "ToolCallCap",
    "TypeMismatchError",
    "ViolationError",
    # Pattern matching
    "method_matches_pattern",
    "pattern_is_subset",
]
