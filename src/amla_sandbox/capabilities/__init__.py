"""Capability enforcement for the Amla sandbox.

This module provides:
- :class:`MethodCapability` for protecting JSON-RPC method calls
- :class:`ConstraintSet` and :class:`Constraint` for parameter constraints
- :class:`Param` for ergonomic constraint building
- Pattern matching utilities for glob patterns

Example::

    >>> from amla_sandbox.capabilities import MethodCapability, ConstraintSet, Param
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
from .method import (
    CallLimitExceededError,
    CapabilityError,
    MethodCapability,
    METHOD_CAPABILITY_TYPE,
)
from .patterns import method_matches_pattern, pattern_is_subset

__all__ = [
    # Main types
    "MethodCapability",
    "METHOD_CAPABILITY_TYPE",
    # Constraints
    "Constraint",
    "ConstraintSet",
    "Param",
    # Errors
    "CapabilityError",
    "CallLimitExceededError",
    "ConstraintError",
    "MissingParamError",
    "TypeMismatchError",
    "ViolationError",
    # Pattern matching
    "method_matches_pattern",
    "pattern_is_subset",
]
