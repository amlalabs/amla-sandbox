"""Constraint system for capability enforcement.

Design Principles:
- **Attenuation is conjunction**: Attenuating adds new clauses AND-ed to existing ones
- **Or allowed within clauses**: Individual constraints can use Or for flexibility
- **No regex**: Regex containment is undecidable, use StartsWith, EndsWith, Contains instead
- **Fail-closed**: Missing parameters cause constraint failure

Example::

    >>> from amla_sandbox.capabilities.constraints import Constraint, ConstraintSet, Param
    >>>
    >>> # Using the Param DSL (ergonomic builder)
    >>> constraints = ConstraintSet([
    ...     Param("amount") >= 100,
    ...     Param("amount") <= 10000,
    ...     Param("currency").is_in(["USD", "EUR"]),
    ... ])
    >>>
    >>> # Evaluate against parameters
    >>> params = {"amount": 500, "currency": "USD"}
    >>> constraints.evaluate(params)  # Returns None on success
    >>>
    >>> # Violation
    >>> bad_params = {"amount": 50, "currency": "USD"}
    >>> constraints.evaluate(bad_params)  # Raises ConstraintError
    Traceback (most recent call last):
    ...
    amla_sandbox.capabilities.constraints.ConstraintError: ...
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, cast


class ConstraintError(Exception):
    """Constraint evaluation error."""

    pass


class MissingParamError(ConstraintError):
    """Required parameter is missing."""

    def __init__(self, param: str) -> None:
        self.param = param
        super().__init__(f"Missing parameter: {param}")


class TypeMismatchError(ConstraintError):
    """Parameter has wrong type."""

    def __init__(self, param: str, expected: str, actual: str) -> None:
        self.param = param
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"Type mismatch for {param}: expected {expected}, got {actual}"
        )


class ViolationError(ConstraintError):
    """Constraint check failed."""

    def __init__(self, param: str, rule: str, actual: str) -> None:
        self.param = param
        self.rule = rule
        self.actual = actual
        super().__init__(f"Constraint violation: {param} {rule}, actual: {actual}")


ConstraintType = Literal[
    "lt",
    "le",
    "gt",
    "ge",
    "eq",
    "ne",
    "in",
    "not_in",
    "starts_with",
    "ends_with",
    "contains",
    "exists",
    "not_exists",
    "and",
    "or",
]


@dataclass
class Constraint:
    """Atomic constraint on a single parameter.

    Constraints are evaluated against JSON-like parameters (dicts).
    Each constraint type has specific evaluation semantics.

    Constraint Types:
        - lt: Less than
        - le: Less than or equal
        - gt: Greater than
        - ge: Greater than or equal
        - eq: Equal
        - ne: Not equal
        - in: Set membership
        - not_in: Set exclusion
        - starts_with: String prefix
        - ends_with: String suffix
        - contains: String contains
        - exists: Parameter exists
        - not_exists: Parameter absent
        - and: All sub-constraints must pass
        - or: At least one sub-constraint must pass
    """

    type: ConstraintType
    param: str = ""
    value: Any = None
    values: list[Any] = field(default_factory=lambda: [])
    prefix: str = ""
    suffix: str = ""
    substring: str = ""
    constraints: list[Constraint] = field(default_factory=lambda: [])

    # Factory methods for each constraint type

    @classmethod
    def lt(cls, param: str, value: Any) -> Constraint:
        """Create a less-than constraint."""
        return cls(type="lt", param=param, value=value)

    @classmethod
    def le(cls, param: str, value: Any) -> Constraint:
        """Create a less-than-or-equal constraint."""
        return cls(type="le", param=param, value=value)

    @classmethod
    def gt(cls, param: str, value: Any) -> Constraint:
        """Create a greater-than constraint."""
        return cls(type="gt", param=param, value=value)

    @classmethod
    def ge(cls, param: str, value: Any) -> Constraint:
        """Create a greater-than-or-equal constraint."""
        return cls(type="ge", param=param, value=value)

    @classmethod
    def eq(cls, param: str, value: Any) -> Constraint:
        """Create an equality constraint."""
        return cls(type="eq", param=param, value=value)

    @classmethod
    def ne(cls, param: str, value: Any) -> Constraint:
        """Create a not-equal constraint."""
        return cls(type="ne", param=param, value=value)

    @classmethod
    def is_in(cls, param: str, values: list[Any]) -> Constraint:
        """Create a set membership constraint."""
        return cls(type="in", param=param, values=list(values))

    @classmethod
    def not_in(cls, param: str, values: list[Any]) -> Constraint:
        """Create a set exclusion constraint."""
        return cls(type="not_in", param=param, values=list(values))

    @classmethod
    def starts_with(cls, param: str, prefix: str) -> Constraint:
        """Create a string prefix constraint."""
        return cls(type="starts_with", param=param, prefix=prefix)

    @classmethod
    def ends_with(cls, param: str, suffix: str) -> Constraint:
        """Create a string suffix constraint."""
        return cls(type="ends_with", param=param, suffix=suffix)

    @classmethod
    def contains(cls, param: str, substring: str) -> Constraint:
        """Create a string contains constraint."""
        return cls(type="contains", param=param, substring=substring)

    @classmethod
    def exists(cls, param: str) -> Constraint:
        """Create an existence constraint."""
        return cls(type="exists", param=param)

    @classmethod
    def not_exists(cls, param: str) -> Constraint:
        """Create a non-existence constraint."""
        return cls(type="not_exists", param=param)

    @classmethod
    def and_(cls, constraints: list[Constraint]) -> Constraint:
        """Create a conjunction constraint (all must pass)."""
        return cls(type="and", constraints=list(constraints))

    @classmethod
    def or_(cls, constraints: list[Constraint]) -> Constraint:
        """Create a disjunction constraint (at least one must pass)."""
        return cls(type="or", constraints=list(constraints))

    def evaluate(self, params: dict[str, Any]) -> None:
        """Evaluate this constraint against the given parameters.

        Args:
            params: Dictionary of parameters to evaluate against

        Raises:
            ConstraintError: If the constraint fails
        """
        if self.type == "exists":
            if _get_param_opt(params, self.param) is None:
                raise MissingParamError(self.param)
            return

        if self.type == "not_exists":
            val = _get_param_opt(params, self.param)
            if val is not None:
                raise ViolationError(self.param, "must not exist", repr(val))
            return

        if self.type == "lt":
            actual = _get_param(params, self.param)
            if not _compare_lt(actual, self.value):
                raise ViolationError(self.param, f"< {self.value!r}", repr(actual))
            return

        if self.type == "le":
            actual = _get_param(params, self.param)
            if not _compare_le(actual, self.value):
                raise ViolationError(self.param, f"<= {self.value!r}", repr(actual))
            return

        if self.type == "gt":
            actual = _get_param(params, self.param)
            if not _compare_gt(actual, self.value):
                raise ViolationError(self.param, f"> {self.value!r}", repr(actual))
            return

        if self.type == "ge":
            actual = _get_param(params, self.param)
            if not _compare_ge(actual, self.value):
                raise ViolationError(self.param, f">= {self.value!r}", repr(actual))
            return

        if self.type == "eq":
            actual = _get_param(params, self.param)
            if actual != self.value:
                raise ViolationError(self.param, f"== {self.value!r}", repr(actual))
            return

        if self.type == "ne":
            actual = _get_param(params, self.param)
            if actual == self.value:
                raise ViolationError(self.param, f"!= {self.value!r}", repr(actual))
            return

        if self.type == "in":
            actual = _get_param(params, self.param)
            if actual not in self.values:
                raise ViolationError(self.param, f"in {self.values!r}", repr(actual))
            return

        if self.type == "not_in":
            actual = _get_param(params, self.param)
            if actual in self.values:
                raise ViolationError(
                    self.param, f"not in {self.values!r}", repr(actual)
                )
            return

        if self.type == "starts_with":
            actual = _get_param_string(params, self.param)
            if not actual.startswith(self.prefix):
                raise ViolationError(self.param, f'starts with "{self.prefix}"', actual)
            return

        if self.type == "ends_with":
            actual = _get_param_string(params, self.param)
            if not actual.endswith(self.suffix):
                raise ViolationError(self.param, f'ends with "{self.suffix}"', actual)
            return

        if self.type == "contains":
            actual = _get_param_string(params, self.param)
            if self.substring not in actual:
                raise ViolationError(self.param, f'contains "{self.substring}"', actual)
            return

        if self.type == "and":
            for c in self.constraints:
                c.evaluate(params)
            return

        if self.type == "or":
            if not self.constraints:
                raise ViolationError(
                    "(or)", "at least one constraint must match", "empty Or"
                )

            last_error: ConstraintError | None = None
            for c in self.constraints:
                try:
                    c.evaluate(params)
                    return  # At least one passed
                except ConstraintError as e:
                    last_error = e

            if last_error:
                raise last_error
            return

        raise ValueError(f"Unknown constraint type: {self.type}")

    def subsumes(self, other: Constraint) -> bool:
        """Check if this constraint subsumes another (this >= other).

        Used for attenuation validation: parent must subsume child.
        Returns True if any value that satisfies `other` also satisfies `self`.
        """
        # Handle composite constraints specially
        if self.type == "and" and other.type == "and":
            # Each parent constraint must be subsumed by some child constraint
            return all(
                any(pc.subsumes(cc) for cc in other.constraints)
                for pc in self.constraints
            )

        if self.type == "or" and other.type == "or":
            # Each child alternative must be subsumed by some parent alternative
            return all(
                any(pc.subsumes(cc) for pc in self.constraints)
                for cc in other.constraints
            )

        # Single constraint subsumes And if it subsumes any child
        if self.param and other.type == "and":
            return any(self.subsumes(c) for c in other.constraints)

        # Single constraint subsumes Or if it subsumes all alternatives
        if self.param and other.type == "or":
            return all(self.subsumes(c) for c in other.constraints)

        # For non-composite, constraints must be on the same parameter
        self_param = self.param_name
        other_param = other.param_name

        if not self_param or not other_param:
            return False
        if self_param != other_param:
            return False

        return self._subsumes_same_param(other)

    def _subsumes_same_param(self, other: Constraint) -> bool:
        """Check subsumption for constraints on the same parameter."""
        # Same constraint types
        if self.type == "lt" and other.type == "lt":
            return _compare_le(other.value, self.value)

        if self.type == "le" and other.type in ("le", "eq"):
            return _compare_le(other.value, self.value)

        if self.type == "gt" and other.type == "gt":
            return _compare_ge(other.value, self.value)

        if self.type == "ge" and other.type in ("ge", "eq"):
            return _compare_ge(other.value, self.value)

        if self.type == "eq" and other.type == "eq":
            return self.value == other.value

        if self.type == "ne" and other.type == "ne":
            return self.value == other.value

        # Set membership
        if self.type == "in" and other.type == "in":
            # Child values must be subset of parent values
            return all(v in self.values for v in other.values)

        if self.type == "not_in" and other.type == "not_in":
            # Child must exclude at least what parent excludes
            return all(v in other.values for v in self.values)

        # String operations
        if self.type == "starts_with" and other.type == "starts_with":
            return other.prefix.startswith(self.prefix)

        if self.type == "ends_with" and other.type == "ends_with":
            return other.suffix.endswith(self.suffix)

        if self.type == "contains" and other.type == "contains":
            return self.substring in other.substring

        # Existence
        if self.type == "exists" and other.type == "exists":
            return True

        if self.type == "not_exists" and other.type == "not_exists":
            return True

        # Cross-type: Eq subsumes comparisons
        if self.type == "eq":
            if other.type == "lt":
                return _compare_lt(self.value, other.value)
            if other.type == "le":
                return _compare_le(self.value, other.value)
            if other.type == "gt":
                return _compare_gt(self.value, other.value)
            if other.type == "ge":
                return _compare_ge(self.value, other.value)

        return False

    @property
    def param_name(self) -> str | None:
        """Get the parameter name this constraint applies to.

        Returns None for composite constraints (and, or).
        """
        if self.type in ("and", "or"):
            return None
        return self.param

    def referenced_params(self) -> list[str]:
        """Get all parameter names referenced by this constraint."""
        if self.type in ("and", "or"):
            result: list[str] = []
            for c in self.constraints:
                result.extend(c.referenced_params())
            return result
        return [self.param]


@dataclass
class ConstraintSet:
    """A set of constraints with implicit AND semantics.

    All constraints must pass for the set to pass.
    """

    constraints: list[Constraint] = field(default_factory=lambda: [])

    def __init__(self, constraints: list[Constraint] | None = None) -> None:
        self.constraints = list(constraints) if constraints else []

    def is_empty(self) -> bool:
        """Returns True if the constraint set is empty."""
        return len(self.constraints) == 0

    def __len__(self) -> int:
        return len(self.constraints)

    def evaluate(self, params: dict[str, Any]) -> None:
        """Evaluate all constraints against parameters.

        Args:
            params: Dictionary of parameters

        Raises:
            ConstraintError: If any constraint fails
        """
        for constraint in self.constraints:
            constraint.evaluate(params)

    def subsumes(self, child: ConstraintSet) -> bool:
        """Check if this constraint set subsumes another (for attenuation).

        Parent subsumes child if for every child constraint on a parameter
        that the parent also constrains, there exists a parent constraint
        that subsumes it. Child can add constraints on new parameters.
        """
        for child_c in child.constraints:
            # Get params referenced by child constraint
            child_params = child_c.referenced_params()

            # Check if parent constrains any of these params
            parent_constrains_any = any(
                cp in pc.referenced_params()
                for cp in child_params
                for pc in self.constraints
            )

            if parent_constrains_any:
                # For composite constraints, check if parent subsumes the whole thing
                # For simple constraints, check param-by-param
                if child_c.param_name:
                    # Simple constraint - find matching parent constraint
                    parent_subsumes = any(
                        pc.param_name == child_c.param_name and pc.subsumes(child_c)
                        for pc in self.constraints
                    )
                else:
                    # Composite constraint - check if any parent subsumes it
                    parent_subsumes = any(
                        pc.subsumes(child_c) for pc in self.constraints
                    )

                if not parent_subsumes:
                    return False
            # If parent doesn't constrain these params, child can add constraint

        return True

    def extend(self, other: ConstraintSet) -> None:
        """Merge another constraint set into this one (conjunction)."""
        self.constraints.extend(other.constraints)

    def merge(self, other: ConstraintSet) -> ConstraintSet:
        """Create a new constraint set by merging two sets."""
        return ConstraintSet(self.constraints + other.constraints)


class Param:
    """Fluent builder for creating constraints.

    Example::

        >>> from amla_sandbox.capabilities.constraints import Param, ConstraintSet
        >>>
        >>> constraints = ConstraintSet([
        ...     Param("amount") >= 100,
        ...     Param("amount") <= 10000,
        ...     Param("currency").is_in(["USD", "EUR"]),
        ...     Param("path").starts_with("/api/"),
        ... ])
    """

    def __init__(self, name: str) -> None:
        """Create a Param builder for the given parameter name."""
        self.name = name

    def __lt__(self, value: Any) -> Constraint:
        """Create a less-than constraint: Param("x") < 100"""
        return Constraint.lt(self.name, value)

    def __le__(self, value: Any) -> Constraint:
        """Create a less-than-or-equal constraint: Param("x") <= 100"""
        return Constraint.le(self.name, value)

    def __gt__(self, value: Any) -> Constraint:
        """Create a greater-than constraint: Param("x") > 0"""
        return Constraint.gt(self.name, value)

    def __ge__(self, value: Any) -> Constraint:
        """Create a greater-than-or-equal constraint: Param("x") >= 1"""
        return Constraint.ge(self.name, value)

    def __eq__(self, value: Any) -> Constraint:  # type: ignore[override]
        """Create an equality constraint: Param("x") == "value" """
        return Constraint.eq(self.name, value)

    def __ne__(self, value: Any) -> Constraint:  # type: ignore[override]
        """Create a not-equal constraint: Param("x") != "value" """
        return Constraint.ne(self.name, value)

    def is_in(self, values: list[Any]) -> Constraint:
        """Create a set membership constraint: Param("x").is_in(["a", "b"])"""
        return Constraint.is_in(self.name, values)

    def not_in(self, values: list[Any]) -> Constraint:
        """Create a set exclusion constraint: Param("x").not_in(["a", "b"])"""
        return Constraint.not_in(self.name, values)

    def starts_with(self, prefix: str) -> Constraint:
        """Create a string prefix constraint: Param("path").starts_with("/api/")"""
        return Constraint.starts_with(self.name, prefix)

    def ends_with(self, suffix: str) -> Constraint:
        """Create a string suffix constraint: Param("file").ends_with(".json")"""
        return Constraint.ends_with(self.name, suffix)

    def contains(self, substring: str) -> Constraint:
        """Create a string contains constraint: Param("query").contains("SELECT")"""
        return Constraint.contains(self.name, substring)

    def exists(self) -> Constraint:
        """Create an existence constraint: Param("customer_id").exists()"""
        return Constraint.exists(self.name)

    def not_exists(self) -> Constraint:
        """Create a non-existence constraint: Param("deprecated").not_exists()"""
        return Constraint.not_exists(self.name)


# Helper functions


def _get_param(params: dict[str, Any], path: str) -> Any:
    """Get a parameter value, raising if missing."""
    result = _get_param_opt(params, path)
    if result is None:
        raise MissingParamError(path)
    return result


def _get_param_opt(params: dict[str, Any], path: str) -> Any:
    """Get a parameter value, returning None if missing.

    Supports both "foo" and "/foo" style paths for nested access.
    """
    # Support both "/foo" and "foo" style paths
    if path.startswith("/"):
        path = path[1:]

    parts = path.split("/") if "/" in path else [path]
    current: Any = params

    for part in parts:
        if isinstance(current, dict):
            current_dict = cast(dict[str, Any], current)
            if part not in current_dict:
                return None
            current = current_dict[part]
        elif isinstance(current, list):
            current_list = cast(list[Any], current)
            try:
                idx = int(part)
                if idx < 0 or idx >= len(current_list):
                    return None
                current = current_list[idx]
            except ValueError:
                # Non-integer path segment into a list (e.g., "items/foo" where
                # items is a list) means the path doesn't exist. This is intentional:
                # returning None is semantically correct for "parameter not found".
                return None
        else:
            return None

    return current


def _get_param_string(params: dict[str, Any], path: str) -> str:
    """Get a parameter value as string, raising if wrong type."""
    val = _get_param(params, path)
    if not isinstance(val, str):
        raise TypeMismatchError(path, "string", type(val).__name__)
    return val


def _compare_lt(a: Any, b: Any) -> bool:
    """Compare two values for less-than."""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) < float(b)
    if isinstance(a, str) and isinstance(b, str):
        return a < b
    return False


def _compare_le(a: Any, b: Any) -> bool:
    """Compare two values for less-than-or-equal."""
    return a == b or _compare_lt(a, b)


def _compare_gt(a: Any, b: Any) -> bool:
    """Compare two values for greater-than."""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) > float(b)
    if isinstance(a, str) and isinstance(b, str):
        return a > b
    return False


def _compare_ge(a: Any, b: Any) -> bool:
    """Compare two values for greater-than-or-equal."""
    return a == b or _compare_gt(a, b)
