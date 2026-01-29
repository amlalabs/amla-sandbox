"""Glob pattern matching for method names.

Supports patterns like:
- ``stripe/charges/create`` - exact match
- ``stripe/charges/*`` - matches single path segment
- ``stripe/**`` - matches zero or more segments

Example::

    >>> from amla_sandbox.capabilities.patterns import method_matches_pattern, pattern_is_subset
    >>>
    >>> # Single-level wildcard
    >>> method_matches_pattern("stripe/charges/create", "stripe/charges/*")
    True
    >>> method_matches_pattern("stripe/charges/refund/full", "stripe/charges/*")
    False
    >>>
    >>> # Multi-level wildcard
    >>> method_matches_pattern("stripe/charges/create", "stripe/**")
    True
    >>> method_matches_pattern("stripe/charges/refund/full", "stripe/**")
    True
    >>>
    >>> # Subset checking for attenuation
    >>> pattern_is_subset("stripe/charges/create", "stripe/charges/*")
    True
    >>> pattern_is_subset("stripe/charges/*", "stripe/**")
    True
"""

from __future__ import annotations


def method_matches_pattern(method: str, pattern: str) -> bool:
    """Check if a method name matches a glob pattern.

    Pattern syntax:
    - ``*`` matches exactly one path segment (no slashes)
    - ``**`` matches zero or more path segments (including slashes)
    - All other characters match literally

    Args:
        method: The method name to check
        pattern: The glob pattern

    Returns:
        True if the method matches the pattern

    Examples:
        >>> method_matches_pattern("stripe/charges/create", "stripe/charges/create")
        True
        >>> method_matches_pattern("stripe/charges/create", "stripe/charges/*")
        True
        >>> method_matches_pattern("stripe/charges/refund/full", "stripe/charges/*")
        False
        >>> method_matches_pattern("stripe/charges/create", "stripe/**")
        True
        >>> method_matches_pattern("stripe", "stripe/**")
        True
    """
    # Handle empty string specially - it has zero segments
    method_parts = [] if method == "" else method.split("/")
    pattern_parts = [] if pattern == "" else pattern.split("/")

    return _matches_parts(method_parts, pattern_parts)


def _matches_parts(method_parts: list[str], pattern_parts: list[str]) -> bool:
    """Recursive helper to match method parts against pattern parts."""
    # Base cases
    if not pattern_parts:
        return not method_parts

    pattern_first = pattern_parts[0]

    # Handle ** (matches zero or more segments)
    if pattern_first == "**":
        rest_pattern = pattern_parts[1:]

        # ** at end matches everything
        if not rest_pattern:
            return True

        # Try matching ** against 0, 1, 2, ... segments
        for i in range(len(method_parts) + 1):
            if _matches_parts(method_parts[i:], rest_pattern):
                return True
        return False

    # Need at least one method part for non-** patterns
    if not method_parts:
        return False

    method_first = method_parts[0]

    # Handle * (matches exactly one segment)
    if pattern_first == "*":
        return _matches_parts(method_parts[1:], pattern_parts[1:])

    # Literal match
    if pattern_first == method_first:
        return _matches_parts(method_parts[1:], pattern_parts[1:])

    return False


def pattern_is_subset(child: str, parent: str) -> bool:
    """Check if a child pattern is a subset of a parent pattern.

    This is used for attenuation validation: a child capability with
    a more specific pattern can be delegated from a parent with a
    broader pattern.

    A child pattern is a subset of parent if every method that matches
    the child also matches the parent. In practice:

    - Exact pattern is subset of any pattern that would match it
    - ``a/b`` <= ``a/*`` <= ``a/**`` <= ``**``
    - ``a/*/c`` <= ``a/**/c`` (but not <= ``a/**``)

    Args:
        child: The child pattern to check
        parent: The parent pattern

    Returns:
        True if child pattern is a subset of parent pattern

    Examples:
        >>> pattern_is_subset("stripe/charges/create", "stripe/charges/*")
        True
        >>> pattern_is_subset("stripe/charges/*", "stripe/**")
        True
        >>> pattern_is_subset("stripe/**", "stripe/**")
        True
        >>> pattern_is_subset("**", "**")
        True
        >>> pattern_is_subset("**", "stripe/**")
        False
        >>> pattern_is_subset("github/**", "stripe/**")
        False
    """
    # Handle empty string specially - it has zero segments
    child_parts = [] if child == "" else child.split("/")
    parent_parts = [] if parent == "" else parent.split("/")

    return _pattern_parts_subset(child_parts, parent_parts)


def _pattern_parts_subset(child: list[str], parent: list[str]) -> bool:
    """Recursive helper for pattern subset checking."""
    # Base cases
    if not parent and not child:
        return True

    if not parent:
        # Parent exhausted but child has more - only ok if child is all literals
        # that would never match (child is more specific)
        return False

    parent_first = parent[0]

    # Parent ** matches any child suffix
    if parent_first == "**":
        # ** at end of parent accepts any child remainder
        if len(parent) == 1:
            return True

        # Parent has more after **: child must eventually match rest
        # For subset: any suffix of child that matches rest of parent works
        parent_rest = parent[1:]
        for i in range(len(child) + 1):
            if _pattern_parts_subset(child[i:], parent_rest):
                return True
        return False

    if not child:
        # Child exhausted, parent has non-** parts remaining
        return False

    child_first = child[0]

    # Child ** - can only be subset if parent also has ** here
    # (child ** matches more than parent * or literal)
    if child_first == "**":
        # Child ** is only subset of parent ** at same position
        return parent_first == "**" and _pattern_parts_subset(child[1:], parent[1:])

    # Child * can be subset of parent * or parent **
    if child_first == "*":
        if parent_first == "*" or parent_first == "**":
            # If parent is **, it's handled above, but let's be safe
            if parent_first == "**":
                # * is subset of ** (both match single segment at this position)
                # But ** can match more, so continue checking
                return _pattern_parts_subset(child[1:], parent)
            # Both are *, continue
            return _pattern_parts_subset(child[1:], parent[1:])
        # Child * vs parent literal: * matches more than literal, not subset
        return False

    # Child is literal
    if parent_first == "*":
        # Parent * matches any single segment including this literal
        return _pattern_parts_subset(child[1:], parent[1:])

    if parent_first == "**":
        # Already handled above
        raise AssertionError("unreachable")

    # Both literals - must match exactly
    if child_first == parent_first:
        return _pattern_parts_subset(child[1:], parent[1:])

    return False
