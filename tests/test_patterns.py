"""Tests for glob pattern matching."""

from amla_sandbox.capabilities.patterns import method_matches_pattern, pattern_is_subset


class TestMethodMatchesPattern:
    """Tests for method_matches_pattern function."""

    def test_exact_match(self) -> None:
        assert method_matches_pattern("stripe/charges/create", "stripe/charges/create")
        assert not method_matches_pattern(
            "stripe/charges/create", "stripe/charges/refund"
        )

    def test_single_wildcard(self) -> None:
        # * matches exactly one segment
        assert method_matches_pattern("stripe/charges/create", "stripe/charges/*")
        assert method_matches_pattern("stripe/charges/refund", "stripe/charges/*")

        # * does not match zero segments
        assert not method_matches_pattern("stripe/charges", "stripe/charges/*")

        # * does not match multiple segments
        assert not method_matches_pattern(
            "stripe/charges/refund/full", "stripe/charges/*"
        )

        # * at different positions
        assert method_matches_pattern("stripe/v2/charges", "stripe/*/charges")
        assert method_matches_pattern("github/v1/repos", "github/*/repos")

    def test_double_wildcard(self) -> None:
        # ** matches zero or more segments
        assert method_matches_pattern("stripe", "stripe/**")
        assert method_matches_pattern("stripe/charges", "stripe/**")
        assert method_matches_pattern("stripe/charges/create", "stripe/**")
        assert method_matches_pattern("stripe/a/b/c/d/e", "stripe/**")

        # ** alone matches everything
        assert method_matches_pattern("anything", "**")
        assert method_matches_pattern("a/b/c/d", "**")
        assert method_matches_pattern("", "**")

        # ** at the start
        assert method_matches_pattern("a/b/create", "**/create")
        assert method_matches_pattern("create", "**/create")

        # ** in the middle
        assert method_matches_pattern("stripe/a/b/c/charges", "stripe/**/charges")
        assert method_matches_pattern("stripe/charges", "stripe/**/charges")

    def test_mixed_wildcards(self) -> None:
        assert method_matches_pattern("api/v2/users/get", "api/*/users/*")
        assert not method_matches_pattern("api/v2/v3/users/get", "api/*/users/*")

        assert method_matches_pattern("api/v2/x/y/users/get", "api/**/users/*")
        assert method_matches_pattern("api/users/get", "api/**/users/*")

    def test_no_match(self) -> None:
        assert not method_matches_pattern("github/repos", "stripe/**")
        assert not method_matches_pattern("stripe/charges", "github/*")

    def test_empty_cases(self) -> None:
        assert method_matches_pattern("", "")
        assert method_matches_pattern("", "**")
        assert not method_matches_pattern("", "*")
        assert not method_matches_pattern("foo", "")


class TestPatternIsSubset:
    """Tests for pattern_is_subset function."""

    def test_exact_is_subset_of_wildcard(self) -> None:
        assert pattern_is_subset("stripe/charges/create", "stripe/charges/*")
        assert pattern_is_subset("stripe/charges/create", "stripe/**")
        assert pattern_is_subset("stripe/charges/create", "**")

    def test_star_is_subset_of_double_star(self) -> None:
        assert pattern_is_subset("stripe/charges/*", "stripe/charges/**")
        assert pattern_is_subset("stripe/*", "stripe/**")
        assert pattern_is_subset("*", "**")

    def test_narrower_double_star_subset(self) -> None:
        assert pattern_is_subset("stripe/charges/**", "stripe/**")
        assert pattern_is_subset("stripe/**", "**")

    def test_same_pattern_is_subset(self) -> None:
        assert pattern_is_subset("stripe/charges/*", "stripe/charges/*")
        assert pattern_is_subset("stripe/**", "stripe/**")
        assert pattern_is_subset("**", "**")
        assert pattern_is_subset("stripe/charges/create", "stripe/charges/create")

    def test_double_star_not_subset_of_star(self) -> None:
        # ** matches more than *, so ** is NOT subset of *
        assert not pattern_is_subset("stripe/charges/**", "stripe/charges/*")
        assert not pattern_is_subset("stripe/**", "stripe/*")

    def test_disjoint_patterns_not_subset(self) -> None:
        assert not pattern_is_subset("github/**", "stripe/**")
        assert not pattern_is_subset("stripe/charges/*", "stripe/refunds/*")

    def test_global_double_star(self) -> None:
        # ** is only subset of **
        assert pattern_is_subset("**", "**")
        assert not pattern_is_subset("**", "stripe/**")
        assert not pattern_is_subset("**", "*")

    def test_complex_subset_cases(self) -> None:
        # api/*/users/* is subset of api/**/users/*
        assert pattern_is_subset("api/v1/users/*", "api/*/users/*")
        assert pattern_is_subset("api/*/users/*", "api/**/users/*")

        # But not the other way
        assert not pattern_is_subset("api/**/users/*", "api/*/users/*")

    def test_empty_patterns(self) -> None:
        assert pattern_is_subset("", "")
        assert pattern_is_subset("", "**")
        assert not pattern_is_subset("foo", "")
