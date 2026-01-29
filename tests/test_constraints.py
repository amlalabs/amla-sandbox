"""Tests for constraint system."""

import pytest

from amla_sandbox.capabilities.constraints import (
    Constraint,
    ConstraintError,
    ConstraintSet,
    MissingParamError,
    Param,
    TypeMismatchError,
    ViolationError,
)


class TestConstraintEvaluation:
    """Tests for constraint evaluation."""

    def test_le_constraint(self) -> None:
        constraint = Constraint.le("amount", 100)

        # Pass
        constraint.evaluate({"amount": 50})
        constraint.evaluate({"amount": 100})

        # Fail
        with pytest.raises(ViolationError):
            constraint.evaluate({"amount": 101})
        with pytest.raises(MissingParamError):
            constraint.evaluate({})

    def test_lt_constraint(self) -> None:
        constraint = Constraint.lt("x", 100)

        constraint.evaluate({"x": 50})
        constraint.evaluate({"x": 99})

        with pytest.raises(ViolationError):
            constraint.evaluate({"x": 100})  # Equal is not less than
        with pytest.raises(ViolationError):
            constraint.evaluate({"x": 101})

    def test_ge_constraint(self) -> None:
        constraint = Constraint.ge("amount", 100)

        constraint.evaluate({"amount": 100})
        constraint.evaluate({"amount": 200})

        with pytest.raises(ViolationError):
            constraint.evaluate({"amount": 50})

    def test_gt_constraint(self) -> None:
        constraint = Constraint.gt("x", 0)

        constraint.evaluate({"x": 1})
        constraint.evaluate({"x": 100})

        with pytest.raises(ViolationError):
            constraint.evaluate({"x": 0})  # Equal is not greater than
        with pytest.raises(ViolationError):
            constraint.evaluate({"x": -1})

    def test_eq_constraint(self) -> None:
        constraint = Constraint.eq("status", "active")

        constraint.evaluate({"status": "active"})

        with pytest.raises(ViolationError):
            constraint.evaluate({"status": "inactive"})

    def test_ne_constraint(self) -> None:
        constraint = Constraint.ne("status", "deleted")

        constraint.evaluate({"status": "active"})
        constraint.evaluate({"status": "pending"})

        with pytest.raises(ViolationError):
            constraint.evaluate({"status": "deleted"})

    def test_in_constraint(self) -> None:
        constraint = Constraint.is_in("currency", ["USD", "EUR"])

        constraint.evaluate({"currency": "USD"})
        constraint.evaluate({"currency": "EUR"})

        with pytest.raises(ViolationError):
            constraint.evaluate({"currency": "GBP"})

    def test_not_in_constraint(self) -> None:
        constraint = Constraint.not_in("method", ["DELETE", "DROP"])

        constraint.evaluate({"method": "SELECT"})
        constraint.evaluate({"method": "INSERT"})

        with pytest.raises(ViolationError):
            constraint.evaluate({"method": "DELETE"})

    def test_starts_with_constraint(self) -> None:
        constraint = Constraint.starts_with("path", "/api/v2/")

        constraint.evaluate({"path": "/api/v2/users"})

        with pytest.raises(ViolationError):
            constraint.evaluate({"path": "/api/v1/users"})

    def test_ends_with_constraint(self) -> None:
        constraint = Constraint.ends_with("file", ".json")

        constraint.evaluate({"file": "data.json"})
        constraint.evaluate({"file": "config/settings.json"})

        with pytest.raises(ViolationError):
            constraint.evaluate({"file": "data.xml"})

    def test_contains_constraint(self) -> None:
        constraint = Constraint.contains("query", "SELECT")

        constraint.evaluate({"query": "SELECT * FROM users"})
        constraint.evaluate({"query": "Running SELECT query"})

        with pytest.raises(ViolationError):
            constraint.evaluate({"query": "DELETE FROM users"})

    def test_exists_constraint(self) -> None:
        constraint = Constraint.exists("customer_id")

        constraint.evaluate({"customer_id": "cus_123"})

        with pytest.raises(MissingParamError):
            constraint.evaluate({})

    def test_not_exists_constraint(self) -> None:
        constraint = Constraint.not_exists("deprecated")

        constraint.evaluate({})
        constraint.evaluate({"other": "value"})

        with pytest.raises(ViolationError):
            constraint.evaluate({"deprecated": "yes"})

    def test_and_constraint(self) -> None:
        constraint = Constraint.and_(
            [
                Constraint.ge("amount", 100),
                Constraint.le("amount", 1000),
            ]
        )

        constraint.evaluate({"amount": 500})
        constraint.evaluate({"amount": 100})
        constraint.evaluate({"amount": 1000})

        with pytest.raises(ConstraintError):
            constraint.evaluate({"amount": 50})
        with pytest.raises(ConstraintError):
            constraint.evaluate({"amount": 1500})

    def test_or_constraint(self) -> None:
        constraint = Constraint.or_(
            [
                Constraint.eq("currency", "USD"),
                Constraint.eq("currency", "EUR"),
            ]
        )

        constraint.evaluate({"currency": "USD"})
        constraint.evaluate({"currency": "EUR"})

        with pytest.raises(ConstraintError):
            constraint.evaluate({"currency": "GBP"})

    def test_empty_or_fails(self) -> None:
        constraint = Constraint.or_([])

        with pytest.raises(ViolationError):
            constraint.evaluate({"x": 1})

    def test_nested_path(self) -> None:
        constraint = Constraint.eq("nested/key", "value")

        constraint.evaluate({"nested": {"key": "value"}})

        with pytest.raises(ViolationError):
            constraint.evaluate({"nested": {"key": "other"}})

    def test_type_mismatch(self) -> None:
        constraint = Constraint.starts_with("path", "/api/")

        with pytest.raises(TypeMismatchError):
            constraint.evaluate({"path": 123})


class TestConstraintSet:
    """Tests for ConstraintSet."""

    def test_empty_set_passes_all(self) -> None:
        cs = ConstraintSet()
        cs.evaluate({"any": "value"})
        assert cs.is_empty()
        assert len(cs) == 0

    def test_all_constraints_must_pass(self) -> None:
        cs = ConstraintSet(
            [
                Constraint.ge("amount", 100),
                Constraint.le("amount", 10000),
            ]
        )

        cs.evaluate({"amount": 500})

        with pytest.raises(ConstraintError):
            cs.evaluate({"amount": 50})
        with pytest.raises(ConstraintError):
            cs.evaluate({"amount": 50000})

    def test_merge(self) -> None:
        cs1 = ConstraintSet([Constraint.ge("x", 0)])
        cs2 = ConstraintSet([Constraint.le("x", 100)])

        merged = cs1.merge(cs2)
        assert len(merged) == 2

        merged.evaluate({"x": 50})

        with pytest.raises(ConstraintError):
            merged.evaluate({"x": -1})
        with pytest.raises(ConstraintError):
            merged.evaluate({"x": 101})

    def test_extend(self) -> None:
        cs1 = ConstraintSet([Constraint.ge("x", 0)])
        cs2 = ConstraintSet([Constraint.le("x", 100)])

        cs1.extend(cs2)
        assert len(cs1) == 2


class TestParamDSL:
    """Tests for the Param fluent builder."""

    def test_comparison_operators(self) -> None:
        assert (Param("x") < 100).type == "lt"
        assert (Param("x") <= 100).type == "le"
        assert (Param("x") > 0).type == "gt"
        assert (Param("x") >= 1).type == "ge"
        assert (Param("x") == "value").type == "eq"
        assert (Param("x") != "value").type == "ne"

    def test_set_operations(self) -> None:
        assert Param("x").is_in(["a", "b"]).type == "in"
        assert Param("x").not_in(["a", "b"]).type == "not_in"

    def test_string_operations(self) -> None:
        assert Param("path").starts_with("/api/").type == "starts_with"
        assert Param("file").ends_with(".json").type == "ends_with"
        assert Param("query").contains("SELECT").type == "contains"

    def test_existence_operations(self) -> None:
        assert Param("id").exists().type == "exists"
        assert Param("deprecated").not_exists().type == "not_exists"

    def test_param_dsl_evaluation(self) -> None:
        constraints = ConstraintSet(
            [
                Param("amount") >= 100,
                Param("amount") <= 10000,
                Param("currency").is_in(["USD", "EUR"]),
            ]
        )

        constraints.evaluate({"amount": 500, "currency": "USD"})

        with pytest.raises(ConstraintError):
            constraints.evaluate({"amount": 50, "currency": "USD"})


class TestConstraintSubsumption:
    """Tests for constraint subsumption."""

    def test_le_subsumes_le(self) -> None:
        parent = Constraint.le("x", 100)
        child = Constraint.le("x", 50)
        looser = Constraint.le("x", 150)

        assert parent.subsumes(child)
        assert not parent.subsumes(looser)

    def test_ge_subsumes_ge(self) -> None:
        parent = Constraint.ge("x", 0)
        child = Constraint.ge("x", 10)
        looser = Constraint.ge("x", -10)

        assert parent.subsumes(child)
        assert not parent.subsumes(looser)

    def test_in_subsumes_subset(self) -> None:
        parent = Constraint.is_in("x", ["a", "b", "c"])
        child = Constraint.is_in("x", ["a", "b"])

        assert parent.subsumes(child)
        assert not child.subsumes(parent)

    def test_not_in_subsumes_superset(self) -> None:
        parent = Constraint.not_in("x", ["a", "b"])
        child = Constraint.not_in("x", ["a", "b", "c"])
        looser = Constraint.not_in("x", ["a"])

        assert parent.subsumes(child)
        assert not parent.subsumes(looser)

    def test_starts_with_subsumes(self) -> None:
        parent = Constraint.starts_with("path", "/api")
        child = Constraint.starts_with("path", "/api/v2")
        looser = Constraint.starts_with("path", "/ap")

        assert parent.subsumes(child)
        assert not parent.subsumes(looser)

    def test_ends_with_subsumes(self) -> None:
        parent = Constraint.ends_with("file", ".json")
        child = Constraint.ends_with("file", "config.json")

        assert parent.subsumes(child)

    def test_different_params_no_subsumption(self) -> None:
        c1 = Constraint.le("x", 100)
        c2 = Constraint.le("y", 50)

        assert not c1.subsumes(c2)

    def test_constraint_set_subsumes(self) -> None:
        parent = ConstraintSet([Constraint.le("x", 100)])

        # Child can make constraint stricter
        child = ConstraintSet([Constraint.le("x", 50)])
        assert parent.subsumes(child)

        # Child cannot make constraint looser
        looser = ConstraintSet([Constraint.le("x", 200)])
        assert not parent.subsumes(looser)

        # Child can add new constraints
        child_extra = ConstraintSet(
            [
                Constraint.le("x", 50),
                Constraint.ge("y", 0),
            ]
        )
        assert parent.subsumes(child_extra)
