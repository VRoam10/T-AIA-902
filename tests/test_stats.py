"""Tests for the statistical aggregation helpers (core.stats)."""

from core.stats import aggregate, is_scalar, numeric_keys, summary_line


def test_is_scalar_excludes_bool_and_non_numeric():
    assert is_scalar(1)
    assert is_scalar(1.5)
    assert not is_scalar(True)
    assert not is_scalar("x")
    assert not is_scalar([1, 2])
    assert not is_scalar(None)


def test_numeric_keys_picks_only_numeric():
    run = {"a": 1, "b": 2.0, "c": "x", "d": True, "e": [1]}
    assert numeric_keys(run) == ["a", "b"]


def test_aggregate_basic_stats():
    runs = [{"x": 10}, {"x": 20}, {"x": 30}]
    agg = aggregate(runs, ["x"])
    assert agg["x"]["mean"] == 20.0
    assert agg["x"]["min"] == 10.0
    assert agg["x"]["max"] == 30.0
    assert agg["x"]["n"] == 3
    assert agg["x"]["std"] > 0
    assert agg["x"]["ci95"] > 0


def test_aggregate_skips_none_values():
    runs = [{"x": 1, "y": None}, {"x": 3, "y": 5}]
    agg = aggregate(runs, ["x", "y"])
    assert agg["x"]["n"] == 2
    assert agg["y"]["n"] == 1
    assert agg["y"]["ci95"] == 0.0


def test_aggregate_infers_keys_from_first_run():
    runs = [{"x": 1, "label": "a"}, {"x": 3, "label": "b"}]
    agg = aggregate(runs)
    assert "x" in agg
    assert "label" not in agg


def test_summary_line_format():
    assert summary_line({"mean": 1.0, "std": 0.5}) == "1.0 ± 0.5"
