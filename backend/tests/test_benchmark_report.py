"""Checks for the benchmark report's measured inputs and cost projection."""

import csv
import json
from decimal import Decimal

import pytest

from benchmarks.generate import INPUTS, projected_cost, read_runs, write_tables


def test_report_uses_saved_measurements(tmp_path):
    runs = read_runs(reversed(INPUTS))
    assert [run["top_k"] for run in runs] == [3, 5, 10]
    write_tables(runs, tmp_path / "output")

    with (tmp_path / "output" / "summary.csv").open(newline="", encoding="utf-8") as file:
        summary = list(csv.DictReader(file))
    assert [row["fact_match"] for row in summary] == ["0.8", "0.9", "0.9"]
    assert [row["average_estimated_ai_cost_per_request"] for row in summary] == [
        "0.000053582", "0.000075662", "0.000121792"
    ]

    with (tmp_path / "output" / "monthly_api_projection.csv").open(
        newline="", encoding="utf-8"
    ) as file:
        projections = list(csv.DictReader(file))
    assert projections[0] == {
        "top_k": "3",
        "requests_per_month": "1000",
        "estimated_ai_api_cost_usd": "0.053582000",
    }
    assert projections[-1] == {
        "top_k": "10",
        "requests_per_month": "500000",
        "estimated_ai_api_cost_usd": "60.896000000",
    }
    assert projected_cost(0.000075662, 100_000) == Decimal("7.5662")


def test_missing_measured_cost_is_rejected(tmp_path):
    path = tmp_path / "missing_cost.json"
    data = json.loads(INPUTS[0].read_text(encoding="utf-8"))
    del data["average_estimated_ai_cost_per_request"]
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="Missing benchmark metric"):
        read_runs([path])
