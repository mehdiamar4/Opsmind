"""Build benchmark tables and PNG charts from existing evaluation JSON files.

Run from backend/: python benchmarks/generate.py
"""

import argparse
import csv
import json
import math
from decimal import Decimal
from pathlib import Path


BACKEND = Path(__file__).resolve().parents[1]
INPUTS = [BACKEND / "evals" / f"results_topk{k}.json" for k in (3, 5, 10)]
OUTPUT_DIR = Path(__file__).resolve().parent / "results"
REQUEST_COUNTS = (1_000, 10_000, 100_000, 500_000)
METRICS = (
    "recall_at_1",
    "recall_at_k",
    "section_at_1",
    "fact_match",
    "llm_judge_accuracy",
)
SUMMARY_FIELDS = (
    "top_k",
    *METRICS,
    "average_rag_latency_ms",
    "average_estimated_ai_cost_per_request",
)


def read_runs(paths=INPUTS):
    """Read measured aggregates and reject missing or mismatched reports."""
    runs = []
    for path in paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if any(field not in data for field in SUMMARY_FIELDS):
            raise ValueError(f"Missing benchmark metric in {path}")
        if type(data["top_k"]) is not int or data["top_k"] < 1:
            raise ValueError(f"Invalid top_k in {path}")
        if data.get("total") != len(data.get("results", [])) or not data["total"]:
            raise ValueError(f"Question count mismatch in {path}")
        for field in METRICS:
            value = data[field]
            if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid {field} in {path}")
        for field in ("average_rag_latency_ms", "average_estimated_ai_cost_per_request"):
            value = data[field]
            if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Missing or invalid {field} in {path}")
        runs.append({field: data[field] for field in SUMMARY_FIELDS})
    runs.sort(key=lambda run: run["top_k"])
    if len({run["top_k"] for run in runs}) != len(runs):
        raise ValueError("Each benchmark report must have a distinct top_k")
    return runs


def projected_cost(cost_per_request, requests):
    return Decimal(str(cost_per_request)) * requests


def decimal_text(value):
    return format(Decimal(str(value)), "f")


def write_tables(runs, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
        writer.writeheader()
        for run in runs:
            writer.writerow({
                **run,
                "average_estimated_ai_cost_per_request": decimal_text(
                    run["average_estimated_ai_cost_per_request"]
                ),
            })

    with (output_dir / "monthly_api_projection.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(("top_k", "requests_per_month", "estimated_ai_api_cost_usd"))
        for run in runs:
            for count in REQUEST_COUNTS:
                writer.writerow((
                    run["top_k"],
                    count,
                    format(projected_cost(run["average_estimated_ai_cost_per_request"], count), "f"),
                ))

    lines = [
        "# OpsMind Day 7 benchmark",
        "",
        "Measured source files: `backend/evals/results_topk3.json`, "
        "`backend/evals/results_topk5.json`, `backend/evals/results_topk10.json`.",
        "Each run contains 10 questions and uses the optional LLM judge.",
        "",
        "## Measured results",
        "",
        "| top_k | Recall@1 | Recall@k | Section@1 | FactMatch | LLM judge | "
        "Avg RAG latency (ms) | Avg estimated AI API cost/request (USD) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in runs:
        lines.append(
            "| {top_k} | {recall_at_1:.0%} | {recall_at_k:.0%} | "
            "{section_at_1:.0%} | {fact_match:.0%} | {llm_judge_accuracy:.0%} | "
            "{average_rag_latency_ms} | {cost} |".format(
                **run,
                cost=decimal_text(run["average_estimated_ai_cost_per_request"]),
            )
        )
    lines += [
        "",
        "## Monthly API cost projection",
        "",
        "Measured average estimated AI API cost/request × monthly request count; "
        "this is a linear projection, not a new measurement.",
        "",
        "| top_k | 1,000 requests | 10,000 requests | 100,000 requests | 500,000 requests |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in runs:
        costs = [
            format(projected_cost(run["average_estimated_ai_cost_per_request"], count), "f")
            for count in REQUEST_COUNTS
        ]
        lines.append(f"| {run['top_k']} | " + " | ".join(f"${cost}" for cost in costs) + " |")
    lines += [
        "",
        "## Scope",
        "",
        "- Costs are estimated OpenAI API costs for the measured RAG flow, not total infrastructure costs.",
        "- The optional LLM judge is paid but excluded from measured RAG latency and cost; PDF upload embeddings are also not logged in `ai_usage_events`.",
        "- The projection assumes the measured question mix and per-request cost remain constant; it excludes database, hosting, and other operational costs.",
        "- Source JSON values are copied without smoothing; chart bars show those same aggregates.",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _fonts():
    from PIL import ImageFont

    def font(size):
        for name in ("Arial.ttf", "DejaVuSans.ttf"):
            try:
                return ImageFont.truetype(name, size)
            except OSError:
                pass
        return ImageFont.load_default()

    return font


def _canvas(title, subtitle):
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (1500, 850), "#ffffff")
    draw = ImageDraw.Draw(image)
    font = _fonts()
    draw.text((95, 65), title, font=font(38), fill="#152536")
    draw.text((95, 120), subtitle, font=font(22), fill="#5a6975")
    return image, draw, font


def _draw_single_bars(runs, field, title, subtitle, filename, output_dir, label):
    image, draw, font = _canvas(title, subtitle)
    left, right, top, bottom = 160, 1400, 220, 660
    values = [run[field] for run in runs]
    ceiling = max(values) * 1.2
    for fraction in (0, 0.25, 0.5, 0.75, 1):
        y = bottom - round((bottom - top) * fraction)
        draw.line((left, y, right, y), fill="#dce2e7", width=2)
        tick = ceiling * fraction
        tick_text = f"{tick:.0f}" if field == "average_rag_latency_ms" else f"{tick:.7f}"
        draw.text((35, y - 13), tick_text, font=font(17), fill="#566674")
    bar_width = 170
    for index, run in enumerate(runs):
        center = left + round((right - left) * (index + 0.5) / len(runs))
        height = round((bottom - top) * values[index] / ceiling)
        draw.rectangle((center - bar_width // 2, bottom - height, center + bar_width // 2, bottom), fill="#28628b")
        draw.text((center, bottom - height - 18), label(values[index]), anchor="ms", font=font(20), fill="#152536")
        draw.text((center, bottom + 24), f"top_k = {run['top_k']}", anchor="mt", font=font(22), fill="#152536")
    draw.text((95, 785), "Bars use the measured aggregate in each source JSON file.", font=font(18), fill="#5a6975")
    image.save(output_dir / filename)


def _draw_quality(runs, output_dir):
    image, draw, font = _canvas(
        "Measured retrieval and answer quality",
        "Percent correct across the same 10 evaluation questions at each top_k",
    )
    left, right, top, bottom = 160, 1400, 240, 660
    colors = ("#245a80", "#4791a5", "#7b739e", "#c17445", "#3e795d")
    names = ("Recall@1", "Recall@k", "Section@1", "FactMatch", "LLM judge")
    for tick in (0, 25, 50, 75, 100):
        y = bottom - round((bottom - top) * tick / 100)
        draw.line((left, y, right, y), fill="#dce2e7", width=2)
        draw.text((90, y - 13), f"{tick}%", font=font(18), fill="#566674")
    for index, run in enumerate(runs):
        center = left + round((right - left) * (index + 0.5) / len(runs))
        for metric_index, metric in enumerate(METRICS):
            value = run[metric]
            x = center - 117 + metric_index * 48
            height = round((bottom - top) * value)
            draw.rectangle((x, bottom - height, x + 38, bottom), fill=colors[metric_index])
            draw.text((x + 19, bottom - height - 12), f"{value:.0%}", anchor="ms", font=font(15), fill="#152536")
        draw.text((center, bottom + 24), f"top_k = {run['top_k']}", anchor="mt", font=font(22), fill="#152536")
    for index, (name, color) in enumerate(zip(names, colors)):
        x = 175 + index * 248
        draw.rectangle((x, 746, x + 20, 766), fill=color)
        draw.text((x + 30, 744), name, font=font(18), fill="#152536")
    image.save(output_dir / "quality_vs_top_k.png")


def write_charts(runs, output_dir):
    _draw_single_bars(
        runs,
        "average_estimated_ai_cost_per_request",
        "Measured estimated AI API cost per RAG request",
        "USD per request; optional judge cost and infrastructure excluded",
        "cost_vs_top_k.png",
        output_dir,
        decimal_text,
    )
    _draw_single_bars(
        runs,
        "average_rag_latency_ms",
        "Measured average RAG latency",
        "Milliseconds; includes embedding, retrieval, answer, and usage recording",
        "latency_vs_top_k.png",
        output_dir,
        lambda value: f"{value:.2f} ms",
    )
    _draw_quality(runs, output_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    runs = read_runs()
    write_tables(runs, args.output_dir)
    write_charts(runs, args.output_dir)
    print(f"Benchmark report saved to {args.output_dir}")


if __name__ == "__main__":
    main()
