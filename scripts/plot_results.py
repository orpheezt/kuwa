#!/usr/bin/env python3
# /// script
# requires-python = ">=3.14"
# dependencies = ["typer>=0.27.0", "matplotlib>=3.11.1"]
# ///
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import typer

PROJECT_ROOT = Path(__file__).resolve().parent.parent

app = typer.Typer()


def _extract_metric(
    entry: dict, metric: str
) -> tuple[float, float | None, float | None]:
    if metric == "timing":
        mean = entry["timing_ms"]["mean"]
        std = entry["timing_ms"].get("std", 0)
        return mean, std, None
    else:
        f = entry["flops"]
        mean = f["gflops"]
        lo = f.get("gflops_lo")
        hi = f.get("gflops_hi")
        err_lo = mean - lo if lo is not None else None
        err_hi = hi - mean if hi is not None else None
        return mean, err_lo, err_hi


def plot_single(data: dict, output: str, metric: str) -> None:
    backend = f"{data['backend']}/{data.get('variant', 'default')}"
    ks = data["kernel_size"]

    mean, err_lo, err_hi = _extract_metric(data, metric)

    fig, ax = plt.subplots(figsize=(6, 4))
    if err_lo is not None and err_hi is not None:
        ax.bar(backend, mean, yerr=[[err_lo], [err_hi]], capsize=5, color="steelblue")
    elif err_lo is not None:
        ax.bar(backend, mean, yerr=err_lo, capsize=5, color="steelblue")

    if metric == "timing":
        ax.set_ylabel("Time (ms)")
        unit = "ms"
    else:
        ax.set_ylabel("GFLOPS")
        unit = "GFLOPS"

    ax.set_title(f"{backend}  kernel_size={ks}")
    ax.grid(axis="y", alpha=0.3)

    t = data["timing_ms"]
    info = (
        f"Mean: {t['mean']} ms  Std: {t['std']} ms\n"
        f"Min: {t['min']} ms  Max: {t['max']} ms\n"
        f"95% CI: [{data['ci_95'][0]}, {data['ci_95'][1]}] ms"
    )
    if "flops" in data:
        f = data["flops"]
        info += f"\nGFLOPS: {f['gflops']}"
    ax.text(
        0.02,
        0.97,
        info,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        family="monospace",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )

    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Saved {output}")


def plot_multi(
    results: list[dict], output: str, log_scale: bool, error: str, metric: str
) -> None:
    by_backend: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        label = f"{r['backend']}/{r.get('variant', 'default')}"
        by_backend[label].append(r)

    fig, ax = plt.subplots(figsize=(8, 5))
    markers = ["o", "s", "^", "D", "v", "<", ">"]

    for i, (backend, entries) in enumerate(sorted(by_backend.items())):
        entries.sort(key=lambda e: e["kernel_size"])
        ks = [e["kernel_size"] for e in entries]
        mean = [
            e["timing_ms"]["mean"] if metric == "timing" else e["flops"]["gflops"]
            for e in entries
        ]

        match (metric, error):
            case ("timing", "std"):
                err = [e["timing_ms"]["std"] for e in entries]
            case ("timing", "ci"):
                err = [(e["ci_95"][1] - e["ci_95"][0]) / 2 for e in entries]
            case ("timing", _):
                err = None
            case (_, "none"):
                err = None
            case _:
                lower = [
                    e["flops"]["gflops"] - e["flops"]["gflops_lo"] for e in entries
                ]
                upper = [
                    e["flops"]["gflops_hi"] - e["flops"]["gflops"] for e in entries
                ]
                err = np.array([lower, upper])

        ax.errorbar(
            ks,
            mean,
            yerr=err,
            marker=markers[i % len(markers)],
            label=backend,
            capsize=3,
            linewidth=1.5,
        )

    ax.set_xlabel("Kernel size")
    if metric == "timing":
        ax.set_ylabel("Mean time (ms)")
    else:
        ax.set_ylabel("GFLOPS")
    if log_scale:
        ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Saved {output}")


def _resolve_run_dir(run_id: str | None) -> Path:
    runs_dir = PROJECT_ROOT / "out" / "runs"
    if run_id is not None:
        return runs_dir / run_id
    latest = runs_dir / "latest"
    if latest.is_symlink() and latest.exists():
        return latest.resolve()
    existing = sorted(
        [d for d in runs_dir.iterdir() if d.is_dir() and d.name != "latest"]
    )
    if not existing:
        print("No runs found", file=sys.stderr)
        raise typer.Exit(code=1)
    return existing[-1]


@app.command()
def plot(
    run_id: str = typer.Option(
        None, "--run-id", help="Specific run ID (default: latest)"
    ),
    input: str = typer.Option(
        None,
        "--input",
        help="Input JSON file (default: <run>/data/benchmark_results.json)",
    ),
    output: str = typer.Option(
        None,
        "--output",
        help="Output image path (default: <run>/plot/benchmark_plot_<metric>.png)",
    ),
    log_scale: bool = typer.Option(False, "--log-scale", help="Log scale y-axis"),
    error: str = typer.Option(
        "std", "--error", help="Error bar type: std, ci, or none"
    ),
    metric: str = typer.Option(
        "timing", "--metric", help="Metric to plot: timing or flops"
    ),
) -> None:
    run_dir = _resolve_run_dir(run_id)
    input_path = Path(input) if input else run_dir / "data" / "benchmark_results.json"
    if output:
        output_path = Path(output)
    else:
        output_path = run_dir / "plot" / f"benchmark_plot_{metric}.png"

    with open(input_path) as f:
        data = json.load(f)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    match data:
        case {"results": results}:
            if not results:
                print("No results found in aggregated file", file=sys.stderr)
                raise typer.Exit(code=1)
            plot_multi(results, str(output_path), log_scale, error, metric)
        case _:
            plot_single(data, str(output_path), metric)


@app.command()
def list(
    after: str = typer.Option(None, "--after", help="Show runs after ISO datetime"),
    before: str = typer.Option(None, "--before", help="Show runs before ISO datetime"),
    limit: int = typer.Option(None, "--limit", help="Max number of runs to show"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON array"),
) -> None:
    runs_dir = PROJECT_ROOT / "out" / "runs"
    if not runs_dir.is_dir():
        print("No runs directory", file=sys.stderr)
        raise typer.Exit(code=1)

    entries = []
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir() or d.name == "latest":
            continue
        events_path = d / "run_events.jsonl"
        results_path = d / "data" / "benchmark_results.json"

        status = "unknown"
        started_at = None
        events = []
        if events_path.exists():
            with open(events_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    ev = json.loads(line)
                    events.append(ev)
                    if ev["event"] == "started":
                        started_at = ev["timestamp"]
            if events:
                last_event = events[-1]["event"]
                if last_event == "complete":
                    status = "complete"
                elif last_event == "interrupted":
                    status = "interrupted"
                else:
                    status = "in_progress"

        metadata = {}
        if results_path.exists():
            with open(results_path) as f:
                payload = json.load(f)
                metadata = payload.get("metadata", {})

        ts = started_at or metadata.get("timestamp")
        if ts is None:
            continue

        if after is not None and ts < after:
            continue
        if before is not None and ts > before:
            continue

        entries.append(
            {
                "timestamp": ts,
                "run_id": d.name,
                "status": status,
                "name": metadata.get("benchmark_name"),
                "images": metadata.get("images", []),
                "kernel_sizes": metadata.get("kernel_sizes", []),
                "num_backends": len(metadata.get("backends", [])),
            }
        )

    entries.sort(key=lambda e: e["timestamp"], reverse=True)
    if limit is not None:
        entries = entries[:limit]

    if json_output:
        print(json.dumps(entries, indent=2))
    else:
        header = f"{'TIMESTAMP':<32} {'RUN_ID':<38} {'STATUS':<14} {'NAME':<20} {'IMAGES':<8} {'BACKENDS':<9}"
        print(header)
        print("-" * len(header))
        for e in entries:
            name = e["name"] or "(none)"
            num_img = str(len(e["images"]))
            num_be = str(e["num_backends"])
            print(
                f"{e['timestamp']:<32} {e['run_id']:<38} {e['status']:<14} {name:<20} {num_img:<8} {num_be:<9}"
            )


if __name__ == "__main__":
    app()
