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
import typer

PROJECT_ROOT = Path(__file__).resolve().parent.parent

app = typer.Typer()


def plot_single(data: dict, output: str) -> None:
    backend = f"{data['backend']}/{data.get('variant', 'default')}"
    ks = data["kernel_size"]
    t = data["timing_ms"]
    ci = data["ci_95"]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(backend, t["mean"], yerr=t["std"], capsize=5, color="steelblue")
    ax.set_ylabel("Time (ms)")
    ax.set_title(f"{backend}  kernel_size={ks}")
    ax.grid(axis="y", alpha=0.3)

    info = (
        f"Mean: {t['mean']} ms  Std: {t['std']} ms\n"
        f"Min: {t['min']} ms  Max: {t['max']} ms\n"
        f"95% CI: [{ci[0]}, {ci[1]}] ms"
    )
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


def plot_multi(results: list[dict], output: str, log_scale: bool, error: str) -> None:
    by_backend: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        label = f"{r['backend']}/{r.get('variant', 'default')}"
        by_backend[label].append(r)

    fig, ax = plt.subplots(figsize=(8, 5))
    markers = ["o", "s", "^", "D", "v", "<", ">"]

    for i, (backend, entries) in enumerate(sorted(by_backend.items())):
        entries.sort(key=lambda e: e["kernel_size"])
        ks = [e["kernel_size"] for e in entries]
        mean = [e["timing_ms"]["mean"] for e in entries]
        if error == "std":
            err = [e["timing_ms"]["std"] for e in entries]
        elif error == "ci":
            err = [(e["ci_95"][1] - e["ci_95"][0]) / 2 for e in entries]
        else:
            err = None

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
    ax.set_ylabel("Mean time (ms)")
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
        help="Output image path (default: <run>/plot/benchmark_plot.png)",
    ),
    log_scale: bool = typer.Option(False, "--log-scale", help="Log scale y-axis"),
    error: str = typer.Option(
        "std", "--error", help="Error bar type: std, ci, or none"
    ),
) -> None:
    run_dir = _resolve_run_dir(run_id)
    input_path = Path(input) if input else run_dir / "data" / "benchmark_results.json"
    output_path = Path(output) if output else run_dir / "plot" / "benchmark_plot.png"

    with open(input_path) as f:
        data = json.load(f)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    match data:
        case {"results": results}:
            if not results:
                print("No results found in aggregated file", file=sys.stderr)
                raise typer.Exit(code=1)
            plot_multi(results, str(output_path), log_scale, error)
        case _:
            plot_single(data, str(output_path))


if __name__ == "__main__":
    app()
