#!/usr/bin/env python3
# /// script
# requires-python = ">=3.14"
# dependencies = ["typer>=0.27.0", "pyyaml>=6.0.3"]
# ///
import json
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import typer
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "bench" / "default.yaml"
app = typer.Typer()


def run_bench(
    backend_dir: str,
    cmd: str,
    variant: str,
    image: str,
    kernel_size: int,
    warmup: int,
    runs: int,
    save_output_dir: str | None = None,
) -> dict | None:
    cmd_args = [
        "uv",
        "run",
        "--directory",
        str(PROJECT_ROOT / backend_dir),
        cmd,
        image,
        "--kernel-size",
        str(kernel_size),
        "--warmup",
        str(warmup),
        "--runs",
        str(runs),
        "--variant",
        variant,
        "--json",
    ]
    if save_output_dir is not None:
        cmd_args.extend(["--save-output-dir", save_output_dir])
    try:
        result = subprocess.run(cmd_args, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        if "Unknown variant" in e.stderr:
            print(f"  Skipping {variant} (not available)", file=sys.stderr)
            return None
        print(e.stderr, file=sys.stderr)
        raise
    return json.loads(result.stdout)


def _append_event(events_path: Path, event: dict) -> None:
    events_path.parent.mkdir(parents=True, exist_ok=True)
    with open(events_path, "a") as f:
        f.write(json.dumps(event, default=str) + "\n")


def _load_completed(events_path: Path) -> set[tuple[str, str, str, int]]:
    if not events_path.exists():
        return set()
    completed: set[tuple[str, str, str, int]] = set()
    with open(events_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ev = json.loads(line)
            if ev.get("event") == "result":
                completed.add(
                    (
                        ev["image"],
                        ev["backend"],
                        ev["variant"],
                        ev["kernel_size"],
                    )
                )
    return completed


def _has_results(run_dir: Path) -> bool:
    events = run_dir / "run_events.jsonl"
    if not events.exists():
        return False
    with open(events) as f:
        return any(
            json.loads(line).get("event") == "result" for line in f if line.strip()
        )


def _resolve_latest_run(runs_dir: Path) -> Path:
    candidates = []
    latest_link = runs_dir / "latest"
    if latest_link.is_symlink() and latest_link.exists():
        candidates.append(latest_link.resolve())
    if runs_dir.is_dir():
        for d in sorted(
            (d for d in runs_dir.iterdir() if d.is_dir() and d.name != "latest"),
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        ):
            if d not in candidates:
                candidates.append(d)
    for d in candidates:
        if _has_results(d):
            return d
    if candidates:
        return candidates[0]
    print("No latest run to resume", file=sys.stderr)
    raise typer.Exit(code=1)


def _load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        print(f"Empty config: {config_path}", file=sys.stderr)
        raise typer.Exit(code=1)
    required_keys = ["backends", "images"]
    for key in required_keys:
        if key not in cfg:
            print(f"Missing '{key}' in {config_path}", file=sys.stderr)
            raise typer.Exit(code=1)
    return cfg


@app.command()
def run(
    config: str = typer.Option(
        str(DEFAULT_CONFIG),
        "--config",
        "-c",
        help="Path to YAML benchmark config",
    ),
    images: str = typer.Option(
        None,
        "--images",
        help="Comma-separated image paths (overrides config)",
    ),
    kernel_sizes: str = typer.Option(
        None,
        "--kernel-sizes",
        help="Comma-separated kernel sizes (overrides config)",
    ),
    warmup: int = typer.Option(
        None,
        "--warmup",
        help="Number of warmup runs (overrides config)",
    ),
    runs: int = typer.Option(
        None,
        "--runs",
        help="Number of timed runs (overrides config)",
    ),
    project: str = typer.Option(
        None,
        "--project",
        help="Filter backends by project dir (comma-separated, overrides config)",
    ),
    new: bool = typer.Option(
        False,
        "--new",
        help="Start a new run instead of resuming",
    ),
    resume: str = typer.Option(
        None,
        "--resume",
        help="Resume a previous run (run ID or 'latest')",
    ),
) -> None:
    config_path = Path(config).resolve()
    cfg = _load_config(config_path)

    benchmark_name = cfg.get("name")
    if images is not None:
        image_list = [str(Path(p).resolve()) for p in images.split(",")]
    else:
        image_list = [str(Path(p).resolve()) for p in cfg["images"]]
    if kernel_sizes is not None:
        ks_list = [int(x) for x in kernel_sizes.split(",")]
    else:
        ks_list = cfg["kernel_sizes"]
    if warmup is None:
        warmup = cfg.get("warmup", 3)
    if runs is None:
        runs = cfg.get("runs", 10)
    backends = [(b["dir"], b["cmd"], b["variants"]) for b in cfg["backends"]]

    if project is not None:
        project_dirs = set(project.split(","))
        backends = [b for b in backends if b[0] in project_dirs]
        if not backends:
            print(f"No matching backends for --project {project}", file=sys.stderr)
            raise typer.Exit(code=1)

    run_dir: Path | None = None

    if not new:
        runs_dir = PROJECT_ROOT / "out" / "runs"
        resume_id = "latest" if resume is None else resume
        if resume_id == "latest":
            try:
                run_dir = _resolve_latest_run(runs_dir)
            except typer.Exit:
                if resume is not None:
                    raise
                run_dir = None
        else:
            candidate = runs_dir / resume_id
            if not candidate.is_dir():
                print(f"Run {resume_id} not found", file=sys.stderr)
                raise typer.Exit(code=1)
            run_dir = candidate

    if run_dir is None:
        run_id = str(uuid.uuid7())
        run_dir = PROJECT_ROOT / "out" / "runs" / run_id
        data_dir = run_dir / "data"
        images_dir = run_dir / "images"
        plot_dir = run_dir / "plot"
        for d in [data_dir, images_dir, plot_dir]:
            d.mkdir(parents=True, exist_ok=True)
        events_path = run_dir / "run_events.jsonl"
        _append_event(
            events_path,
            {
                "event": "started",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
        completed = set()
        existing_results = []
        print(f"Run {run_id}", file=sys.stderr)
    else:
        data_dir = run_dir / "data"
        images_dir = run_dir / "images"
        events_path = run_dir / "run_events.jsonl"

        completed = _load_completed(events_path)
        _append_event(
            events_path,
            {
                "event": "resumed",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

        results_path = data_dir / "benchmark_results.json"
        existing_results = []
        if results_path.exists():
            with open(results_path) as f:
                existing_payload = json.load(f)
                existing_results = existing_payload.get("results", [])

        print(
            f"Resuming run {run_dir.name} — {len(completed)} combos already done",
            file=sys.stderr,
        )

    results = list(existing_results)
    total = len(image_list) * len(ks_list) * sum(len(v) for _, _, v in backends)

    try:
        for img in image_list:
            for ks in ks_list:
                for backend_dir, cmd, variants in backends:
                    for variant in variants:
                        key = (img, backend_dir, variant, ks)
                        if key in completed:
                            continue
                        label = f"{backend_dir}/{variant}"
                        print(
                            f"  {label} kernel_size={ks} image={Path(img).name} ...",
                            file=sys.stderr,
                        )
                        data = run_bench(
                            backend_dir,
                            cmd,
                            variant,
                            img,
                            ks,
                            warmup,
                            runs,
                            save_output_dir=str(images_dir),
                        )
                        if data is not None:
                            results.append(data)
                            _append_event(
                                events_path,
                                {
                                    "event": "result",
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                    "image": img,
                                    "backend": backend_dir,
                                    "variant": variant,
                                    "kernel_size": ks,
                                },
                            )
                            payload = {
                                "metadata": {
                                    "run_id": run_dir.name,
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                    "benchmark_name": benchmark_name,
                                    "config_path": str(config_path),
                                    "images": image_list,
                                    "kernel_sizes": ks_list,
                                    "backends": [
                                        {"backend": b[0], "cmd": b[1], "variants": b[2]}
                                        for b in backends
                                    ],
                                    "warmup": warmup,
                                    "runs": runs,
                                },
                                "results": results,
                            }
                            (data_dir / "benchmark_results.json").write_text(
                                json.dumps(payload, indent=2)
                            )

        _append_event(
            events_path,
            {
                "event": "complete",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
        print(f"Completed {run_dir.name}", file=sys.stderr)
    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        _append_event(
            events_path,
            {
                "event": "interrupted",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
        if results:
            payload = {
                "metadata": {
                    "run_id": run_dir.name,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "benchmark_name": benchmark_name,
                    "config_path": str(config_path),
                    "images": image_list,
                    "kernel_sizes": ks_list,
                    "backends": [
                        {"backend": b[0], "cmd": b[1], "variants": b[2]}
                        for b in backends
                    ],
                    "warmup": warmup,
                    "runs": runs,
                },
                "results": results,
            }
            (data_dir / "benchmark_results.json").write_text(
                json.dumps(payload, indent=2)
            )
        latest_link = PROJECT_ROOT / "out" / "runs" / "latest"
        if latest_link.is_symlink() or latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(run_dir, target_is_directory=True)
        print(f"Partial results saved to {run_dir}", file=sys.stderr)
        raise typer.Exit(code=0)

    latest_link = PROJECT_ROOT / "out" / "runs" / "latest"
    if latest_link.is_symlink() or latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(run_dir, target_is_directory=True)
    print(f"Updated {latest_link} -> {run_dir}", file=sys.stderr)


if __name__ == "__main__":
    app()
