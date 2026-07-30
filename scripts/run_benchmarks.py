#!/usr/bin/env python3
# /// script
# requires-python = ">=3.14"
# dependencies = ["typer>=0.27.0"]
# ///
from __future__ import annotations

import json
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import typer

BACKENDS = [
    ("kuwahara-torch", "bench-torch", ["eager", "inductor", "turbine_cpu"]),
    ("kuwahara-numba", "bench-numba", ["default"]),
    ("kuwahara-jax", "bench-jax", ["default"]),
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
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


@app.command()
def run(
    image: str = typer.Option(
        str(PROJECT_ROOT / "assets/cowboy.jpg"), "--image", help="Input image path"
    ),
    kernel_sizes: str = typer.Option(
        "3,5,7,9,11,15,21,31",
        "--kernel-sizes",
        help="Comma-separated kernel sizes",
    ),
    warmup: int = typer.Option(3, "--warmup", help="Number of warmup runs"),
    runs: int = typer.Option(10, "--runs", help="Number of timed runs"),
) -> None:
    ks_list = [int(x) for x in kernel_sizes.split(",")]
    image_path = str(Path(image).resolve())

    run_id = str(uuid.uuid7())
    run_dir = PROJECT_ROOT / "out" / "runs" / run_id
    data_dir = run_dir / "data"
    images_dir = run_dir / "images"
    plot_dir = run_dir / "plot"
    for d in [data_dir, images_dir, plot_dir]:
        d.mkdir(parents=True, exist_ok=True)

    results = []
    for ks in ks_list:
        for backend_dir, cmd, variants in BACKENDS:
            for variant in variants:
                label = f"{backend_dir}/{variant}"
                print(f"  {label} kernel_size={ks} ...", file=sys.stderr)
                data = run_bench(
                    backend_dir,
                    cmd,
                    variant,
                    image_path,
                    ks,
                    warmup,
                    runs,
                    save_output_dir=str(images_dir),
                )
                if data is not None:
                    results.append(data)

    payload = {
        "metadata": {
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "image_path": image_path,
            "kernel_sizes": ks_list,
            "backends": [
                {"backend": b[0], "cmd": b[1], "variants": b[2]} for b in BACKENDS
            ],
            "warmup": warmup,
            "runs": runs,
        },
        "results": results,
    }

    out_path = data_dir / "benchmark_results.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out_path}", file=sys.stderr)

    latest_link = PROJECT_ROOT / "out" / "runs" / "latest"
    if latest_link.is_symlink() or latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(run_dir, target_is_directory=True)
    print(f"Updated {latest_link} -> {run_dir}", file=sys.stderr)


if __name__ == "__main__":
    app()
