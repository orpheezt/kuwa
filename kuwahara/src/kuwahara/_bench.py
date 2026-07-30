import json
import time
from pathlib import Path

import numpy as np
import typer
from PIL import Image

from ._protocol import KuwaharaFilter


def make_bench_app(variants: dict[str, KuwaharaFilter], name: str) -> typer.Typer:
    app = typer.Typer()
    variant_names = list(variants.keys())

    @app.command()
    def run(
        image: str = typer.Argument(..., help="Path to input image"),
        kernel_size: int = typer.Option(5, help="Kernel size (odd, >=3)"),
        warmup: int = typer.Option(3, help="Number of warmup runs"),
        runs: int = typer.Option(10, help="Number of timed runs"),
        variant: str = typer.Option(
            variant_names[0],
            help=f"Variant to run: {', '.join(variant_names)}",
        ),
        json_output: bool = typer.Option(False, "--json", help="Output JSON"),
        list_variants: bool = typer.Option(
            False, "--list-variants", help="List available variants and exit"
        ),
        save_output_dir: str = typer.Option(
            None, "--save-output-dir", help="Directory to save output image"
        ),
    ):
        if list_variants:
            print(f"Available variants for {name}: {', '.join(variant_names)}")
            raise typer.Exit()

        if variant not in variants:
            valid = ", ".join(variant_names)
            typer.echo(f"Unknown variant '{variant}'. Valid: {valid}", err=True)
            raise typer.Exit(code=1)

        filter_fn = variants[variant]
        backend_label = f"{name}/{variant}"

        img = np.asarray(Image.open(image).convert("RGB"), dtype=np.float32) / 255.0

        for _ in range(warmup):
            filter_fn(img, kernel_size)

        samples = []
        for _ in range(runs):
            t0 = time.perf_counter()
            filter_fn(img, kernel_size)
            t1 = time.perf_counter()
            samples.append(t1 - t0)

        samples = np.array(samples)
        mean = samples.mean()
        std = samples.std(ddof=1)
        lo = mean - 1.96 * std / np.sqrt(runs)
        hi = mean + 1.96 * std / np.sqrt(runs)

        if save_output_dir is not None:
            out_path = Path(save_output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            out_img = (np.clip(filter_fn(img, kernel_size), 0, 1) * 255).astype(
                np.uint8
            )
            Image.fromarray(out_img).save(out_path / f"{variant}_k{kernel_size}.png")

        if json_output:
            data = {
                "backend": name,
                "variant": variant,
                "kernel_size": kernel_size,
                "image": {"path": image, "shape": list(img.shape)},
                "config": {"warmup": warmup, "runs": runs},
                "timing_ms": {
                    "mean": round(mean * 1000, 4),
                    "std": round(std * 1000, 4),
                    "min": round(samples.min() * 1000, 4),
                    "max": round(samples.max() * 1000, 4),
                },
                "ci_95": [
                    round(lo * 1000, 4),
                    round(hi * 1000, 4),
                ],
            }
            print(json.dumps(data))
        else:
            print(f"{backend_label} (kernel_size={kernel_size}):")
            print(f"  Shape: {img.shape}")
            print(f"  Warmup: {warmup}, Runs: {runs}")
            print(f"  Mean: {mean * 1000:8.2f} ms")
            print(f"  Std:  {std * 1000:8.2f} ms")
            print(f"  Min:  {samples.min() * 1000:8.2f} ms")
            print(f"  Max:  {samples.max() * 1000:8.2f} ms")
            print(f"  95% CI: [{lo * 1000:8.2f}, {hi * 1000:8.2f}] ms")

    return app
