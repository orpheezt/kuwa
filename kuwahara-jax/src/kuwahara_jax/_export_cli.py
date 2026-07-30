import typer
from kuwahara import Dtype

from ._export import export_model

app = typer.Typer()


@app.command()
def export(
    height: int = typer.Option(512, "--height"),
    width: int = typer.Option(512, "--width"),
    channels: int = typer.Option(3, "--channels"),
    kernel_size: int = typer.Option(5, "--kernel-size"),
    output: str = typer.Option("out/generated/jax", "--output"),
    dtype: Dtype = typer.Option(Dtype.FLOAT32, "--dtype"),
):
    export_model(height, width, channels, kernel_size, output, dtype)
