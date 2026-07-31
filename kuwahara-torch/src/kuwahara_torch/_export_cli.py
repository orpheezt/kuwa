from typing import Annotated

import typer
from kuwahara import Dtype

from ._export import export_model

app = typer.Typer()


@app.command()
def export(
    height: Annotated[int, typer.Option("--height")] = 512,
    width: Annotated[int, typer.Option("--width")] = 512,
    channels: Annotated[int, typer.Option("--channels")] = 3,
    kernel_size: Annotated[int, typer.Option("--kernel-size")] = 5,
    output: Annotated[str, typer.Option("--output")] = "out/generated/torch",
    dtype: Annotated[Dtype, typer.Option("--dtype")] = Dtype.FLOAT32,
):
    export_model(height, width, channels, kernel_size, output, dtype)
