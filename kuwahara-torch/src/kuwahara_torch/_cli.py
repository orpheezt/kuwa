from collections.abc import Callable

from kuwahara import make_bench_app

from ._kernel import (
    get_compile_variants,
    kuwahara_filter_torch_eager,
)

variants: dict[str, Callable] = {"eager": kuwahara_filter_torch_eager}
for name, fn in get_compile_variants().items():
    variants[name] = fn

app = make_bench_app(variants, "kuwahara-torch")  # type: ignore[arg-type]
