from kuwahara import make_bench_app

from ._kernel import kuwahara_filter_jax

app = make_bench_app({"default": kuwahara_filter_jax}, "kuwahara-jax")
