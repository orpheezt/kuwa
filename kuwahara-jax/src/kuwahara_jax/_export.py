from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
from kuwahara import Dtype, gen_dummy

from ._kernel import _filter_jax


def _to_jnp_dtype(dtype: Dtype) -> jnp.dtype:
    match dtype:
        case Dtype.FLOAT16:
            return jnp.float16
        case Dtype.FLOAT32:
            return jnp.float32
        case Dtype.FLOAT64:
            return jnp.float64
        case Dtype.BFLOAT16:
            return jnp.bfloat16


def export_model(
    height: int = 512,
    width: int = 512,
    channels: int = 3,
    kernel_size: int = 5,
    output: str = "out/generated/jax",
    dtype: Dtype = Dtype.FLOAT32,
) -> None:
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    dummy = gen_dummy(height, width, channels, dtype)
    jnp_dtype = _to_jnp_dtype(dtype)
    x = jnp.array(dummy, dtype=jnp_dtype).transpose(2, 0, 1)[None, ...]

    fn = jax.jit(partial(_filter_jax, kernel_sz=kernel_size))
    spec = jax.ShapeDtypeStruct(x.shape, x.dtype)
    exported = jax.export.export(fn)(spec)

    (out_dir / "kuwahara.mlir").write_text(exported.mlir_module())
    (out_dir / "kuwahara.jax_export").write_bytes(bytes(exported.serialize()))
