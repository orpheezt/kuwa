from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


@partial(jax.jit, static_argnames=("kernel_sz",))
def _filter_jax(x: jnp.ndarray, kernel_sz: int) -> jnp.ndarray:
    b, c, h, w = x.shape
    r = (kernel_sz - 1) // 2
    sub_size = r + 1

    padded_x = jnp.pad(x, ((0, 0), (0, 0), (r, r), (r, r)), mode="reflect")

    if c == 3:
        weights = jnp.array([0.299, 0.587, 0.114], dtype=x.dtype).reshape(1, 3, 1, 1)
        intensity = jnp.sum(padded_x * weights, axis=1, keepdims=True)
    else:
        intensity = jnp.mean(padded_x, axis=1, keepdims=True)

    zero = jnp.zeros((), dtype=x.dtype)
    window = (1, 1, sub_size, sub_size)
    strides = (1, 1, 1, 1)

    mean_intensity_full = jax.lax.reduce_window(
        intensity, zero, jax.lax.add, window, strides, "VALID"
    ) / (sub_size**2)
    mean_sq_intensity_full = jax.lax.reduce_window(
        intensity**2, zero, jax.lax.add, window, strides, "VALID"
    ) / (sub_size**2)
    mean_rgb_full = (
        jax.lax.reduce_window(
            padded_x.reshape(b * c, 1, h + 2 * r, w + 2 * r),
            zero,
            jax.lax.add,
            window,
            strides,
            "VALID",
        )
        / (sub_size**2)
    ).reshape(b, c, h + r, w + r)

    v0 = (
        mean_sq_intensity_full[:, :, 0:h, 0:w]
        - mean_intensity_full[:, :, 0:h, 0:w] ** 2
    )
    q0 = mean_rgb_full[:, :, 0:h, 0:w]
    v1 = (
        mean_sq_intensity_full[:, :, 0:h, r : w + r]
        - mean_intensity_full[:, :, 0:h, r : w + r] ** 2
    )
    q1 = mean_rgb_full[:, :, 0:h, r : w + r]
    v2 = (
        mean_sq_intensity_full[:, :, r : h + r, 0:w]
        - mean_intensity_full[:, :, r : h + r, 0:w] ** 2
    )
    q2 = mean_rgb_full[:, :, r : h + r, 0:w]
    v3 = (
        mean_sq_intensity_full[:, :, r : h + r, r : w + r]
        - mean_intensity_full[:, :, r : h + r, r : w + r] ** 2
    )
    q3 = mean_rgb_full[:, :, r : h + r, r : w + r]

    variances = jnp.stack([v0, v1, v2, v3], axis=0)
    means = jnp.stack([q0, q1, q2, q3], axis=0)

    variances = jnp.squeeze(variances, axis=2)
    variances = jnp.transpose(variances, (1, 0, 2, 3))
    means = jnp.transpose(means, (1, 0, 2, 3, 4))

    min_indices = jnp.argmin(variances, axis=1, keepdims=True)
    min_indices_expanded = jnp.expand_dims(min_indices, axis=2)
    output = jnp.take_along_axis(means, min_indices_expanded, axis=1)
    output = jnp.squeeze(output, axis=1)
    return output


def kuwahara_filter_jax(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    if image.ndim == 2:
        x = jnp.expand_dims(jnp.expand_dims(jnp.array(image, dtype=jnp.float32), 0), 0)
    else:
        x = jnp.expand_dims(
            jnp.transpose(jnp.array(image, dtype=jnp.float32), (2, 0, 1)), 0
        )

    out = _filter_jax(x, kernel_size)

    out = jnp.squeeze(out, axis=0)
    out = jnp.transpose(out, (1, 2, 0))
    out = np.asarray(out)
    if image.ndim == 2:
        out = out.squeeze(-1)
    return out
