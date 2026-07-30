import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True)
def kuwahara_filter_numba(img: np.ndarray, kernel_sz: int = 5) -> np.ndarray:
    if kernel_sz % 2 == 0 or kernel_sz < 3:
        raise ValueError("kernel_size must be an odd integer >= 3")

    h, w = img.shape[:2]
    is_multichannel = img.ndim == 3
    c = img.shape[2] if is_multichannel else 1

    r = (kernel_sz - 1) // 2
    sub_size = r + 1
    num_pixels = sub_size * sub_size

    intensity = np.empty((h, w), dtype=img.dtype)
    if is_multichannel and c == 3:
        for i in prange(h):
            for j in range(w):
                intensity[i, j] = (
                    0.299 * img[i, j, 0] + 0.587 * img[i, j, 1] + 0.114 * img[i, j, 2]
                )
    elif is_multichannel:
        for i in prange(h):
            for j in range(w):
                val = 0.0
                for ch in range(c):
                    val += img[i, j, ch]
                intensity[i, j] = val / c
    else:
        intensity = img.copy()

    output = np.empty_like(img)

    quadrant_offsets = np.array(
        [
            [-r, 0, -r, 0],
            [-r, 0, 0, r],
            [0, r, -r, 0],
            [0, r, 0, r],
        ],
        dtype=np.int64,
    )

    for i in prange(h):
        for j in range(w):
            min_var = 1e18
            best_q = 0

            for q in range(4):
                r_start = quadrant_offsets[q, 0]
                r_end = quadrant_offsets[q, 1]
                c_start = quadrant_offsets[q, 2]
                c_end = quadrant_offsets[q, 3]

                sum_int = 0.0
                sum_sq_int = 0.0

                for y in range(r_start, r_end + 1):
                    py = abs(i + y)
                    if py >= h:
                        py = 2 * h - 2 - py
                    for x in range(c_start, c_end + 1):
                        px = abs(j + x)
                        if px >= w:
                            px = 2 * w - 2 - px
                        val = intensity[py, px]
                        sum_int += val
                        sum_sq_int += val * val

                mean_int = sum_int / num_pixels
                var = (sum_sq_int / num_pixels) - (mean_int * mean_int)

                if var < min_var:
                    min_var = var
                    best_q = q

            r_start = quadrant_offsets[best_q, 0]
            r_end = quadrant_offsets[best_q, 1]
            c_start = quadrant_offsets[best_q, 2]
            c_end = quadrant_offsets[best_q, 3]

            if is_multichannel:
                for ch in range(c):
                    sum_channel = 0.0
                    for y in range(r_start, r_end + 1):
                        py = abs(i + y)
                        if py >= h:
                            py = 2 * h - 2 - py
                        for x in range(c_start, c_end + 1):
                            px = abs(j + x)
                            if px >= w:
                                px = 2 * w - 2 - px
                            sum_channel += img[py, px, ch]
                    output[i, j, ch] = sum_channel / num_pixels
            else:
                sum_val = 0.0
                for y in range(r_start, r_end + 1):
                    py = abs(i + y)
                    if py >= h:
                        py = 2 * h - 2 - py
                    for x in range(c_start, c_end + 1):
                        px = abs(j + x)
                        if px >= w:
                            px = 2 * w - 2 - px
                        sum_val += img[py, px]
                output[i, j] = sum_val / num_pixels

    return output
