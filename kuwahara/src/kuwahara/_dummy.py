import enum

import numpy as np


class Dtype(enum.StrEnum):
    FLOAT16 = "float16"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    BFLOAT16 = "bfloat16"


def gen_dummy(
    height: int = 512,
    width: int = 512,
    channels: int = 3,
    dtype: Dtype = Dtype.FLOAT32,
) -> np.ndarray:
    match dtype:
        case Dtype.FLOAT16:
            np_dtype = np.float16
        case Dtype.FLOAT32:
            np_dtype = np.float32
        case Dtype.FLOAT64:
            np_dtype = np.float64
        case Dtype.BFLOAT16:
            np_dtype = np.float32
    return np.random.randn(height, width, channels).astype(np_dtype)
