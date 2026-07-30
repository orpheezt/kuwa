from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class KuwaharaFilter(Protocol):
    def __call__(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray: ...
