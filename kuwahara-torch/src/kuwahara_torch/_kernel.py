from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F

from ._backends import get_compile_backends


def _filter_torch(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    B, C, H, W = x.shape
    r = (kernel_size - 1) // 2
    sub_size = r + 1

    padded_x = F.pad(x, (r, r, r, r), mode="reflect")

    if C == 3:
        weights = torch.tensor(
            [0.299, 0.587, 0.114], device=x.device, dtype=x.dtype
        ).view(1, 3, 1, 1)
        intensity = (padded_x * weights).sum(dim=1, keepdim=True)
    else:
        intensity = padded_x.mean(dim=1, keepdim=True)

    avg_kernel = torch.ones(
        (1, 1, sub_size, sub_size), device=x.device, dtype=x.dtype
    ) / (sub_size**2)

    quadrants = [(0, 0), (0, r), (r, 0), (r, r)]
    means, variances = [], []

    for dy, dx in quadrants:
        cropped_x = padded_x[:, :, dy : dy + H + r, dx : dx + W + r]
        cropped_intensity = intensity[:, :, dy : dy + H + r, dx : dx + W + r]
        mean_intensity = F.conv2d(cropped_intensity, avg_kernel)
        mean_sq_intensity = F.conv2d(cropped_intensity**2, avg_kernel)
        variance = mean_sq_intensity - (mean_intensity**2)
        mean_rgb = F.conv2d(
            cropped_x.reshape(B * C, 1, H + r, W + r), avg_kernel
        ).reshape(B, C, H, W)
        means.append(mean_rgb)
        variances.append(variance)

    variances = torch.cat(variances, dim=1)
    means = torch.stack(means, dim=1)

    min_indices = torch.argmin(variances, dim=1, keepdim=True)
    min_indices_expanded = min_indices.unsqueeze(2).expand(B, 1, C, H, W)
    output = torch.gather(means, dim=1, index=min_indices_expanded).squeeze(1)
    return output


def _run_filter(filter_fn, image: np.ndarray, kernel_size: int) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if image.ndim == 2:
        x = torch.from_numpy(image).float().to(device).unsqueeze(0).unsqueeze(0)
    else:
        x = torch.from_numpy(image).float().to(device).permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        out = filter_fn(x, kernel_size)

    out = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
    if image.ndim == 2:
        out = out.squeeze(-1)
    return out


def kuwahara_filter_torch_eager(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    return _run_filter(_filter_torch, image, kernel_size)


def _make_compiled(backend: str):
    return torch.compile(_filter_torch, backend=backend)


_compiled_fns: dict[str, Callable] = {}
for _name in get_compile_backends():
    _compiled_fns[_name] = _make_compiled(_name)


def _make_compile_variant(compiled_fn):
    def variant(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        return _run_filter(compiled_fn, image, kernel_size)

    return variant


kuwahara_filter_torch_inductor = _make_compile_variant(
    _compiled_fns.get("inductor", _filter_torch)
)
kuwahara_filter_torch_turbine_cpu = (
    _make_compile_variant(_compiled_fns["turbine_cpu"])
    if "turbine_cpu" in _compiled_fns
    else None
)

kuwahara_filter_torch = kuwahara_filter_torch_eager


def get_compile_variants() -> dict[str, Callable]:
    result: dict[str, Callable] = {}
    if kuwahara_filter_torch_inductor is not None:
        result["inductor"] = kuwahara_filter_torch_inductor
    if kuwahara_filter_torch_turbine_cpu is not None:
        result["turbine_cpu"] = kuwahara_filter_torch_turbine_cpu
    return result
