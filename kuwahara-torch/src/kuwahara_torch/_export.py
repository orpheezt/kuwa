from pathlib import Path

import torch
from kuwahara import Dtype, gen_dummy

from ._kernel import _filter_torch


def _to_torch_dtype(dtype: Dtype) -> torch.dtype:
    match dtype:
        case Dtype.FLOAT16:
            return torch.float16
        case Dtype.FLOAT32:
            return torch.float32
        case Dtype.FLOAT64:
            return torch.float64
        case Dtype.BFLOAT16:
            return torch.bfloat16


class KuwaharaFilterModule(torch.nn.Module):
    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _filter_torch(x, self.kernel_size)


def export_model(
    height: int = 512,
    width: int = 512,
    channels: int = 3,
    kernel_size: int = 5,
    output: str = "out/generated/torch",
    dtype: Dtype = Dtype.FLOAT32,
) -> None:
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    dummy = gen_dummy(height, width, channels, dtype)
    x = torch.from_numpy(dummy).to(_to_torch_dtype(dtype)).permute(2, 0, 1).unsqueeze(0)

    module = KuwaharaFilterModule(kernel_size).to(x.device)
    module.eval()

    with torch.no_grad():
        exported = torch.export.export(module, (x,))

    torch.export.save(exported, out_dir / "kuwahara.pt2")

    try:
        import iree.turbine.aot as aot

        export_output = aot.export(module, x)
        (out_dir / "kuwahara.mlir").write_text(str(export_output.mlir_module))
    except ImportError:
        pass
