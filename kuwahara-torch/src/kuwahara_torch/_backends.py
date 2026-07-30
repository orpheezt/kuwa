from __future__ import annotations

import torch.compiler


def get_compile_backends() -> dict[str, dict]:
    backends = set(torch.compiler.list_backends())

    result: dict[str, dict] = {}
    if "inductor" in backends:
        result["inductor"] = {"cpu": True}
    if "turbine_cpu" in backends:
        result["turbine_cpu"] = {"cpu": True}
    return result
