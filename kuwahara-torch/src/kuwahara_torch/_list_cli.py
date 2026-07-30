def list_inductor_backends() -> None:
    from ._backends import get_compile_backends

    backends = get_compile_backends()

    print("kuwahara-torch backends:")
    print("  eager       - Direct PyTorch execution")
    print("  inductor    - torch.compile(backend=inductor)")

    if "turbine_cpu" in backends:
        print("  turbine_cpu - torch.compile(backend=turbine_cpu)")
    else:
        print("  turbine_cpu - (install kuwahara-torch[turbine] to enable)")
