from collections.abc import Callable

import torch
import triton
import triton.testing

from cs336_basics.layer.attention import scaled_dot_product_attention

# 1. Imports
from cs336_systems.attn.flash_attn import TritonFlashAttn2Fn

configs = [
    triton.testing.Benchmark(
        x_names=["seq_len"],
        x_vals=[2**i for i in range(7, 17)],  # 128 to 65536
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton Flash v2", "Naive PyTorch"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="Latency (ms)",
        plot_name=f"flash-attn-d{head_dim}-{dtype}",
        args={"head_dim": head_dim, "dtype": dtype, "batch_size": 1},
    )
    for head_dim in [16, 32, 64, 128]
    for dtype in [torch.float32, torch.bfloat16]
]


@triton.testing.perf_report(configs)
def benchmark(
    seq_len: int, head_dim: int, dtype: torch.dtype, batch_size: int, provider: str
) -> tuple[float, float, float]:
    device = torch.device("cuda")

    # Clear cache to ensure fair start for this config
    torch.cuda.empty_cache()

    try:
        num_heads = 4
        shape = (batch_size, seq_len, num_heads, head_dim)

        q = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
        k = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
        v = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
        do = torch.randn_like(q)

        fn_fwd: Callable[[], torch.Tensor]
        fn_bwd: Callable[[], None]
        fn_e2e: Callable[[], None]

        if provider == "torch":
            # Adapter: Naive expects (B, H, S, D)
            compiled_scaled_dot_product_attention = torch.compile(scaled_dot_product_attention)

            q_t = q.transpose(1, 2).contiguous().detach().requires_grad_(True)
            k_t = k.transpose(1, 2).contiguous().detach().requires_grad_(True)
            v_t = v.transpose(1, 2).contiguous().detach().requires_grad_(True)
            do_t = do.transpose(1, 2).contiguous()

            def fn_fwd() -> torch.Tensor:
                # Ensure mask handling matches your implementation
                return compiled_scaled_dot_product_attention(q_t, k_t, v_t, mask=None)

            # Pre-run to generate graph for backward bench
            o_ref = fn_fwd()

            def fn_bwd() -> None:
                o_ref.backward(do_t, retain_graph=True)

            def fn_e2e() -> None:
                o = fn_fwd()
                o.backward(do_t)

        elif provider == "triton":
            q_in = q.transpose(1, 2).reshape(-1, seq_len, head_dim).contiguous().detach().requires_grad_(True)
            k_in = k.transpose(1, 2).reshape(-1, seq_len, head_dim).contiguous().detach().requires_grad_(True)
            v_in = v.transpose(1, 2).reshape(-1, seq_len, head_dim).contiguous().detach().requires_grad_(True)
            do_in = do.transpose(1, 2).reshape(-1, seq_len, head_dim).contiguous()

            def fn_fwd() -> torch.Tensor:
                return TritonFlashAttn2Fn.apply(q_in, k_in, v_in, True)

            o_ref = fn_fwd()

            def fn_bwd() -> None:
                o_ref.backward(do_in, retain_graph=True)

            def fn_e2e() -> None:
                o = fn_fwd()
                o.backward(do_in)

        ms_fwd = triton.testing.do_bench(fn_fwd, quantiles=[0.5], rep=20, warmup=10)  # pyright: ignore[reportPossiblyUnboundVariable]
        ms_bwd = triton.testing.do_bench(fn_bwd, quantiles=[0.5], rep=20, warmup=10)  # pyright: ignore[reportPossiblyUnboundVariable]
        ms_e2e = triton.testing.do_bench(fn_e2e, quantiles=[0.5], rep=20, warmup=10)  # pyright: ignore[reportPossiblyUnboundVariable]

        return ms_fwd[0], ms_bwd[0], ms_e2e[0]  # pyright: ignore[reportOptionalSubscript]

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return float("inf"), float("inf"), float("inf")


if __name__ == "__main__":
    benchmark.run(print_data=True, show_plots=False, save_path=".")
