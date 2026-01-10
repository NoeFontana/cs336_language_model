import torch
import triton
import triton.testing

from cs336_systems.attn.flash_attn import TritonFlashAttn2Fn


def test_timing_flash_forward_backward() -> None:
    n_heads = 16
    d_head = 64
    sequence_length = 16384
    dtype = torch.bfloat16

    if not torch.cuda.is_available():
        print("CUDA not available. Skipping benchmark.")
        return

    # Generate inputs as specified in the assignment
    qkv = torch.randn(3, n_heads, sequence_length, d_head, device="cuda", dtype=dtype, requires_grad=True)
    q, k, v = qkv.unbind(0)

    # TritonFlashAttn2Fn expects (batch, seq, d_head)
    # We treat n_heads as the batch dimension.
    flash = torch.compile(TritonFlashAttn2Fn.apply)

    def flash_forward_backward() -> None:
        o: torch.Tensor = flash(q, k, v, True)  # pyright: ignore[reportAssignmentType]
        loss = o.sum()
        loss.backward()
        # Reset gradients to ensure consistent benchmarking and avoid OOM/accumulation
        q.grad = None
        k.grad = None
        v.grad = None

    print(
        f"Benchmarking TritonFlashAttn2Fn with seq_len={sequence_length}, "
        f"n_heads={n_heads}, d_head={d_head}, dtype={dtype}"
    )
    print("Running benchmark...")
    results = triton.testing.do_bench(flash_forward_backward, rep=10_000, warmup=1_000)
    print(f"Latency (ms): {results}")


if __name__ == "__main__":
    test_timing_flash_forward_backward()
