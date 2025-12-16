import typing

import torch
import triton
import triton.language as tl


@triton.jit
def flash_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    L_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_ob,
    stride_oq,
    stride_od,
    stride_lb,
    stride_lq,
    N_QUERIES,
    N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    """
    Triton kernel for FlashAttention-2 forward pass.

    Args:
        Q_ptr: Pointer to query matrix in HBM.
        K_ptr: Pointer to key matrix in HBM.
        V_ptr: Pointer to value matrix in HBM.
        O_ptr: Pointer to output matrix in HBM.
        L_ptr: Pointer to logsumexp vector in HBM.
        stride_qb: Stride for query batch dimension.
        stride_qq: Stride for query sequence dimension.
        stride_qd: Stride for query head dimension.
        stride_kb: Stride for key batch dimension.
        stride_kk: Stride for key sequence dimension.
        stride_kd: Stride for key head dimension.
        stride_vb: Stride for value batch dimension.
        stride_vk: Stride for value sequence dimension.
        stride_vd: Stride for value head dimension.
        stride_ob: Stride for output batch dimension.
        stride_oq: Stride for output sequence dimension.
        stride_od: Stride for output head dimension.
        stride_lb: Stride for logsumexp batch dimension.
        stride_lq: Stride for logsumexp sequence dimension.
        N_QUERIES: Number of queries.
        N_KEYS: Number of keys.
        scale: Scaling factor (1/sqrt(d)).
        D: Head dimension.
        Q_TILE_SIZE: Tile size for queries.
        K_TILE_SIZE: Tile size for keys.
        is_causal: Whether to apply causal masking.
    """
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),  # Note: Dimension 1 is contiguous
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),  # Note: Dimension 1 is contiguous
    )

    # setup HBM outputs
    o = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    l = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)  # noqa: E741
    m = tl.full((Q_TILE_SIZE,), -float("inf"), dtype=tl.float32)

    qi = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")

    iter_limit = tl.minimum(N_KEYS, (query_tile_index + 1) * Q_TILE_SIZE) if is_causal else N_KEYS

    q_range = (query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE))[:, None]
    for key_tile_index in range(0, iter_limit, K_TILE_SIZE):
        kj = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        vj = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

        # Accumulate in float32 for precision as inpyut to softmax
        sij = tl.dot(qi, tl.trans(kj), out_dtype=tl.float32) * scale

        if is_causal:
            offs_n = key_tile_index + tl.arange(0, K_TILE_SIZE)
            mask = offs_n[None, :] > q_range
            sij = tl.where(mask, float("-inf"), sij)

        rowmax = tl.max(sij, axis=-1)
        m_new = tl.maximum(m, rowmax)
        pij = tl.exp(sij - tl.expand_dims(m_new, axis=-1))

        scale_diff = tl.exp(m - m_new)
        l = scale_diff * l + tl.sum(pij, axis=-1)  # noqa: E741

        o = tl.expand_dims(scale_diff, axis=-1) * o
        o = tl.dot(pij.to(vj.dtype), vj, acc=o)

        # book-keeping
        m = m_new

        ## Next tile
        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

    Oi = o / tl.expand_dims(l, axis=-1)
    Li = m + tl.log(l)

    tl.store(O_block_ptr, Oi.to(O_ptr.type.element_ty), boundary_check=(0, 1))
    tl.store(L_block_ptr, Li, boundary_check=(0,))


class TritonFlashAttn2Fn(torch.autograd.Function):
    """A triton implementation of FlashAttention v2."""

    @staticmethod
    def forward(
        ctx: typing.Any, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False
    ) -> torch.Tensor:
        """
        FlashAttention-2 forward pass.

        Args:
            ctx: Context to save tensors for backward pass.
            q: Query tensor of shape (batch_size, seq_len, d_head).
            k: Key tensor of shape (batch_size, seq_len, d_head).
            v: Value tensor of shape (batch_size, seq_len, d_head).
            is_causal: Whether to apply causal masking.

        Returns:
            out: Output tensor of shape (batch_size, seq_len, d_head).
            logsumexp: Logsumexp tensor of shape (batch_size, seq_len).
        """

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        batch_size, seq_len_q, d_head = q.shape
        _, seq_len_k, _ = k.shape
        # Tiles of size (Bq * d)
        Bq = 16  # **2
        Tq = (seq_len_q + Bq - 1) // Bq

        # Tiles of size (Bk * d)
        Bk = 16  # **2

        rsqrt_d = d_head**-0.5

        out = torch.empty_like(q)
        logsumexp = torch.empty((batch_size, seq_len_q), device=q.device, dtype=torch.float32)

        # lg = (Tq , batch_size)
        flash_fwd_kernel[(Tq, batch_size)](
            Q_ptr=q,
            K_ptr=k,
            V_ptr=v,
            O_ptr=out,
            L_ptr=logsumexp,
            stride_qb=q.stride(0),
            stride_qq=q.stride(1),
            stride_qd=q.stride(2),
            stride_kb=k.stride(0),
            stride_kk=k.stride(1),
            stride_kd=k.stride(2),
            stride_vb=v.stride(0),
            stride_vk=v.stride(1),
            stride_vd=v.stride(2),
            stride_ob=out.stride(0),
            stride_oq=out.stride(1),
            stride_od=out.stride(2),
            stride_lb=logsumexp.stride(0),
            stride_lq=logsumexp.stride(1),
            N_QUERIES=seq_len_q,
            N_KEYS=seq_len_k,
            scale=rsqrt_d,
            D=d_head,  # type: ignore[reportArgumentType]
            Q_TILE_SIZE=Bq,  # type: ignore[reportArgumentType]
            K_TILE_SIZE=Bk,  # type: ignore[reportArgumentType]
            is_causal=is_causal,  # type: ignore[reportArgumentType]
        )

        ctx.save_for_backward(q, k, v, out, logsumexp)
        ctx.is_causal = is_causal

        return out

    @staticmethod
    def backward(ctx: typing.Any, *grad_outputs: typing.Any) -> typing.Any:
        """
        FlashAttention-2 backward pass.

        Args:
            ctx: Context containing saved tensors.
            grad_outputs: Gradients with respect to the outputs of forward.
                Expected to contain (grad_out, grad_logsumexp).

        Returns:
            grad_q: Gradient with respect to Q.
            grad_k: Gradient with respect to K.
            grad_v: Gradient with respect to V.
            grad_is_causal: Gradient with respect to is_causal (None).
        """
        raise NotImplementedError


class TorchFlashAttn2Fn(torch.autograd.Function):
    """A pure PyTorch implementation of FlashAttention v2.

    This is present for didactic and debugging purposes only.
    """

    @staticmethod
    def forward(
        ctx: typing.Any, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False
    ) -> torch.Tensor:
        """
        FlashAttention-2 forward pass.

        Args:
            ctx: Context to save tensors for backward pass.
            q: Query tensor of shape (batch_size, seq_len, d_head).
            k: Key tensor of shape (batch_size, seq_len, d_head).
            v: Value tensor of shape (batch_size, seq_len, d_head).
            is_causal: Whether to apply causal masking.

        Returns:
            out: Output tensor of shape (batch_size, seq_len, d_head).
            logsumexp: Logsumexp tensor of shape (batch_size, seq_len).
        """

        batch_size, seq_len_q, d_head = q.shape
        _, seq_len_k, _ = k.shape
        # Tiles of size (Bq * d)
        Bq = 16**2
        Tq = (seq_len_q + Bq - 1) // Bq

        # Tiles of size (Bk * d)
        Bk = 16**2
        Tk = (seq_len_k + Bk - 1) // Bk

        rsqrt_d = d_head**-0.5

        out = torch.empty_like(q)
        logsumexp = torch.empty((batch_size, seq_len_q), device=q.device, dtype=torch.float32)

        for b in range(batch_size):
            q_b = q[b]
            k_b = k[b]
            v_b = v[b]

            for i in range(Tq):
                qi = q_b[i * Bq : (i + 1) * Bq]
                # Account for possible non divisibility of Nq
                current_bq = qi.size(0)

                # setup HBM outputs
                oi = torch.zeros_like(qi, dtype=torch.float32)
                li = torch.zeros((current_bq,), device=qi.device, dtype=torch.float32)
                mi = torch.full((current_bq,), -torch.inf, device=qi.device, dtype=torch.float32)

                limit_k = Tk
                if is_causal:
                    limit_k = min(Tk, i + 1)

                for j in range(limit_k):
                    kj = k_b[j * Bk : (j + 1) * Bk]
                    vj = v_b[j * Bk : (j + 1) * Bk]

                    sij = qi @ kj.T * rsqrt_d

                    if is_causal:
                        row_idx = i * Bq + torch.arange(current_bq, device=qi.device)[:, None]
                        col_idx = j * Bk + torch.arange(kj.size(0), device=qi.device)[None, :]
                        mask = col_idx > row_idx
                        sij = torch.where(mask, float("-inf"), sij)

                    rowmax = torch.amax(sij, dim=-1)
                    mij = torch.maximum(mi, rowmax)
                    pij = torch.exp(sij - mij[..., None])

                    scale_diff = torch.exp(mi - mij)
                    li = scale_diff * li + torch.sum(pij, dim=-1)

                    oi = scale_diff[:, None] * oi + pij @ vj

                    # book-keeping
                    mi = mij

                Oi = oi / li[..., None]
                Li = mi + torch.log(li)

                out[b, i * Bq : (i + 1) * Bq] = Oi
                logsumexp[b, i * Bq : (i + 1) * Bq] = Li

        ctx.save_for_backward(q, k, v, out, logsumexp)
        ctx.is_causal = is_causal

        return out

    @staticmethod
    def backward(ctx: typing.Any, *grad_outputs: typing.Any) -> typing.Any:
        """
        FlashAttention-2 backward pass.

        Args:
            ctx: Context containing saved tensors.
            grad_outputs: Gradients with respect to the outputs of forward.
                Expected to contain (grad_out, grad_logsumexp).

        Returns:
            grad_q: Gradient with respect to Q.
            grad_k: Gradient with respect to K.
            grad_v: Gradient with respect to V.
            grad_is_causal: Gradient with respect to is_causal (None).
        """
        raise NotImplementedError
