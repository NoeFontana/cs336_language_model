import typing

import torch


class TorchFlashAttn2Fn(torch.autograd.Function):
    """A pure PyTorch implementation of FlashAttentionv2.

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
        q_shape = q.shape

        # Tiles of size (Bq * d)
        Bq = 16**2
        Tq = torch.ceil(torch.tensor(q_shape[1] / Bq)).to(torch.long)

        # Tiles of size (Bk * d)
        Bk = 16**2
        Tk = torch.ceil(torch.tensor(k.shape[1] / Bk)).to(torch.long)

        d = q.size(-1)
        rsqrt_d = torch.rsqrt(torch.tensor(d))

        out = torch.empty_like(q)
        logsumexp = torch.empty(q_shape[:-1])

        batch_size = q_shape[0]
        for b in range(batch_size):
            q_b = q[b]
            k_b = k[b]
            v_b = v[b]

            for i in range(Tq):
                qi = q_b[i * Bq : (i + 1) * Bq]
                # Account for possible non divisibility of Nq
                current_bq = qi.size(0)

                # setup HBM outputs
                oi = torch.zeros_like(qi)
                li = torch.zeros((current_bq,), device=qi.device)
                mi = torch.full((current_bq,), -torch.inf, device=qi.device)

                for j in range(Tk):
                    kj = k_b[j * Bk : (j + 1) * Bk]
                    vj = v_b[j * Bk : (j + 1) * Bk]

                    sij = qi @ kj.T * rsqrt_d

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
