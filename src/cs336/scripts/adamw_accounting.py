"""
a. How much peak memory does running AdamW require? Decompose your answer based on the memory usage of the parameters,
activations, gradients, and optimizer state. Express your answer in terms of the batch_size and the model
hyperparameters (vocab_size, context_length, num_layers, d_model, num_heads). Assume d_ff = 4 × d_model.
"""

import logging
from typing import cast

import sympy  # type: ignore


def accounting_a(
    batch_size: sympy.Symbol,
    vocab_size: sympy.Symbol,
    context_length: sympy.Symbol,
    num_layers: sympy.Symbol,
    d_model: sympy.Symbol,
    num_heads: sympy.Symbol,
) -> tuple[sympy.Expr, sympy.Expr]:
    """Accounting for the number of trainable parameters"""
    logger = logging.getLogger("accounting_a")
    logger.setLevel(logging.INFO)

    params = {}

    # We first start with counts then will translate them into memory estimates

    # Token and positional embeddings
    params["embedding/parameters"] = vocab_size * d_model
    # params["embedding/activations"] = batch_size * context_length * d_model
    params["embedding/gradients"] = vocab_size * d_model
    params["embedding/optimizer_state"] = 2 * vocab_size * d_model

    # Transformer blocks
    # Each layer has 2 RMSNorm layers
    params["block_rms_norm/parameters"] = 2 * num_layers * d_model
    params["block_rms_norm/activations"] = 2 * num_layers * batch_size * context_length * d_model
    params["block_rms_norm/gradients"] = 2 * num_layers * d_model
    params["block_rms_norm/optimizer_state"] = 2 * 2 * num_layers * d_model

    # Each MHSA has QKV projection and one output projection
    params["msha_projections/parameters"] = 4 * num_layers * d_model * d_model
    # QKV share the same input, so we can store only 2 inputs instead of 4
    # # params["msha_projections/activations"] = 4 * num_layers * batch_size * context_length * d_model
    params["msha_projections/activations"] = (
        2 * num_layers * batch_size * context_length * d_model
        + num_layers * batch_size * num_heads * context_length * context_length
    )
    params["msha_projections/gradients"] = 4 * num_layers * d_model * d_model
    params["msha_projections/optimizer_state"] = 2 * 4 * num_layers * d_model * d_model

    # Each FFN is implemented via SwiGLU
    params["ffn_swiglu/parameters"] = 3 * num_layers * d_model * 4 * d_model
    # d_model -> 4 d_model -> d_model
    params["ffn_swiglu/activations"] = 6 * num_layers * batch_size * context_length * d_model
    params["ffn_swiglu/gradients"] = 3 * num_layers * d_model * 4 * d_model
    params["ffn_swiglu/optimizer_state"] = 2 * 3 * num_layers * d_model * 4 * d_model

    # Final LayerNorm
    params["final_norm/parameters"] = d_model
    params["final_norm/activations"] = batch_size * context_length * d_model
    params["final_norm/gradients"] = d_model
    params["final_norm/optimizer_state"] = 2 * d_model

    # Final linear layer to produce logits
    params["final_linear/parameters"] = d_model * vocab_size
    params["final_linear/activations"] = batch_size * context_length * d_model
    params["final_linear/gradients"] = d_model * vocab_size
    params["final_linear/optimizer_state"] = 2 * d_model * vocab_size

    # Cross-entropy on logits
    params["cross_entropy_loss/parameters"] = 0
    params["cross_entropy_loss/activations"] = batch_size * context_length * vocab_size
    params["cross_entropy_loss/gradients"] = 0
    params["cross_entropy_loss/optimizer_state"] = 0

    # Group by memory type
    parameter_count_by_type = {
        "parameters": sum(v for k, v in params.items() if "parameters" in k),
        "activations": sum(v for k, v in params.items() if "activations" in k),
        "gradients": sum(v for k, v in params.items() if "gradients" in k),
        "optimizer_state": sum(v for k, v in params.items() if "optimizer_state" in k),
    }

    total_memory_elements = sum(parameter_count_by_type.values())
    total_memory_elements = cast(sympy.Expr, total_memory_elements)

    logger.info("-" * 40)
    logger.info("Algebraic expressions for memory usage (assumed float32):")
    for mem_type, expression in parameter_count_by_type.items():
        logger.info(f"{mem_type.capitalize():<16}: {sympy.expand(4 * expression)}")

    logger.info(f"{'Total':<16}: {sympy.expand(4 * total_memory_elements)}")
    logger.info("-" * 40)

    # (c) Adam FLOPs
    # AdamW updates conduct 14 operations on the parameters
    adam_w_flops = 14 * parameter_count_by_type["parameters"]
    adam_w_flops = cast(sympy.Expr, adam_w_flops)

    return total_memory_elements, adam_w_flops


def accounting_b(
    total_parameter_expr: sympy.Expr,
    adam_w_flops: sympy.Expr,
    batch_size: sympy.Symbol,
    context_length: int,
    num_layers: int,
    d_model: int,
    num_heads: int,
) -> None:
    """
    Instantiates the memory usage expression for a GPT-2 XL-shaped model
    and calculates the maximum batch size for an 80GB memory budget.
    """
    logger = logging.getLogger("accounting_b")
    logger.setLevel(logging.INFO)

    vocab_size_sym = sympy.Symbol("vocab_size")
    context_length_sym = sympy.Symbol("context_length")
    num_layers_sym = sympy.Symbol("num_layers")
    d_model_sym = sympy.Symbol("d_model")
    num_heads_sym = sympy.Symbol("num_heads")

    # Substitute GPT-2 XL hyperparameters
    vocab_size = 50257
    memory_expr_b = (4 * total_parameter_expr).subs(
        [
            (batch_size, batch_size),
            (vocab_size_sym, vocab_size),
            (context_length_sym, context_length),
            (num_layers_sym, num_layers),
            (d_model_sym, d_model),
            (num_heads_sym, num_heads),
        ]
    )

    logger.info("-" * 40)
    logger.info(f"Memory expression for GPT-2 XL: {memory_expr_b} (bytes)")

    # --- Solve for max batch size ---
    memory_budget_gib = 80
    memory_budget_bytes = memory_budget_gib * (1024**3)

    max_batch_size = 0
    while memory_expr_b.subs(batch_size, max_batch_size + 1) < memory_budget_bytes:
        max_batch_size += 1
    logger.info(f"With an 80 GiB memory budget the maximum batch size is: {max_batch_size}")

    # (c) AdamW FLOPs
    logger.info(f"AdamW FLOPs: {sympy.expand(adam_w_flops)}")
    adam_w_flops_gpt2 = sympy.expand(adam_w_flops).subs(
        [
            (num_layers_sym, num_layers),
            (d_model_sym, d_model),
            (vocab_size_sym, vocab_size),
        ]
    )

    logger.info(f"For GPT-2 XL: {adam_w_flops_gpt2}")


def accounting_d() -> None:
    logger = logging.getLogger("accounting_d")
    logger.setLevel(logging.INFO)

    A100_peak = 19.5 * 10**12  # 19.5 tflops
    A100_half_MFU = 0.5 * A100_peak

    flops_forward = 1_024 * 4_513_336_524_800  # Matrix multiplies
    flops_backwards = 2 * flops_forward
    flops_adamw = 29_778_806_400

    total_flops = (flops_forward + flops_backwards + flops_adamw) * 400_000

    seconds_of_training = total_flops / A100_half_MFU
    num_days_of_training = seconds_of_training / (60 * 60 * 24)

    logger.info(f"Training days: {num_days_of_training:.2f}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Define symbolic variables for model hyperparameters to derive algebraic expressions.
    batch_size_sym = sympy.Symbol("batch_size")
    vocab_size_sym = sympy.Symbol("vocab_size")
    context_length_sym = sympy.Symbol("context_length")
    num_layers_sym = sympy.Symbol("num_layers")
    d_model_sym = sympy.Symbol("d_model")
    num_heads_sym = sympy.Symbol("num_heads")

    total_memory_expr, adam_w_flops = accounting_a(
        batch_size=batch_size_sym,
        vocab_size=vocab_size_sym,
        context_length=context_length_sym,
        num_layers=num_layers_sym,
        d_model=d_model_sym,
        num_heads=num_heads_sym,
    )

    # --- Accounting Question (b) ---
    accounting_b(
        total_parameter_expr=total_memory_expr,
        adam_w_flops=adam_w_flops,
        batch_size=batch_size_sym,
        context_length=1024,
        num_layers=48,
        d_model=1600,
        num_heads=25,
    )

    accounting_d()
