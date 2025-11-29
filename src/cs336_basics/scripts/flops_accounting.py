"""
Consider GPT-2 XL, which has the following configuration:
vocab_size : 50,257
context_length : 1,024
num_layers : 48
d_model : 1,600
num_heads : 25
d_ff 6,400

Suppose we constructed our model using this configuration.
How many trainable parameters would our model have?
Assuming each parameter is represented using single-precision floating point,
how much memory is required to just load this model?
"""

import logging


def accounting_a(
    vocab_size: int,
    num_layers: int,
    d_model: int,
    d_ff: int,
) -> None:
    """Accounting for the number of trainable parameters"""
    logger = logging.getLogger("accounting_a")
    logger.setLevel(logging.INFO)

    params = {}

    # Token and positional embeddings
    params["embedding"] = vocab_size * d_model

    # Transformer blocks
    # Each layer has 2 RMSNorm layers
    params["block_rms_norm"] = 2 * num_layers * d_model

    # Each MHSA has QKV projection and one output projection
    params["msha_projections"] = 4 * num_layers * d_model * d_model

    # Each FFN is implemented via SwiGLU
    params["ffn_swiglu"] = 3 * num_layers * d_model * d_ff

    # Final LayerNorm
    params["final_norm"] = d_model

    # Final linear layer to produce logits
    params["final_linear"] = d_model * vocab_size

    total_params = sum(params.values())

    # Assuming single-precision floating point (4 bytes per parameter)
    memory_gb = total_params * 4 / (1024**3)

    logger.info(f"Total trainable parameters: {total_params:,}")
    logger.info(f"Memory to load model: {memory_gb:.2f} GiB")
    logger.info("-" * 40)
    logger.info("Parameter distribution (descending):")

    # Sort parameters by value in descending order
    sorted_params = sorted(params.items(), key=lambda item: item[1], reverse=True)

    for name, count in sorted_params:
        percentage = (count / total_params) * 100
        logger.info(f"{name:<20}: {count:>15,} ({percentage:.2f}%)")


def accounting_b(
    vocab_size: int,
    context_length: int,
    num_layers: int,
    d_model: int,
    d_ff: int,
) -> None:
    """Accounting for the FLOPs of matrices' operations"""
    logger = logging.getLogger("accounting_b")
    logger.setLevel(logging.INFO)

    flops = {}

    flops["mhsa_projections"] = num_layers * 2 * 4 * context_length * d_model * d_model
    flops["mhsa_attention"] = num_layers * 2 * 2 * context_length * context_length * d_model

    flops["ffn"] = num_layers * 2 * 3 * context_length * d_model * d_ff
    flops["final_linear"] = 2 * context_length * d_model * vocab_size

    total_flops = sum(flops.values())

    logger.info(f"\nTotal FLOPs for a single forward pass: {total_flops:,.0f}")
    logger.info("-" * 40)
    logger.info("FLOPs distribution (descending):")

    sorted_flops = sorted(flops.items(), key=lambda item: item[1], reverse=True)

    for name, count in sorted_flops:
        percentage = (count / total_flops) * 100
        logger.info(f"{name:<20}: {count:>20,.0f} ({percentage:.2f}%)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    # Common parameters for GPT-2 models
    VOCAB_SIZE = 50_257
    CONTEXT_LENGTH = 1_024

    # --- GPT-2 Small ---
    logger.info("\n" + "=" * 20 + " GPT-2 Small Analysis " + "=" * 20)
    accounting_a(vocab_size=VOCAB_SIZE, num_layers=12, d_model=768, d_ff=768 * 4)
    accounting_b(vocab_size=VOCAB_SIZE, context_length=CONTEXT_LENGTH, num_layers=12, d_model=768, d_ff=768 * 4)

    # --- GPT-2 Medium ---
    logger.info("\n" + "=" * 20 + " GPT-2 Medium Analysis " + "=" * 20)
    accounting_a(vocab_size=VOCAB_SIZE, num_layers=24, d_model=1024, d_ff=1024 * 4)
    accounting_b(vocab_size=VOCAB_SIZE, context_length=CONTEXT_LENGTH, num_layers=24, d_model=1024, d_ff=1024 * 4)

    # --- GPT-2 Large ---
    logger.info("\n" + "=" * 20 + " GPT-2 Large Analysis " + "=" * 20)
    accounting_a(vocab_size=VOCAB_SIZE, num_layers=36, d_model=1280, d_ff=1280 * 4)
    accounting_b(vocab_size=VOCAB_SIZE, context_length=CONTEXT_LENGTH, num_layers=36, d_model=1280, d_ff=1280 * 4)

    # --- GPT-2 XL ---
    logger.info("\n" + "=" * 20 + " GPT-2 XL Analysis " + "=" * 20)
    accounting_a(vocab_size=VOCAB_SIZE, num_layers=48, d_model=1600, d_ff=1600 * 4)
    accounting_b(vocab_size=VOCAB_SIZE, context_length=CONTEXT_LENGTH, num_layers=48, d_model=1600, d_ff=1600 * 4)

    # --- GPT-2 XL context length 16,384---
    logger.info("\n" + "=" * 20 + " GPT-2 XL Analysis " + "=" * 20)
    accounting_b(vocab_size=VOCAB_SIZE, context_length=16384, num_layers=48, d_model=1600, d_ff=1600 * 4)
