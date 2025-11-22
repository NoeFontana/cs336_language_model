import argparse
import logging

import torch

from cs336.layer.transformer import softmax
from cs336.tokenizer import Tokenizer
from cs336.transformer import TransformerLM


def get_args() -> argparse.Namespace:
    """Parse and return command-line arguments for decoding."""
    parser = argparse.ArgumentParser(description="Generate text from a Transformer Language Model.")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time",
        help="The prompt to start generation from.",
    )
    parser.add_argument(
        "--checkpoint_file",
        type=str,
        required=True,
        help="Path to the model checkpoint file (.pt).",
    )
    parser.add_argument(
        "--vocab_file",
        type=str,
        default="results/owt.json",
        help="Path to the vocabulary file (JSON format).",
    )
    parser.add_argument(
        "--merges_file",
        type=str,
        help="Path to the merges file. Optional if merges are in the vocab file.",
    )
    parser.add_argument("--vocab_size", type=int, default=32000, help="Vocabulary size.")
    parser.add_argument("--context_length", type=int, default=256, help="Context length for the model.")
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension.")
    parser.add_argument("--d_ff", type=int, default=2048, help="Feed-forward dimension.")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads.")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of transformer layers.")
    parser.add_argument("--theta", type=float, default=10000.0, help="Theta for RoPE.")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum number of new tokens to generate.")
    parser.add_argument("--end_of_text_token", type=str, default="<|endoftext|>", help="End-of-text token.")
    parser.add_argument("--max_seq_len", type=int, default=256, help="Maximum sequence length for the model.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling probability.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature. 1.0 means no change.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to train on.")
    return parser.parse_args()


@torch.inference_mode()
def decode(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    end_of_text_token_id: int,
    max_seq_len: int,
    temperature: float,
    top_p: float,
    device: str,
) -> str:
    """
    Generates text from a model given a prompt.

    Args:
        model: The language model.
        tokenizer: The tokenizer.
        prompt: The starting text for generation.
        max_new_tokens: The maximum number of tokens to generate.
        end_of_text_token_id: The token id indicating the end of text.
        max_seq_len: The maximum sequence length for the model.
        temperature: The temperature for sampling. Higher values ( > 1.0) make
            the output more random, lower values ( < 1.0) make it more deterministic.
        top_p: The cumulative probability for nucleus sampling. 0.0 < top_p <= 1.0.
            If set to 1.0, it's equivalent to standard sampling.
        device: The device to run the generation on.

    Returns:
        The generated text, including the prompt.
    """
    model.eval()

    prompt_tokens = tokenizer.encode(prompt)
    tokens = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(max_new_tokens):
        # Crop context if it exceeds max_seq_len
        current_tokens = tokens if tokens.size(1) <= max_seq_len else tokens[:, -max_seq_len:]

        logits = model(current_tokens)
        last_token_logits = logits[:, -1, :]

        last_token_logits /= temperature

        probs = softmax(last_token_logits, dim=-1)

        probs, indices = torch.sort(probs, descending=True, dim=-1)

        cumulative_probs = torch.cumsum(probs, dim=-1)

        # Ensures that the first token is always kept, and we include
        # the token that pushes the cumulative probability over top_p.
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        keep_mask = ~sorted_indices_to_remove
        probs[~keep_mask] = 0.0
        renormalized_probs = probs / probs.sum(dim=-1, keepdim=True)

        sampled_idx = torch.multinomial(renormalized_probs, num_samples=1)
        next_token = indices.gather(1, sampled_idx)

        tokens = torch.cat((tokens, next_token), dim=1)

        if next_token.item() == end_of_text_token_id:
            break

    generated_text = tokenizer.decode(tokens[0].tolist())
    return generated_text


def main() -> None:
    """
    Main function to generate text from a trained Transformer Language Model.
    """
    args = get_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)

    logger.info("Loading tokenizer...")
    tokenizer = Tokenizer.from_files(vocab_filepath=args.vocab_file, merges_filepath=args.merges_file)
    end_of_text_token_id = tokenizer.encode(args.end_of_text_token)[0]

    logger.info("Initializing model...")
    model: TransformerLM = TransformerLM(
        vocab_size=args.vocab_size,
        max_seq_len=args.context_length,
        d_model=args.d_model,
        d_ff=args.d_ff,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        theta=args.theta,
    ).to(args.device)

    logger.info(f"Loading model checkpoint from {args.checkpoint_file}...")
    model.load_state_dict(torch.load(args.checkpoint_file, map_location=args.device)["model"])
    model.eval()

    logger.info("Generating completion...")
    generated_text = decode(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        end_of_text_token_id=end_of_text_token_id,
        max_seq_len=args.max_seq_len,
        temperature=args.temperature,
        top_p=args.top_p,
        device=args.device,
    )

    print("\n--- Generated Text ---")
    print(generated_text)
    print("----------------------")


if __name__ == "__main__":
    main()
