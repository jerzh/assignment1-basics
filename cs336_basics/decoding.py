from __future__ import annotations

import argparse
import logging

import torch

from cs336_basics.modules import softmax
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.transformer import TransformerLM


def generate_completion(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 1,
    top_p_threshold: float = 1,
    device: str | None = None,
):
    """ Generates completion given a prompt. Not batched """
    if max_tokens > model.context_length:
        raise ValueError("Max tokens cannot be longer than context length")
    completion = torch.tensor(tokenizer.encode(prompt), device=device).unsqueeze(0)
    end_token = tokenizer.encode("<|endoftext|>")[0]
    next_token = completion[:, -1]
    model.eval()
    with torch.no_grad():
        while completion.shape[1] < max_tokens and next_token.item() != end_token:
            logits = model.forward(completion)[:, -1]
            probs = softmax(logits / temperature, -1)
            if top_p_threshold < 1:
                sorted_probs, indices = torch.sort(probs, dim=1)
                partials = torch.cumsum(sorted_probs, dim=1)
                sorted_probs = torch.where(partials > 1 - top_p_threshold, sorted_probs, 0)
                sorted_probs /= torch.sum(sorted_probs, dim=1, keepdim=True)
                probs = sorted_probs.gather(dim=1, index=torch.argsort(indices))
            next_token = probs.multinomial(num_samples=1)
            completion = torch.concat([completion, next_token], dim=1)
    return tokenizer.decode(completion.squeeze(0).tolist())


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    # ---- data ----
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--vocab-path", type=str, default="TinyStories_tokenizer_vocab.pkl")
    p.add_argument("--merges-path", type=str, default="TinyStories_tokenizer_merges.pkl")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=1)
    p.add_argument("--top-p-threshold", type=float, default=1)

    # ---- model architecture ----
    p.add_argument("--vocab-size", type=int, default=10_000)
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=16)
    p.add_argument("--d-ff", type=int, default=1344)   # ~ (8/3) * d_model, rounded to multiple of 64
    p.add_argument("--rope-theta", type=float, default=10000.0)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", type=str, default="float32")
    args = p.parse_args()

    if args.dtype == "float32":
        dtype = torch.float32
    else:
        raise ValueError("Only f32 supported")

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=args.device,
        dtype=torch.float32,
    )
    tokenizer = Tokenizer.from_files(
        vocab_filepath=args.vocab_path,
        merges_filepath=args.merges_path,
        special_tokens=["<|endoftext|>"],
    )
    obj = torch.load(args.model_path, weights_only=True)
    model.load_state_dict(obj["model"])
    completion = generate_completion(
        model,
        tokenizer,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p_threshold=args.top_p_threshold,
        device=args.device,
    )
    print(completion)
