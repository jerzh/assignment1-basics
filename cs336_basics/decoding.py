from __future__ import annotations

import numpy as np
import torch

from cs336_basics.modules import softmax
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.transformer import TransformerLM


def generate_completion(model: TransformerLM, tokenizer: Tokenizer, prompt: str,
                        max_tokens: int, temperature: float = 1, top_p_threshold: float = 1):
    """ Generates completion given a prompt. Not batched """
    completion = torch.tensor(tokenizer.encode(prompt))
    next_token_str = None
    while len(completion) < max_tokens and next_token_str != "<|endoftext|>":
        logits = model.forward(completion)[-1]
        probs = softmax(logits / temperature, -1)
        if top_p_threshold < 1:
            sorted_probs, indices = torch.sort(probs)
            partials = torch.cumsum(sorted_probs, dim=0)
            sorted_probs = torch.where(partials > 1 - top_p_threshold, sorted_probs, 0)
            sorted_probs /= torch.sum(sorted_probs)
            probs = sorted_probs[torch.argsort(indices)]
        next_token = probs.multinomial(num_samples=1)
        next_token_str = tokenizer.decode(next_token)
        completion = torch.concat([completion, next_token])
    return tokenizer.decode(completion)
