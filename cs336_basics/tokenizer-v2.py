from __future__ import annotations

import logging
import os
import time
import regex as re
from collections import defaultdict
from itertools import pairwise
from typing import Iterator, BinaryIO

logger = logging.getLogger(__name__)


# from example code
def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    verbose: bool = False,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    log = logger.info if verbose else (lambda *a, **kw: None)
    t0 = time.perf_counter()

    merges = []
    # pre-tokenize
    pretoken_counts: dict[str, int] = defaultdict(int)
    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            for pretoken in re.finditer(PAT, chunk):
                pretoken_counts[pretoken.group()] += 1

    t_pretokenize = time.perf_counter()
    log(f"[timing] pre-tokenize: {t_pretokenize - t0:.3f}s  ({len(pretoken_counts)} unique pretokens)")

    # initial tokenize
    # vocab = token_to_pretokens.keys()
    pretoken_to_tokens: dict[str, list[bytes]] = {}
    token_to_pretokens: dict[bytes, set[str]] = defaultdict(set)
    for pretoken in pretoken_counts.keys():
        tokens_list = [bytes([t]) for t in bytes(pretoken, "utf-8")]
        pretoken_to_tokens[pretoken] = tokens_list
        for token in tokens_list:
            token_to_pretokens[token].add(pretoken)

    t_init = time.perf_counter()
    log(f"[timing] initial tokenize: {t_init - t_pretokenize:.3f}s  (initial vocab size {len(token_to_pretokens)})")

    # count pairs
    token_pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
    for pretoken, pretoken_count in pretoken_counts.items():
        for token_pair in pairwise(pretoken_to_tokens[pretoken]):
            token_pair_counts[token_pair] += pretoken_count

    # big loop, keep repeating until we hit desired vocab size
    t_loop_start = time.perf_counter()
    iter_count = 0
    t_count_total = 0.0
    t_merge_total = 0.0

    while len(token_to_pretokens) < vocab_size:
        # merge highest count, break ties lexicographically
        t_count_start = time.perf_counter()
        merge_pair, _ = max(token_pair_counts.items(), key=lambda p: (-p[1], p[0]))
        t_count_total += time.perf_counter() - t_count_start

        merges.append(merge_pair)
        new_token = merge_pair[0] + merge_pair[1]

        # actually execute the merge
        t_merge_start = time.perf_counter()
        merge_pretokens = (token_to_pretokens[merge_pair[0]], token_to_pretokens[merge_pair[1]])
        # remove all merge pretokens. we will re-add them manually during the merge
        token_to_pretokens.pop(merge_pair[0])
        if merge_pair[1] in token_to_pretokens:
            token_to_pretokens.pop(merge_pair[1])
        # temp variable to help with timing
        old_pretoken_to_tokens: dict[str, list[bytes]] = {}
        # only consider pretokens containing either merged token
        for pretoken in merge_pretokens[0] | merge_pretokens[1]:
            # update pretoken_to_tokens
            tokens_list = pretoken_to_tokens[pretoken]
            new_tokens_list = []
            i = 0
            while i < len(tokens_list):
                if i < len(tokens_list) - 1 and (tokens_list[i], tokens_list[i+1]) == merge_pair:
                    new_tokens_list.append(new_token)
                    i += 1
                else:
                    new_tokens_list.append(tokens_list[i])
                i += 1
            old_pretoken_to_tokens[pretoken] = tokens_list
            pretoken_to_tokens[pretoken] = new_tokens_list
            # update token_to_pretokens
            for token in new_tokens_list:
                token_to_pretokens[token].add(pretoken)
        t_merge_total += time.perf_counter() - t_merge_start

        def adj_count(_d, _k, _v):
            _d[_k] += _v
            if _d[_k] == 0:
                _d.pop(_k)

        # marginal changes to pair counts
        t_count_start = time.perf_counter()
        for pretoken in merge_pretokens[0] | merge_pretokens[1]:
            tokens_list = old_pretoken_to_tokens[pretoken]
            new_tokens_list = pretoken_to_tokens[pretoken]
            # remove old token pairs
            for token_pair in pairwise(tokens_list):
                adj_count(token_pair_counts, token_pair, -pretoken_counts[pretoken])
            # add new token pairs
            for token_pair in pairwise(new_tokens_list):
                adj_count(token_pair_counts, token_pair, pretoken_counts[pretoken])
        t_count_total += time.perf_counter() - t_count_start

        iter_count += 1
        if iter_count % 50 == 0:
            elapsed = time.perf_counter() - t_loop_start
            log(f"[timing] iter {iter_count:4d}  vocab={len(token_to_pretokens)}  "
                f"elapsed={elapsed:.1f}s  count={t_count_total:.2f}s  merge={t_merge_total:.2f}s")

    t_loop_end = time.perf_counter()
    log(f"[timing] merge loop total: {t_loop_end - t_loop_start:.3f}s over {iter_count} iters  "
        f"(count={t_count_total:.2f}s  merge={t_merge_total:.2f}s)")

    vocab = {
        i: val for i, val in enumerate(sorted(token_to_pretokens.keys()))
    }
    log(f"[timing] total: {time.perf_counter() - t0:.3f}s")
    return vocab, merges


# temporary test code
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    vocab, merges = train_bpe(
        input_path="/Users/jeremyzhou/repos/claude-projects/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt",
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
        verbose=True,
    )
    print(vocab)
