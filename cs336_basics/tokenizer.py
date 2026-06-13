from __future__ import annotations

import logging
import multiprocessing
import os
import time
import regex as re
from collections import defaultdict, Counter
from itertools import pairwise
from typing import Iterator, BinaryIO

logger = logging.getLogger(__name__)


# from example code
def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_tokens: list[bytes],
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert all(isinstance(t, bytes) for t in split_special_tokens), "Must represent special token as a bytestring"

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
            found_at = min(mini_chunk.find(t) for t in split_special_tokens)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def count_pretokens(chunk):
    _pretoken_counts: dict[str, int] = defaultdict(int)
    # Run pre-tokenization on your chunk and store the counts for each pre-token
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for pretoken in re.finditer(PAT, chunk):
        _pretoken_counts[pretoken.group()] += 1
    return _pretoken_counts


def pretokenize(input_path, special_tokens: list[str]):
    special_tokens_bytes = [bytes(t, "utf-8") for t in special_tokens]

    # pre-tokenize
    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, special_tokens_bytes)

        chunks = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            PAT = "|".join(re.escape(d) for d in special_tokens)
            chunks.extend(re.split(PAT, chunk))

        with multiprocessing.Pool() as pool:
            pretoken_counts_list = pool.map(count_pretokens, chunks)
            pretoken_counts = sum((Counter(p) for p in pretoken_counts_list), start=Counter())
    return pretoken_counts


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    verbose: bool = False,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    log = logger.info if verbose else (lambda *a, **kw: None)
    t0 = time.perf_counter()

    merges = []
    pretoken_counts = pretokenize(input_path, special_tokens)

    t_pretokenize = time.perf_counter()
    log(f"[timing] pre-tokenize: {t_pretokenize - t0:.3f}s  ({len(pretoken_counts)} unique pretokens)")

    # initial tokenize
    special_tokens_bytes = [bytes(t, "utf-8") for t in special_tokens]
    vocab = set(special_tokens_bytes + [bytes([i]) for i in range(256)])
    pretoken_to_tokens: dict[str, list[bytes]] = {}
    for pretoken in pretoken_counts.keys():
        tokens_list = [bytes([t]) for t in bytes(pretoken, "utf-8")]
        pretoken_to_tokens[pretoken] = tokens_list

    t_init = time.perf_counter()
    log(f"[timing] initial tokenize: {t_init - t_pretokenize:.3f}s  (initial vocab size {len(vocab)})")

    # big loop, keep repeating until we hit desired vocab size
    t_loop_start = time.perf_counter()
    iter_count = 0
    t_count_total = 0.0
    t_merge_total = 0.0

    while len(vocab) < vocab_size:
        # count pairs
        t_count_start = time.perf_counter()
        token_pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
        for pretoken, pretoken_count in pretoken_counts.items():
            for token_pair in pairwise(pretoken_to_tokens[pretoken]):
                token_pair_counts[token_pair] += pretoken_count

        # merge highest count, break ties lexicographically
        merge_pair, _ = max(token_pair_counts.items(), key=lambda p: (p[1], p[0]))
        t_count_total += time.perf_counter() - t_count_start

        merges.append(merge_pair)
        new_token = merge_pair[0] + merge_pair[1]
        vocab.add(new_token)

        # actually execute the merge
        t_merge_start = time.perf_counter()
        for pretoken, tokens_list in pretoken_to_tokens.items():
            i = 0
            while i < len(tokens_list):
                if i < len(tokens_list) - 1 and (tokens_list[i], tokens_list[i+1]) == merge_pair:
                    tokens_list[i] = new_token
                    tokens_list.pop(i+1)
                i += 1
        t_merge_total += time.perf_counter() - t_merge_start

        iter_count += 1
        if iter_count % 50 == 0:
            elapsed = time.perf_counter() - t_loop_start
            log(f"[timing] iter {iter_count:4d}  vocab={len(vocab)}  "
                f"elapsed={elapsed:.1f}s  count={t_count_total:.2f}s  merge={t_merge_total:.2f}s")

    t_loop_end = time.perf_counter()
    log(f"[timing] merge loop total: {t_loop_end - t_loop_start:.3f}s over {iter_count} iters  "
        f"(count={t_count_total:.2f}s  merge={t_merge_total:.2f}s)")

    vocab = {
        i: val for i, val in enumerate(sorted(vocab))
    }
    log(f"[timing] total: {time.perf_counter() - t0:.3f}s")
    return vocab, merges


# temporary test code
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    vocab, merges = train_bpe(
        input_path="tests/fixtures/corpus.en",
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
        verbose=True,
    )
    print("\n".join(str(p) for p in merges))
