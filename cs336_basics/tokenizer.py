from __future__ import annotations

import logging
import multiprocessing
import os
import pickle
import time
import regex as re
from collections import defaultdict, Counter
from itertools import pairwise, repeat
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
            found_at = min(
                (idx for t in split_special_tokens if (idx := mini_chunk.find(t)) != -1),
                default=-1
            )
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def count_pretokens(start, end, input_path, special_tokens):
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        PAT = "|".join(re.escape(d) for d in special_tokens)
        chunks = re.split(PAT, chunk)

    _pretoken_counts: Counter[str] = Counter()
    for chunk in chunks:
        # Run pre-tokenization on your chunk and store the counts for each pre-token
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        for pretoken in re.finditer(PAT, chunk):
            _pretoken_counts[pretoken.group()] += 1
    return _pretoken_counts


def pretokenize(input_path, special_tokens: list[str]):
    special_tokens_bytes = [bytes(t, "utf-8") for t in special_tokens]

    with open(input_path, "rb") as f:
        num_processes = 8
        boundaries = find_chunk_boundaries(f, num_processes, special_tokens_bytes)

    with multiprocessing.Pool() as pool:
        pretoken_counts_list = pool.starmap(count_pretokens, zip(
            boundaries[:-1], boundaries[1:], repeat(input_path), repeat(special_tokens),
        ))
        pretoken_counts = sum((Counter(p) for p in pretoken_counts_list), start=Counter())
    return pretoken_counts


def apply_merge(tokens_list, merge_pair):
    new_token = merge_pair[0] + merge_pair[1]
    new_tokens_list = []
    i = 0
    while i < len(tokens_list):
        if i < len(tokens_list) - 1 and (tokens_list[i], tokens_list[i+1]) == merge_pair:
            new_tokens_list.append(new_token)
            i += 1
        else:
            new_tokens_list.append(tokens_list[i])
        i += 1
    return new_tokens_list


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    verbose: bool = False,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    t0 = time.perf_counter()

    merges = []
    pretoken_counts = pretokenize(input_path, special_tokens)

    t_pretokenize = time.perf_counter()
    logger.info(f"[timing] pre-tokenize: {t_pretokenize - t0:.3f}s  ({len(pretoken_counts)} unique pretokens)")

    # initial tokenize
    special_tokens_bytes = [bytes(t, "utf-8") for t in special_tokens]
    vocab = set(special_tokens_bytes + [bytes([i]) for i in range(256)])
    pretoken_to_tokens: dict[str, list[bytes]] = {}
    for pretoken in pretoken_counts.keys():
        tokens_list = [bytes([t]) for t in bytes(pretoken, "utf-8")]
        pretoken_to_tokens[pretoken] = tokens_list

    t_init = time.perf_counter()
    logger.info(f"[timing] initial tokenize: {t_init - t_pretokenize:.3f}s  (initial vocab size {len(vocab)})")

    # big loop, keep repeating until we hit desired vocab size
    t_loop_start = time.perf_counter()
    iter_count = 0
    t_count_total = 0.0
    t_merge_total = 0.0

    while len(vocab) < vocab_size:
        # count pairs
        t_count_start = time.perf_counter()
        token_pair_counts: Counter[tuple[bytes, bytes]] = Counter()
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
            pretoken_to_tokens[pretoken] = apply_merge(tokens_list, merge_pair)
        t_merge_total += time.perf_counter() - t_merge_start

        iter_count += 1
        if iter_count % 50 == 0:
            elapsed = time.perf_counter() - t_loop_start
            logger.info(f"[timing] iter {iter_count:4d}  vocab={len(vocab)}  "
                f"elapsed={elapsed:.1f}s  count={t_count_total:.2f}s  merge={t_merge_total:.2f}s")

    t_loop_end = time.perf_counter()
    logger.info(f"[timing] merge loop total: {t_loop_end - t_loop_start:.3f}s over {iter_count} iters  "
        f"(count={t_count_total:.2f}s  merge={t_merge_total:.2f}s)")

    vocab = {
        i: val for i, val in enumerate(sorted(vocab))
    }
    logger.info(f"[timing] total: {time.perf_counter() - t0:.3f}s")
    return vocab, merges


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        for special_token in special_tokens or []:
            if special_token not in vocab:
                vocab[len(vocab)] = special_token
        self.vocab: dict[int, bytes] = vocab
        self.inverse_vocab: dict[bytes, int] = {v: k for k, v in vocab.items()}
        self.merges: list[tuple[bytes, bytes]] = merges
        self.inverse_merges: dict[tuple[bytes, bytes], int] = {v: k for k, v in enumerate(self.merges)}
        self.special_tokens: list[str] = special_tokens or []
        self.pretoken_encodings: dict[str, list[int]] = defaultdict(list)

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return Tokenizer(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        return list(self.encode_iterable(iter([text])))

    def next_chunk(self, iterable: Iterator[str]):
        # Read until hit special token or EOF
        chunk = ""
        for minichunk in iterable:
            while minichunk:
                # Find the special token in the mini chunk
                found_at, _, split_token = min(
                    ((idx, -len(t), t) for t in self.special_tokens if (idx := minichunk.find(t)) != -1),
                    default=(-1, 0, None),
                )
                if found_at != -1:
                    yield chunk + minichunk[:found_at]
                    yield split_token
                    chunk = ""
                    minichunk = minichunk[found_at + len(split_token):]
                else:
                    chunk += minichunk
                    minichunk = ""
        if chunk:
            yield chunk

    def encode_iterable(self, iterable: Iterator[str]) -> Iterator[int]:
        t_start = time.perf_counter()
        t_last_log = t_start
        bytes_seen = 0
        tokens_emitted = 0
        chunks_seen = 0
        cache_hits = 0
        cache_misses = 0
        LOG_EVERY = 5.0  # seconds

        for chunk in self.next_chunk(iterable):
            chunks_seen += 1
            bytes_seen += len(chunk.encode("utf-8"))
            if chunk in self.special_tokens:
                yield self.inverse_vocab[bytes(chunk, "utf-8")]
                tokens_emitted += 1
                continue
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            for match in re.finditer(PAT, chunk):
                pretoken = match.group()
                if pretoken in self.pretoken_encodings:
                    cache_hits += 1
                    cached = self.pretoken_encodings[pretoken]
                    yield from cached
                    tokens_emitted += len(cached)
                    continue
                cache_misses += 1
                tokens_list = [bytes([t]) for t in bytes(pretoken, "utf-8")]
                # Apply merges
                while True:
                    merge_pair = min(
                        (pair for pair in pairwise(tokens_list) if pair in self.inverse_merges),
                        key=lambda pair: self.inverse_merges[pair],
                        default=None
                    )
                    if not merge_pair:
                        break
                    tokens_list = apply_merge(tokens_list, merge_pair)
                encoding = [self.inverse_vocab[token] for token in tokens_list]
                yield from encoding
                tokens_emitted += len(encoding)
                self.pretoken_encodings[pretoken] = encoding

            now = time.perf_counter()
            if now - t_last_log >= LOG_EVERY:
                elapsed = now - t_start
                mb = bytes_seen / (1 << 20)
                rate_mb = mb / elapsed if elapsed else 0.0
                rate_tok = tokens_emitted / elapsed if elapsed else 0.0
                total_lookups = cache_hits + cache_misses
                hit_rate = cache_hits / total_lookups if total_lookups else 0.0
                logger.info(
                    f"[encode] elapsed={elapsed:6.1f}s  "
                    f"bytes={mb:8.2f}MB ({rate_mb:6.2f}MB/s)  "
                    f"tokens={tokens_emitted:>10d} ({rate_tok:8.0f}/s)  "
                    f"cache_hit={hit_rate:.3f}  cache_size={len(self.pretoken_encodings)}"
                )
                t_last_log = now

        elapsed = time.perf_counter() - t_start
        mb = bytes_seen / (1 << 20)
        rate_mb = mb / elapsed if elapsed else 0.0
        logger.info(
            f"[encode] DONE  elapsed={elapsed:.1f}s  "
            f"bytes={mb:.2f}MB ({rate_mb:.2f}MB/s)  tokens={tokens_emitted}"
        )

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[token] for token in ids).decode("utf-8", errors="replace")
