import array
import pickle
import logging
import numpy as np

from cs336_basics.tokenizer import Tokenizer
from cs336_basics.tokenizer_v2 import train_bpe


def train_tokenizer():
    vocab, merges = train_bpe(
        input_path="data/TinyStoriesV2-GPT4-train.txt",
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
        verbose=True,
    )

    with open("data/TinyStories_tokenizer_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open("data/TinyStories_tokenizer_merges.pkl", "wb") as f:
        pickle.dump(merges, f)


def sample_10():
    # Sample 10 documents (delimited by <|endoftext|>)
    SEP = "<|endoftext|>"
    n_docs = 10
    docs: list[str] = []
    buf = ""
    with open("data/TinyStoriesV2-GPT4-train.txt", "r") as f:
        while len(docs) < n_docs:
            chunk = f.read(1 << 20)  # 1 MB
            if not chunk:
                break
            buf += chunk
            while len(docs) < n_docs and SEP in buf:
                doc, buf = buf.split(SEP, 1)
                docs.append(doc)

    total_bytes = 0
    total_tokens = 0
    for i, doc in enumerate(docs):
        n_bytes = len(doc.encode("utf-8"))
        n_tokens = len(tokenizer.encode(doc))
        total_bytes += n_bytes
        total_tokens += n_tokens
        ratio = n_bytes / n_tokens if n_tokens else float("inf")
        print(f"doc {i:2d}: {n_bytes:6d} bytes / {n_tokens:5d} tokens = {ratio:.3f} bytes/token")

    overall = total_bytes / total_tokens if total_tokens else float("inf")
    print(f"\nTOTAL: {total_bytes} bytes / {total_tokens} tokens = {overall:.3f} bytes/token")


# temporary test code
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    tokenizer = Tokenizer.from_files(
        vocab_filepath="data/TinyStories_tokenizer_vocab.pkl",
        merges_filepath="data/TinyStories_tokenizer_merges.pkl",
        special_tokens=["<|endoftext|>"],
    )
    assert len(tokenizer.vocab) <= 65536, "vocab too large for uint16"
    BATCH = 1_000_000  # tokens per flush (~2 MB at uint16)
    buf = array.array("H")  # unsigned short = uint16, native byte order
    out_path = "data/TinyStories_encoded.bin"
    with open("data/TinyStoriesV2-GPT4-train.txt", "r") as f, open(out_path, "wb") as out:
        for tid in tokenizer.encode_iterable(f):
            buf.append(tid)
            if len(buf) >= BATCH:
                buf.tofile(out)
                del buf[:]
        if buf:
            buf.tofile(out)
    # Load with: np.fromfile("data/TinyStories_encoded.bin", dtype=np.uint16)
