import array
import pickle
import logging

from cs336_basics.tokenizer import Tokenizer
from cs336_basics.tokenizer_v2 import train_bpe


def train_tokenizer(in_path, vocab_size, vocab_path, merges_path):
    vocab, merges = train_bpe(
        input_path=in_path,
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>"],
        verbose=True,
    )

    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    with open(merges_path, "wb") as f:
        pickle.dump(merges, f)


def sample_10(in_path, tokenizer):
    # Sample 10 documents (delimited by <|endoftext|>)
    SEP = "<|endoftext|>"
    n_docs = 10
    docs: list[str] = []
    buf = ""
    with open(in_path, "r") as f:
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


def tokenize(tokenizer: Tokenizer, in_path, out_path):
    assert len(tokenizer.vocab) <= 65536, "vocab too large for uint16"
    BATCH = 1_000_000  # tokens per flush (~2 MB at uint16)
    buf = array.array("H")  # unsigned short = uint16, native byte order
    with open(in_path, "r") as f, open(out_path, "wb") as out:
        for tid in tokenizer.encode_iterable(f):
            buf.append(tid)
            if len(buf) >= BATCH:
                buf.tofile(out)
                del buf[:]
        if buf:
            buf.tofile(out)
    # Load with: np.fromfile("data/TinyStories_encoded.bin", dtype=np.uint16)


if __name__ == "__main__":
    train_path = "data/owt_train.txt"
    valid_path = "data/owt_valid.txt"
    vocab_size = 32000
    out_path = "data/owt_encoded_train.bin"
    vocab_path = "data/owt_tokenizer_vocab.pkl"
    merges_path = "data/owt_tokenizer_merges.pkl"

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    train_tokenizer(train_path, vocab_size, vocab_path, merges_path)

    # tokenizer = Tokenizer.from_files(
    #     vocab_filepath=vocab_path,
    #     merges_filepath=merges_path,
    #     special_tokens=["<|endoftext|>"],
    # )
    # sample_10(train_path, tokenizer)
    # tokenize(valid_path, out_path, vocab_path, merges_path)
