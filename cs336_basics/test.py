import pickle
import logging

from cs336_basics.tokenizer import Tokenizer
from cs336_basics.tokenizer_v2 import train_bpe


# temporary test code
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # vocab, merges = train_bpe(
    #     input_path="data/TinyStoriesV2-GPT4-train.txt",
    #     vocab_size=10000,
    #     special_tokens=["<|endoftext|>"],
    #     verbose=True,
    # )

    # with open("data/TinyStories_tokenizer_vocab.pkl", "wb") as f:
    #     pickle.dump(vocab, f)
    # with open("data/TinyStories_tokenizer_merges.pkl", "wb") as f:
    #     pickle.dump(merges, f)

    tokenizer = Tokenizer.from_files(
        vocab_filepath="data/TinyStories_tokenizer_vocab.pkl",
        merges_filepath="data/TinyStories_tokenizer_merges.pkl",
        special_tokens=["<|endoftext|>"],
    )
    with open("data/TinyStoriesV2-GPT4-valid.txt", "r") as f:
        encoding = tokenizer.encode_iterable(f)
        for _ in range(500):
            print(next(encoding), end=", ")
        print()
