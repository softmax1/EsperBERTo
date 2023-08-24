from os import getenv
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv
from tap import Tap
from tokenizers.implementations import ByteLevelBPETokenizer


class Parser(Tap):
    test_pipeline: bool = False  # More quickly test the pipeline by not saving


def batch_iterator(dataset, batch_size: int = 1000):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i: i + batch_size]["text"]


def train(test_pipeline: bool = False):
    batch_size = 10000
    split = f'train[:{batch_size}]' if test_pipeline else 'train'
    dataset = load_dataset(f"{getenv('HUGGINGFACE_USER')}/esperanto", split=split)

    # Initialize a tokenizer.json
    tokenizer = ByteLevelBPETokenizer()

    # Customize training
    special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>", ]
    tokenizer.train_from_iterator(batch_iterator(dataset, batch_size), vocab_size=52032, min_frequency=2, special_tokens=special_tokens)

    if not test_pipeline:
        # Now let's save files to disk
        tokenizer_dir = Path.cwd() / "tokenizer"
        tokenizer_dir.mkdir(exist_ok=True)
        tokenizer.save_model(directory=str(tokenizer_dir))


def main():
    load_dotenv()
    args = Parser().parse_args()
    train(test_pipeline=args.test_pipeline)


if __name__ == '__main__':
    main()
