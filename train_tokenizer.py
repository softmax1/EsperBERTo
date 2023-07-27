from os import getenv
from pathlib import Path
from warnings import warn

from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login, logout
from tap import Tap
from tokenizers.implementations import ByteLevelBPETokenizer


class Parser(Tap):
    test_pipeline: bool = False  # More quickly test the pipeline by not saving


def train(test_pipeline: bool = False):
    paths = [str(x) for x in Path.cwd().glob("**/*.txt")]

    # Initialize a tokenizer.json
    tokenizer = ByteLevelBPETokenizer()

    # Customize training
    special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>", ]
    tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2, special_tokens=special_tokens)

    if not test_pipeline:
        # Now let's save files to disk
        tokenizer_dir = Path.cwd() / "tokenizer"
        tokenizer_dir.mkdir(exist_ok=True)
        tokenizer.save_model(directory=str(tokenizer_dir))

        # Let's upload the dataset while we're at it
        try:
            login(token=getenv("HUGGINGFACE_TOKEN"))
            data_dir = Path.cwd() / "data"
            dataset = load_dataset(path=str(data_dir))
            dataset.push_to_hub(repo_id=f"{getenv('HUGGINGFACE_USER')}/esperberto")
        except ValueError as e:
            warn(f"Unable to upload dataset due to, {e}.", UserWarning)
        finally:
            logout()


def main():
    load_dotenv()
    args = Parser().parse_args()
    train(test_pipeline=args.test_pipeline)


if __name__ == '__main__':
    main()
