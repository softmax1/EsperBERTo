from os import getenv
from pathlib import Path
from warnings import warn

from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login, logout


def main():
    load_dotenv()

    data_dir = Path.cwd() / "data"

    # The Leipzig Corpora needs a bit of preprocessing.
    for path in data_dir.glob("epo_*.txt"):
        with path.open(mode='r') as fr:
            # Preprocessing only needs to be done once. The index -1 handles that.
            text_list = [line.split('\t')[-1] for line in fr]
        with path.open(mode='w') as fw:
            fw.write(''.join(text_list))

    # Now try to upload to HF hub.
    dataset = load_dataset(path=str(data_dir))
    try:
        login(token=getenv("HUGGINGFACE_TOKEN"))
        dataset.push_to_hub(repo_id=f"{getenv('HUGGINGFACE_USER')}/esperanto")
    except ValueError as e:
        warn(f"Unable to upload dataset due to, {e}.", UserWarning)
    finally:
        logout()


if __name__ == '__main__':
    main()
