from os import getenv
from pathlib import Path
from warnings import warn

from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login, logout
from tap import Tap
from transformers import (
    RobertaConfig,
    RobertaTokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    RobertaForMaskedLM
)

from src.modeling_roberta import RobertaForMaskedLM as RobertaForMaskedLMSoftmax1
from src.statistics import compute_statistics, save_statistics


class Parser(Tap):
    use_softmax1: bool = False  # Whether to the Softmax1 activation function in the Attention mechanism
    test_pipeline: bool = False  # Quickly test the pipeline by training on 2 samples and not saving


def train(use_softmax1: bool = False, test_pipeline: bool = False):
    # We'll define the following config for the model
    config = RobertaConfig(
        vocab_size=52_000,
        max_position_embeddings=514,
        num_attention_heads=12,
        num_hidden_layers=6,
        type_vocab_size=1,
    )

    # Now let's re-create our tokenizer.json in transformers
    tokenizer_dir = Path.cwd() / "tokenizer"
    tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_dir, max_len=512)

    # Finally let's initialize our model.
    # As we are training from scratch, we only initialize from a config, not from an existing pretrained model or checkpoint.
    if use_softmax1:
        model = RobertaForMaskedLMSoftmax1(config=config)
    else:
        model = RobertaForMaskedLM(config=config)

    # Load the raw dataset
    data_dir = Path.cwd() / "data"
    split = 'train[:2]' if test_pipeline else 'train'
    dataset = load_dataset(path=str(data_dir), split=split)

    # We'll build our dataset by applying our tokenizer.json to our text file.
    def process_data(examples):
        return tokenizer(
            examples["text"],
            return_special_tokens_mask=True,
            truncation=True,
            max_length=tokenizer.model_max_length
        )

    tokenized_dataset = dataset.map(process_data, batched=True)

    # we need to define a data_collator to batch different samples of the dataset together into an object that PyTorch knows how to perform backprop on.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    # Finally, we are all set to initialize our Trainer
    n = 1 if use_softmax1 else 0
    output_dir = f"{getenv('HUGGINGFACE_USER')}/esperberto-softmax{n}"

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=128,
        auto_find_batch_size=True,
        logging_strategy="steps",
        logging_steps=1000,
        logging_first_step=True,
        save_strategy="no",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=tokenized_dataset
    )

    # Start training
    trainer.train()

    # Compute kurtosis and other stats
    stats = compute_statistics(model)
    save_statistics(stats, output_dir)

    # Save final model (+ tokenizer.json + config)
    if not test_pipeline:
        try:
            login(token=getenv("HUGGINGFACE_TOKEN"))
            trainer.push_to_hub()
        except (ValueError, RuntimeError, OSError, FileNotFoundError, TypeError) as e:  # I've seen it all XD
            warn(f"Unable to upload model due to, {e}. Trying to write to disk instead.", UserWarning)
            trainer.save_model()
        finally:
            logout()


def main():
    load_dotenv()
    args = Parser().parse_args()
    train(use_softmax1=args.use_softmax1, test_pipeline=args.test_pipeline)


if __name__ == '__main__':
    main()
