# EsperBERTo
A test of the [Attention Is Off By One](https://www.evanmiller.org/attention-is-off-by-one.html) hypothesis.
We’ll train a "small" model (84 M parameters = 6 layers, 768 hidden size, 12 attention heads) on Esperanto.
In fact, we'll train two models, a baseline that uses the default softmax in its Attention mechanism, and a challenger that instead uses the proposed softmax1.

This pipeline is based on this [blog post](https://huggingface.co/blog/how-to-train).
The main difference is their dataloader is deprecated.

## Running
I'm running on an AWS _g5.2xlarge_ EC2 instance with 1x Nvidia A10G GPU.

Steps:
```
git clone https://github.com/softmax1/EsperBERTo.git
cd EsperBERTo
source activate pytorch
pip install -r requirements.txt
emacs .env
>>> HUGGINGFACE_TOKEN=<mySecretTokenValue>
>>> HUGGINGFACE_USER=<myHFUserName>
bash download_dataset.sh
python train_tokenizer.py
python train_model.py
python train_model.py --use_softmax1
```

To test `train_*.py` before running in earnest add the `--test_pipeline` flag.

## Output
The dataset will be available on Hugging Face at `chriswmurphy/esperberto`.
The models will also be available on Hugging Fact at `chriswmurphy/esperberto-softmax0` and `chriswmurphy/esperberto-softmax1`.
