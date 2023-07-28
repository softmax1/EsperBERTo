# EsperBERTo
A test of the [Attention Is Off By One](https://www.evanmiller.org/attention-is-off-by-one.html) hypothesis.
We’ll train a "small" model (84 M parameters = 6 layers, 768 hidden size, 12 attention heads) on Esperanto.
In fact, we'll train two models, a baseline that uses the default softmax in its Attention mechanism, and a challenger that instead uses the proposed softmax1.

This pipeline is based on this [blog post](https://huggingface.co/blog/how-to-train) with two differences.
The biggest change is I subclassed the Roberta model to use a custom softmax1 activation function.
Also, their dataloader is deprecated, so I swapped in the datasets package.

## Running
I'm running on an AWS _g5.2xlarge_ EC2 instance with 1x Nvidia A10G GPU.
Running takes about 5 hours and costs approximately $6.

You can use the following script to reproduce my results: `screen -S run "bash runner.sh"`
Don't forget to add your Hugging Face token and username to the env vars before running, e.g.
```
echo "HUGGINGFACE_TOKEN=<mySecretTokenVariable>" > .env
echo "HUGGINGFACE_USER=<myHFUserName>" >> .env
```
To test before running in earnest, add the test_pipeline flag, e.g. `python train_model.py --test_pipeline`.


## Output
The dataset is available on Hugging Face at [chriswmurphy/esperberto](https://huggingface.co/datasets/chriswmurphy/esperberto).
The models will also be available on Hugging Face at [chriswmurphy/esperberto-softmax0](https://huggingface.co/chriswmurphy/esperberto-softmax0) and [chriswmurphy/esperberto-softmax1](https://huggingface.co/chriswmurphy/esperberto-softmax1).
