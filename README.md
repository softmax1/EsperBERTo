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
To run the unit tests do `pytest tests`.


## Output
The dataset is available on Hugging Face at [chriswmurphy/esperanto](https://huggingface.co/datasets/chriswmurphy/esperanto).
The models are also available on Hugging Face at [chriswmurphy/esperberto-softmax0](https://huggingface.co/chriswmurphy/esperberto-softmax0) and [chriswmurphy/esperberto-softmax1](https://huggingface.co/chriswmurphy/esperberto-softmax1).

## Results
Here we report the average (excess) kurtosis in the Attention output layers from our initial run.

| Model                   | Dense Weight      | Dense Bias      | LayerNorm Weight | LayerNorm Bias  |
|-------------------------|-------------------|-----------------|------------------|-----------------|
| EsperBERTo w/ softmax_0 | $0.012 \pm 0.020$ | $0.21 \pm 0.27$ | $3.5 \pm 3.3$    | $0.08 \pm 0.26$ |
| EsperBERTo w/ softmax_1 | $0.011 \pm 0.007$ | $0.35 \pm 0.42$ | $3.2 \pm 2.6$    | $0.39 \pm 0.47$ |

The weights in the dense Attention layers are Gaussian to a good approximation.
However, I don't think any conclusions can be drawn from this because the Esperanto dataset is a mere 200 MB, which is tiny by LLM training standards.