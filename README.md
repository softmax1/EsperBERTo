# EsperBERTo
A test of the [Attention Is Off By One](https://www.evanmiller.org/attention-is-off-by-one.html) hypothesis with RoBERTA and Esperanto.

## Dataset
The dataset is the Esperanto portion of the [OSCAR](https://cdn-datasets.huggingface.co/EsperBERTo/data/oscar.eo.txt) corpus from INRIA, which is a part of Common Crawl.
Additionally, the dataset contains the Esperanto sub-corpus of the [Leipzig Corpora Collection](https://wortschatz.uni-leipzig.de/en/download/Esperanto).
In particular, along with OSCAR, I use the following `epo_*-sentences.txt` files from the Leipzig Corpora:

| Dataset          | Year | # of Sentences |
|------------------|------|----------------|
| OSCAR            | 2020 | 974k           |
| LCC - Literature | 2011 | 300k           |
| LCC - Mixed      | 2012 | 1M             |
| LCC - Newscrawl  | 2017 | 1M             |
| LCC - Web        | 2012 | 1M             |
| LCC - Wikipedia  | 2021 | 300k           |
| total            | -    | 4.57M          |

The dataset is 473 MB.


## Model
The models are RoBERTa with 84M parameters.
The first model is a baseline that uses the default softmax in its Attention mechanism, and a challenger that instead uses the proposed softmax1.
The models are also available on Hugging Face at [chriswmurphy/esperberto-softmax0](https://huggingface.co/chriswmurphy/esperberto-softmax0) and [chriswmurphy/esperberto-softmax1](https://huggingface.co/chriswmurphy/esperberto-softmax1). 
The idea to use RoBERTa with Esperanto came from this [blog post](https://huggingface.co/blog/how-to-train).

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

## Results
Here we report the average (excess) kurtosis in the Attention output layers from our initial run.

| Model                   | Dense Weight      | Dense Bias      | LayerNorm Weight | LayerNorm Bias  |
|-------------------------|-------------------|-----------------|------------------|-----------------|
| EsperBERTo w/ softmax_0 | $0.012 \pm 0.020$ | $0.21 \pm 0.27$ | $3.5 \pm 3.3$    | $0.08 \pm 0.26$ |
| EsperBERTo w/ softmax_1 | $0.011 \pm 0.007$ | $0.35 \pm 0.42$ | $3.2 \pm 2.6$    | $0.39 \pm 0.47$ |

The weights in the dense Attention layers are Gaussian to a good approximation.
However, I don't think any conclusions can be drawn from this because the Esperanto dataset is a mere 200 MB, which is tiny by LLM training standards.