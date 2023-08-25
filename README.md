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

## Results

### Training
As expected, softmax1 does not impact model performance at single-precision.

| Model                  | Loss | Runtime | Cost   |
|------------------------|------|---------|--------|
| EsperBERTo w/ softmax0 | 4.46 | 9h 16m  | $11.22 |
| EsperBERTo w/ softmax1 | 4.44 | 9h 16m  | $11.24 |

### Kurtosis - Weights
Here we report the average excess kurtosis in the Attention output weights from our initial run.
The weights in the dense Attention layers are Gaussian to a good approximation.

| Model                  | Dense Weight      | Dense Bias    | LayerNorm Weight | LayerNorm Bias |
|------------------------|-------------------|---------------|------------------|----------------|
| EsperBERTo w/ softmax0 | $0.031 \pm 0.021$ | $0.6 \pm 1.1$ | $6.8 \pm 3.9$    | $0.5 \pm 0.8$  |
| EsperBERTo w/ softmax1 | $0.040 \pm 0.027$ | $1.4 \pm 1.4$ | $4.4 \pm 2.9$    | $1.9 \pm 2.3$  |

### Kurtosis - Activations
Finally, we report the average excess kurtosis in the Attention output activations from our initial run.
Once again, there is no meaningful difference between the softmax0 and softmax1 models here, and the kurtosis in the activation of the Attention output is consistent with being Gaussian.

| Model                  | Dense           | Dropout         | Output (LayerNorm) |
|------------------------|-----------------|-----------------|--------------------|
| EsperBERTo w/ softmax0 | $0.74 \pm 0.29$ | $1.15 \pm 0.32$ | $0.77 \pm 1.35$    |
| EsperBERTo w/ softmax1 | $1.96 \pm 1.58$ | $2.51 \pm 1.76$ | $0.90 \pm 1.31$    |

## Running
I'm running on an AWS _g5.2xlarge_ EC2 instance with 1x Nvidia A10G GPU.

You can use the following script to reproduce my results: `screen -S run "bash runner.sh"`
Don't forget to add your Hugging Face token and username to the env vars before running, e.g.
```
echo "HUGGINGFACE_TOKEN=<mySecretTokenVariable>" > .env
echo "HUGGINGFACE_USER=<myHFUserName>" >> .env
```
To test before running in earnest, add the test_pipeline flag, e.g. `python train_model.py --test_pipeline`.
To run the unit tests do `pytest tests`.