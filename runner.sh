#!/bin/bash
set -eo pipefail

# Clone the repo, and move into it
git clone https://github.com/softmax1/EsperBERTo.git
cd EsperBERTo

# Find a dataset
wget -c -P data/ https://cdn-datasets.huggingface.co/EsperBERTo/data/oscar.eo.txt

# Activate the venv on AWS
source activate pytorch

# Install packages
pip install -r requirements.txt

# Train a tokenizer
python train_tokenizer.py

# Train a baseline language model from scratch
python train_model.py

# Train a challenger to the baseline
python train_model.py --use_softmax1

# Wait for everything to finish, and deactivate the venv
wait
conda deactivate