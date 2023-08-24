#!/bin/bash
set -eo pipefail

# We need to use the Git Large File System to save our model to Hugging Face hub
# These are AWS-specific instructions
sudo amazon-linux-extras install epel -y
sudo yum-config-manager --enable epel
sudo yum install git-lfs
git lfs install

# Clone the repo, and move into it
git clone https://github.com/softmax1/EsperBERTo.git
cd EsperBERTo

# Download the dataset.
# The Leipzig Corpora website didn't have an obvious way to programmatically download files, so I manually did those ones.
wget -c -P data/ https://cdn-datasets.huggingface.co/EsperBERTo/data/oscar.eo.txt

# Activate the venv on AWS
source activate pytorch

# Install packages
pip install -r requirements.txt

# Upload the raw data files to Hugging Face hug, so they're easier to work with.
python upload_dataset.py

# Train a tokenizer
python train_tokenizer.py

# Train a baseline language model from scratch
python train_model.py

# Train a challenger to the baseline
python train_model.py --use_softmax1

# Wait for everything to finish, and deactivate the venv
wait
conda deactivate