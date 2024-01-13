#!/bin/bash
# Setup FREEVC for training in Sagemaker Studio Lab Environment
FREEVC_ENV=free-vc
DATASET_URL=https://drive.google.com/file/d/1bfByWAngeFzGBFhuqwzW0p80p0DTxK6Z&export=download
MODEL_URL=https://drive.google.com/file/d/1ci-qtZYSsJYPAMufH-NIsYUWKdSzHhLe&export=download
DATASET_ZIP_FILENAME=dataset.zip
MODEL_ZIP_FILENAME=model.zip
MODEL_PATH=logs/freevc


# Setup Dataset and Model
wget $DATASET_URL -O $DATASET_ZIP_FILENAME
wget $MODEL_URL -O $MODEL_ZIP_FILENAME
mkdir -p $MODEL_PATH
unzip $DATASET_ZIP_FILENAME
unzip $MODEL_ZIP_FILENAME -d $MODEL_PATH


# Setup Environment and Install Dependencies
conda create --name $FREEVC_ENV python=3.9
conda activate $FREEVC_ENV
pip install -r requirements.txt


# Train model
python train.py -c configs/freevc.json -m freevc