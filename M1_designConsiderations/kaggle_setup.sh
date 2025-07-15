#!/bin/bash
# Upload kaggle.json and download dataset

mkdir -p ~/.kaggle
chmod 600 ~/.kaggle/kaggle.json

kaggle competitions download -c ieee-fraud-detection
unzip ieee-fraud-detection.zip -d ieee_fraud
