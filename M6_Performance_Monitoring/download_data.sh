#!/bin/bash
set -e

echo "📥 Downloading IEEE-CIS Fraud Detection dataset..."

if ! command -v kaggle &>/dev/null; then
  pip install kaggle
fi

if [ ! -f ~/.kaggle/kaggle.json ]; then
  echo "❌ Kaggle credentials missing!"
  exit 1
fi

mkdir -p data/raw
kaggle competitions download -c ieee-fraud-detection -p data/raw/
cd data/raw && unzip -o ieee-fraud-detection.zip && cd ../..

echo "🔄 Preprocessing..."
python3 -c "
import pandas as pd
tt = pd.read_csv('data/raw/train_transaction.csv')
ti = pd.read_csv('data/raw/train_identity.csv')
df = tt.merge(ti, on='TransactionID', how='left').fillna(0)
df.to_csv('data/kaggle_fraud.csv', index=False)
"
echo "✅ Data ready: data/kaggle_fraud.csv"
