"""Basic Data Ingestion + CloudWatch Logging"""

import boto3
import pandas as pd
import logging
import watchtower

BUCKET = 'fraud-demo-data-yourname'
KEY = 'raw/train_transaction.csv'

s3 = boto3.client('s3')
obj = s3.get_object(Bucket=BUCKET, Key=KEY)
df = pd.read_csv(obj['Body'], nrows=5)
print(df)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(watchtower.CloudWatchLogHandler(log_group='fraud-demo-logs'))
logger.info("Loaded 5 rows from S3 successfully")
