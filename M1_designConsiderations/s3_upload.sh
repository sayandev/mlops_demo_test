#!/bin/bash
# Upload unzipped dataset to S3

aws s3 cp ieee_fraud/ s3://fraud-demo-data-yourname/raw/ --recursive
aws s3 ls s3://fraud-demo-data-yourname/raw/
