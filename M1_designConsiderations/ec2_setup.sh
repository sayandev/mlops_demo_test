#!/bin/bash
# Setup Python env on EC2

sudo apt update && sudo apt install -y python3-pip unzip
pip3 install -r requirements.txt
