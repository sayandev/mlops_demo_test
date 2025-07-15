# End-to-End Demo: Model Training and Experimentation with SageMaker and MLflow

## 🧩 Problem

Training at scale and tracking experiments is hard without automation. We use **AWS SageMaker** for scalable training (CPU/GPU clusters) and **MLflow** for experiment tracking to ensure reproducibility and auditability.

---

## 1. 🔧 Environment Setup

### Containerize Code with Docker

```Dockerfile
FROM python:3.10-slim

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app
WORKDIR /app

ENTRYPOINT ["python", "train.py"]
```

### Build & Push to AWS ECR

```bash
aws ecr get-login-password | docker login --username AWS --password-stdin <aws_account_id>.dkr.ecr.<region>.amazonaws.com

docker build -t my-ml-image .
docker tag my-ml-image:latest <aws_account_id>.dkr.ecr.<region>.amazonaws.com/my-ml-image:latest
docker push <aws_account_id>.dkr.ecr.<region>.amazonaws.com/my-ml-image:latest
```

---

## 2. 🧠 Launch Training Jobs with SageMaker

```python
from sagemaker.estimator import Estimator

estimator = Estimator(
    image_uri='<ecr_image_url>',
    role='<sagemaker_execution_role>',
    instance_count=1,
    instance_type='ml.p3.2xlarge',
    output_path='s3://<bucket>/output/',
    hyperparameters={'lr': 0.001, 'batch_size': 64}
)

estimator.fit({'training': 's3://<bucket>/data/train'})
```

### For Distributed or Ray Training

* Use `SageMaker PyTorch` or `TensorFlow` estimators with `distribution` config.
* Use [Amazon SageMaker Ray](https://docs.ray.io/en/latest/cluster/sagemaker.html).

---

## 3. 📈 Experiment Tracking with MLflow

### In `train.py`

```python
import mlflow
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.001)
args = parser.parse_args()

mlflow.set_tracking_uri("http://<mlflow-server>:5000")
mlflow.set_experiment("fraud-detection")

with mlflow.start_run():
    mlflow.log_param("lr", args.lr)
    acc = 0.93  # Dummy metric
    mlflow.log_metric("accuracy", acc)
    mlflow.log_artifacts("models/")
```

### Model Deployment

```bash
mlflow sagemaker deploy -m runs:/<run_id>/model -e prod-endpoint --region <region>
```

---

## 4. 🔍 Hyperparameter Tuning

```python
from sagemaker.tuner import HyperparameterTuner, ContinuousParameter

tuner = HyperparameterTuner(
    estimator,
    objective_metric_name='validation:accuracy',
    hyperparameter_ranges={'lr': ContinuousParameter(0.0001, 0.01)},
    max_jobs=10,
    max_parallel_jobs=2,
    objective_type='Maximize'
)

tuner.fit({'training': 's3://<bucket>/data/train'})
```

---

## 5. 🏷️ Register Models in MLflow

```python
from mlflow.tracking import MlflowClient

mlflow.register_model(
    "runs:/<run_id>/model",
    "FraudDetectionModel"
)

client = MlflowClient()
client.transition_model_version_stage(
    name="FraudDetectionModel",
    version=1,
    stage="Staging"
)
```

---

## ✅ Benefits

| Challenge           | Solution                                      |
| ------------------- | --------------------------------------------- |
| Reproducibility     | Dockerized environments + MLflow metadata     |
| Experiment Tracking | MLflow + tags for lineage                     |
| Scalable Training   | SageMaker clusters (CPU/GPU) + tuning jobs    |
| Auditability        | MLflow + S3 + DVC for data/code/model linkage |

---

Let me know if you want Terraform/CDK templates or the MLflow EC2 deploy script.
