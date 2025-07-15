# #!/usr/bin/env python3
# """
# Ray Tune hyperparameter optimization for fraud detection
# No AWS dependencies - pure open source
# """

# import os
# import tempfile
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import f1_score
# import ray
# from ray import tune
# from ray.tune.schedulers import ASHAScheduler
# from typing import Dict, Any
# import mlflow
# import mlflow.sklearn
# import logging

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# class FraudTuner:
#     def __init__(self, data_path: str, mlflow_uri: str = "http://localhost:5000"):
#         self.data_path = data_path
#         self.mlflow_uri = mlflow_uri
        
#         # Load data once
#         self.X_train, self.X_test, self.y_train, self.y_test = self._load_data()
        
#         # Set up MLflow
#         mlflow.set_tracking_uri(mlflow_uri)
#         try:
#             self.experiment_id = mlflow.create_experiment("fraud_detection_tuning")
#         except mlflow.exceptions.MlflowException:
#             self.experiment_id = mlflow.get_experiment_by_name("fraud_detection_tuning").experiment_id
    
#     def _load_data(self) -> tuple:
#         """Load and prepare data for tuning"""
#         logger.info(f"Loading data from {self.data_path}")
#         df = pd.read_csv(self.data_path)
        
#         # Basic preprocessing
#         df = df.dropna(thresh=len(df) * 0.5, axis=1)
#         df = df.fillna(0)
        
#         # Prepare features and target
#         if 'isFraud' in df.columns:
#             y = df['isFraud']
#             X = df.drop('isFraud', axis=1)
#         else:
#             # Create synthetic target for demo
#             y = np.random.choice([0, 1], size=len(df), p=[0.9, 0.1])
#             X = df.select_dtypes(include=[np.number])
        
#         # Take a sample for faster tuning (adjust size as needed)
#         if len(X) > 10000:
#             X_sample, _, y_sample, _ = train_test_split(
#                 X, y, train_size=10000, random_state=42, stratify=y
#             )
#             X, y = X_sample, y_sample
        
#         return train_test_split(X, y, test_size=0.2, random_state=42)
    
#     def objective(self, config: Dict[str, Any]) -> Dict[str, float]:
#         """Objective function for Ray Tune"""
#         try:
#             # Train model with given hyperparameters
#             model = RandomForestClassifier(
#                 n_estimators=int(config['n_estimators']),
#                 max_depth=int(config['max_depth']) if config['max_depth'] > 0 else None,
#                 min_samples_split=int(config['min_samples_split']),
#                 min_samples_leaf=int(config['min_samples_leaf']),
#                 random_state=42
#             )
            
#             model.fit(self.X_train, self.y_train)
#             y_pred = model.predict(self.X_test)
            
#             # Calculate metrics
#             f1 = f1_score(self.y_test, y_pred, average='weighted')
            
#             # Log to MLflow
#             with mlflow.start_run(experiment_id=self.experiment_id):
#                 mlflow.log_params(config)
#                 mlflow.log_metric("f1_score", f1)
#                 mlflow.sklearn.log_model(model, "model")
            
#             return {"f1_score": f1}
            
#         except Exception as e:
#             logger.error(f"Error in objective function: {e}")
#             return {"f1_score": 0.0}
    
#     def run_tuning(self, num_samples: int = 20, max_epochs: int = 10):
#         """Run hyperparameter tuning with Ray Tune"""
#         logger.info("Starting hyperparameter tuning...")
        
#         # Initialize Ray
#         if not ray.is_initialized():
#             ray.init(ignore_reinit_error=True)
        
#         # Define search space
#         search_space = {
#             'n_estimators': tune.randint(50, 200),
#             'max_depth': tune.randint(3, 15),
#             'min_samples_split': tune.randint(2, 10),
#             'min_samples_leaf': tune.randint(1, 5)
#         }
        
#         # Configure scheduler
#         scheduler = ASHAScheduler(
#             time_attr='training_iteration',
#             metric='f1_score',
#             mode='max',
#             max_t=max_epochs,
#             grace_period=1,
#             reduction_factor=2
#         )
        
#         # Run tuning
#         tuner = tune.Tuner(
#             tune.with_resources(
#                 self.objective,
#                 resources={"cpu": 1}
#             ),
#             param_space=search_space,
#             tune_config=tune.TuneConfig(
#                 scheduler=scheduler,
#                 num_samples=num_samples,
#                 metric='f1_score',
#                 mode='max'
#             )
#         )
        
#         results = tuner.fit()
        
#         # Get best result
#         best_result = results.get_best_result()
#         logger.info(f"Best trial config: {best_result.config}")
#         logger.info(f"Best trial final f1_score: {best_result.metrics['f1_score']}")
        
#         return best_result


# def main():
#     import argparse
    
#     parser = argparse.ArgumentParser(description='Tune fraud detection model')
#     parser.add_argument('--data', required=True, help='Path to training data')
#     parser.add_argument('--num_samples', type=int, default=20, help='Number of trials')
#     parser.add_argument('--max_epochs', type=int, default=10, help='Max epochs per trial')
#     parser.add_argument('--mlflow_uri', default='http://localhost:5000', help='MLflow URI')
    
#     args = parser.parse_args()
    
#     # Create tuner and run
#     tuner = FraudTuner(args.data, args.mlflow_uri)
#     best_result = tuner.run_tuning(args.num_samples, args.max_epochs)
    
#     logger.info("Tuning completed!")
#     logger.info(f"Best hyperparameters: {best_result.config}")
#     logger.info(f"Best F1 score: {best_result.metrics['f1_score']}")


# if __name__ == "__main__":
#     main()
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import mlflow
import mlflow.sklearn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(data_path):
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    df = df.dropna(thresh=len(df) * 0.5, axis=1)
    df = df.fillna(0)
    if 'isFraud' in df.columns:
        y = df['isFraud']
        X = df.drop('isFraud', axis=1)
    else:
        y = np.random.choice([0, 1], size=len(df), p=[0.9, 0.1])
        X = df.select_dtypes(include=[np.number])
    if len(X) > 10000:
        X_sample, _, y_sample, _ = train_test_split(
            X, y, train_size=10000, random_state=42, stratify=y
        )
        X, y = X_sample, y_sample
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_objective(config, data=None, experiment_id=None):
    X_train, X_test, y_train, y_test = data
    model = RandomForestClassifier(
        n_estimators=int(config['n_estimators']),
        max_depth=int(config['max_depth']) if config['max_depth'] > 0 else None,
        min_samples_split=int(config['min_samples_split']),
        min_samples_leaf=int(config['min_samples_leaf']),
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"Trial config: {config}, F1: {f1}")  # This will show in Ray dashboard logs
    with mlflow.start_run(experiment_id=experiment_id):
        mlflow.log_params(config)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")
    tune.report(f1_score=f1)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Tune fraud detection model')
    parser.add_argument('--data', required=True, help='Path to training data')
    parser.add_argument('--num_samples', type=int, default=20, help='Number of trials')
    parser.add_argument('--max_epochs', type=int, default=10, help='Max epochs per trial')
    parser.add_argument('--mlflow_uri', default='http://mlflow:5050', help='MLflow URI')
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.mlflow_uri)
    try:
        experiment_id = mlflow.create_experiment("fraud_detection_tuning")
    except mlflow.exceptions.MlflowException:
        experiment_id = mlflow.get_experiment_by_name("fraud_detection_tuning").experiment_id

    data = load_data(args.data)

    if not ray.is_initialized():
        ray.init(address="auto", ignore_reinit_error=True)

    search_space = {
        'n_estimators': tune.randint(50, 200),
        'max_depth': tune.randint(3, 15),
        'min_samples_split': tune.randint(2, 10),
        'min_samples_leaf': tune.randint(1, 5)
    }

    scheduler = ASHAScheduler(
        time_attr='training_iteration',
        metric='f1_score',
        mode='max',
        max_t=args.max_epochs,
        grace_period=1,
        reduction_factor=2
    )

    tuner = tune.Tuner(
        tune.with_parameters(train_objective, data=data, experiment_id=experiment_id),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            num_samples=args.num_samples,
            metric='f1_score',
            mode='max'
        )
    )

    results = tuner.fit()
    best_result = results.get_best_result()
    logger.info(f"Best trial config: {best_result.config}")
    logger.info(f"Best trial final f1_score: {best_result.metrics['f1_score']}")

if __name__ == "__main__":
    main()