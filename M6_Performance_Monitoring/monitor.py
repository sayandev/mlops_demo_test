import boto3, logging, watchtower
from sklearn.metrics import accuracy_score

y_true = [0, 1, 0, 1, 1]
y_pred = [0, 1, 0, 0, 1]

acc = accuracy_score(y_true, y_pred)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(watchtower.CloudWatchLogHandler(log_group='fraud-monitoring'))
logger.info(f"Model Accuracy: {acc}")
