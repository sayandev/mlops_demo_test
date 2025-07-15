import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('sample_train_data.csv')  # Use pre-processed sample
X, y = df.drop('isFraud', axis=1), df['isFraud']
model = LogisticRegression().fit(X, y)
print("Model trained successfully.")
