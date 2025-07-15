import pandas as pd
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
for batch in ['data/batch1.csv', 'data/batch2.csv']:
    df = pd.read_csv(batch)
    X, y = df.drop('isFraud', axis=1), df['isFraud']
    model.fit(X, y)
    print(f"Retrained on {batch}")
