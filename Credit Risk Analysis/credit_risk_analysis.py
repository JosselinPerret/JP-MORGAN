import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

df = pd.read_csv('Task 3 and 4_Loan_Data.csv')

df_drop = df.drop(columns=["customer_id"])
df_drop.head()
y = df_drop['default'].values
X = df_drop.drop(columns=['default']).values
rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf.fit(X, y)

def loss_estimated(loan, credit_lines_outstanding, loan_amt_outstanding, total_debt_outstanding, income, years_employed, fico_score, recovery_rate=0.1):
    return rf.predict_proba(np.array([[credit_lines_outstanding, loan_amt_outstanding, total_debt_outstanding, income, years_employed, fico_score]]))[:, 1][0] * (1 - recovery_rate) * loan