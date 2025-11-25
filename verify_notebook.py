import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.ensemble import HistGradientBoostingClassifier
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
sample_sub = pd.read_csv('sample_submission.csv')
print(f'Train shape: {train_df.shape}')
print(f'Test shape: {test_df.shape}')

# Preprocessing
print("Preprocessing...")
scaler = RobustScaler()
train_df['scaled_amount'] = scaler.fit_transform(train_df['Transaction_Amount'].values.reshape(-1,1))
train_df['scaled_time'] = scaler.fit_transform(train_df['Time'].values.reshape(-1,1))
test_df['scaled_amount'] = scaler.transform(test_df['Transaction_Amount'].values.reshape(-1,1))
test_df['scaled_time'] = scaler.transform(test_df['Time'].values.reshape(-1,1))
train_df.drop(['Time','Transaction_Amount'], axis=1, inplace=True)
test_df.drop(['Time','Transaction_Amount'], axis=1, inplace=True)

X = train_df.drop(['IsFraud', 'id'], axis=1)
y = train_df['IsFraud']
X_test = test_df.drop(['id'], axis=1)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Training Logistic Regression...")
lr = LogisticRegression(class_weight='balanced', max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict_proba(X_val)[:,1]
print('Logistic Regression ROC AUC:', roc_auc_score(y_val, y_pred_lr))

print("Training HistGradientBoostingClassifier...")
# HistGradientBoostingClassifier handles missing values and is efficient
hgb_clf = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=100, max_depth=6, random_state=42)
hgb_clf.fit(X_train, y_train)
y_pred_hgb = hgb_clf.predict_proba(X_val)[:,1]
print('HistGradientBoosting ROC AUC:', roc_auc_score(y_val, y_pred_hgb))

print("Generating submission...")
test_preds = hgb_clf.predict_proba(X_test)[:,1]
submission = pd.DataFrame({'id': test_df['id'], 'IsFraud': test_preds})
submission.to_csv('submission.csv', index=False)
print('Submission file saved!')
