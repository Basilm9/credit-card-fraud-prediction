import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
import warnings
import os

warnings.filterwarnings('ignore')

print("Loading data...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Class distribution
print("Generating class_distribution.png...")
plt.figure(figsize=(8, 6))
sns.countplot(x='IsFraud', data=train_df)
plt.title('Class Distribution')
plt.savefig('class_distribution.png')
plt.close()

# Feature distribution
print("Generating feature_distribution.png...")
fig, ax = plt.subplots(1, 2, figsize=(18,4))
amount_val = train_df['Transaction_Amount'].values
time_val = train_df['Time'].values
sns.distplot(amount_val, ax=ax[0], color='r')
ax[0].set_title('Distribution of Transaction Amount', fontsize=14)
sns.distplot(time_val, ax=ax[1], color='b')
ax[1].set_title('Distribution of Transaction Time', fontsize=14)
plt.savefig('feature_distribution.png')
plt.close()

# Correlation Matrix
print("Generating correlation_matrix.png...")
plt.figure(figsize=(20, 10))
corr = train_df.corr()
sns.heatmap(corr, cmap='coolwarm_r', annot_kws={'size':20})
plt.title('Correlation Matrix', fontsize=14)
plt.savefig('correlation_matrix.png')
plt.close()

# Preprocessing
scaler = RobustScaler()
train_df['scaled_amount'] = scaler.fit_transform(train_df['Transaction_Amount'].values.reshape(-1,1))
train_df['scaled_time'] = scaler.fit_transform(train_df['Time'].values.reshape(-1,1))
train_df.drop(['Time','Transaction_Amount'], axis=1, inplace=True)

X = train_df.drop(['IsFraud', 'id'], axis=1)
y = train_df['IsFraud']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Model Training
print("Training model...")
hgb_clf = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=100, max_depth=6, random_state=42)
hgb_clf.fit(X_train, y_train)
y_pred_hgb = hgb_clf.predict_proba(X_val)[:,1]

# ROC Curve
print("Generating roc_curve.png...")
fpr, tpr, thresholds = roc_curve(y_val, y_pred_hgb)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='HistGradientBoosting (AUC = %0.4f)' % roc_auc_score(y_val, y_pred_hgb))
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.savefig('roc_curve.png')
plt.close()

# Confusion Matrix
print("Generating confusion_matrix.png...")
y_pred_binary = (y_pred_hgb > 0.5).astype(int)
cm = confusion_matrix(y_val, y_pred_binary)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.close()

# Feature Importance
print("Generating feature_importance.png...")
result = permutation_importance(hgb_clf, X_val, y_val, n_repeats=5, random_state=42, n_jobs=-1) # Reduced repeats for speed
sorted_idx = result.importances_mean.argsort()
plt.figure(figsize=(10, 8))
plt.boxplot(result.importances[sorted_idx].T, vert=False, labels=X_val.columns[sorted_idx])
plt.title("Permutation Importance (test set)")
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

print("Verification complete. Checking files...")
expected_files = ['class_distribution.png', 'feature_distribution.png', 'correlation_matrix.png', 'roc_curve.png', 'confusion_matrix.png', 'feature_importance.png']
for f in expected_files:
    if os.path.exists(f):
        print(f"Found {f}")
    else:
        print(f"MISSING {f}")
