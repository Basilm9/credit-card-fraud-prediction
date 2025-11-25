import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
from joblib import Parallel, delayed
import warnings
import os

warnings.filterwarnings('ignore')

print("Loading data...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Preprocessing
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

# ProbaBoost Implementation
class ConditionProbabilitiesClassifier:
    def __init__(self, df, cat_feats, num_feats, target, nb_buckets=10):
        self.df = df
        self.cat_feats = cat_feats
        self.num_feats = num_feats
        self.target = target
        self.nb_buckets = nb_buckets
        self.cat_cond_probs = {}
        self.num_cond_probs = {}
        self.num_buckets = {}

    def fit(self):
        for col in self.cat_feats:
            self.cat_cond_probs[col] = self.df.groupby(col)[self.target].mean().to_dict()

        for col in self.num_feats:
            self.df[f'{col}_bucket'], bins = pd.qcut(self.df[col], self.nb_buckets, duplicates='drop', retbins=True, labels=False)
            self.num_buckets[col] = bins
            self.num_cond_probs[col] = self.df.groupby(f'{col}_bucket')[self.target].mean().to_dict()

    def predict_buckets(self, df, col):
        bins = self.num_buckets[col]
        bins[0] = -np.inf
        bins[-1] = np.inf
        return pd.cut(df[col], bins=bins, labels=False, include_lowest=True).fillna(0).astype(int)

    def predict(self, df):
        pred_df = df.copy()
        for col in self.cat_feats:
            global_mean = self.df[self.target].mean()
            pred_df[col] = pred_df[col].map(self.cat_cond_probs[col]).fillna(global_mean)
        for col in self.num_feats:
            buckets = self.predict_buckets(pred_df, col)
            global_mean = self.df[self.target].mean()
            pred_df[col] = buckets.map(self.num_cond_probs[col]).fillna(global_mean)
        preds = pred_df[self.cat_feats + self.num_feats].mean(axis=1)
        return preds

class ProbaBoost:
    def __init__(self, df, cat_feats, num_feats, target, nb_boosters=10, nb_buckets=10, random_state=42):
        self.df = df
        self.cat_feats = cat_feats
        self.num_feats = num_feats
        self.target = target
        self.nb_boosters = nb_boosters
        self.nb_buckets = nb_buckets
        self.random_state = random_state
        self.classifiers = []

    def train_booster(self, i):
        sample_df = self.df.sample(len(self.df), replace=True, random_state=self.random_state + i)
        clf = ConditionProbabilitiesClassifier(sample_df, self.cat_feats, self.num_feats, self.target, self.nb_buckets)
        clf.fit()
        return clf

    def fit(self):
        self.classifiers = Parallel(n_jobs=-1)(delayed(self.train_booster)(i) for i in range(self.nb_boosters))

    def predict(self, df):
        preds_list = Parallel(n_jobs=-1)(delayed(clf.predict)(df) for clf in self.classifiers)
        return np.mean(preds_list, axis=0)

# Train ProbaBoost
print("Training ProbaBoost...")
train_data = X_train.copy()
train_data['IsFraud'] = y_train
num_cols = X_train.columns.tolist()
cat_cols = []

proba_boost = ProbaBoost(
    df=train_data,
    cat_feats=cat_cols,
    num_feats=num_cols,
    target='IsFraud',
    nb_boosters=10, # Reduced for speed in verification
    nb_buckets=10
)
proba_boost.fit()
print("Training complete.")

# Evaluate
y_pred_proba = proba_boost.predict(X_val)
print('ProbaBoost ROC AUC:', roc_auc_score(y_val, y_pred_proba))

# Submission
print("Generating submission...")
test_preds = proba_boost.predict(X_test)
submission = pd.DataFrame({'id': test_df['id'], 'IsFraud': test_preds})
submission.to_csv('submission.csv', index=False)
print('Submission file saved!')
