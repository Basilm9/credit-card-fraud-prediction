import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('train.csv')

# Features of interest
features = ['feat21', 'feat26', 'feat23', 'feat8', 'feat7', 'feat3', 'feat10']

# Plotting
plt.figure(figsize=(15, 10))
for i, col in enumerate(features):
    plt.subplot(3, 3, i+1)
    sns.boxplot(x='IsFraud', y=col, data=df, showfliers=False) # Hiding outliers for better visibility of the box
    plt.title(col)

plt.tight_layout()
plt.savefig('selected_features_analysis.png')
print("Saved selected_features_analysis.png")
