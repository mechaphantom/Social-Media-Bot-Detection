import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from sklearn.ensemble import IsolationForest

df = pd.read_csv("twitter_human_bots_dataset.csv")
df.head()

# Convert Verified from True/False → 1/0
df["Verified_int"] = df["Verified"].astype(int)

# Select numeric features for clustering + anomaly detection
feature_cols = ["Retweet Count", "Mention Count", "Follower Count", "Verified_int"]

X = df[feature_cols].copy()
y = df["Bot Label"].astype(int)

print("Feature columns:", feature_cols)
X.head()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-means with 2 clusters (bot-like vs human-like)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# Compare clusters with real labels
km_df = pd.DataFrame(X_scaled, columns=feature_cols)
km_df["cluster"] = clusters
km_df["bot_label"] = y.values

print("Cluster vs Bot Label:")
print(pd.crosstab(km_df["cluster"], km_df["bot_label"]))

# Determine which cluster is 'bot heavy'
crosstab = pd.crosstab(km_df["cluster"], km_df["bot_label"])
bot_cluster = crosstab[1].idxmax()

# Map clusters to predicted bot label
km_df["pred_bot"] = (km_df["cluster"] == bot_cluster).astype(int)

print("\nK-means Classification Report:")
print(classification_report(km_df["bot_label"], km_df["pred_bot"]))

print("K-means Confusion Matrix:")
print(confusion_matrix(km_df["bot_label"], km_df["pred_bot"]))

iso = IsolationForest(
n_estimators=200, contamination="auto", random_state=42
)
iso.fit(X_scaled)
# IsolationForest returns 1 (normal) and -1 (anomaly)
iso_raw = iso. predict(X_scaled)
# Convert to bot label: -1 = bot
iso_pred_bot = (iso_raw == -1).astype(int)
print("Isolation Forest Classification Report:")
print(classification_report(y, iso_pred_bot))
print ("Isolation Forest Confusion Matrix:")
print(confusion_matrix(y, iso_pred_bot))
