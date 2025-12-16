#Imports
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier # Or RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from ydata_profiling import ProfileReport
#%%xs
#Load dataset from csv file
csv_file = r"twitter_human_bots_dataset.csv" 
chunksize = 10 ** 3
df_chunks = pd.read_csv(csv_file, chunksize=chunksize)
df = pd.concat(df_chunks)
df.head(10)
#%%
sample_size = 10  # Adjust based on what fits your memory
df_sample = df.sample(n=sample_size, random_state=42)

# VSCode supports viewing data through pandas_profiling
ProfileReport(df_sample)
#%% Prepare the data for Random Forest
df_prepared = df[['lang', 'account_type']].copy()

label_encoder = LabelEncoder()
df_prepared['account_type'] = label_encoder.fit_transform(df_prepared['account_type'])

lang_encoded_cols = pd.get_dummies(df_prepared['lang'], prefix='lang')

df_prepared = pd.concat([df_prepared.drop('lang', axis=1), lang_encoded_cols], axis=1)

print("Shape of df_prepared:", df_prepared.shape)
print("First 5 rows of df_prepared:")
print(df_prepared.head())
#%% Split the data into training and testing sets
X = df_prepared.drop('account_type', axis=1)
y = df_prepared['account_type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)
#%%
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

print("RandomForestClassifier model trained successfully.")
#%% Most Influential Variables in Random Forest Model
correlations = df_prepared.corr(numeric_only=True)['account_type'].drop('account_type')
sorted_correlations = correlations.abs().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=sorted_correlations.values, y=sorted_correlations.index, hue=sorted_correlations.index, palette='viridis', legend=False)
plt.title('Correlation of Features with Account Type (Absolute Values)')
plt.xlabel('Absolute Pearson Correlation Coefficient')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()
#%% Account Age with Boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='account_type', y='account_age_days', data=df_prepared)
plt.title('Distribution of Account Age by Account Type')
plt.xlabel('Account Type (0: Bot, 1: Human)')
plt.ylabel('Account Age (Days)')
plt.xticks(ticks=[0, 1], labels=['Bot', 'Human'])
plt.tight_layout()
plt.show()
#%% Bar graphs for Boolean Features
boolean_features = ['default_profile', 'default_profile_image', 'geo_enabled', 'verified']

plt.figure(figsize=(15, 10))
for i, feature in enumerate(boolean_features):
    plt.subplot(2, 2, i + 1) # Arrange plots in a 2x2 grid
    sns.countplot(x=feature, hue='account_type', data=df_prepared, palette='viridis')
    plt.title(f'Distribution of {feature} by Account Type')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.xticks(ticks=[0, 1], labels=['False', 'True'])
    plt.legend(title='Account Type', labels=['Bot', 'Human'])
plt.tight_layout()
plt.show()
#%% Score and Metrics for Random Forest Model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")