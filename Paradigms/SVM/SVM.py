#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt
#%%
df = pd.read_csv(r"twitter_human_bots_dataset.csv" )
print("Dataset shape:", df.shape)
df.head()
print("\n=== Dataset Info ===")
df.info()
#%%
print("\n=== Missing Values per Column ===")
missing_vals = df.isnull().sum().sort_values(ascending=False)
print(missing_vals)

plt.figure(figsize=(10,5))
missing_vals.plot(kind='bar')
plt.title("Missing Values per Column")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
#%%
class_counts = df['account_type'].value_counts()
print("\n=== Class Distribution ===")
print(class_counts)

plt.figure(figsize=(6,4))
class_counts.plot(kind='bar', color=['blue', 'red'])
plt.title("Class Distribution: Bot vs Human")
plt.xlabel("Account Type")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
#%%
numeric_cols = df.select_dtypes(include=['int64','float64']).columns

print("\n=== Statistical Summary (Numeric Features) ===")
df[numeric_cols].describe()
#%%
plt.figure(figsize=(14,10))
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, cmap='coolwarm', annot=False)
plt.title("Correlation Heatmap (Numeric Features)")
plt.show()
#%%
# handling missing calues
df['description'] = df['description'].fillna("")
df['lang'] = df['lang'].fillna("unknown")

# Droping text-like columns manually
cols_to_drop = [
    'description',
    'created_at',
    'profile_image_path',
    'profile_background_image_path',
    'id'
]
df_svm = df.drop(columns=cols_to_drop)
#Boolean to integer conversion
bool_columns = [
    'default_profile',
    'default_profile_image',
    'geo_enabled',
    'verified'
]
for col in bool_columns:
    df_svm[col] = df_svm[col].astype(int)
    
# I derived these features
df_svm['follow_ratio'] = df_svm['followers_count'] / (df_svm['friends_count'] + 1)
df_svm['favourite_rate'] = df_svm['favourites_count'] / (df_svm['statuses_count'] + 1)
df_svm['description_length'] = df['description'].apply(len)
df_svm['statuses_per_follower'] = df_svm['statuses_count'] / (df_svm['followers_count'] + 1)
y = df_svm['account_type'].map({'human': 0, 'bot': 1})
X = df_svm.select_dtypes(include=['int64', 'float64']) # I am just keeping the numeric values
X = X.drop(columns=['split'], errors='ignore')

print("Numeric X Shape:", X.shape)
print("Columns used in SVM:", list(X.columns))
#%%
#Scailing is an important part for SVM
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Scaling Completed. Final Feature Matrix Shape:", X_scaled.shape)
#%%
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print("\nTrain/Test Shapes:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)
#%%
# Creating baseline model for linear SVM

linear_svm = LinearSVC(class_weight='balanced', max_iter=5000)
linear_svm.fit(X_train, y_train)

y_pred_linear = linear_svm.predict(X_test)

print("\n=== Linear SVM Performance ===")
print("Accuracy:", accuracy_score(y_test, y_pred_linear))
print("Precision:", precision_score(y_test, y_pred_linear))
print("Recall:", recall_score(y_test, y_pred_linear))
print("F1 Score:", f1_score(y_test, y_pred_linear))
print("\nClassification Report:\n", classification_report(y_test, y_pred_linear))

# Creating a traditional SVM. It's an RBF Kernel SVM

rbf_svm = SVC(kernel='rbf', class_weight='balanced')
rbf_svm.fit(X_train, y_train)

y_pred_rbf = rbf_svm.predict(X_test)

print("\n=== RBF SVM Performance ===")
print("Accuracy:", accuracy_score(y_test, y_pred_rbf))
print("Precision:", precision_score(y_test, y_pred_rbf))
print("Recall:", recall_score(y_test, y_pred_rbf))
print("F1 Score:", f1_score(y_test, y_pred_rbf))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rbf))
#%%
# Confusion matrix for the RBF model
cm = confusion_matrix(y_test, y_pred_rbf)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Human", "Bot"], yticklabels=["Human", "Bot"])
plt.title("Confusion Matrix — RBF SVM")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()
#%%
# I run a GridSearchCV  for RBF SVM to achieve hyperparameter tuning
param_grid = {
    'C': [0.1, 1, 5, 10],
    'gamma': ['scale', 0.1, 0.01, 0.001]
}

grid = GridSearchCV(
    SVC(kernel='rbf', class_weight='balanced'),
    param_grid,
    scoring='f1',
    cv=3,
    verbose=1,
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("\nBest Parameters:", grid.best_params_)
print("Best F1 Score:", grid.best_score_)
#%%
#evaluation of final tuned model
best_svm = grid.best_estimator_
y_pred_best = best_svm.predict(X_test)

print("\n=== Tuned RBF SVM Performance ===")
print("Accuracy:", accuracy_score(y_test, y_pred_best))
print("Precision:", precision_score(y_test, y_pred_best))
print("Recall:", recall_score(y_test, y_pred_best))
print("F1 Score:", f1_score(y_test, y_pred_best))
print("\nClassification Report:\n", classification_report(y_test, y_pred_best))
#%%