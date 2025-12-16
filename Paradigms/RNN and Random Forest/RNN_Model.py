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
#%%
#Load dataset from csv file
csv_file = r"twitter_human_bots_dataset.csv"
chunksize = 10 ** 3 
df_chunks = pd.read_csv(csv_file, chunksize=chunksize)
df = pd.concat(df_chunks)
df.head(10)
#%%
rnn_df = df[['lang', 'account_type']].copy()

label_encoder_rnn = LabelEncoder()
rnn_df['account_type_encoded'] = label_encoder_rnn.fit_transform(rnn_df['account_type'])
y_rnn = rnn_df['account_type_encoded']

# Fill NaN values in 'lang' column with empty strings before tokenization
rnn_df['lang'] = rnn_df['lang'].fillna('')

# Initialize Tokenizer with num_words (adjust as needed)
tokenizer = Tokenizer(num_words=5000, oov_token="<unk>") # OOV token for out-of-vocabulary words
tokenizer.fit_on_texts(rnn_df['lang'])

sequences = tokenizer.texts_to_sequences(rnn_df['lang'])

# Determine max sequence length and pad sequences
# Calculate max_sequence_length only if sequences is not empty
if sequences:
    max_sequence_length = max([len(x) for x in sequences])
else:
    max_sequence_length = 1 # Default or handle as appropriate for empty sequences
X_rnn = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')

# Split data into training and testing sets
X_train_rnn, X_test_rnn, y_train_rnn, y_test_rnn = train_test_split(X_rnn, y_rnn, test_size=0.2, random_state=42)

print("Shape of X_rnn (padded sequences):", X_rnn.shape)
print("Shape of y_rnn (encoded labels):", y_rnn.shape)
print("Shape of X_train_rnn: ", X_train_rnn.shape)
print("Shape of X_test_rnn:", X_test_rnn.shape)
print("Shape of y_train_rnn:", y_train_rnn.shape)
print("Shape of y_test_rnn:", y_test_rnn.shape)
#%%
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 100 # You can adjust this value

print(f"DEBUG: Final calculated vocab_size for Embedding layer: {vocab_size}")
print(f"DEBUG: Number of words in tokenizer.word_index: {len(tokenizer.word_index)}")

model_rnn = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim), # input_dim must match max index + 1
    LSTM(128), # You can adjust the number of LSTM units
    Dense(1, activation='sigmoid') # Binary classification output
])

model_rnn.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

model_rnn.summary()
print("RNN model built and compiled successfully.")
#%%
history = model_rnn.fit(X_train_rnn, y_train_rnn, epochs=10, batch_size=32, validation_split=0.2)
print("RNN model trained successfully.")
#%% RNN Model Evaluation
# Predict probabilities on the test set
y_pred_proba_rnn = model_rnn.predict(X_test_rnn)

# Convert probabilities to binary predictions (0 or 1)
y_pred_rnn = (y_pred_proba_rnn > 0.5).astype(int)

# Flatten y_pred_rnn to match y_test_rnn shape
y_pred_rnn_flat = y_pred_rnn.flatten()

accuracy_rnn = accuracy_score(y_test_rnn, y_pred_rnn_flat)
precision_rnn = precision_score(y_test_rnn, y_pred_rnn_flat)
recall_rnn = recall_score(y_test_rnn, y_pred_rnn_flat)
f1_rnn = f1_score(y_test_rnn, y_pred_rnn_flat)

print(f"\nRNN Model Evaluation:")
print(f"Accuracy: {accuracy_rnn:.4f}")
print(f"Precision: {precision_rnn:.4f}")
print(f"Recall: {recall_rnn:.4f}")
print(f"F1-Score: {f1_rnn:.4f}")
#%% Confusion Matrix for RNN Model
# Calculate the confusion matrix
cm_rnn = confusion_matrix(y_test_rnn, y_pred_rnn_flat)

# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_rnn, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Bot', 'Predicted Human'],
            yticklabels=['Actual Bot', 'Actual Human'])
plt.title('Confusion Matrix for RNN Model')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()