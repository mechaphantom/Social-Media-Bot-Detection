import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import torch
from transformers import BertTokenizer
from datasets import Dataset
import torch
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
os.environ["WANDB_DISABLED"] = "true"
#%%
df = pd.read_csv("twitter_human_bots_dataset.csv")
print("Dataset Shape:", df.shape)
df.head()
#%%
# Replacing NaN descriptions with empty string
df['description'] = df['description'].fillna("")
df['description'] = df['description'].astype(str)

# Preparing labels
df['label'] = df['account_type'].map({'human': 0, 'bot': 1})

print("\nLabel Distribution:")
print(df['label'].value_counts())

plt.figure(figsize=(6,4))
df['label'].value_counts().plot(kind='bar', color=['blue','red'])
plt.title("Bot vs Human Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

# Description Length Analysis

df['desc_length'] = df['description'].apply(len)

print("\nDescription Length Statistics:")
print(df['desc_length'].describe())

plt.figure(figsize=(8,4))
sns.histplot(df['desc_length'], bins=50, kde=True)
plt.title("Distribution of Description Length")
plt.xlabel("Number of Characters")
plt.ylabel("Frequency")
plt.show()

train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['description'].tolist(),
    df['label'].tolist(),
    test_size=0.20,
    random_state=42,
    stratify=df['label']
)
#%%
print("\nTrain size:", len(train_texts))
print("Test size:", len(test_texts))

print("\nSample training text:", train_texts[0])
print("Sample label:", train_labels[0])

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
def tokenize_batch(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )
#%%
# In here, I am converting our python lists to huggingface dateset format
train_dataset = Dataset.from_dict({
    "text": train_texts,
    "labels": train_labels
})

test_dataset = Dataset.from_dict({
    "text": test_texts,
    "labels": test_labels
})

# format setting for pytorch
train_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"]
)

test_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"]
)
print("Train dataset example:")
print(train_dataset[0])

print("\nTest dataset example:")
print(test_dataset[0])
#%%
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)
#%%
# training arguments

training_args = TrainingArguments(
    output_dir="./bert_output",
    eval_strategy="epoch",
    save_strategy="no",
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_steps=50,
    report_to="none"
)

# trainer setup

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)
trainer.train()

predictions = trainer.predict(test_dataset)
pred_labels = torch.argmax(torch.tensor(predictions.predictions), dim=1)
#%%
acc = accuracy_score(test_labels, pred_labels)
prec = precision_score(test_labels, pred_labels)
rec = recall_score(test_labels, pred_labels)
f1 = f1_score(test_labels, pred_labels)

print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("F1 Score:", f1)
#%%
cm = confusion_matrix(test_labels, pred_labels)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Human", "Bot"],
            yticklabels=["Human", "Bot"])
plt.title("Confusion Matrix — BERT Fine-Tuned Model")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()
#%%