from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import torch
import numpy as np

from utils import compute_metrics, SymptomDataset


# Load the dataset
df = pd.read_csv('C:\\Users\\matth\\OneDrive\\Documenten\\UU MSc Artificial Intelligence\\Thesis\\Thesis-code\\data\\raw\\patient_data\\example_dataset.csv')

# Preprocess the dataset
df['labels'] = df['Koorts'].apply(lambda x: [0 if i=='0' else 1 if i=='1' else 2 for i in x])
df = df[['DEDUCE_omschrijving', 'labels']]

train_texts, test_texts, train_labels, test_labels = train_test_split(df['DEDUCE_omschrijving'].tolist(), df['labels'].tolist(), test_size=0.2)

# Load the pre-trained model and tokenizer
tokenizer = RobertaTokenizerFast.from_pretrained("CLTL/MedRoBERTa.nl")
model = RobertaForSequenceClassification.from_pretrained("CLTL/MedRoBERTa.nl", num_labels=3)

# Tokenize the dataset
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

# Create a Dataset class
class SymptomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    


train_dataset = SymptomDataset(train_encodings, train_labels)
test_dataset = SymptomDataset(test_encodings, test_labels)



# Fine-tune the model on the dataset
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01, 
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()

print(eval_results)

# Save the fine-tuned model
model.save_pretrained('./fine_tuned_model')