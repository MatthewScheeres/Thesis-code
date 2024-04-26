from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import torch
import numpy as np
import matplotlib.pyplot as plt

from utils import compute_metrics, SymptomDataset
import json

LABEL_TO_CLASSIFY = 'Hoesten'

# Load the dataset
df = pd.read_csv('C:\\Users\\matth\\OneDrive\\Documenten\\UU MSc Artificial Intelligence\\Thesis\\Thesis-code\\data\\raw\\patient_data\\example_dataset.csv')

# Preprocess the dataset
df = df[['DEDUCE_omschrijving', LABEL_TO_CLASSIFY]]

train_texts, test_texts, train_labels, test_labels = train_test_split(df['DEDUCE_omschrijving'].tolist(), df[LABEL_TO_CLASSIFY].tolist(), test_size=0.2)

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
    evaluation_strategy='steps',
    save_strategy='epoch',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01, 
    logging_steps=10,
    logging_strategy='steps',
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

print(trainer.state.log_history)

# Evaluate the model
eval_results = trainer.evaluate()


save_path = './fine_tuned_model'

# Save the fine-tuned model
model.save_pretrained(save_path)

### log saving
# Initialize lists for each type of log
train_logs = []
eval_logs = []
total_logs = []

# Iterate over the log history
for log in trainer.state.log_history:
    # Check the keys in the log to determine its type
    if 'loss' in log and 'learning_rate' in log:
        train_logs.append(log)
    elif 'eval_loss' in log:
        eval_logs.append(log)
    elif 'train_runtime' in log:
        total_logs.append(log)

# Save the logs to the file
with open(f'{save_path}/log_history.json', 'w') as f:
    json.dump({'Train Logs': train_logs, 'Eval Logs': eval_logs, 'Total Logs': total_logs}, f)




# Extract the training and evaluation loss from the logs
train_loss = [log['loss'] for log in train_logs]
eval_loss = [log['eval_loss'] for log in eval_logs]

# Create a new figure
plt.figure()

# Plot the training and evaluation loss
plt.plot(train_loss, label='Train Loss')
plt.plot(eval_loss, label='Eval Loss')

# Add a legend
plt.legend()

# Show the plot
plt.show()