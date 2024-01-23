from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import json
from sklearn.model_selection import train_test_split
import evaluate
import numpy as np


# Load dataset
def gen() -> dict:
    '''
    Read the json file at path and yield each line.
    '''
    with open('datasets/HealthCareMagic-100k/HealthCareMagic100k_translated_nl.json', 'r') as f:
        for line in f:
            line = json.loads(line)
            yield {"context": line["system_prompt"], "prompt": line["question_text"], "output": line["orig_answer_texts"]}

def preprocess_function(example):
    text = ' '.join(example['context']) + ' ' + ' '.join(example['prompt'])
    target = example['output']
    model_inputs = tokenizer(text, truncation=True, padding="max_length", max_length=1000)
    model_inputs["labels"] = tokenizer(target, truncation=True, padding="max_length", max_length=1000)["input_ids"]
    
    return model_inputs


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# Load pre-trained model and tokenizer
model_type = "CLTL/MedRoBERTa.nl"
model = AutoModelForCausalLM.from_pretrained(model_type)
tokenizer = AutoTokenizer.from_pretrained(model_type)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load and map dataset
dataset = Dataset.from_generator(gen)
dataset = dataset.map(preprocess_function, batched=True)

# Split the dataset into train and test
dataset = dataset.train_test_split(test_size=0.1)
train_dataset, test_dataset = dataset['train'], dataset['test']


# Define training arguments and instantiate Trainer
metric = evaluate.load("accuracy")
training_args = TrainingArguments(output_dir="test-trainer", evaluation_strategy="epoch")
trainer = Trainer(
    model=model, 
    args=training_args, 
    train_dataset=train_dataset, 
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Generate sample response
prompt = "Your prompt text here"
inputs = tokenizer.encode(prompt, return_tensors="pt")
outputs = model.generate(inputs, max_length=150, num_return_sequences=3)
for output in outputs:
    print(tokenizer.decode(output))