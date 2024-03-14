from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import json
import evaluate
import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: ", device)

if torch.cuda.is_available():
    print("GPU Type: ", torch.cuda.get_device_name())

print("CUDA Version: ", torch.version.cuda)

# Load dataset
def gen():
    '''
    Read the json file at path and yield each line.
    '''
    print("Loading dataset...")
    with open('data/raw/HealthCareMagic-100k/HealthCareMagic100k_translated_nl_sample mini.json', 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            yield {"context": line["system_prompt"], "prompt": line["question_text"], "output": line["orig_answer_texts"]}

def preprocess_function(example):
    print("Preprocessing data...")
    text = [context + ' ' + prompt for context, prompt in zip(example['context'], example['prompt'])]
    
    target = example['output']
    model_inputs = tokenizer(text, truncation=True, padding="max_length", max_length=512)
    input_ids = tokenizer(target, truncation=True, padding="max_length", max_length=512)["input_ids"]
    model_inputs["labels"] = input_ids
    return model_inputs


def compute_metrics(eval_pred):
    print("Computing metrics...")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1).flatten()
    print(np.shape(predictions), np.shape(labels))
    return metric.compute(predictions=predictions, references=labels)


# Load pre-trained model and tokenizer
print("Loading pre-trained model and tokenizer...")
model_type = "CLTL/MedRoBERTa.nl"
model = AutoModelForCausalLM.from_pretrained(model_type).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_type)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Load and map dataset
print("Mapping dataset...")
dataset = Dataset.from_generator(gen)
dataset = dataset.map(preprocess_function, batched=True)

# Split the dataset into train and test
print("Splitting dataset into train and test...")
dataset = dataset.train_test_split(test_size=0.1)
train_dataset, test_dataset = dataset['train'], dataset['test']
print("Train dataset: ", train_dataset)

# Define training arguments and instantiate Trainer
print("Defining training arguments and instantiating Trainer...")
metric = evaluate.load("accuracy")
training_args = TrainingArguments(
    output_dir="test-trainer", 
    evaluation_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    save_strategy='epoch',
    save_steps=500,
    load_best_model_at_end=True,
    gradient_checkpointing=True,
    fp16=True,
    gradient_accumulation_steps=4,
)

trainer = Trainer(
    model=model, 
    args=training_args, 
    train_dataset=train_dataset, 
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)

# Train the model
print("Training the model...")
trainer.train(
    resume_from_checkpoint=False,
)

print("Saving the model...")
trainer.save_model("models/medroberta")

# Generate sample response
print("Generating sample response...")
prompt = "Your prompt text here"
inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
outputs = model.generate(inputs, max_length=150, num_return_sequences=3)
for output in outputs:
    print(tokenizer.decode(output))