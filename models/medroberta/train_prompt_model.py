from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.model_selection import train_test_split

def preprocess_function(example):
    text = ' '.join(example['system_prompt']) + ' ' + ' '.join(example['question_text'])
    target = example['orig_answer_texts']
    model_inputs = tokenizer(text, truncation=True, padding="max_length", max_length=1000)
    model_inputs["labels"] = tokenizer(target, truncation=True, padding="max_length", max_length=1000)["input_ids"]
    return model_inputs


# Load pre-trained model and tokenizer
model = AutoModelForCausalLM.from_pretrained("CLTL/MedRoBERTa.nl")
tokenizer = AutoTokenizer.from_pretrained("CLTL/MedRoBERTa.nl")

# Load dataset
dataset = load_dataset('json', data_files='datasets/HealthCareMagic-100k/HealthCareMagic100k_translated_nl.json')

# Split the dataset into train and test
train_dataset, test_dataset = train_test_split(dataset['train'], test_size=0.2)

# Preprocess and map for both train and test datasets
train_dataset = train_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)

# Define training arguments and instantiate Trainer
training_args = TrainingArguments("test-trainer")
trainer = Trainer(
    model=model, args=training_args, train_dataset=train_dataset, eval_dataset=test_dataset
)

# Train the model
trainer.train()

# After training, you can generate responses like this:
prompt = "Your prompt text here"
inputs = tokenizer.encode(prompt, return_tensors="pt")
outputs = model.generate(inputs, max_length=150, num_return_sequences=3)
for output in outputs:
    print(tokenizer.decode(output))