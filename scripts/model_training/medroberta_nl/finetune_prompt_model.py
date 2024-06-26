from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import AutoModelForSeq2SeqLM, RobertaModel, EncoderDecoderModel
from datasets import Dataset
import json
import evaluate
import numpy as np
import torch




# Load dataset
def gen():
    '''
    Read the json file at path and yield each line.
    '''
    print("Loading dataset...")
    with open('data/raw/HealthCareMagic-100k/HealthCareMagic100k_translated_nl_sample 1k.json', 'r', encoding='utf-8') as f:
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
    model_inputs["encoder_input_ids"] = input_ids
    model_inputs["decoder_input_ids"] = input_ids
    return model_inputs


def compute_metrics(eval_pred):
    print("Computing metrics...")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1).flatten()
    return metric.compute(predictions=predictions, references=labels.flatten())


# Load pre-trained model and tokenizer
print("Loading pre-trained model and tokenizer...")
encoder = "CLTL/MedRoBERTa.nl"
decoder = "google-bert/bert-base-uncased"

model = EncoderDecoderModel.from_encoder_decoder_pretrained(encoder, decoder)


tokenizer = AutoTokenizer.from_pretrained(encoder)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)

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
training_args = Seq2SeqTrainingArguments(
    output_dir="test-trainer", 
    evaluation_strategy="epoch",
    eval_accumulation_steps=20,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    save_strategy='epoch',
    save_steps=500,
    load_best_model_at_end=True,
    gradient_checkpointing=True,
    fp16=False, # Can only use this with CUDA
    gradient_accumulation_steps=4,
    label_smoothing_factor=0.1,
)

trainer = Seq2SeqTrainer(
    model=model, 
    args=training_args, 
    train_dataset=train_dataset, 
    eval_dataset=test_dataset,
    compute_metrics=None,
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
prompt = "Hallo dokter, ik heb last van een zere keel en koorts. Wat moet ik doen?"
inputs = tokenizer.encode(prompt, return_tensors="pt")
outputs = model.generate(
    inputs, 
    max_length=150, 
    num_return_sequences=1, 
    do_sample=True,  # Enable sampling
    top_p=0.95,  # Use top-p sampling
    no_repeat_ngram_size=2,
)
for output in outputs:
    print(tokenizer.decode(output))

# Generate output using beam search
print("Generating sample response using beam search...")
beam_outputs = model.generate(
    inputs,
    max_length=150,
    num_beams=5,  # Number of beams for beam search
    num_return_sequences=1,
    do_sample=False,  # Disable sampling
)
for output in beam_outputs:
    print(tokenizer.decode(output))