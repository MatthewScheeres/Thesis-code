from transformers import EncoderDecoderModel, AutoTokenizer
import torch

# Load the trained model and tokenizer
model = EncoderDecoderModel.from_pretrained("models/medroberta")
tokenizer = AutoTokenizer.from_pretrained("CLTL/MedRoBERTa.nl")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Access the model configuration
config = model.config

# Get the BOS token ID
bos_token_id = config.bos_token_id
model.generation_config.decoder_start_token_id = tokenizer.cls_token_id

def generate_response(prompt):
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors='pt', padding='max_length', truncation=True, max_length=512)

    # Pass the tokenized prompt to the model
    outputs = model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], do_sample=True, top_p=0.95, no_repeat_ngram_size=2)

    # Decode the output ids to get the generated response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response    


def predict_token_probability(prompt, token):
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
    tokenized_token = tokenizer(token, return_tensors='pt', padding='max_length', truncation=True, max_length=512)

    # Get the token id for the token
    token_id = tokenizer.encode(token, add_special_tokens=False)[0]

    # Pass the tokenized prompt to the model
    with torch.no_grad():  # No need to calculate gradients
        outputs = model(input_ids=inputs['input_ids'], decoder_input_ids=tokenized_token['input_ids'], attention_mask=inputs['attention_mask'], output_hidden_states=False, return_dict=True)

    # Get the logits (unnormalized probabilities)
    logits = outputs.logits

    # Apply softmax to get probabilities
    probabilities = torch.nn.functional.softmax(logits, dim=-1)

    # Get the probability of the specific token
    token_probability = probabilities[0, -1, token_id].item()

    return token_probability

prompt = "De patient heeft last van kortademigheid."
token = "ĠKortademigheid afwezig"
print(predict_token_probability(prompt, token))




