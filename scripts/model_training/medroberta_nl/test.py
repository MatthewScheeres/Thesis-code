from transformers import pipeline, AutoTokenizer, EncoderDecoderModel
import torch.nn.functional as F
import torch

# Replace "your_model_name" with the actual name of your fine-tuned model
tokenizer = AutoTokenizer.from_pretrained("CLTL/MedRoBERTa.nl")
model = EncoderDecoderModel.from_pretrained("models/medroberta")

# Access the model configuration
config = model.config

# Get the BOS token ID
bos_token_id = config.bos_token_id
model.generation_config.decoder_start_token_id = tokenizer.cls_token_id

symptom_classes = ["NOT_DESCRIBED", "POSITIVE", "NEGATIVE"]

def preprocess_note(note):
    return note.lower()

def predict_symptom(doctor_note, symptom_name):
  processed_note = preprocess_note(doctor_note)  # Replace with your preprocessing function
  prompt = f"Given the context and labeled examples, does the following doctor's note describe the symptom '{symptom_name}' as: \n {processed_note}"
  inputs = tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
  outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
  logits = outputs.logits.squeeze(0)
  probabilities = F.softmax(logits, dim=-1)
  predicted_class = symptom_classes[torch.argmax(probabilities).item()]
  confidence_score = probabilities[torch.argmax(probabilities)].item()
  
  if confidence_score > 0.8:
    print(f"Symptom '{symptom_name}' is predicted as: {predicted_class} with confidence {confidence_score:.2f}")
  else:
    print(f"Model is uncertain about symptom '{symptom_name}' with confidence score {confidence_score:.2f}")

# Example usage
doctor_note = "The patient reported experiencing a cough and fever for the past three days."
symptom_name = "cough"
predict_symptom(doctor_note, symptom_name)
