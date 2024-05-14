from transformers import EncoderDecoderModel, AutoTokenizer, PreTrainedModel
import torch

from prompt_classification_utils import symptom_dict, construct_prompt

class SymptomPromptModel:
    def __init__(self, model: PreTrainedModel, tokenizer: AutoTokenizer, symptom: str):
        self.model = model
        self.tokenizer = tokenizer
        self.symptom = symptom


    def classify(self, note: str, examples: list):
        prompt = construct_prompt(note, examples)
        prob_pos = self._predict_token_probability(prompt, symptom_dict[self.symptom]["pos"], self.model, self.tokenizer)
        prob_neg = self._predict_token_probability(prompt, symptom_dict[self.symptom]["neg"], self.model, self.tokenizer)
        prob_abs = self._predict_token_probability(prompt, symptom_dict[self.symptom]["abs"], self.model, self.tokenizer)
        
        # return the label with the highest probability
        if prob_neg > prob_pos and prob_neg > prob_abs:
            return 0    # negatief
        elif prob_pos > prob_neg and prob_pos > prob_abs:
            return 1    # positief
        else:
            if prob_abs == prob_neg and prob_abs == prob_pos:
                print("[Warning] All probabilities are equal. Returning 'afwezig' as default label.")
            return 2    # afwezig
    
    def _predict_token_probability(self, prompt: str, token: str, model: PreTrainedModel, tokenizer: AutoTokenizer):
        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
        tokenized_token = self.tokenizer(token, return_tensors='pt', padding='max_length', truncation=True, max_length=512)

        # Get the token id for the token
        token_id = self.tokenizer.encode(token, add_special_tokens=False)[0]

        # Pass the tokenized prompt to the model
        with torch.no_grad():  # No need to calculate gradients
            outputs = self.model(input_ids=inputs['input_ids'], decoder_input_ids=tokenized_token['input_ids'], attention_mask=inputs['attention_mask'], output_hidden_states=False, return_dict=True)

        # Get the logits (unnormalized probabilities)
        logits = outputs.logits

        # Apply softmax to get probabilities
        probabilities = torch.nn.functional.softmax(logits, dim=-1)

        # Get the probability of the specific token
        token_probability = probabilities[0, -1, token_id].item()

        return token_probability


    def evaluate(self, dataset):
        results = []
        for ehr in dataset.iterrows():
            note = ehr['DEDUCE_omschrijving']
            label = ehr['label']
            examples = ...
            response, probability = self.predict(note, examples)
            results.append((response, probability, label))
        return results
    
    
    def _generate_response(prompt, model: PreTrainedModel, tokenizer: AutoTokenizer):
        # Tokenize the prompt
        inputs = tokenizer(prompt, return_tensors='pt', padding='max_length', truncation=True, max_length=512)

        # Pass the tokenized prompt to the model
        outputs = model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], do_sample=True, top_p=0.95, no_repeat_ngram_size=2)

        # Decode the output ids to get the generated response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return response    




def predict


def main():
    symptom = "koorts"
    # Load the trained model and tokenizer
    model = EncoderDecoderModel.from_pretrained("models/medroberta")
    tokenizer = AutoTokenizer.from_pretrained("CLTL/MedRoBERTa.nl")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model.generation_config.decoder_start_token_id = tokenizer.cls_token_id
    
    # TODO Load the dataset
    dataset = ...
    
    results = []
    
    
    for ehr in dataset.iterrows():
        note = ehr['DEDUCE_omschrijving']
        label = ehr['label']
        
        # TODO Think of how to best to get examples
        examples = ...
        
        
        
        prompt = construct_prompt(note, examples)
        response = predict_token_probability(note, model, tokenizer)
        print(response)
    prompt = "De patient heeft last van kortademigheid."
    response = predict_token_probability(prompt, "ĠKoorts positief", model, tokenizer)
    print(response)
    
    

if __name__ == "__main__":
    main()




