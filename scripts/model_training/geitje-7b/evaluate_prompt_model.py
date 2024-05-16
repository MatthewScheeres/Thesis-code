from transformers import pipeline, Conversation
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Union
import warnings
import pandas as pd
import os
import torch
from tqdm import tqdm

from prompt_classification_utils import symptom_dict, construct_prompt

warnings.filterwarnings("ignore")



class promptBasedClassification:
    def __init__(self, model_name: str, symptom: str):
        self.model = pipeline(task='conversational', model=model_name,
                              device_map='auto')
        self.symptom = symptom
        # Initialize the model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    
    def generate(self, text: str, examples: Union[list[tuple], None]) -> str:
        self.prepare_prompt(text['DEDUCE_omschrijving'], examples)
        
        conversation = Conversation(self.prompt)
        return self.model(conversation)
    
    def classify(self, text: str, examples: Union[list[tuple], None]) -> str:
        prompt = construct_prompt(text, examples)
         

        # Get the probabilities of the words
        prob_pos = self._predict_token_probability(prompt, symptom_dict[self.symptom]["pos"])
        prob_neg = self._predict_token_probability(prompt, symptom_dict[self.symptom]["neg"])
        prob_abs = self._predict_token_probability(prompt, symptom_dict[self.symptom]["abs"])
        print(prob_pos == prob_neg, prob_pos == prob_abs)   # Probabilities are all the same, something's wrong. 
        print(f"Probabilities: pos={prob_pos}, neg={prob_neg}, abs={prob_abs}")

        # return the label with the highest probability
        if prob_neg > prob_pos and prob_neg > prob_abs:
            return 0    # negatief
        elif prob_pos > prob_neg and prob_pos > prob_abs:
            return 1    # positief
        else:
            if prob_abs == prob_neg and prob_abs == prob_pos:
                print("[Warning] All probabilities are equal. Returning 'afwezig' as default label.")
            return 2    # afwezig
        
    def _predict_token_probability(self, prompt: str, token: str):
        inputs = self.tokenizer(prompt, return_tensors='pt', padding='max_length', truncation=True, max_length=512)

        tokenized_token = self.tokenizer(token, return_tensors='pt', padding='max_length', truncation=True, max_length=512)

        # Get token id for the token
        token_id = self.tokenizer.encode(token, add_special_tokens=False)[0]

        # Pass tokenized prompt to the model
        with torch.no_grad():  # No need to calculate gradients
            outputs = self.model(input_ids=inputs['input_ids'], decoder_input_ids=tokenized_token['input_ids'], attention_mask=inputs['attention_mask'], output_hidden_states=False, return_dict=True)

        # Get logits (non-normalized probabilities)
        logits = outputs.logits

        # Apply softmax to get probabilities
        probabilities = torch.nn.functional.softmax(logits, dim=-1)

        # Get the probability of the specific token
        token_probability = probabilities[0, -1, token_id].item()

        return token_probability
        
    def evaluate(self, dataset: pd.DataFrame):
        predicted_labels = []
        correct_labels = []

        for _, ehr in tqdm(dataset.iterrows(), total=len(dataset), desc="Predicting..."):
            note = ehr.DEDUCE_omschrijving
            label = ehr[self.symptom]
            print(note, label)

            examples = dataset.iloc[-3:]  # TODO Think of how to best to get examples
            predicted_label = self.classify(note, examples)
            correct_labels.append(label)
            predicted_labels.append(predicted_label)
        return predicted_labels, correct_labels

def main():
    symptom='Koorts'


    # Load the dataset
    df = pd.read_csv("data/raw/patient_data/fake_set_complete.csv")
    
    
    clf = promptBasedClassification(model_name='Rijgersberg/GEITje-7B-chat-v2', symptom=symptom)
    results = clf.evaluate(df)
    print(results)

if __name__ == '__main__':
    main()