from transformers import EncoderDecoderModel, AutoTokenizer, PreTrainedModel
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import json
from datetime import datetime
import os
from typing import Union

from prompt_classification_utils import symptom_dict, construct_prompt

PARENT_DIR = 'Z:/E_ResearchData/2_ResearchData/NLP project Tuur/Thesis Matthew/'
PARENT_DIR = 'C:/Users/matth/OneDrive/Documenten/UU MSc Artificial Intelligence/Thesis/Thesis-code/'

class SymptomPromptModel:
    def __init__(self, model: PreTrainedModel, tokenizer: AutoTokenizer, symptom: str):
        self.model = model
        self.tokenizer = tokenizer
        self.symptom = symptom


    def classify(self, note: str, examples: Union[list[tuple], None]):
        prompt = construct_prompt(note, self.symptom, examples)
        prob_pos = self._predict_token_probability(prompt, symptom_dict[self.symptom]["pos"], self.model, self.tokenizer)
        prob_neg = self._predict_token_probability(prompt, symptom_dict[self.symptom]["neg"], self.model, self.tokenizer)
        prob_abs = self._predict_token_probability(prompt, symptom_dict[self.symptom]["abs"], self.model, self.tokenizer)
        print(f"Prompt: {prompt}")
        print(f"Probabilities: pos={prob_pos}, neg={prob_neg}, abs={prob_abs}")
        print("Prompt length: ", len(prompt))
        
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
            
            examples = dataset.iloc[-3:]  # TODO Think of how to best to get examples
            prediction = self.classify(note, examples)
            correct_labels.append(label)
            predicted_labels.append(prediction)
        return predicted_labels, correct_labels
    
    
    def _generate_response(prompt, model: PreTrainedModel, tokenizer: AutoTokenizer):
        # Tokenize the prompt
        inputs = tokenizer(prompt, return_tensors='pt', padding='max_length', truncation=True, max_length=512)

        # Pass the tokenized prompt to the model
        outputs = model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], do_sample=True, top_p=0.95, no_repeat_ngram_size=2)

        # Decode the output ids to get the generated response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return response    

def generate_run_dir(model_name: str, label, DS_SIZE, k=5) -> str:
    current = datetime.now().strftime('%Y%m%d_%H%M')
    dir_name = f'runs/pbrun_{label}_{model_name}_{DS_SIZE}-samples_{current}/'
    
    path = os.path.join(PARENT_DIR, dir_name)
    os.mkdir(path)
    
    # Create subdirs for folds
    for i in range(k):
        fold_path = os.path.join(PARENT_DIR, dir_name, f'fold_{i}')
        os.mkdir(fold_path)
    
    
    print(f'Created folder \'{path}\' for current run.')
    return path


def main():
    symptom = "Koorts"
    k=5
    model_name = "medroberta"
    DS_SIZE = 10
    run_dir = generate_run_dir(model_name, symptom, DS_SIZE)

    for current_fold in range(k):
        print(f'Fold {current_fold}')
        # Load the trained model and tokenizer
        model = EncoderDecoderModel.from_pretrained("models/"+model_name)
        tokenizer = AutoTokenizer.from_pretrained("CLTL/MedRoBERTa.nl")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model.generation_config.decoder_start_token_id = tokenizer.cls_token_id

        # Load the dataset
        ds_path = f"data/raw/patient_data/fake_test_{current_fold+1}.csv"#TODO switch to actual filename in DRE
        dataset = pd.read_csv(ds_path, sep=',')[:DS_SIZE]#TODO switch to | in DRE

        # Initialize the model
        model = SymptomPromptModel(model, tokenizer, symptom)
        predicted_labels, correct_labels = model.evaluate(dataset)
        precision, recall, f1, _ = precision_recall_fscore_support(correct_labels, predicted_labels, average='weighted')
        accuracy = accuracy_score(correct_labels, predicted_labels)
        conf_matrix = confusion_matrix(correct_labels, predicted_labels)

        results_json = {
            "Eval Logs": [
                {
                    "eval_accuracy": accuracy,
                    "eval_f1": f1,
                    "eval_precision": precision,
                    "eval_recall": recall,
                    "eval_confusion_matrix": conf_matrix.tolist()
                }
            ]
        }

        save_path = os.path.join(run_dir, f'fold_{current_fold}/log_history.json')
        with open(save_path, "w") as f:
            json.dump(results_json, f)

if __name__ == "__main__":
    main()




