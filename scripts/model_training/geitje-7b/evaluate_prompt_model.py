from transformers import pipeline, Conversation
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Union
import warnings
import pandas as pd
import os
import torch

warnings.filterwarnings("ignore")



class promptBasedClassification:
    def __init__(self, model_name: str, symptom: str):
        self.model = pipeline(task='conversational', model=model_name,
                              device_map='auto')
        self.symptom = symptom
        self.base_prompt = \
        f"""
Classificeer de volgende teksten op basis van de aanwezigheid van het symptoom '{symptom}' als volgt:
Als '{symptom}' wordt vermeld als positief, label het als 'positief'.
Als '{symptom}' wordt vermeld als negatief, label het als 'negatief'.
Als '{symptom}' niet wordt vermeld in de tekst, label het als 'afwezig'.
        
        """
        self.model_name = model_name
        
    def prepare_prompt(self, text: str, examples: Union[pd.DataFrame, None]) -> None:
        prompt = self.base_prompt
        if examples is not None:
            prompt += f"Gebruik de volgende voorbeelden om de classificatie te illustreren:\n"
            for i, row in examples.iterrows():
                if row[self.symptom] == 2:
                    label = 'afwezig'
                if row[self.symptom] == 1:
                    label = 'positief'
                if row[self.symptom] == 0:
                    label = 'negatief'
                prompt += f"Tekst: {row['DEDUCE_omschrijving']}\tLabel: {label}\n"
        prompt += f"\nPrint exclusief het label bijbehorende aan de volgende tekst: {text}"
        self.prompt = prompt
    
    def classify(self, text: str, examples: Union[list[tuple], None]) -> str:
        self.prepare_prompt(text['DEDUCE_omschrijving'], examples)
        
        conversation = Conversation(self.prompt)
        return self.model(conversation)
    
    def classify(self, text: str, examples: Union[list[tuple], None]) -> str:
        self.prepare_prompt(text['DEDUCE_omschrijving'], examples)
        
        # Initialize the model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Prepare the input
        inputs = tokenizer.encode(self.prompt, return_tensors='pt')

        # Pass the input to the model
        outputs = model(inputs)

        # Get the logits
        logits = outputs.logits

        # Apply softmax to get probabilities
        probabilities = torch.nn.functional.softmax(logits, dim=-1)

        # Get the indices of the words
        pos_index = tokenizer.encode(' positief')[0]
        neg_index = tokenizer.encode(' negatief')[0]
        abs_index = tokenizer.encode(' afwezig')[0]

        # Get the probabilities of the words
        pos_prob = probabilities[0, -1, pos_index].item()
        neg_prob = probabilities[0, -1, neg_index].item()
        abs_prob = probabilities[0, -1, abs_index].item()

        # Return the word with the highest probability
        max_prob = max(pos_prob, neg_prob, abs_prob)
        if max_prob == pos_prob:
            return 'positief'
        elif max_prob == neg_prob:
            return 'negatief'
        else:
            return 'afwezig'

def main():
    # Load the dataset
    df = pd.read_csv("data/raw/patient_data/fake_set_complete.csv")

    # Preprocess the dataset
    df = df[['DEDUCE_omschrijving', 'Koorts', 'Hoesten', 'Kortademigheid']]
    
    
    clf = promptBasedClassification(model_name='Rijgersberg/GEITje-7B-chat-v2', symptom='Koorts')
    classification = clf.classify(df.iloc[90][:30], df.iloc[20:24])
    print(clf.prompt)
    print(classification)

if __name__ == '__main__':
    main()