from transformers import pipeline, Conversation
from typing import Union
import warnings
import pandas as pd
import os

warnings.filterwarnings("ignore")



class promptBasedClassification:
    def __init__(self, model_name: str, symptom: str):
        self.model = pipeline(task='conversational', model=model_name,
                              device_map='auto')
        self.symptom = symptom
        self.base_prompt = \
        f"""
Classificeer de volgende teksten op basis van de aanwezigheid van het symptoom '{symptom}' als volgt:
Als '{symptom}' wordt vermeld als voorkomend bij de patiënt, label het als 'Aanwezig.'
Als '{symptom}' wordt vermeld als niet voorkomend bij de patiënt, label het als 'Niet aanwezig.'
Als '{symptom}' helemaal niet wordt vermeld in de tekst, label het als 'Niet vermeld.'
        
        """
        
    def prepare_prompt(self, text: str, examples: Union[pd.DataFrame, None]) -> None:
        prompt = self.base_prompt
        if examples is not None:
            prompt += f"Gebruik de volgende voorbeelden om de classificatie te illustreren:\n"
            for i, row in examples.iterrows():
                if row[self.symptom] == 2:
                    label = 'Niet vermeld.'
                if row[self.symptom] == 1:
                    label = 'Aanwezig.'
                if row[self.symptom] == 0:
                    label = 'Niet aanwezig.'
                prompt += f"Tekst: {row['DEDUCE_omschrijving']}\tLabel: {label}\n"
        prompt += f"\nPrint exclusief het label bijbehorende aan de volgende tekst: {text}"
        self.prompt = prompt
    
    def classify(self, text: str, examples: Union[list[tuple], None]) -> str:
        self.prepare_prompt(text['DEDUCE_omschrijving'], examples)
        
        #TODO Add check to split longer prompts appropriately
        
        conversation = Conversation(self.prompt)
        return self.model(conversation)
    
def run_crossvalidation(symptom: str, shot_amount: int = 0) -> str:
    """Runs prompt-based crossvalidation on a given symptom. 

    Args:
        symptom (str): One of Koorts, Hoesten or Kortademigheid.
        shot_amount (int): The amount of samples to provide the LLM with. Defaults to 0.

    Returns:
        str: Path to the evaluation results JSON file.
    """
    

def main():
    # Load the dataset
    df = pd.read_csv("data/raw/patient_data/fake_train.csv")

    # Preprocess the dataset
    df = df[['DEDUCE_omschrijving', 'Koorts', 'Hoesten', 'Kortademigheid']]
    
    
    clf = promptBasedClassification(model_name='Rijgersberg/GEITje-7B-chat-v2', symptom='Koorts')
    classification = clf.classify(df.iloc[0], df.iloc[1:3])
    print(classification)

if __name__ == '__main__':
    main()