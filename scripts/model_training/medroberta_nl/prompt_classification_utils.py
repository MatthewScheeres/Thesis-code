import pandas as pd
from typing import Union

symptom_dict = {
    "hoesten": {
        "pos": "hoesten positief",
        "neg": "hoesten negatief", 
        "abs": "hoesten afwezig",
    },
    "koorts": {
        "pos": "ĠKoorts positief",
        "neg": "ĠKoorts negatief",
        "abs": "ĠKoorts afwezig",
    },
    "kortademigheid": {
        "pos": "ĠKortademigheid positief",
        "neg": "ĠKortademigheid negatief",
        "abs": "ĠKortademigheid afwezig",
    },
}

def construct_prompt(note: str, symptom: str, examples: Union[pd.DataFrame, None] = None):
    prompt = \
        f"""
        Classificeer de volgende teksten op basis van de aanwezigheid van het symptoom '{symptom}' als volgt:
        Als '{symptom}' wordt vermeld als positief, label het als '{symptom_dict['symptom']['pos']}'.
        Als '{symptom}' wordt vermeld als negatief, label het als '{symptom_dict['symptom']['neg']}'.
        Als '{symptom}' niet wordt vermeld in de tekst, label het als '{symptom_dict['symptom']['abs']}'.
        """
        
    if examples is not None:
        prompt += f"Gebruik de volgende voorbeelden om de classificatie te illustreren:\n"
        for i, row in examples.iterrows():
            if row[symptom] == 2:
                label = 'afwezig'
            if row[symptom] == 1:
                label = 'positief'
            if row[symptom] == 0:
                label = 'negatief'
            prompt += f"Tekst: {row['DEDUCE_omschrijving']}\tLabel: {label}\n"
    prompt += f"\nPrint exclusief het label bijbehorende aan de volgende tekst: {note}"
    return prompt