import pandas as pd
from typing import Union

symptom_dict = {
    "Hoesten": {
        "pos": "hoesten positief",
        "neg": "hoesten negatief", 
        "abs": "hoesten afwezig",
    },
    "Koorts": {
        "pos": "ĠKoorts positief",
        "neg": "ĠKoorts negatief",
        "abs": "ĠKoorts afwezig",
    },
    "Kortademigheid": {
        "pos": "ĠKortademigheid positief",
        "neg": "ĠKortademigheid negatief",
        "abs": "ĠKortademigheid afwezig",
    },
}

def construct_prompt(note: str, symptom: str, examples: Union[pd.DataFrame, None] = None):
    prompt = \
        f"""
        Classificeer de volgende teksten op basis van aanwezigheid van het symptoom '{symptom}'.
        Word '{symptom}' vermeld als positief, antwoord '{symptom_dict[symptom]['pos']}'.
        Word '{symptom}' wordt vermeld als negatief, antwoord '{symptom_dict[symptom]['neg']}'.
        Word '{symptom}' niet vermeld in de tekst, antwoord '{symptom_dict[symptom]['abs']}'.
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
            prompt += f"Tekst: '{row['DEDUCE_omschrijving']}'\tBijbehorend label: {label}\n"
    prompt += f"Print exclusief het label bijbehorende aan de volgende tekst: '{note}'"
    return prompt