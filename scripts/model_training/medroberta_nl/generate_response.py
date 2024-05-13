from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, EncoderDecoderModel
from torch.nn.functional import softmax

def main():
    # Load the model and tokenizer
    model_name = "models/medroberta"
    tokenizer = AutoTokenizer.from_pretrained("CLTL/MedRoBERTa.nl")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    #model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model = EncoderDecoderModel.from_pretrained(model_name)
    
    prompt = """
    Classificeer de volgende teksten op basis van de aanwezigheid van het symptoom 'hoesten' als volgt:

    Als 'hoesten' wordt vermeld als voorkomend bij de patiënt, label het als 'aanwezig'.
    Als 'hoesten' wordt vermeld als niet voorkomend bij de patiënt, label het als 'niet aanwezig'.
    Als 'hoesten' helemaal niet wordt vermeld in de tekst, label het als 'niet vermeld'.

    Print exclusief het label bijbehorende aan de volgende tekst:
    "Vierkant naar grap beer bureau extreem vergelijken. Moeder structuur hoog traan verdrietig iets hoe verstoppen. Helft rots benzine paard altijd. Volgorde laag geliefde aan kok. Koorts Hoesten Kortademigheid."
    """

    possible_outcomes = ["aanwezig", "niet aanwezig", "niet vermeld"]
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
    
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.decoder_start_token_id = tokenizer.cls_token_id

    # Pass the tokenized prompt to the model
    # Pass the tokenized prompt to the model
    outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])

    # Get the logits
    logits = outputs.logits

    # Apply softmax to convert logits into probabilities
    probabilities = softmax(logits, dim=-1)

    # Find the index of the maximum probability
    max_prob_index = probabilities.argmax()

    # Get the corresponding outcome
    outcome = possible_outcomes[max_prob_index]

    # Print the outcome and its probability
    print(f"Outcome: {outcome}, Probability: {probabilities[0, max_prob_index]}")
    
    

    
if __name__ == "__main__":
    main()
