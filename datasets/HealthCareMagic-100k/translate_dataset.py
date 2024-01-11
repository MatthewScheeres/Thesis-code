# This code is deprecated, the dataset is now translated using the Large_Dataset_translator repository


import json
from pygoogletranslation import Translator
import codecs
from tqdm import tqdm
import logging
import os
from time import sleep
from random import randint

# Set up logging
logging.basicConfig(filename='translation.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

translator = Translator(
    service_url= ['translate.google.com', 'translate.google.nl'],
)

# Translate the "instruction" value once at the start
instruction_translation = translator.translate("If you are a doctor, please answer the medical questions based on the patient's description.", src='en', dest='nl').text

def translate_entry(entry, index):
    logger.info(f"Translating entry {index}")
    entry["instruction"] = instruction_translation
    try:
        entry["input"], entry["output"] = (x.text for x in translator.translate([entry["input"], entry["output"]], src='en', dest='nl'))
    except Exception as e:
        logger.error(f"Error translating entry {index}: {e}")
    return entry

def count_entries(filename):
    with open(filename, 'r') as f:
        return f.read().count('{')

def main():
    BUFFER_SIZE = 75  # Adjust this value based on your requirements and available memory
    

    filename = 'HealthCareMagic-100k'
    output_filename = f'datasets/{filename}_translated.json'

    # Check if the output file exists and count the entries if it does
    start_index = count_entries(output_filename) if os.path.exists(output_filename) else 0


    with codecs.open(f'datasets/{filename}.json', 'r', encoding='utf8') as f_in, \
        codecs.open(f'datasets/{filename}_translated.json', 'a' if start_index > 0 else 'w', encoding='utf8') as f_out:
        try:
            data = json.load(f_in)
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON: {e}")
            return

        if start_index == 0:
            f_out.write('[\n')
        

        buffer = []
        for i in tqdm(range(start_index, len(data))):
            try:
                data[i] = translate_entry(data[i], i)
                buffer.append(data[i])
                if len(buffer) >= BUFFER_SIZE:
                    logger.info(f"Writing {len(buffer)} entries to output file")
                    f_out.write(',\n'.join(json.dumps(item, ensure_ascii=False) for item in buffer) + ',\n')
                    buffer.clear()
                    #sleep(randint(900, 1100)/100)  # Sleep to avoid getting blocked by Google
            except Exception as e:
                logger.error(f"Error at index {start_index + i}: {e}")
                
                

        # Write any remaining entries in the buffer
        if buffer:
            f_out.write(',\n'.join(json.dumps(item, ensure_ascii=False) for item in buffer) + ',\n')
        f_out.write(']')
        logger.info("Done!")
            
if __name__ == '__main__':  
    main()
        
