from datasets import Dataset
from tqdm import tqdm
import json



def gen() -> dict:
    '''
    Read the json file at path and yield each line.
    '''
    with open('datasets/HealthCareMagic-100k/HealthCareMagic100k_translated_nl.json', 'r') as f:
        for line in f:
            line = json.loads(line)
            yield {"context": line["system_prompt"], "prompt": line["question_text"], "output": line["orig_answer_texts"]}
    
    
ds = Dataset.from_generator(gen)
print(ds[0])