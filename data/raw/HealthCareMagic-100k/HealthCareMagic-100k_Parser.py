import json
import random
import sys
sys.path.insert(0,r'./')
from tqdm.auto import tqdm

from configs import BaseConfig
from translator import DataParser
from providers import Provider, GoogleProvider, MultipleProviders

PARSER_NAME = "HealthCareMagic100k"


class HealthCareMagic100k(DataParser):
    def __init__(self, file_path: str, output_path: str, target_lang: str="nl",
                 max_example_per_thread=400, large_chunks_threshold=20000,
                 translator: Provider = GoogleProvider):
        super().__init__(file_path, output_path,
                         parser_name=PARSER_NAME,
                         target_config=BaseConfig,  # The data config to be validated to check if self implement "convert" function is correct or not,
                                                    # you must map the data form to the correct fields of the @dataclass in the configs/base_config.py
                         target_fields=['question_text', 'orig_answer_texts'],     # The data fields to be translated (The fields belong to BaseConfig)
                         do_translate=True,
                         target_lang=target_lang,
                         max_example_per_thread=max_example_per_thread,
                         large_chunks_threshold=large_chunks_threshold,
                         translator=translator)

        self.max_ctxs = 5

    # Read function must assign data that has been read to self.data_read
    def read(self) -> None:
        # The read function must call the read function in DataParser class
        # I just want to be sure that the file path is correct
        super(HealthCareMagic100k, self).read()
        print(self.file_path)
        with open(self.file_path, 'r', encoding='utf-8') as jfile:
            self.data_read = json.load(jfile)
        #print(type(self.data_read));exit()
        return None

    def convert(self) -> None:
        # The convert function must call the convert function in DataParser class
        # I just want to be sure the read function has actually assigned the self.data_read
        super(HealthCareMagic100k, self).convert()

        data_converted = []

        for data in tqdm(self.data_read, desc=f"Converting data"):
            data_dict = {}
            data_dict['system_prompt'] = "Als u een arts bent, beantwoord dan de medische vragen op basis van de beschrijving van de patiënten."

            # The DataParser class has an id_generator method which can create random id for you
            data_dict['qas_id'] = self.id_generator()

            data_dict['question_text'] = data['input']

            data_dict['orig_answer_texts'] = data['output']
            data_dict['answer_lengths'] = None
            data_converted.append(data_dict)

        # Be sure to assign the final data list to self.converted_data
        self.converted_data = data_converted

        return None


if __name__ == '__main__':
    HealthCareMagic100k_parser = HealthCareMagic100k(r"HealthCareMagic-100k/HealthCareMagic-100k.json",
                              r"HealthCareMagic-100k",
                              max_example_per_thread=100,
                              large_chunks_threshold=1000,
                              target_lang="nl")
    HealthCareMagic100k_parser.read()
    HealthCareMagic100k_parser.convert()
    HealthCareMagic100k_parser.save