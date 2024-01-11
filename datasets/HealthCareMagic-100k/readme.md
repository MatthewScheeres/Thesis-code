# HealthCareMagic-100k Dutch Dataset

This repository contains the Dutch version of the HealthCareMagic-100k dataset. The dataset was gathered from the ChatDoctor paper (https://arxiv.org/abs/2303.14070) and can be accessed at [https://drive.google.com/file/d/1lyfqIwlLSClhgrCutWuEe_IACNq6XNUt/view?usp=sharing](https://drive.google.com/file/d/1lyfqIwlLSClhgrCutWuEe_IACNq6XNUt/view?usp=sharing). The dataset was translated using the HealthCareMagic-100k_Parser.py file, which makes use of the Large_dataset_translator repository located at [https://github.com/vTuanpham/Large_dataset_translator](https://github.com/vTuanpham/Large_dataset_translator).

## Dataset Description

The HealthCareMagic-100k dataset is a collection of medical questions and answers. It is a valuable resource for natural language processing tasks in the healthcare domain. The Dutch version of the dataset provides a translated version of the original dataset, allowing researchers and developers to work with the data in the Dutch language.

## Instructions to Generate Translated Version

To generate your own translated version of the HealthCareMagic-100k dataset, follow these steps:

1. Run the following code:
    ```
    git clone https://github.com/vTuanpham/Large_dataset_translator.git
 
    cd Large_dataset_translator

    # setup virtual env
    virtualenv trans-env

    # Activate virtual env
    source trans-env/bin/activate

    # Install package into virtual env
    pip install -r requirements.txt
    ```

2. Place the HealthCareMagic-100k_Parser.py file in the Large_dataset_translator directory.

3. Run the HealthCareMagic-100k_Parser.py file, specifying the path to the original dataset and the desired output path for the translated dataset.

5. The script will process the original dataset and generate the translated version in the specified output path.

## License

The HealthCareMagic-100k dataset and the translation script are provided under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).
