from transformers import GPT2TokenizerFast, GPT2ForSequenceClassification, RobertaTokenizerFast, RobertaForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import tqdm
import json

# custom imports
from utils import compute_metrics, SymptomDataset

class ModelTrainer:
    def __init__(
            self,
            model_name: str,
            symptom_name: str,
            sample_sizes: list[int],
        ) -> None:
            """Train a classification model on the given dataset len(sample_sizes) times and return the evaluation results for each sample size.
            Uses cuda if available.

            Args:
                train_set (SymptomDataset): A SymptomDataset object containing the training data (should be exactly the same each time it's used).
                test_set (SymptomDataset): A SymptomDataset object containing the test data (should be exactly the same each time it's used).
                model_name (str): The name of the model to train. Should be one of 'gpt2', 'medroberta', or 'robbert'.
                symptom_name (str): The name of the symptom to train the model for. Should be one of 'Koorts', 'Hoesten', or 'Kortademigheid' (for now).
                sample_sizes (list[int]): A list of integers containing the sample sizes to use for training the model.
            """
            self.training_results = {}

            self.model_name = model_name
            self.symptom_name = symptom_name
            self.sample_sizes = sample_sizes

            

            # Initialize tokenizer and model
            if self.model_name == 'gpt2':
                self.tokenizer = GPT2TokenizerFast.from_pretrained("GroNLP/gpt2-small-dutch")
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model = GPT2ForSequenceClassification.from_pretrained("GroNLP/gpt2-small-dutch", num_labels=3)
            elif self.model_name == 'medroberta':
                self.tokenizer = RobertaTokenizerFast.from_pretrained("CLTL/MedRoBERTa.nl")
                self.model = RobertaForSequenceClassification.from_pretrained("CLTL/MedRoBERTa.nl", num_labels=3)
            elif self.model_name == 'robbert':
                self.tokenizer = RobertaTokenizerFast.from_pretrained("pdelobelle/robbert-v2-dutch-base")
                self.model = RobertaForSequenceClassification.from_pretrained("pdelobelle/robbert-v2-dutch-base", num_labels=3)
            else:
                raise ValueError("Invalid model name. Should be one of 'gpt2', 'medroberta', or 'robbert'.")
            
            self.training_args = TrainingArguments(
                output_dir='./results',
                num_train_epochs=3,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=64,
                warmup_steps=500,
                weight_decay=0.01, 
                logging_dir='./logs',
            )
            
            
    
    def preprocess_dataset(
        self,
        train_set: pd.DataFrame,
        test_set: pd.DataFrame,
        train_labels: str,
        test_labels: str,
    ) -> None:
        """Tokenize the dataset and create a SymptomDataset object for training the model.

        Args:
            train_set (pd.DataFrame): _description_
            test_set (pd.DataFrame): _description_
            train_labels (str): _description_
            test_labels (str): _description_
        """
        # Tokenize the dataset
        train_encodings = self.tokenizer(train_set, truncation=True, padding=True)
        test_encodings = self.tokenizer(test_set, truncation=True, padding=True)
            
        self.train_dataset = SymptomDataset(train_encodings, train_labels)
        self.test_dataset = SymptomDataset(test_encodings, test_labels)

        # Check if sample_sizes contains values larger than len(train_set)
        if any([sample_size > len(train_set) for sample_size in self.sample_sizes]):
            raise ValueError(f"Values in sample_sizes should not be larger than the size of the training set. \
                             Please reinstantiate the ModelTrainer object with sample_sizes values <= {len(train_set)}.")

    def limit_dataset_size(
            self,
            sample_size: int,
    ) -> SymptomDataset:
        """Limit the size of the training and test datasets to the given sample size.

        Args:
            sample_size (int): The size to limit the datasets to.
        """
        return self.train_dataset[:sample_size]

                 

    def train_classification_model(
            self,
            train_data: SymptomDataset,
    ) -> dict:
        """Train a classification model on the given dataset len(sample_sizes) times and return the evaluation results for each sample size.
        Uses cuda if available.

        Args:
            train_data (SymptomDataset): A SymptomDataset object containing the training data.
        Returns:
            dict: A dictionary containing the evaluation results for each sample size.

        TODO: Implement k-fold cross-validation.
        """
        # Move the model and datasets to the GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            self.model.to(device)
        else:
            print("No GPU found, using CPU for training.")
        
        # Fine-tune the model on the dataset
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_data,
            eval_dataset=self.test_dataset,
            compute_metrics=compute_metrics,
        )

        trainer.train()

        # Evaluate the model
        eval_results = trainer.evaluate()

        return eval_results
    
    def train_classification(self, use_crossval: bool=False) -> dict:
        """Train a classification model on the given dataset len(sample_sizes) times and return the evaluation results for each sample size.

        Args:
            use_crossval (bool, optional): Whether or not to use 5-fold cross-validation. As the test fold is always kept the same, sample sizes should be divisible by 4 when applying cross-validation. Defaults to False.

        Returns:
            dict: _description_
        """
        if torch.cuda.is_available():
            print(f"GPU found, using {torch.cuda.get_device_name()} for training.")
        
        # Check if sample_sizes are divisible by 4 when using cross-validation
        if use_crossval and any([sample_size % 4 != 0 for sample_size in self.sample_sizes]):
            raise ValueError("Sample sizes should be divisible by 4 when using cross-validation.")
        
        for sample_size in self.sample_sizes:
            print(f"Training model for sample size {sample_size}...")
            training_set = self.limit_dataset_size(sample_size)
            if use_crossval:
                # Implement k-fold cross-validation
                training_subset_results = []
                for i in range(4):
                    training_subset = training_set[i::4]
                    training_subset_results.append(self.train_classification_model(training_subset))
                self.training_results[sample_size] = training_subset_results
                
            else:
                self.training_results[sample_size] = self.train_classification_model(training_set)
        
        return self.training_results

        

def main():
    # Settings
    model_type = 'robbert'
    symptom = 'Koorts'
    sample_sizes = [100, 200, 300, 400, 500, 600, 700, 800,]

    # Load the datasets
    train_set = pd.read_csv('data/raw/patient_data/fake_train.csv')
    test_set = pd.read_csv('data/raw/patient_data/fake_test.csv')

    # Prepare the datasets
    train_set['labels'] = train_set[symptom]
    test_set['labels'] = test_set[symptom]
    train_texts = train_set['DEDUCE_omschrijving'].tolist()
    train_labels = train_set['labels'].tolist()
    test_texts = test_set['DEDUCE_omschrijving'].tolist()
    test_labels = test_set['labels'].tolist()

    # Train the model
    trainer = ModelTrainer(model_type, symptom, sample_sizes)
    trainer.preprocess_dataset(train_texts, test_texts, train_labels, test_labels)

    results = trainer.train_classification(use_crossval=True)

    # Save the results to a file
    with open(f'results/{model_type}_{symptom}_results.json', 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()