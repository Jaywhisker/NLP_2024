# This is the python file containing the Dataset Class and Dataset Loader
# This is a modification of both ImdbData and TestIMDBDataset in notebooks/dc_roberta.ipynb
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer
import pandas as pd
import os

# Dataset Class
class customDataset(Dataset):
    def __init__(self,
                data_type:str,
                data_path:str,
                is_train:bool,
                tokenizer:RobertaTokenizer,
                max_len:int):
        """
        Initializes an instance of the dataset class. 
        Initializes the dataset by reading and tokenizing the data from the specified directory.

        Args:
        data_type (str): Determines the data type, either an folder (IMDB) or a csv file (released test data). Accepts either "folder" or "csv" as input.
        data_path (str): The directory where the dataset is stored if folder, else the csv filepath. It is used to load the data (either training or testing).
        is_train (bool): A flag indicating whether the dataset is for training (`True`) or testing (`False`).
        tokenizer (RobertaTokenizer): The tokenizer to process text data for the RoBERTa model, used to encode and convert raw text into tokens.
        max_len (int): The maximum sequence length after tokenization. Text will be padded or truncated to this length.
        """

        self.tokenizer = tokenizer

        if data_type.lower() == 'folder':
            self.data = self._read_imdb(data_path, is_train)
            self.text = self.data.review
            self.targets = self.data.label
        elif data_type.lower() == 'csv':
            self.text, self.targets = self._read_csv(data_path)
        else:
            raise Exception("Invalid data_type provided. Only folder or csv is accepted")
        
        self.max_len = max_len

    def _read_imdb(self,data_dir, is_train):
        data = []
        labels = []
        data_folder = 'train' if is_train else 'test'

        for label, label_folder in enumerate(['neg', 'pos']):
            # Retrieve full path
            full_path = os.path.join(data_dir, data_folder, label_folder)
            for text_file in os.listdir(full_path):
                # Read text
                with open(os.path.join(full_path, text_file), 'r', encoding='utf-8') as f:
                    # Add text and label
                    data.append(f.read())
                    labels.append(label)
        df=pd.DataFrame({'review': data, 'label': labels})
        df.head()
        return df
    
    def _read_csv(self, csv_path):
        dataset = pd.read_csv(csv_path)
        data = dataset['text']
        labels = dataset['label']
        return data, labels

    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }
    
# Dataloader Creator Function
def createDataLoaders(only_test:bool,
                      train_data_type:str,
                      test_data_type:str,
                      train_data_path:str,
                      test_data_path:str,
                      tokenizer:RobertaTokenizer,
                      max_len:int,
                      train_batch_size:int,
                      test_batch_size:int):
    """
    Creates PyTorch DataLoaders for the training and testing datasets.

        Args:
            only_test (bool): Only runs test dataloader (for evaluation)
            train_data_type (str): Determines the data type, either an folder (IMDB) or a csv file (released test data) for train data
            test_data_type (str): Determines the data type, either an folder (IMDB) or a csv file (released test data) for test data
            train_data_path (str): The directory where the train dataset is stored if folder, else the csv filepath.
            test_data_path (str): The directory where the test dataset is stored if folder, else the csv filepath.
            tokenizer (RobertaTokenizer): The tokenizer to process text data for the RoBERTa model, used to tokenize and encode the text.
            max_len (int): The maximum length for the tokenized sequences. Texts will be padded or truncated to this length.
            train_batch_size (int): The batch size for the training dataset.
            test_batch_size (int): The batch size for the testing dataset.

        Returns:
            train_iter (DataLoader): A DataLoader for the training dataset, providing batches of data.
            test_iter (DataLoader): A DataLoader for the testing dataset, providing batches of data.

        """
    # Initialisation
    train_iter = None
    
    testing_set = customDataset(data_type=test_data_type,
                                 data_path=test_data_path,
                                 is_train=False,
                                 tokenizer=tokenizer,
                                 max_len=max_len
                                 )
    
    test_params = {'batch_size': test_batch_size,
                    'shuffle': False,
                    'num_workers': 0
                    }
    
    test_iter = DataLoader(testing_set, **test_params)

    if not only_test:
        training_set = customDataset(data_type=train_data_type,
                                    data_path=train_data_path,
                                    is_train=True,
                                    tokenizer=tokenizer,
                                    max_len=max_len
                                    )
    
        train_params = {'batch_size': train_batch_size,
                        'shuffle': True,
                        'num_workers': 0
                        }

        train_iter = DataLoader(training_set, **train_params)
    
    return train_iter, test_iter

if __name__ == "__main__":
    pass