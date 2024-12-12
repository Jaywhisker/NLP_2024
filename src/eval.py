# Code file for evaluiating finetuned Roberta
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import RobertaTokenizer
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import argparse

from src.utils.model import RobertaClass
from src.utils.dataloader import createDataLoaders

def parse_arguments():
    """
    Function defining all arguments for the data
    """
    parser = argparse.ArgumentParser(description="Training parameters for finetuning roberta.")
    
    # Define the arguments
    parser.add_argument('--test_data_type', type=str, default='csv', help='Train data type, folder or csv. Defaults to csv.')
    parser.add_argument('--test_data_path', type=str, default='./data/test_data_movie.csv', help='Test data path, folder directory or csv filepath. Defaults to ./data/test_data_movie.csv')
    parser.add_argument('--max_length', type=int, default=256, help='Maximum token length in each input. Defaults to 256.')
    parser.add_argument('--test_batch_size', type=int, default=8, help='Test dataloader batch size. Defaults to 8.')
    parser.add_argument('--model_path', type=str, default='./models/Best_Roberta.pt', help='Path to model for evaluation. Defaults to ./models/Best_Roberta.pt')
    parser.add_argument('--download_results', type=bool, default=True, help='Boolean to download model output. Defaults to True.')
    parser.add_argument('--download_csv_filepath', type=str, default='./results/Roberta.csv', help='CSV filepath to model outputs. Defaults to ./results/Roberta.csv')

    return parser.parse_args()


def main():
    print("Retrieving Arguments")
    args = parse_arguments()
    test_data_type = args.test_data_type
    test_data_path = args.test_data_path
    max_length = args.max_length
    test_batch_size = args.test_batch_size
    model_path = args.model_path
    download_results = args.download_results
    result_path = args.download_csv_filepath

    print("Loading Roberta Tokenizer")
    roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base', truncation=True, do_lower_case=True)
    
    print("Creating Testing dataloaders, this might take a while...")
    _, test_iter = createDataLoaders(True,
                                              None,
                                              test_data_type,
                                              None,
                                              test_data_path,
                                              roberta_tokenizer,
                                              max_length,
                                              None,
                                              test_batch_size
                                              )

    print(f"Dataloaders created, loading model from {model_path}")
    roberta_model = torch.load(model_path, weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Initialising evluation on {device}")

    accuracy, recall, precision, f1 = eval_model(roberta_model,
                                                 test_iter,
                                                 device,
                                                 True,
                                                 download_results,
                                                 result_path)
    
    print(f"Evaluation completed, {'results downloaded to ' + result_path if download_results else 'no results downloaded.'}")
    print(f"Accuracy: {accuracy}")
    print(f"Recall: {recall}")
    print(f"Precision: {precision}")
    print(f"F1 Score: {f1}")


def eval_model(model:RobertaClass,
               test_iter:DataLoader,
               device:torch.device,
               calc_confusion_matrix:bool=False,
               download_results:bool=False,
               result_path:str=None):

    """
    Function to evaluate model, returning the Accuracy, Precision, Recall and F1 score

    Args:
        model (RobertaClass): Instance of model
        test_iter (DataLoader): test dataloader
        device (torch.device): cpu or cuda
        calc_confusion_matrix (bool, Optional): Boolean if Precision, Recall and F1 score should also be calculated. Defaults to False.
        download_results (bool, Optional): Boolean to save predictions into a csv file. Defaults to False.
        result_path (str, Optional): CSV filepath for results. Defaults to None.

    Returns:
        accuracy, recall (None if calc_confusion_matrix == False), precision (None if calc_confusion_matrix == False), f1 (None if calc_confusion_matrix == False)

    """
    if download_results and result_path == None and '.csv' not in result_path:
        raise Exception("Invalid result filepath")

    prediction_list = []
    actual_list = []
    recall, precision, f1 = None, None, None
    # For downloading results
    sentence_list = []
    predicted_labels = []
    actual_labels = []
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base', truncation=True, do_lower_case=True)
    
    # Setup model eval to prevent training
    model.eval()
    with torch.no_grad():
        for idx, data in tqdm(enumerate(test_iter), total=len(test_iter)):
            # Append actual scores
            actual_list.append(data['targets'].numpy())
            
            # Convert to device an retrieve prediction
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            outputs = model(ids, mask, token_type_ids)
            
            # Append predicted scores
            predicted = torch.argmax(outputs, dim=-1)
            prediction_list.append(predicted.cpu().numpy())
            
            # Append results
            if download_results:
                sentence_list.extend(tokenizer.batch_decode(data['ids'], skip_special_tokens=True))
                # Labels
                predicted_labels.extend(["negative" if pred == 0 else "positive" for pred in predicted.cpu().numpy()])
                actual_labels.extend(["negative" if pred == 0 else "positive" for pred in data['targets'].cpu().numpy()])

    # Flatten the lists into arrays
    actual_list = np.concatenate(actual_list)
    prediction_list = np.concatenate(prediction_list)

    # Calculate accuracy
    accuracy = accuracy_score(actual_list, prediction_list)

    # Calculate other metrics
    if calc_confusion_matrix:
        recall = recall_score(actual_list, prediction_list)
        precision = precision_score(actual_list, prediction_list)
        f1 = f1_score(actual_list, prediction_list)
        
    # Download results
    if download_results:
        data = pd.DataFrame({
            'Sentence': sentence_list,
            'Actual': actual_labels,
            "Predicted": predicted_labels
        })

        data.to_csv(result_path, index=False)

    return accuracy, recall, precision, f1

if __name__ == "__main__":
    main()