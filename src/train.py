# Code file for finetuning Roberta
import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

from src.utils.model import RobertaClass
from src.utils.dataloader import createDataLoaders
from src.eval import eval_model

def parse_arguments():
    """
    Function defining all arguments for the data
    """
    parser = argparse.ArgumentParser(description="Training parameters for finetuning roberta.")
    
    # Define the arguments
    parser.add_argument('--train_data_type', type=str, default='folder', help='Train data type, folder or csv. Defaults to folder.')
    parser.add_argument('--test_data_type', type=str, default='folder', help='Train data type, folder or csv. Defaults to folder.')
    parser.add_argument('--train_data_path', type=str, default='./data/aclImdb', help='Train data path, folder directory or csv filepath. Defaults to ./data/aclImdb')
    parser.add_argument('--test_data_path', type=str, default='./data/aclImdb', help='Test data path, folder directory or csv filepath. Defaults to ./data/aclImdb')
    parser.add_argument('--max_length', type=int, default=256, help='Maximum token length in each input. Defaults to 256.')
    parser.add_argument('--train_batch_size', type=int, default=8, help='Train dataloader batch size. Defaults to 8.')
    parser.add_argument('--test_batch_size', type=int, default=8, help='Test dataloader batch size. Defaults to 8.')
    parser.add_argument('--dropout', type=float, default=0.4, help='Dropout probability for Roberta. Defaults to 0.4')
    parser.add_argument('--unfreeze_roberta', type=bool, default=True, help='Determines if we are unfreezing roberta. Defaults to True.')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of training epochs. Defaults to 5.')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate for Roberta. Defaults to 1e-5')
    parser.add_argument('--show_training_graph', type=bool, default=True, help='Boolean to show training graph of loss and accuracy after training. Defaults to True.')
    parser.add_argument('--best_model_path', type=str, default='./models/Best_Roberta.pt', help='Best model path. Defaults to ./models/Best_Roberta.pt')
    parser.add_argument('--final_model_path', type=str, default='./models/Final_Roberta.pt', help='Final model path. Defaults to ./models/Final_Roberta.pt')

    return parser.parse_args()


def main():
    print("Retrieving Arguments")
    args = parse_arguments()
    train_data_type = args.train_data_type
    test_data_type = args.test_data_type
    train_data_path = args.train_data_path
    test_data_path = args.test_data_path
    max_length = args.max_length
    train_batch_size = args.train_batch_size
    test_batch_size = args.test_batch_size
    dropout = args.dropout
    unfreeze_all = args.unfreeze_roberta
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    show_training_graph  = args.show_training_graph
    best_model_path = args.best_model_path
    final_model_path = args.final_model_path

    print("Loading Roberta Tokenizer")
    roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base', truncation=True, do_lower_case=True)

    print("Creating Training and Testing dataloaders, this might take a while...")
    train_iter, test_iter = createDataLoaders(False,
                                              train_data_type,
                                              test_data_type,
                                              train_data_path,
                                              test_data_path,
                                              roberta_tokenizer,
                                              max_length,
                                              train_batch_size,
                                              test_batch_size
                                              )
    print(f"Dataloaders created, creating Roberta model with {dropout} dropout. Unfreezing roberta: {unfreeze_all}")
    roberta_model = RobertaClass(dropout,
                                 unfreeze_all)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initialising training on {device} for {num_epochs} epochs")
    
    criterion = torch.nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(params =  roberta_model.parameters(), lr=learning_rate)
    
    train_model(roberta_model,
                train_iter,
                test_iter,
                criterion,
                optimiser,
                num_epochs,
                device,
                show_training_graph,
                best_model_path,
                final_model_path)

    print(f"Training completed. Best model saved to {best_model_path} and final model to {final_model_path}")


def train_model(model:RobertaClass,
                train_iter:DataLoader,
                test_iter:DataLoader,
                criterion,
                optimiser,
                num_epochs:int,
                device:torch.device,
                show_training_graph:bool,
                best_model_path:str,
                final_model_path:str):

    """
    Function to train and evaluate model
    The model will train on the train_iter, be evaluated on test_iter
    Best model with highest accuracy on test_iter will be saved in best_model_path
    Final model after last epoch will be saved in final_model_path

    Args:
        model (RobertaClass): model to be trained
        train_iter (DataLoader): train dataset
        test_iter (DataLoader): test dataset
        criterion: criterion (loss) function
        optimiser: optimiser
        num_epochs (int): Number of epochs to train
        device (torch.device): cuda or cpu
        show_training_graph (bool): boolean to show the training loss, training acc and test acc graph across epochs
        best_model_path (str): path to best model
        final_model_path (str): path to model aft final epoch training
    """
    # Initalisation of variables
    train_loss_container = []
    train_acc_container = []
    test_acc_container = []

    best_test_acc = 0
    
    for epoch in range(num_epochs):
        total_train_loss = 0
        total_train_correct = 0
        total_train_samples = 0

        # Setup training
        model.train()
        model.to(device)

        for idx, data in tqdm(enumerate(train_iter), total=len(train_iter)):
            # Convert inputs to device
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.long)
            
            # Get predictions and loss
            outputs = model(ids, mask, token_type_ids)
            loss = criterion(outputs, targets)
            total_train_loss += loss.item()

            # Train
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            # Count accuracy
            predicted = torch.argmax(outputs, dim=-1)
            total_train_correct += (predicted == targets).sum().item()
            total_train_samples += targets.shape[0]

        # Evaluation for test
        test_acc, _, _, _ = eval_model(model, test_iter, device)

        train_loss_container.append(total_train_loss/len(train_iter)) #total num of batches
        train_acc_container.append(total_train_correct/total_train_samples)
        test_acc_container.append(test_acc)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model, best_model_path)
            print(f"Best Model Saved at epoch {epoch+1}")

        print(f"Epoch {epoch+1} completed, train_loss: {total_train_loss/len(train_iter)}, train_acc: {total_train_correct/total_train_samples}, test_acc: {test_acc}")

    # Save Final model
    torch.save(model, final_model_path)

    if show_training_graph:
        # Graph with matplotlib
        # Plot each list
        plt.plot(train_loss_container, label='Train Loss', color='blue')
        plt.plot(train_acc_container, label='Train Acc', color='green')
        plt.plot(test_acc_container, label='Test Acc', color='red')

        # Add labels and legend
        plt.xlabel("Epoch")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()