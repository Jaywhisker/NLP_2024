# NLP Final Project 2024
This is the readme folder for 50.040 Natural Language Processing Final Project. <br>
The memebers in this team are: Cheng Wei Xuan (1006040), Onn Hui Ning (1006132), Koh Jin Rui Vinny (1006036)

In this folder, you can retrieve the following:
1. Solution for final_project.ipynb as `final_project_notebook.pdf`
2. Final Design Challenge report as `final_report.pdf`
3. Source code (.py files) in `src/`
4. Output files (RNN.csv, CNN.csv, Roberta.csv) in `results/`
5. Design Challenge model (Best_Roberta.pt) in `models/`

The team reports that our best model on the released test set is a fine-tuned Roberta Model, with the following results
```
Accuracy: 0.9673
Recall: 0.9703718532605976
Precision: 0.964386968415817
F1 Score: 0.9673701541685377
```

The following sections covers how to run the python files and the overall file structure of the folder.

# Initialisation
Before running any code, ensure you have all the required packages by running the following
```
pip install -r requirements.txt
```

# Training
Python files to finetune a Roberta-base model from hugging face is provided in `src/train.py`.

The following arguments are provided for your training, with the default values being the arguments that we used to finetune our best model.
```
--train_data_type (str): Determines Train data type, accepts only folder or csv (folder is for aclImdb folder dataset, csv is for released test data). Defaults to folder.
--test_data_type (str): Determines Test data type, accepts only folder or csv (folder is for aclimdb folder dataset, csv is for released test data). Defaults to folder. 
--train_data_path (str): Train data path, folder directory or csv filepath. Defaults to ./data/aclImdb
--train_data_path (str): Test data path, folder directory or csv filepath. Defaults to ./data/aclImdb
--max_length (int): Maximum token length in each input for tokenizer. Defaults to 256.
--train_batch_size (int): Train dataloader batch size. Defaults to 8.
--test_batch_size (int): Test dataloader batch size. Defaults to 8.
--dropout (float): Dropout probability hyperparameter for Roberta. Defaults to 0.4
--unfreeze_roberta (bool): Hyperparameter to determine if we are unfreezing roberta. Defaults to True.
--num_epochs (int): Number of training epochs. Defaults to 5.
--learning_rate (float): Learning rate hyperparameter for Roberta. Defaults to 1e-5
--show_training_graph (bool): Boolean to show training graph of loss and accuracy after training. Defaults to True.
--best_model_path (str): Filepath to save best model (highest test accuracy). Defaults to ./models/Best_Roberta.pt
--final_model_path (str): Filepath to save final model (after last epoch). Defaults to ./models/Final_Roberta.pt
```
To finetune Roberta with default parameters, run the following in your terminal (Ensure you are in the ROOT of this folder):
```
python -m src.train 
```

# Evaluation
Python files to evaluate your Roberta-base model is provided in `src/eval.py`.

The following arguments are provided for your training, with the default values being the arguments that we used.
```
--test_data_type (str): Determines Test data type, accepts only folder or csv (folder is for aclimdb folder dataset, csv is for released test data). Defaults to csv for new dataset. 
--train_data_path (str): Test data path, folder directory or csv filepath.Defaults to ./data/test_data_movie.csv
--max_length (int): Maximum token length in each input for tokenizer. Defaults to 256.
--test_batch_size (int): Test dataloader batch size. Defaults to 8.
--model_path (str): Filepath to model for evaluation. Defaults to ./models/Best_Roberta.pt
--download_results (bool): Boolean to download model predictions on test dataset. Defaults to True.
--download_csv_filepath (str): Filepath to save model predictions. Defaults to ./results/Roberta.csv
```
To evaluate your Roberta model with default parameters, run the following in your terminal (Ensure you are in the ROOT of this folder):
```
python -m src.eval 
```

# Folder Structure
Other than the python files, we left our other exploration notebooks in the notebook folder. Explorations detailed in the appendix of the final_report can be found there.

```
├── data                           <- base folder containing all the dataset
│   ├── aclImdb                    <- folder containing aclImdb train and test data 
│   │
│   └── test_data_movie.csv        <- released test dataset
|
├── models                         <- base folder containing final model
│   └── Best_Roberta.pt            <- best model for design challenge
|
├── notebooks                      <- base folder containing all notebooks
│   ├── dc_xx.ipynb                <- python notebooks containing the different explorations done for design challenge
│   │
│   └── final_project.ipynb        <- python notebook for final project (that is converted to pdf)
│
├── results
│   ├── RNN.csv                    <- Prediction Output of RNN
│   │
│   ├── CNN.csv                    <- Prediction Output of CNN
│   │                     
│   └── Roberta.csv                <- Prediction Output of best Roberta for design challenge
|
├── src                            <- base folder containing all the main code
│   ├── utils                      <- folder containing utils file
|   |   ├── dataloader.py          <- python file to load dataset into dataloader for training / eval
│   │   |
|   |   └── model.py               <- python file containing roberta class
|   |
│   ├── train.py                   <- python file to finetune roberta
│   │
│   └── eval.py                    <- python file to evaluate roberta
│   │
│   └── main.py                    <- main python file that consist of all the endpoints for the fastAPI
|
├── final_project_notebook.pdf     <- final_project.ipynb in pdf format (notebook is in notebooks/)
|
├── final_report.pdf               <- final report
|                            
└── requirements.txt               <- text file containing required python packages

```