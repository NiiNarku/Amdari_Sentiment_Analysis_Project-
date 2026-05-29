import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import logging
import torch 
import os 
from config.constant import input_data, model_name, train_data, test_data
from src.data_ingestion import data_ingestion
from src.data_cleaning import clean_data


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
)


class data_processor:
    def __init__(self):
        # Reading the dataset using the the data ingestion function from data_ingestion.py
        self.data = data_ingestion()
        # Cleaning the dataset using the clean_data function from data_cleaning.py
        self.clean_data = clean_data(self.data)

    def split_data(self):
        try:
            # Splitting the cleaned dataset into features and labels
            x = self.clean_data['final_text'].astype(str)
            y = self.clean_data['label']
            # Using train_test_split to split the data into training and testing sets
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=0.2, random_state=42
    
            )

            print(x_train.head())
            print(x_test.head())
            logging.info("Data successfully split into training and testing sets.")
            return x_train, x_test, y_train, y_test
        except Exception as e:
            logging.error(f"Error occurred while splitting the dataset: {e}")



class Tokenizerwrapper:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    

    def encode(self, text):
        return self.tokenizer(text.tolist(), truncation=True, padding=True, max_length=128)
    

class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings

        # Ensuring labels are in a list to avoid any pandas index issues
        if hasattr(labels, 'tolist'):
            self.labels = labels.tolist()
        elif hasattr(labels, "__iter__") and not isinstance(labels, (list, tuple)):
            self.labels = list(labels)
        else:
            self.labels = labels


    def __len__(self):
      return len(self.labels)

    def __getitem__(self, idx):
      item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
      item['labels'] = torch.tensor(self.labels[idx])
      return item


def Prepare_sentiment_data():
    try:
        processor = data_processor()
        x_train, x_test, y_train, y_test = processor.split_data()
        tokenizer = Tokenizerwrapper()
        train_encodings = tokenizer.encode(x_train)
        test_encodings = tokenizer.encode(x_test)
        train_dataset = SentimentDataset(train_encodings, y_train)
        test_dataset = SentimentDataset(test_encodings, y_test)
        os.makedirs(os.path.dirname(train_data), exist_ok=True)
        os.makedirs(os.path.dirname(test_data), exist_ok=True)
        torch.save(train_dataset, train_data)
        torch.save(test_dataset, test_data)
        logging.info("Data successfully prepared and saved for model training.")
        return train_dataset, test_dataset
    except Exception as e:
        logging.error(f"Error occurred while preparing the dataset: {e}")









       
       
       
       
       
       
       
       
       
       
       
       
