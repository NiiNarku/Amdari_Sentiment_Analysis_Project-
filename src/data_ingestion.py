import pandas as pd
import numpy as np
import logging 
import re
from pathlib import Path
from config.constant import input_data

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
)



def data_ingestion():
    try:
        data = pd.read_csv(input_data)
        logging.info(f"Data successfully loaded...")
        print(data.head())
        return data
    except Exception as e:
        logging.error(f"Error occurred while loading the dataset{e}")

     

data_ingestion()
    
        
