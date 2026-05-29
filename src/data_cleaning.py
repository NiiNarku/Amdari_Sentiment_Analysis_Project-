import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from spacy import tokens
from src.data_ingestion import data_ingestion
import logging
import sys
from config.constant import Clean_data



logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

sentiment_data = data_ingestion()

class DataCleaning:
    def __init__(self):
        self._ensure_nltk()

    def _load_nlp(self) -> spacy.language.Language:
        for model in ("en_core_web_sm",  "xx_ent_wiki_sm"):
            try:
                 return spacy.load(model)
            except OSError:
                continue
        nlp_fallback = spacy.blank("xx")
        return nlp_fallback


#  NLTK Stop word and tokenizer installation
    def _ensure_nltk(self) -> None:
        try:
            _ =stopwords.words('english')
        except LookupError:
            nltk.download('stopwords')
        try:
            word_tokenize('test')
        except LookupError:
            nltk.download('punkt')


# Newer NLTK version split tokenizer tables into 'punkt tab'

        try:
            nltk.data.find('tokenizers/punkt_tab/english')
        except LookupError:
            try:
                nltk.download('punkt_tab')
            except Exception:
                pass


    def clean_text(self, text: str) -> str:
        """convert to lowercase, remove url, special characters and white space
        keeps accented letter and regular expressions and return cleaned text
        """
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    


    def lemmatize(self, text: str) -> str:
        """ Reduce words to thir base form using SpaCy's lemmatization."""
        nlp = self._load_nlp()
        doc = nlp(text)
        return " ".join(token.lemma_ if token.lemma_ else token.text for token in doc)
    

    # Remove stop

    def remove_stopwords(self, text: str) -> str:
        """ remove low-signal words
        example: the, in, is, and, but
        """
        tokens = word_tokenize(text)
        sw = set(stopwords.words('english'))
        tokens = [t for t in tokens if t not in sw]
        return " ".join(tokens)
    

def clean_data(data: pd.DataFrame):
    try:
        cleaner = DataCleaning()
        data['clean_text'] = data['review'].apply(cleaner.clean_text)
        data['lemma_text'] = data['clean_text'].apply(cleaner.lemmatize)
        data['final_text'] = data['lemma_text'].apply(cleaner.remove_stopwords)



        data['label'] = data['rating'].apply(lambda r:0 if r in (1,2) else (1 if r == 3 else 2 ))
        data = data[['review','final_text', 'label']]
        data.to_csv(Clean_data)

        print(data.head())
        return data

    except Exception as e:
          logging.error(f"Error occurred during data cleaning: {e}")
 

