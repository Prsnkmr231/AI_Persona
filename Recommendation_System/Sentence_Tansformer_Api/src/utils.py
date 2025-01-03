import pandas as pd
import os
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
import re

def load_dataframe(path):
    """Load a CSV file into a DataFrame."""
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        raise FileNotFoundError(f"File path does not exist: {path}")

def clean_text(text):
    """Clean and preprocess text."""
    stop_words = set(stopwords.words('english'))
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

def load_model(model_name):
    """Load a pre-trained SentenceTransformer model."""
    return SentenceTransformer(model_name)
