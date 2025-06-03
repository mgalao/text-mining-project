import pandas as pd
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from wordcloud import WordCloud
import plotly.express as px
import nltk
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords
import re
from nltk.stem import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import ftfy #to fix encoding issues
import emoji
from tqdm import tqdm
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException


def get_top_words_by_class(df, label_col, text_col, top_criteria=10):
    result = []
    for label in df[label_col].unique():
        words = ' '.join(df[df[label_col] == label][text_col]).lower().split()
        most_common = Counter(words).most_common(top_criteria)
        for word, freq in most_common:
            result.append({'label': label, 'word': word, 'freq': freq})
    return pd.DataFrame(result)

# Function to safely detect language
def detect_language(text):
    try:
        return detect(text)
    except LangDetectException:
        return "error"


def clean_text(text_list, lemmatize=True, stem=False):
    stop = set(stopwords.words('english'))
    stemmer_obj = SnowballStemmer('english')
    lemmatizer = WordNetLemmatizer()
    
    cleaned = []
    
    for text in tqdm(text_list):
        # Fix encoding
        text = ftfy.fix_text(text)
        
        # Remove emojis
        text = emoji.replace_emoji(text, replace='')
        
        # Lowercase
        text = text.lower()

        #Preserve common Financial Indexes
        text = re.sub(r's[\.\&]?\s*p[\.\s]*500', 'sp500', text)
        text = re.sub(r'dow[\s\-]?jones', 'dowjones', text)
        text = re.sub(r'nasdaq[\s\-]?composite', 'nasdaq_composite', text)
        text = re.sub(r'\bnasdaq\b', 'nasdaq', text)
        text = re.sub(r'\bdow\b', 'dow', text)

        # Remove URLs
        text = re.sub(r'http\S+', '', text)

        # Preserve US and UK
        text = re.sub(r'\b(u\.?\s?s\.?)\b', 'USA', text)
        text = re.sub(r'\bus\b', 'USA', text)
        text = re.sub(r'\b(u\.?\s?k\.?)\b', 'UK', text)
        text = re.sub(r'\buk\b', 'UK', text)

        # Replace Q: as question
        text = re.sub(r'\bq\s*:\s*', 'question', text)

        #Preserve Quarter Information
        text = re.sub(r'\b([1-4])q(?:20)?(\d{2})\b', r'quarter_\1 20\2', text)

        # Q1 2020, q2 2021 → quarter_1 2020
        text = re.sub(r'\bq([1-4])\s*20(\d{2})\b', r'quarter_\1 20\2', text)

        # q12020 → quarter_1 2020
        text = re.sub(r'\bq([1-4])(20\d{2})\b', r'quarter_\1 \2', text)

        # 3Q2020 → quarter_3 2020
        text = re.sub(r'\b(\d)q[\-]?(20\d{2})\b', r'quarter_\1 \2', text)

        # 3Q → quarter_3
        text = re.sub(r'\b(\d)q\b', r'quarter_\1', text)

        # Q1, Q2, ... → quarter_1
        text = re.sub(r'\bq([1-4])\b', r'quarter_\1', text)

        # Q/Q → quarter_over_quarter
        text = re.sub(r'\bq\s*/\s*q\b', 'quarter_over_quarter', text)

        # Preserve cashtags
        cashtags = re.findall(r'\$\w+', text)
        text = re.sub(r'\$\w+', ' ', text)  # remove cashtags temporarily

        # Preserve percentages (useful in stock market analysis)
        text = re.sub(r'\+(\d+)%', r'PCT_INCREASE \1percent', text)
        text = re.sub(r'\-(\d+)%', r'PCT_DECREASE \1percent', text)
        text = re.sub(r'(\d+)%', r'percent \1percent', text)

        # Replace monetary values 
        text = re.sub(r'[$€£]\d+', 'costvalue', text)

        # Replace other numeric values
       # text = re.sub(r'\b(?!(?:19|20)\d{2})\d+\b', 'NUMERICVALUE', text)

        # Remove all non-letter characters
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)

        # Add cashtags back
        text += ' ' + ' '.join([tag.replace('$', 'TICKER_') for tag in cashtags])

        # Remove @mentions
        text = re.sub(r'@\w+', '', text)

        # Remove #hashtags but keep the word
        text = re.sub(r'#(\w+)', r'\1', text)

        # Reduce repeated letters (e.g., "soooo good" becomes "soo good")
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)

        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()

        important_terms = {
            'up', 'down', 'rise', 'fall', 'increase', 'decrease', 'drop', 'plunge',
            'crash', 'tank', 'soar', 'rally', 'surge', 'slump', 'gain', 'loss'
        }
        
        # Tokenize and remove stopwords
        tokens = [word for word in text.split() if word not in stop or word in important_terms]

        # Apply lemmatization or stemming
        if lemmatize:
            tokens = [lemmatizer.lemmatize(word) for word in tokens]
        elif stem:
            tokens = [stemmer_obj.stem(word) for word in tokens]

        # Reconstruct and save
        cleaned.append(" ".join(tokens))

    return cleaned

