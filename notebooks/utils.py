# basic libraries
import os
import time
import math
import pickle
import random
import numpy as np
import pandas as pd
import re
import string
from collections import Counter
from tqdm import tqdm

# environment and Azure OpenAI
from dotenv import load_dotenv
from openai import AzureOpenAI

# visualization
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# NLP and text preprocessing
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
import ftfy  # Fix encoding issues
import emoji
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

# feature engineering
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from keras_preprocessing.sequence import pad_sequences
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE


# embeddings
import gensim.downloader
from gensim.models import Word2Vec
from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch

# models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# deep learning
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Input, LSTM, Dense, TimeDistributed, Bidirectional, Masking

# datasets from Hugging Face
from datasets import Dataset, DatasetDict


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


# Function to convert tokens into mean embedding vector
def tweet_to_vec(tokens, model, size):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(size)

#get word embeddings from documents
def corpus2vec(corpus, glove_model):
    index_set = set(glove_model.index_to_key)  # fast lookup
    word_vec = glove_model.get_vector          # alias for speed

    return [
        [word_vec(word) for word in word_tokenize(doc) if word in index_set]
        for doc in tqdm(corpus)
    ]

def average_tweet_vectors(corpus_vectors, vector_size):
    return np.array([
        np.mean(tweet, axis=0) if len(tweet) > 0 else np.zeros(vector_size)
        for tweet in corpus_vectors
    ])

# embeddings
def embedding_bow(x_train, y_train, x_val, model, ngram_range=(1,1), oversampling_function=None):
    bow = CountVectorizer(binary=True, ngram_range=ngram_range) # each term is marked as present or not per document - good for short text
    X_bow = bow.fit_transform(x_train['text'])

    if oversampling_function:
        X_bow_resampled, y_train_resampled = oversampling_function(X_bow.toarray(), y_train)
    else:
        X_bow_resampled, y_train_resampled = X_bow, y_train

    model.fit(X_bow_resampled, y_train_resampled)

    y_train_pred = model.predict(X_bow.toarray())
    y_val_pred = model.predict(bow.transform(x_val['text']))

    return X_bow, y_train_pred, y_val_pred, bow

def embedding_tfidf(x_train, y_train, x_val, model, max_df, ngram_range=(1,1), oversampling_function=None):
    tfidf = TfidfVectorizer(max_df=max_df, ngram_range=ngram_range) 
    X_tfidf = tfidf.fit_transform(x_train['text']).toarray()

    if oversampling_function:
        X_tfidf_resampled, y_train_resampled = oversampling_function(X_tfidf, y_train)
    else:
        X_tfidf_resampled, y_train_resampled = X_tfidf, y_train

    model.fit(X_tfidf_resampled,y_train_resampled)

    y_train_pred = model.predict(tfidf.transform(x_train['text']))
    y_val_pred = model.predict(tfidf.transform(x_val['text']))

    return X_tfidf, y_train_pred, y_val_pred, tfidf

def embedding_word2vec(x_train, y_train, x_val, window, min_count, model, vector_size=None, sg=1, oversampling_function=None):
    tokenized_train = [word_tokenize(tweet.lower()) for tweet in x_train['text']]
    tokenized_val = [word_tokenize(tweet.lower()) for tweet in x_val['text']]

    corpus = x_train['text']

    #get list with lenghts of sentences
    train_len = []
    for i in corpus:
        train_len.append(len(i))

    if vector_size == None:
        word2vec = Word2Vec(
            sentences=tokenized_train,
            vector_size=max(train_len),    # size of the embedding vectors 
            window=window,        
            min_count=min_count,     
            sg=sg            
        )
    else:
        word2vec = Word2Vec(
            sentences=tokenized_train,
            vector_size=vector_size,   
            window=window,        
            min_count=min_count,     
            sg=sg            
        )

    # Apply to train and validation sets
    X_train_vec = np.array([tweet_to_vec(tokens, word2vec, word2vec.vector_size) for tokens in tokenized_train])
    X_val_vec = np.array([tweet_to_vec(tokens, word2vec, word2vec.vector_size) for tokens in tokenized_val])

    if oversampling_function:
        X_train_vec, y_train = oversampling_function(X_train_vec, y_train)

    model.fit(X_train_vec,y_train)

    y_train_pred = model.predict(X_train_vec)
    y_val_pred = model.predict(X_val_vec)

    return X_train_vec, y_train_pred, y_val_pred

def embedding_glove(x_train, y_train, x_val, model_glove, emb_size, model, oversampling_function=None):
    x_train = corpus2vec(x_train['text'], model_glove)
    x_val = corpus2vec(x_val['text'], model_glove)

    X_train_avg = average_tweet_vectors(x_train, emb_size)
    X_val_avg = average_tweet_vectors(x_val, emb_size)

    if oversampling_function:
        X_train_avg, y_train = oversampling_function(X_train_avg, y_train)


    model.fit(X_train_avg, y_train)

    y_train_pred = model.predict(X_train_avg)
    y_val_pred = model.predict(X_val_avg)

    return X_train_avg, y_train_pred, y_val_pred

def embedding_te3s(texts, cache_file, client, model, delay=1.0, batch_size=32, force_reload=False):
    # Check if the cache file exists and if we should force reload
    if not force_reload and os.path.exists(cache_file):
        print(f"Loading embeddings from {cache_file}...")
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    
    # If not, generate embeddings
    print(f"Generating embeddings in batches and saving to {cache_file}...")
    embeddings = []
    num_batches = math.ceil(len(texts) / batch_size)
    
    for i in tqdm(range(num_batches)):
        # Get the current batch of texts
        batch_texts = texts[i*batch_size : (i+1)*batch_size]
        try:
            response = client.embeddings.create(
                input=batch_texts,
                model=model
            )
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
        except Exception as e:
            print(f"Error on batch {i}: {e}")
            # Append zero vectors for this batch
            embeddings.extend([[0.0]*1536]*len(batch_texts))
        
        # Respect the rate limit
        time.sleep(delay)
    
    # Save the embeddings to the cache file
    with open(cache_file, "wb") as f:
        pickle.dump(embeddings, f)
    
    return embeddings

def embedding_roberta(texts, cache_file, tokenizer, model, batch_size=32, force_reload=False, max_length=512):
    # Check if the cache file exists and if we should force reload
    if not force_reload and os.path.exists(cache_file):
        print(f"Loading RoBERTa embeddings from {cache_file}...")
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    
    # If not, generate embeddings
    print(f"Generating RoBERTa embeddings and saving to {cache_file}...")
    embeddings = []
    num_batches = math.ceil(len(texts) / batch_size)

    for i in tqdm(range(num_batches)):
        # Get the current batch of texts
        batch_texts = texts[i*batch_size:(i+1)*batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        
        # Move inputs to the same device as the model
        with torch.no_grad():
            outputs = model(**inputs)

        # Get the CLS token embeddings
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        embeddings.extend(cls_embeddings.numpy())
    
    # Save the embeddings to the cache file
    with open(cache_file, "wb") as f:
        pickle.dump(embeddings, f)

    return embeddings



from imblearn.over_sampling import BorderlineSMOTE

def oversample(X, y):
    return BorderlineSMOTE(random_state=42).fit_resample(X, y)



# Metrics

def compute_metrics(y_true, y_pred):
    """
    Compute F1, precision, recall, and accuracy for given true and predicted labels.
    Parameters:
    - y_true: true labels
    - y_pred: predicted labels
    Returns:
    - A tuple of (F1, precision, recall, accuracy) rounded to 4 decimal places.
    """
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    return round(f1, 4), round(precision, 4), round(recall, 4), round(accuracy, 4)

def get_metrics_df(model_name, y_train, y_train_pred, y_val, y_val_pred):
    """
    Generate a DataFrame with classification metrics for a single model.
    Parameters:
    - model_name: str
    - y_train, y_train_pred, y_val, y_val_pred: arrays/lists of true and predicted labels
    Returns:
    - A one-row DataFrame with evaluation metrics.
    """
    # Compute metrics for train and validation sets
    train_f1, train_prec, train_rec, train_acc = compute_metrics(y_train, y_train_pred)
    val_f1, val_prec, val_rec, val_acc = compute_metrics(y_val, y_val_pred)

    # Create a dictionary with the metrics
    data = {
        "Model": model_name,
        "Train F1 (Macro)": train_f1,
        "Val F1 (Macro)": val_f1,
        "Train Precision": train_prec,
        "Val Precision": val_prec,
        "Train Recall": train_rec,
        "Val Recall": val_rec,
        "Train Accuracy": train_acc,
        "Val Accuracy": val_acc
    }

    return pd.DataFrame([data])


# Plots

def plot_metrics(y_train, y_train_pred, y_val, y_val_pred, title="Model Performance"):
    """
    Plots accuracy, precision, recall, and F1 score for train and validation sets.

    Parameters:
    - y_train: True training labels
    - y_train_pred: Predicted training labels
    - y_val: True validation labels
    - y_val_pred: Predicted validation labels
    - title: Plot title
    """
    # Set Seaborn style
    sns.set(style="whitegrid")

    # Compute metrics
    train_scores = compute_metrics(y_train, y_train_pred)
    val_scores = compute_metrics(y_val, y_val_pred)

    labels = ['F1 Score (Macro)', 'Precision', 'Recall', 'Accuracy']
    x = range(len(labels))
    
    bar_width = 0.35
    plt.figure(figsize=(9, 5))

    # Plot bars
    train_bars = plt.bar(x, train_scores, width=bar_width, label='Train', color='#1f4e79')
    val_bars = plt.bar([i + bar_width for i in x], val_scores, width=bar_width, label='Validation', color='#A9C4E2')

    # Add values on top of bars
    for bars in [train_bars, val_bars]:
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, f'{yval:.4f}', ha='center', va='bottom', fontsize=9)

    # Formatting
    plt.xticks([i + bar_width / 2 for i in x], labels)
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", labels=None, figsize=(6, 5), cmap="YlGnBu"):
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, 
                xticklabels=labels, yticklabels=labels,
                linewidths=0, linecolor='white', cbar=False)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.show()