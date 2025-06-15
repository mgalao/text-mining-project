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

# data splitting
from sklearn.model_selection import train_test_split

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
import contractions
import string

# feature engineering
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from keras_preprocessing.sequence import pad_sequences
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from collections import defaultdict
from sklearn.utils import resample

# embeddings
import gensim.downloader
from gensim.models import Word2Vec
from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding, AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoConfig
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
from tensorflow.keras.layers import Input, LSTM, Dense, TimeDistributed, Bidirectional, Masking, Dropout
from tensorflow.keras.optimizers import Adam
from keras.metrics import Precision, Recall, AUC, TopKCategoricalAccuracy

# datasets from Hugging Face
from datasets import Dataset, DatasetDict

# optuna
from optuna import Trial



# --------------- PRE PROCESSING ---------------

def get_top_words_by_class(df, label_col, text_col, top_criteria=10):
    result = []
    for label in df[label_col].unique():
        words = ' '.join(df[df[label_col] == label][text_col]).lower().split()
        most_common = Counter(words).most_common(top_criteria)
        for word, freq in most_common:
            result.append({'label': label, 'word': word, 'freq': freq})
    return pd.DataFrame(result)

# Function to detect language
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

        # Expand contractions like "can't" -> "can not"
        text = contractions.fix(text)

        important_terms = {
            'up', 'down', 'rise', 'fall', 'increase', 'decrease', 'drop', 'plunge',
            'crash', 'tank', 'soar', 'rally', 'surge', 'slump', 'gain', 'loss', 'not'
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


# --------------- AUX - EMBEDDINGS --------------- 

# Function to convert tokens into mean embedding vector
def tweet_to_vec(tokens, model, size):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(size)

# get word embeddings from documents
def corpus2vec(corpus, glove_model):
    index_set = set(glove_model.index_to_key)  
    word_vec = glove_model.get_vector    

    return [
        [word_vec(word) for word in word_tokenize(doc) if word in index_set]
        for doc in tqdm(corpus)
    ]

def average_tweet_vectors(corpus, glove_model, vector_size):
    index_set = set(glove_model.index_to_key)
    word_vec = glove_model.get_vector

    averaged_vectors = []
    for doc in tqdm(corpus):
        words = word_tokenize(doc)
        vectors = [word_vec(w) for w in words if w in index_set]
        if vectors:
            averaged_vectors.append(np.mean(vectors, axis=0))
        else:
            averaged_vectors.append(np.zeros(vector_size))
    return np.array(averaged_vectors)

def oversample(X, y):
    return BorderlineSMOTE(random_state=42).fit_resample(X, y)

def oversample_data(texts, labels):
    label_to_texts = defaultdict(list)
    for text, label in zip(texts, labels):
        label_to_texts[label].append(text)

    max_len = max(len(texts) for texts in label_to_texts.values())

    oversampled_texts, oversampled_labels = [], []
    for label, texts in label_to_texts.items():
        resampled = resample(texts, replace=True, n_samples=max_len, random_state=42)
        oversampled_texts.extend(resampled)
        oversampled_labels.extend([label] * max_len)

    return oversampled_texts, oversampled_labels





# --------------- EMBEDDINGS --------------- 

# BoW
def embedding_bow(x_train, y_train, x_val, model, max_df, min_df, ngram_range=(1,1), oversampling_function=None):
    bow = CountVectorizer(binary=True, ngram_range=ngram_range, max_df=max_df, min_df=min_df) # each term is marked as present or not per document - good for short text
    X_bow = bow.fit_transform(x_train['text'])

    if oversampling_function:
        X_bow, y_train = oversampling_function(X_bow.toarray(), y_train)

    model.fit(X_bow, y_train)

    y_train_pred = model.predict(bow.transform(x_train['text']))
    y_val_pred = model.predict(bow.transform(x_val['text']))

    return X_bow, y_train_pred, y_val_pred, bow

# TF-IDF
def embedding_tfidf(x_train, y_train, x_val, model, max_df, min_df, ngram_range=(1,1), oversampling_function=None):
    tfidf = TfidfVectorizer(max_df=max_df, min_df=min_df, ngram_range=ngram_range) 
    X_tfidf = tfidf.fit_transform(x_train['text']).toarray()

    if oversampling_function:
        X_tfidf, y_train = oversampling_function(X_tfidf, y_train)
    else:
        X_tfidf, y_train = X_tfidf, y_train

    model.fit(X_tfidf,y_train)

    y_train_pred = model.predict(tfidf.transform(x_train['text']))
    y_val_pred = model.predict(tfidf.transform(x_val['text']))

    return X_tfidf, y_train_pred, y_val_pred, tfidf


# Word2Vec
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
            vector_size=max(train_len),    
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


def embedding_word2vec_lstm(x_train, y_train, x_val, y_val, window, min_count, model_lstm, n_classes=3, batch_size=16, epochs=10, vector_size=None, sg=1, max_seq_len=None, oversampling_function=None):
    tokenized_train = [word_tokenize(tweet.lower()) for tweet in x_train['text']]
    tokenized_val = [word_tokenize(tweet.lower()) for tweet in x_val['text']]

    if max_seq_len is None:
        max_seq_len = max(len(tokens) for tokens in tokenized_train)
    
    if vector_size is None:
        corpus = x_train['text']

        #get list with lenghts of sentences
        train_len = []
        for i in corpus:
            train_len.append(len(i))
        
        vector_size = max(train_len)

    word2vec_model = Word2Vec(
        sentences=tokenized_train,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg
    )

    emb_size = word2vec_model.vector_size

    def sentence_to_seq(tokens):
        return [word2vec_model.wv[word] if word in word2vec_model.wv else np.zeros(emb_size) for word in tokens]

    X_train_seq = [sentence_to_seq(tokens) for tokens in tokenized_train]
    X_val_seq = [sentence_to_seq(tokens) for tokens in tokenized_val]

    X_train_pad = pad_sequences(X_train_seq, maxlen=max_seq_len, dtype='float32', padding='post')
    X_val_pad = pad_sequences(X_val_seq, maxlen=max_seq_len, dtype='float32', padding='post')

    if oversampling_function:
        X_train_pad, y_train = oversampling_function(X_train_pad, y_train)

    y_train_encoded = tf.one_hot(y_train, depth=n_classes)
    y_val_encoded = tf.one_hot(y_val, depth=n_classes)

    model_lstm.fit(X_train_pad, y_train_encoded, batch_size=batch_size, epochs=epochs, validation_data=(X_val_pad, y_val_encoded))

    y_train_pred = np.argmax(model_lstm.predict(X_train_pad), axis=1)
    y_val_pred = np.argmax(model_lstm.predict(X_val_pad), axis=1)

    return X_train_pad, y_train_pred, y_val_pred



# Glove
def embedding_glove(x_train, y_train, x_val, model_glove, emb_size, model, oversampling_function=None):
    X_train_avg = average_tweet_vectors(x_train['text'], model_glove, emb_size)
    X_val_avg = average_tweet_vectors(x_val['text'], model_glove, emb_size)

    if oversampling_function:
        X_train_avg, y_train = oversampling_function(X_train_avg, y_train)

    model.fit(X_train_avg, y_train)

    y_train_pred = model.predict(X_train_avg)
    y_val_pred = model.predict(X_val_avg)

    return X_train_avg, y_train_pred, y_val_pred


def embedding_glove_lstm(x_train, y_train, x_val, y_val, model_glove, emb_size, model_lstm, n_classes=3, batch_size=16, epochs=10, oversampling_function=None, max_seq_len=None):
    X_train_vec = corpus2vec(x_train['text'], model_glove)
    X_val_vec = corpus2vec(x_val['text'], model_glove)

    if max_seq_len is None:
        max_seq_len = max(len(seq) for seq in X_train_vec)

    if emb_size is None:
        corpus = x_train['text']

        #get list with lenghts of sentences
        train_len = []
        for i in corpus:
            train_len.append(len(i))
        
        emb_size = max(train_len)
    
    # pad sequences (shape: n_samples x max_seq_len x emb_size)
    X_train_pad = pad_sequences(X_train_vec, maxlen=max_seq_len, dtype='float32', padding='post')
    X_val_pad   = pad_sequences(X_val_vec, maxlen=max_seq_len, dtype='float32', padding='post')

    if oversampling_function:
        X_train_pad, y_train = oversampling_function(X_train_pad, y_train)

    # one-hot encode targets 
    y_train_encoded = tf.one_hot(y_train, depth=n_classes)
    y_val_encoded = tf.one_hot(y_val, depth=n_classes)

    model_lstm.fit(X_train_pad, y_train_encoded, batch_size=batch_size, epochs=epochs, validation_data=(X_val_pad, y_val_encoded))

    y_train_pred = tf.argmax(model_lstm.predict(X_train_pad), axis=1).numpy()
    y_val_pred = tf.argmax(model_lstm.predict(X_val_pad), axis=1).numpy()


    return X_train_pad, y_train_pred, y_val_pred



# Text Embedding 3 Small
def embedding_te3s(train_texts, train_labels, val_texts, cache_file_train, cache_file_val, client, model_te3s, batch_size, model, force_reload=False, oversampling_function=None):
    X_train = np.array(embedding_te3s_aux(train_texts, cache_file_train, client, model_te3s, batch_size=batch_size, force_reload=force_reload))
    X_val = np.array(embedding_te3s_aux(val_texts, cache_file_val, client, model_te3s, batch_size=batch_size, force_reload=force_reload))

    if oversampling_function:
        X_train, y_train = oversampling_function(X_train, y_train)

    model.fit(X_train, train_labels)
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    return X_train, X_val, y_train_pred, y_val_pred

def embedding_te3s_aux(texts, cache_file, client, model, delay=1.0, batch_size=32, force_reload=False):
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



# Roberta
def embedding_roberta(train_texts, train_labels, val_texts, cache_file_train, cache_file_val, tokenizer_roberta, model_roberta, batch_size, model, force_reload=False, oversampling_function=None):
    # Get embeddings for train and validation sets
    X_train = np.array(embedding_roberta_aux(train_texts, cache_file_train, tokenizer_roberta, model_roberta, batch_size=batch_size, force_reload=force_reload))
    X_val = np.array(embedding_roberta_aux(val_texts, cache_file_val, tokenizer_roberta, model_roberta, batch_size=batch_size, force_reload=force_reload))

    if oversampling_function:
        X_train, y_train = oversampling_function(X_train, y_train)

    model.fit(X_train, train_labels)
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    return X_train, y_train_pred, y_val_pred


def embedding_roberta_aux(texts, cache_file, tokenizer, model, batch_size=32, force_reload=False, max_length=512):
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



# --------------- CLASSIFICATION --------------- 

# Function to classify texts using GPT-4o with few-shot examples
def classify_with_gpt4o_fewshot(texts, label_options, few_shot_examples=None,
                                delay=1.0, client=None, deployment="gpt-4o", batch_size=16):
    
    predictions = []
    # Calculate the number of batches
    num_batches = math.ceil(len(texts) / batch_size)

    # System prompt
    system_prompt = (
        "You are a financial sentiment classification assistant. Your task is to analyze short social media texts (tweets) "
        "that may influence or reflect investor sentiment regarding the stock market. Based on the content, classify each tweet "
        f"into one of the following categories: {', '.join(map(str, label_options))}.\n\n"
        "- Bearish (0): Suggests negative or pessimistic sentiment about the market or a stock.\n"
        "- Bullish (1): Suggests positive or optimistic sentiment.\n"
        "- Neutral (2): Does not express a clear opinion or is irrelevant to market sentiment.\n\n"
        "Respond only with the correct category label — no explanation."
    )

    for i in tqdm(range(num_batches), desc="Classifying with GPT-4o"):
        # Get the current batch of texts
        batch_texts = texts[i*batch_size : (i+1)*batch_size]

        for text in batch_texts:
            messages = [{"role": "system", "content": system_prompt}]

            # Add few-shot examples
            if few_shot_examples:
                for example in few_shot_examples:
                    messages.append({"role": "user", "content": example['text']})
                    messages.append({"role": "assistant", "content": example['label']})

            # Add actual input
            messages.append({"role": "user", "content": text})

            try:
                # Call the Azure OpenAI API
                response = client.chat.completions.create(
                    model=deployment,
                    messages=messages,
                    max_tokens=10,
                    temperature=0
                )
                output = response.choices[0].message.content.strip()
            except Exception as e:
                print(f"Error on input '{text[:30]}...': {e}")
                output = "unknown" # Fallback in case of error

            # Append the output to predictions
            predictions.append(output)
            # Delay to avoid hitting rate limits
            time.sleep(delay)

    return predictions


# Function to cache or run classification with gpt-4o with few-shot examples
def cached_classification_run(filename, texts, label_options, few_shot_examples=None,
                              delay=1.0, client=None, deployment="gpt-4o", force_reload=False, batch_size=16):
    
    # If the cache file exists and force_reload is False, load from cache
    if not force_reload and os.path.exists(filename):
        print(f"Loading cached results from {filename}")
        with open(filename, "rb") as f:
            predictions = pickle.load(f)
    # Otherwise, run classification and save to cache
    else:
        print(f"{'Force reload enabled.' if force_reload else 'No cache found.'} Running classification and saving to {filename}")
        # Get predictions using the classification function
        predictions = classify_with_gpt4o_fewshot(
            texts, label_options, few_shot_examples=few_shot_examples,
            delay=delay, client=client, deployment=deployment, batch_size=batch_size
        )
        # Save predictions to cache
        with open(filename, "wb") as f:
            pickle.dump(predictions, f)

    return predictions





# --------------- METRICS --------------- 

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

def compute_metrics_transformers(pred):
    preds = np.argmax(pred.predictions, axis=1)
    labels = pred.label_ids
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}

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




# --------------- PLOTS --------------- 

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

def plot_best_f1_by_embedding(df):
    df_best = df.sort_values('Val F1 (Macro)', ascending=False).drop_duplicates('Embedding')
    df_best = df_best.sort_values('Val F1 (Macro)', ascending=True)
    df_best['Label'] = df_best['Embedding'] + ' (' + df_best['Model'] + ')'

    labels = df_best['Label']
    train_f1 = df_best['Train F1 (Macro)']
    val_f1 = df_best['Val F1 (Macro)']

    y = np.arange(len(labels))
    bar_width = 0.35

    plt.figure(figsize=(10, 6))
   
    plt.barh(y + bar_width, train_f1, height=bar_width, label='Train', color='#1f4e79')
    plt.barh(y, val_f1, height=bar_width, label='Val', color='#A9C4E2')

    for i in range(len(labels)):
        plt.text(val_f1.iloc[i] + 0.01, y[i], f'{val_f1.iloc[i]:.2f}', va='center', fontsize=8)
        plt.text(train_f1.iloc[i] + 0.01, y[i] + bar_width, f'{train_f1.iloc[i]:.2f}', va='center', fontsize=8)

    plt.yticks(y + bar_width / 2, labels, fontsize=9)
    plt.xlabel('F1 Macro Score', fontsize=11)
    plt.title('Best Train and Validation F1 per Embedding', fontsize=13)
    plt.xlim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_model_comparison_all_embeddings(df):
    unique_embeddings = df['Embedding'].unique()
    n = len(unique_embeddings)
    cols = 2
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    axes = axes.flatten()

    for i, emb in enumerate(sorted(unique_embeddings)):
        df_sub = df[df['Embedding'] == emb].sort_values('Val F1 (Macro)', ascending=True)
        ax = axes[i]
        bars = ax.barh(df_sub['Model'], df_sub['Val F1 (Macro)'], color='#1f4e79')

        for bar, val in zip(bars, df_sub['Val F1 (Macro)']):
            ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{val:.2f}', va='center', fontsize=8)

        ax.set_title(f'{emb}', fontsize=11)
        ax.set_xlim(0, 1.05)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle('Model Comparison by Embedding (F1 Macro - Validation)', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

