# Text Mining: Project

## Team Members
- Bruna Simões  
- Catarina Ribeirinha  
- Marco Galão  
- Margarida Cardoso  

## Project Overview
This repository contains materials for our Text Mining project, which focuses on predicting market sentiment based on tweets, classifying each tweet as `Bearish (0)`, `Bullish (1)`, or `Neutral (2)`. The aim is to capture the sentiment dynamics influencing market behavior, leveraging NLP techniques learned throughout the cours

## Repository Structure

### `data/`
Contains the datasets provided by the professors:
- `train.csv`: Training data with labeled tweets.
- `test.csv`: Test data for evaluation.

### `notebooks/`
Jupyter notebooks documenting the complete analysis pipeline:
- `1_EDA_&_Pre-Processing.ipynb`  
  Exploratory data analysis and text preprocessing.

- `2_Feature_Engineering_&_Baseline_Models.ipynb`  
  Feature engineering and baseline classification models implementation and evaluation.

- `3_Classification_Transformers.ipynb`  
  Implementation and evaluation of transformer-based classifiers.

- `4_Hyperparameter_Tuning_&_Final_Predictions.ipynb`  
  Hyperparameter tuning of the best baseline models and generation of final predictions.

- `utils.py`  
  Shared helper functions and library imports used throughout the notebooks.

- `train_val_split.pkl`  
  Pickle file with train-validation split **including preprocessing**, to ensure consistent data partitions.

- `train_val_split_no_preproc.pkl`  
  Pickle file with train-validation split **without preprocessing**, used for transformer models that process raw text internally.

- `metrics_df.csv`  
  Aggregated classification metrics (macro F1-score, Precision, Recall, Accuracy) for all non-transformer baseline models on both training and validation sets. This file supports comparative model evaluation and selection of top performing approaches.

### Root Files
- `project_description.pdf` — Official project handout. 
- `requirements.txt` — List of Python dependencies for this project.
- `.gitignore` — Specifies files and folders excluded from version control.
- `README.md` — This file.
