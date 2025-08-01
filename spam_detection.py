# Standard library imports
import os
import time
from datetime import datetime
import re

# Progress Bar
from tqdm import tqdm

# Third-party imports
import numpy as np
import pandas as pd
import psutil
import torch
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from torch.utils.data import Dataset, DataLoader

# Language detection imports
# from langid.langid import LanguageIdentifier, model
from langdetect import detect, detect_langs, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import fasttext

# Spam Labelling
from transformers import pipeline

# sentence embeddings
from sentence_transformers import SentenceTransformer 

# Spam Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# class for input data structure
class Data(Dataset):
    def __init__(self, x_train, y_train):
        self.x = torch.from_numpy(x_train.astype(np.float32))
        self.y = torch.from_numpy(y_train).type(torch.LongTensor)
        self.len = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len 


def log_checkpoint(message):
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    
    log_msg = f"{timestamp} - {message}"
    print(log_msg)
    
    # Also write to log file
    with open("script.log", "a") as f:
        f.write(log_msg + "\n")


def get_data_from_mariadb(host, database, username, password, port=3306, table_name='data_tdr_spam_filter'):
    """
    Fetch SMS data from MariaDB database using SQLAlchemy
    """
    try:
        # Create SQLAlchemy engine
        # Use pymysql as the driver
        connection_string = f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"
        engine = create_engine(connection_string)

        log_checkpoint(f"Successfully connected to MariaDB database: {database}")
        
        # Query to fetch data
        query = f"""
            SELECT payload, smsc_src_addr, LEFT(current_datetime, 16) as tx_datetime
            FROM {table_name}
            LIMIT 100
        """
        
        # Read data into pandas DataFrame using SQLAlchemy engine
        data = pd.read_sql(query, engine)
        log_checkpoint(f"Fetched {len(data)} records from database")
        
        return data
        
    except SQLAlchemyError as e:
        log_checkpoint(f"Error connecting to MariaDB: {e}")
        return None
    except Exception as e:
        log_checkpoint(f"Unexpected error: {e}")
        return None
        
    finally:
        # SQLAlchemy engine handles connection pooling automatically
        if 'engine' in locals():
            engine.dispose()
            log_checkpoint("MariaDB connection closed")

def clean_text_alpha_only(text):
    cleaned = re.sub(r'[^a-zA-Z\s]', '', text).strip()
    return cleaned

DetectorFactory.seed = 0  # for consistent langdetect results
def detect_with_langdetect(text):
    try:
        lang = detect(text)
        langs = detect_langs(text)
        confidence = langs[0].prob if langs else 0.0
        return pd.Series([lang, confidence])
    except LangDetectException:
        return pd.Series(['unknown', 0.0])
    except Exception:
        return pd.Series(['error', 0.0])
    

# Load the pre-trained language identification model
model = fasttext.load_model("lid.176.ftz")

def detect_with_fasttextwheel(text):
    try:
        # top-1
        lang, confidence = model.predict(text, k=1)
        lang = lang[0].replace('__label__', '')
        confidence = confidence[0]
        return pd.Series([lang, confidence])
    except LangDetectException:
        return pd.Series(['unknown', 0.0])


def categorize(text):
    text = text.lower()
    # Replace URLs
    text = re.sub(r"http\S+", " URL ", text)
    # Replace long digit sequences
    text = re.sub(r"\b\d{10,}\b", " PHONE ", text)
    # Replace newlines
    text = re.sub(r"\n", " NEWLINE ", text)
    # Replace punctuation
    text = re.sub(r"[^\w\s]", " PUNCT ", text)
    
    # Replace currency values
    # rm50, rm 50.00
    text = re.sub(r"\brm\s?\d+(\.\d{1,2})?\b", " MONEY ", text)
    # usd 100
    text = re.sub(r"\busd\s?\d+(\.\d{1,2})?\b", " MONEY ", text)
    # $200
    text = re.sub(r"\$\s?\d+(\.\d{1,2})?\b", " MONEY ", text)
    return text


def preproccess(text):
    text = text.lower()
    # Remove URLs
    text = re.sub(r"http\S+", "", text)
    # Remove long digit sequences
    text = re.sub(r"\b\d{10,}\b", "", text)
    # Remove newlines
    text = re.sub(r"\n", " ", text)
    # Remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)
    
    # Replace currency values
    # rm50, rm 50.00
    text = re.sub(r"\brm\s?\d+(\.\d{1,2})?\b", "", text)
    # usd 100
    text = re.sub(r"\busd\s?\d+(\.\d{1,2})?\b", "", text)
    # $200
    text = re.sub(r"\$\s?\d+(\.\d{1,2})?\b", "", text)
    return text


# Load a pre-trained spam classifier
classifier = pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-sms-spam-detection")
def detect_with_transformer(text):
    try:
        result = classifier(
            text,
            truncation=True,
            # Force max token limit
            max_length=512,
            # Optional, helps for consistent batching
            padding=True,
            return_all_scores=False
        )[0]
        payload = result['label']
        confidence = result['score']
        return pd.Series([payload, confidence])
    except LangDetectException:
        return pd.Series(['unknown', 0.0])
    

if __name__ == '__main__':
    # Database configuration
    DB_CONFIG = {
        'host': '10.168.51.196',  
        'database': 'sms_spam_cd',  
        'username': 'wj_wong',  
        'password': 'unified', 
        'port': 3306,  
        'table_name': 'data_tdr_spam_filter' 
    }
    
    # Get dataset from MariaDB
    data = get_data_from_mariadb(**DB_CONFIG)
    
    if data is None:
        log_checkpoint("Failed to fetch data from database. Exiting...")
        exit(1)
    
    # Ensure we have the required columns
    if not all(col in data.columns for col in ['payload']):
        log_checkpoint("Error: Required columns 'payload' not found in data")
        exit(1)
    
    # Display basic info about the dataset
    log_checkpoint(f"Dataset shape: {data.shape}")
    df = pd.DataFrame(data)
    
    # Count number of SMS per user per minute
    sms_per_user_per_min = (df.groupby(['smsc_src_addr', 'tx_datetime']).size().reset_index(name='count'))
    avg_sms_per_user_per_min = (sms_per_user_per_min.groupby('smsc_src_addr')['count'].mean().reset_index(name='avg_same_sms_per_minute'))
    df = df.merge(avg_sms_per_user_per_min, on='smsc_src_addr', how='left')
    # Count number of SMS per user per minute
    sms_per_payload_per_min = (df.groupby(['payload', 'tx_datetime']).size().reset_index(name='count'))
    avg_sms_per_payload_per_min = (sms_per_payload_per_min.groupby('payload')['count'].mean().reset_index(name='avg_same_sms_per_minute'))
    df = df.merge(avg_sms_per_payload_per_min, on='payload', how='left')

    df['payload_ori'] = df['payload']

    # Replace all newlines in 'payload' with space
    df['payload'] = df['payload'].astype(str).str.replace(r'[\r\n]+', ' ', regex=True)

    # Left alphabet only
    df['payload'] = df['payload'].apply(lambda x: clean_text_alpha_only(x))
    
    tqdm.pandas()

    # Checkpoint 1
    log_checkpoint("<langdetect> STARTED...")
    df[['language_with_langdetect', 'confidence_with_langdetect']] = df['payload'].progress_apply(lambda x: pd.Series(detect_with_langdetect(x)))
    log_checkpoint("<langdetect> COMPLETED")

    # Checkpoint 2
    log_checkpoint("<fasttext> STARTED...")
    df[['language_with_fasttext', 'confidence_with_fasttext']] = df['payload'].progress_apply(lambda x: pd.Series(detect_with_fasttextwheel(x)))
    log_checkpoint("<fasttext> COMPLETED")

    # Final Lang
    df['final_lang'] = np.where(df['language_with_langdetect'] == 'unknown', 'unknown', df['language_with_fasttext'])
    # df = df[df['final_lang'].isin(['unknown', 'en'])]
    df['avg_sms_per_user_per_min'] = df['avg_same_sms_per_minute_x']
    df['avg_sms_per_payload_per_min'] = df['avg_same_sms_per_minute_y']
    df = df[['payload_ori', 'avg_sms_per_user_per_min', 'avg_sms_per_payload_per_min', 'final_lang']]

    # categorizeing
    df['payload_categorized'] = df['payload_ori'].apply(categorize)

    # Count each token
    df['url_count'] = df['payload_categorized'].str.count(r'\bURL\b')
    df['phone_count'] = df['payload_categorized'].str.count(r'\bPHONE\b')
    df['newline_count'] = df['payload_categorized'].str.count(r'\bNEWLINE\b')
    df['punct_count'] = df['payload_categorized'].str.count(r'\bPUNCT\b')
    df['money_count'] = df['payload_categorized'].str.count(r'\bMONEY\b')

    # Spam and Ham Labelling
    log_checkpoint("Spam Labelling with <transformer> STARTED...")
    df[['is_spam', 'confidence']] = df['payload_ori'].progress_apply(lambda x: pd.Series(detect_with_transformer(x)))
    df['is_spam'] = df['is_spam'].map({'LABEL_0':0, 'LABEL_1':1})
    log_checkpoint("Spam Labelling with <transformer> COMPLETED")

    # Save as CSV
    log_checkpoint("Downloading data to csv...")
    df.to_csv('C:/Users/wj_wong/Desktop/spam_sms/spam_labelling/df_spam_labeled.csv', index=False)
    log_checkpoint("Download COMPLETED")

    # Turn sentences into vector
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # preproccess
    df['payload_cleaned'] = df['payload_ori'].apply(preproccess)
    X_text = np.vstack(df['payload_cleaned'].apply(model.encode))

    # Prepare Training Data
    feature_cols = ['avg_sms_per_user_per_min', 
                    'avg_sms_per_payload_per_min', 
                    'url_count', 
                    'phone_count', 
                    'newline_count', 
                    'punct_count', 
                    'money_count']
    X_meta = df[feature_cols].values
    X_combined = np.hstack([X_text, X_meta])
    
    y = df['is_spam']

    X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

    n_trees = 100

    # Train a Classifier
    model = RandomForestClassifier(random_state=42)
    for i in tqdm(range(1, n_trees + 1)):
        model.set_params(n_estimators=i)
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Save final results including prediction to CSV
    df_result = df.copy()
    # Align with test set
    df_result = df_result.iloc[y_test.index].copy()
    df_result['predicted_is_spam'] = y_pred

    df_result.to_csv('C:/Users/wj_wong/Desktop/spam_sms/spam_labelling/df_spam_labeled_with_predictions.csv', index=False)
    log_checkpoint("Final prediction CSV saved")