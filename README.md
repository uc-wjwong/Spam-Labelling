# SMS Spam Detection Pipeline

## This project provides a complete machine learning pipeline for SMS spam detection. It fetches SMS data from a MariaDB database, applies language detection, performs text preprocessing, classifies the messages using a transformer model, and trains a Random Forest classifier to predict spam. It then exports labeled and predicted results to CSV.

📦 Requirements

✅ Python Version

Python 3.12

✅ Dependencies

Install all required packages:

pip install tqdm numpy pandas psutil torch scikit-learn sqlalchemy \
  langdetect fasttext transformers sentence-transformers pymysql

⚠ Notes

On Windows with Python 3.12, fasttext may fail to install from PyPI. Use a precompiled wheel:

pip install https://github.com/ffreemt/wheelhouse/releases/download/2023-12-01/fasttext-0.9.2-cp312-cp312-win_amd64.whl

On RHEL 9, install development tools:

sudo dnf install gcc gcc-c++ python3-devel mariadb-devel

🗂 Project Files

spam_detection.py: Main executable script

lid.176.ftz: FastText model for language detection (must be in working directory)

script.log: Log file tracking progress and memory usage

df_spam_labeled.csv: Output from transformer spam labelling

df_spam_labeled_with_predictions.csv: Final results with predictions

🔧 Configuration

💾 Change the SQL Query or Database Connection

Edit this section in spam_detection.py to match your DB setup:

DB_CONFIG = {
    'host': 'your-db-host',
    'database': 'your-database',
    'username': 'your-username',
    'password': 'your-password',
    'port': 3306,
    'table_name': 'data_tdr_spam_filter'
}

You can also modify the SQL query inside the get_data_from_mariadb() function if needed.

🚀 Execution

▶ On Windows 11

python spam_detection.py

▶ On RHEL 9

python3 spam_detection.py

Ensure the working directory contains lid.176.ftz.

📝 Output Files

df_spam_labeled.csv

Contains SMS content, metadata, and transformer spam label

df_spam_labeled_with_predictions.csv

Adds Random Forest prediction to the above output

🔍 Pipeline Overview (Step-by-Step)

1. Database Fetch

Connects to MariaDB using SQLAlchemy and PyMySQL

Pulls SMS message content and metadata (source, timestamp)

2. Data Cleaning

Removes newline characters and unwanted symbols

Creates a cleaned version of the text with only alphabet characters

3. Language Detection

Uses langdetect and fasttext to determine the language of each SMS

Final language is selected based on fasttext (or fallback to langdetect)

4. Feature Engineering

Calculates statistics:

Average number of messages per user per minute

Average number of identical payloads per minute

Extracts counts of specific patterns:

URLs, phone numbers, newlines, punctuation, and monetary terms

5. Spam Classification (Transformer-Based)

Uses HuggingFace's bert-tiny-finetuned-sms-spam-detection transformer

Labels each message as LABEL_0 (not spam) or LABEL_1 (spam)

6. Sentence Embedding

Transforms each SMS into a 384-dimensional vector using the all-MiniLM-L6-v2 model from sentence-transformers

7. Model Training (Random Forest)

Combines sentence embeddings with engineered features

Trains a Random Forest classifier with increasing tree count (1 to 100)

Evaluates using classification_report

8. Final Output

Saves spam-labeled dataset to CSV

Saves final predictions (from Random Forest) in another CSV file

Records progress and memory usage in a log file

📌 Notes

Make sure you have internet access the first time to download the HuggingFace transformer model.

Avoid hardcoding DB passwords in production. Use .env or secrets management.

You can increase LIMIT 100 in the SQL query for more data during training.

🤝 License

This project is for internal use and experimentation. License can be updated as needed.

