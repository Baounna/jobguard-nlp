import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

TEXT_COLUMNS = ['title', 'company_profile', 'description', 'requirements', 'benefits']

def combine_text_columns(df, columns=None):
    if columns is None:
        columns = TEXT_COLUMNS
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            df[col] = ''
        df[col] = df[col].fillna('')
    df['combined_text'] = df[columns].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    return df

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize(text):
    return word_tokenize(text)

def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    return [t for t in tokens if t not in stop_words and len(t) > 2]

def lemmatize(tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(t) for t in tokens]

def full_preprocess(text):
    text = clean_text(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)
    return ' '.join(tokens)

def preprocess_dataframe(df):
    df = combine_text_columns(df)
    df['processed_text'] = df['combined_text'].apply(full_preprocess)
    return df
