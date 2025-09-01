import pandas as pd
import html
import contractions
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import joblib

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Load the TF-IDF vectorizer
vectorizer = joblib.load('/Users/vytran/MScData/MastersProject/mastersproject/models/tfidf_vectorizer.joblib')


# Function to convert NLTK POS tags to WordNet POS tags
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default to noun if POS tag is unknown

# Function to preprocess text
def preprocess_text(text):
    """
    Preprocess a single text review for text analysis.

    Parameters:
        text (str): The text to preprocess.

    Returns:
        str: The preprocessed text.
    """
    # Convert to lowercase
    text = text.lower()

    # Decode HTML entities
    text = html.unescape(text)

    # Expand contractions
    text = contractions.fix(text)

    # Tokenize, remove punctuation, and lemmatize with POS tagging
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    lemmatized_tokens = [lemmatizer.lemmatize(token, get_wordnet_pos(tag)) for token, tag in pos_tags]
    text = " ".join(lemmatized_tokens)

    # Remove extra whitespace
    text = " ".join(text.split())

    # Remove all punctuation, including apostrophes
    text = re.sub(r"[^\w\s]", "", text)

    # Remove numeric characters
    text = re.sub(r'\d+', '', text)

    # Load general English stop words
    stop_words = set(stopwords.words('english'))

    # Define words to keep for sentiment analysis
    important_words = {
        "doesn", "doesn't", "doesnt", "dont", "don't", "not", "wasn't", "wasnt",
        "aren", "aren't", "arent",  "couldn", "couldn't", "couldnt", "didn",
        "didn't", "didnt", "hadn", "hadn't", "hadnt",  "hasn", "hasn't", "hasnt",
        "haven't", "havent", "isn", "isn't", "isnt", "mightn",  "mightn't",
        "mightnt", "mustn", "mustn't", "mustnt", "needn", "needn't", "neednt",
        "shan", "shan't", "shant", "shouldn", "shouldn't", "shouldnt", "wasn",
        "wasn't",  "wasnt", "weren", "weren't", "werent", "won", "won't", "wont",
        "wouldn", "wouldn't", "wouldnt", "good", "bad", "worst", "wonderfull",
        "best", "better", "not", "no", "but", "yet", "never", "none"
    }

    # Remove important words from stop words list
    custom_stop_words = stop_words - important_words

    # Remove customized stop words
    text = ' '.join([word for word in text.split() if word not in custom_stop_words])

    return text

# Function to preprocess and transform reviews
def preprocess_reviews(data, column_name='Review'):
    """
    Preprocess reviews (single or multiple) and transform them using a TF-IDF vectorizer.

    Parameters:
        data (str, pd.DataFrame): The input text (single review) or DataFrame with multiple reviews.
        column_name (str): The column name containing reviews (if input is a DataFrame).

    Returns:
        sparse matrix: TF-IDF transformed text data.
        list or pd.DataFrame: Preprocessed text data.
    """
    if isinstance(data, str):

        # Process a single review
        preprocessed_text = preprocess_text(data)
        tfidf_transformed = vectorizer.transform([preprocessed_text])
        return tfidf_transformed

    elif isinstance(data, pd.DataFrame):

        # Process multiple reviews in a DataFrame
        if column_name not in data.columns:
            raise ValueError(f"Column '{column_name}' not found in DataFrame.")

        df = data.copy()
        df[column_name] = df[column_name].apply(preprocess_text)

        # Transform the preprocessed text using the TF-IDF vectorizer
        tfidf_transformed = vectorizer.transform(df[column_name])

        return tfidf_transformed

    else:
        raise ValueError("Input must be either a string (single review) or a Pandas DataFrame.")
