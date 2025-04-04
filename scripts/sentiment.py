import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')

# Test Pytorch

classifier = pipeline("sentiment-analysis")
classifier("We are very happy to show you the 🤗 Transformers library.")
results = classifier(["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."])
for result in results:
    print(f"label: {result['label']}, with score: {round(result['score'], 4)}")

reviews = pd.read_csv('../rawdata/cleaned_reviews.csv')
summary = pd.read_csv('../rawdata/asin_summary.csv')

# Function to map sentiment labels to numerical values
def map_sentiment(label):
    if label == "NEGATIVE":
        return -1
    elif label == "POSITIVE":
        return 1

# Function to split text into chunks
def split_into_chunks(text, max_length=128):
    words = text.split()
    for i in range(0, len(words), max_length):
        yield " ".join(words[i:i + max_length])

# Function to run sentiment analysis on a single text
def analyze_sentiment(text):
    # Check for NaN, null, or empty strings
    if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
        return np.nan  # Return NaN for invalid inputs

    # Split the text into chunks if it's too long
    chunks = list(split_into_chunks(text, max_length=128))  # Adjust chunk size as needed
    results = classifier(chunks)
    
    # Map sentiment labels to numerical values and calculate the average score
    scores = [map_sentiment(result['label']) for result in results]
    avg_score = round(np.mean(scores))  # Calculate and round the average score
    
    return avg_score

# Apply sentiment analysis to 'title_x' column
reviews['sentiment_title'] = reviews['title_x'].apply(analyze_sentiment)

# Apply sentiment analysis to 'text' column
reviews['sentiment_text'] = reviews['text'].apply(analyze_sentiment)

# Step 1: Combine sentiment_title and sentiment_text for each review
reviews['combined_sentiment'] = reviews['sentiment_title'] + reviews['sentiment_text']

# Step 2: Group by asin and calculate total negative and positive reviews
summary_sentiments = reviews.groupby('asin')['combined_sentiment'].apply(
    lambda x: pd.Series({
        'total_negative': (x < 0).sum(),
        'total_positive': (x > 0).sum()
    })
).unstack()

# Step 3: Merge the calculated sentiments into the summary table
summary = summary.merge(summary_sentiments, on='asin', how='left')

# Convert timestamp to datetime
reviews['timestamp'] = pd.to_datetime(reviews['timestamp'])

# Step 1: Calculate sentiment_ratio_positive
sentiment_summary = reviews.groupby('asin').agg(
    total_positive=('sentiment_title', lambda x: (x + reviews.loc[x.index, 'sentiment_text'] > 0).sum()),
)
sentiment_summary['sentiment_ratio_positive'] = sentiment_summary['total_positive'] / summary.set_index('asin')['num_reviews']

# Step 2: Calculate first_review_date and last_review_date
first_review_date = reviews.groupby('asin')['timestamp'].min()
last_review_date = reviews.groupby('asin')['timestamp'].max()
review_period_hours = (last_review_date - first_review_date).dt.total_seconds() / 3600

# Step 3: Calculate review_frequency (reviews per hour)
review_frequency = summary.set_index('asin')['num_reviews'] / review_period_hours

# Step 4: Calculate value_for_money_score
summary['value_for_money_score'] = summary['avg_rating'] / summary['price']

# Step 5: Merge all calculated fields into the summary DataFrame
summary = summary.merge(sentiment_summary[['sentiment_ratio_positive']], on='asin', how='left')
summary = summary.merge(first_review_date.rename('first_review_date'), on='asin', how='left')
summary = summary.merge(last_review_date.rename('last_review_date'), on='asin', how='left')
summary = summary.merge(review_period_hours.rename('review_period_hours'), on='asin', how='left')
summary = summary.merge(review_frequency.rename('review_frequency'), on='asin', how='left')

summary.to_csv('../rawdata/processed_summary.csv', index=False)
reviews.to_csv('../rawdata/processed_reviews.csv', index=False)