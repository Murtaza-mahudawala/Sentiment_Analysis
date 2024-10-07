import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

# Function to get sentiment based on VADER analysis
def get_sentiment(text):
    sentiment_score = analyzer.polarity_scores(text)
    if sentiment_score['compound'] >= 0.05:
        return 'Positive'
    elif sentiment_score['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Load the dataset
df = pd.read_csv('Sentiment.csv', encoding='ISO-8859-1')


# Apply sentiment analysis to the 'Content' column
df['Sentiment'] = df['Content'].apply(get_sentiment)

# Display the first few rows to verify the results
print(df[['Title', 'Sentiment']])

# Optional: Save the results to a new CSV file
df.to_csv('articles_with_sentiment.csv', index=False)

positive_articles = df[df['Sentiment'] == 'Positive']
negative_articles = df[df['Sentiment'] == 'Negative']
neutral_articles = df[df['Sentiment'] == 'Neutral']

# Print segmented data summaries
print("Positive Articles:\n", positive_articles[['Title']])
print("Negative Articles:\n", negative_articles[['Title']])
print("Neutral Articles:\n", neutral_articles[['Title']])
