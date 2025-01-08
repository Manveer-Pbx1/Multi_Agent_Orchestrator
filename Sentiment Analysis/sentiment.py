import streamlit as st
from transformers import pipeline

# Load sentiment analysis pipeline from Hugging Face
sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Streamlit app
st.title("Sentiment Analysis Tool")
st.write("Enter text to analyze its sentiment (positive, neutral, negative).")

# Text input
user_input = st.text_area("Enter text here:")

def convert_to_sentiment(star_rating):
    if star_rating in ["1 star", "2 stars"]:
        return "Negative"
    elif star_rating == "3 stars":
        return "Neutral"
    else:
        return "Positive"

if st.button("Analyze"):
    if user_input:
        # Perform sentiment analysis
        result = sentiment_pipeline(user_input)
        star_rating = result[0]['label']
        sentiment = convert_to_sentiment(star_rating)
        score = result[0]['score']
        
        # Display result
        st.write(f"Sentiment: {sentiment}")
        st.write(f"Confidence Score: {score:.2f}")
    else:
        st.write("Please enter some text to analyze.")
