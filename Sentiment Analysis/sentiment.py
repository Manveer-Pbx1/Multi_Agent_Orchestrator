import streamlit as st
from transformers import pipeline

try:
    sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
except Exception as e:
    st.error("Error loading the sentiment analysis model. Please try again later.")
    st.stop()

st.title("Sentiment Analysis Tool")
st.write("Enter text to analyze its sentiment (positive, neutral, negative).")

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
        try:
            result = sentiment_pipeline(user_input)
            star_rating = result[0]['label']
            sentiment = convert_to_sentiment(star_rating)
            score = result[0]['score']
            
            st.write(f"Sentiment: {sentiment}")
            st.write(f"Confidence Score: {score:.2f}")
        except Exception as e:
            st.error(f"An error occurred during analysis. Please try with different text or try again later. Error: {str(e)}")
    else:
        st.write("Please enter some text to analyze.")
