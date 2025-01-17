from transformers import pipeline

# Initialize the sentiment pipeline once
try:
    sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
except Exception as e:
    raise RuntimeError("Error loading the sentiment analysis model: " + str(e))

def convert_to_sentiment(star_rating):
    if star_rating in ["1 star", "2 stars"]:
        return "Negative"
    elif star_rating == "3 stars":
        return "Neutral"
    else:
        return "Positive"

def analyze_sentiment(text):
    """
    Analyze the sentiment of the given text.
    
    Args:
        text (str): The text to analyze
        
    Returns:
        dict: A dictionary containing sentiment analysis results with keys:
            - sentiment: str (Positive, Neutral, or Negative)
            - confidence: float (confidence score)
            - success: bool (True if analysis was successful)
            - error: str (error message if any, None otherwise)
    """
    if not text:
        return {
            "sentiment": None,
            "confidence": 0.0,
            "success": False,
            "error": "Empty text provided"
        }
    
    try:
        result = sentiment_pipeline(text)
        star_rating = result[0]['label']
        sentiment = convert_to_sentiment(star_rating)
        score = result[0]['score']
        
        return {
            "sentiment": sentiment,
            "confidence": round(score, 2),
            "success": True,
            "error": None
        }
    except Exception as e:
        return {
            "sentiment": None,
            "confidence": 0.0,
            "success": False,
            "error": str(e)
        }
