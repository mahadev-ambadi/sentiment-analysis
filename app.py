import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import re
import nltk
from nltk.tokenize import word_tokenize

# Download NLTK resources
nltk.download("punkt")

# Load the trained model and TF-IDF vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Load dataset (Ensure your dataset has 'Time' and 'Text' columns)
df = pd.read_csv("Reviews.csv", encoding="latin1", usecols=["Time", "Score", "Text"])
df["Time"] = pd.to_datetime(df["Time"], unit="s")  # Convert timestamps to datetime
df["Sentiment"] = df["Score"].apply(lambda x: "Positive" if x >= 4 else "Negative")

# Streamlit UI
st.title("📝 Product Review Sentiment Analyzer")
st.write("Analyze customer sentiment and get AI-based insights!")

# **1️⃣ Sentiment Prediction**
st.subheader("🔍 Predict Sentiment for a New Review")
user_review = st.text_area("Enter your review here:")

# AI-Based Recommendations Function
def get_recommendations(sentiment):
    if sentiment == "Positive 😊":
        return "🎉 Keep up the great work! Customers love your product."
    else:
        return "⚠️ Consider improving product quality or customer service."

if st.button("Analyze Sentiment"):
    if user_review.strip() == "":
        st.warning("⚠️ Please enter a review before analyzing.")
    else:
        # Preprocess input review
        transformed_review = vectorizer.transform([user_review])

        # Predict sentiment
        prediction = model.predict(transformed_review)[0]
        confidence_scores = model.predict_proba(transformed_review)[0]  # Get confidence scores

        # Extract confidence values
        negative_confidence = confidence_scores[0] * 100  # % confidence for Negative
        positive_confidence = confidence_scores[1] * 100  # % confidence for Positive

        # Determine sentiment
        sentiment = "Positive 😊" if prediction == 1 else "Negative 😡"

        # AI-Based Recommendation
        recommendation = get_recommendations(sentiment)

        # Display result
        st.subheader("Prediction:")
        st.write(f"**{sentiment}**")

        # Show confidence scores
        st.write(f"🔹 **Confidence Scores:**")
        st.write(f"- Positive: **{positive_confidence:.2f}%**")
        st.write(f"- Negative: **{negative_confidence:.2f}%**")

        # Show AI Recommendation
        st.subheader("💡 AI-Based Recommendation:")
        st.write(f"{recommendation}")

# **2️⃣ Sentiment Trends Over Time**
st.subheader("📈 Customer Sentiment Trends Over Time")
df_trend = df.resample("M", on="Time")["Sentiment"].value_counts().unstack()
st.line_chart(df_trend.fillna(0))  # Fill missing values with 0

# **3️⃣ AI-Based Recommendations Based on Past Data**
st.subheader("🤖 AI-Based Recommendations")

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return word_tokenize(text)

# Tokenize all reviews
df["Tokenized_Text"] = df["Text"].apply(preprocess_text)

# Get frequent words
positive_words = Counter([word for row in df[df["Sentiment"] == "Positive"]["Tokenized_Text"] for word in row])
negative_words = Counter([word for row in df[df["Sentiment"] == "Negative"]["Tokenized_Text"] for word in row])

# Show top 5 liked & disliked features
st.write("✅ **Customers love:**", ", ".join([word for word, _ in positive_words.most_common(5)]))
st.write("❌ **Customers dislike:**", ", ".join([word for word, _ in negative_words.most_common(5)]))

# Footer
st.markdown("---")
st.write("💡 Built with Streamlit & Machine Learning")
