import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Function to clean the text
def clean_text(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text)  # Remove mentions
    text = re.sub(r'#', '', text)  # Remove hashtags
    text = re.sub(r'https?:\/\/\S+', '', text)  # Remove links
    text = re.sub(r'[^A-Za-z\s]', '', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    text = " ".join(word for word in text.split() if word not in stop_words)  # Remove stopwords
    return text

# Load the trained model and vectorizer using joblib
model = joblib.load('D:\\Project\\ML\\paper\\.dist\\model.joblib')
vectorizer = joblib.load('D:\\Project\\ML\\paper\\.dist\\vectorizer.joblib')

# Streamlit UI
st.title("Text Sentiment Classifier")

st.write("""
This is a simple sentiment analysis tool. Enter a sentence below, and the model will classify it as Positive, Negative, or Neutral.
""")

# User input text box
user_input = st.text_area("Enter text here:")

# Add an Enter button
if st.button("Enter"):
    if user_input:
        # Clean and transform the user input
        cleaned_input = clean_text(user_input)
        input_vector = vectorizer.transform([cleaned_input])

        # Predict the sentiment
        prediction = model.predict(input_vector)

        # Display result
        if prediction == 1:
            st.write("Sentiment: **Positive**")
        elif prediction == -1:
            st.write("Sentiment: **Negative**")
        else:
            st.write("Sentiment: **Neutral**")
    else:
        st.write("Please enter some text to analyze.")
