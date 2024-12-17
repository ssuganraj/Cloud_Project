import streamlit as st
import pandas as pd
import joblib
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Load spaCy model for NER
nlp = spacy.load("en_core_web_sm")

# Load models and vectorizers
sentiment_model = joblib.load('D:\\Project\\ML\\paper\\.dist\\model.joblib')
sentiment_vectorizer = joblib.load('D:\\Project\\ML\\paper\\.dist\\vectorizer.joblib')

classification_model = joblib.load('D:\\Project\\ML\\paper\\.dist\\text_classification_model.joblib')
classification_vectorizer = joblib.load('D:\\Project\\ML\\paper\\.dist\\text_classification_vectorizer.joblib')

# Define a mapping for classification predictions
classification_mapping = {
    0: 'Teaching',
    1: 'Infrastructure',
    2: 'Exams',
    3: 'Placements',
    -1: 'Miscellaneous'  # Add labels for other predictions as needed
}

# Function to predict sentiment
def predict_sentiment(text):
    text_vectorized = sentiment_vectorizer.transform([text])
    prediction = sentiment_model.predict(text_vectorized)
    if prediction == 1:
        return 'Positive'
    elif prediction == -1:
        return 'Negative'
    else:
        return 'Neutral'

# Function to predict classification category
def predict_classification(text):
    text_vectorized = classification_vectorizer.transform([text])
    prediction = classification_model.predict(text_vectorized)
    # Map numerical prediction to a descriptive category
    return classification_mapping.get(prediction[0], "Unknown")

# Function to extract named entities
def extract_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Streamlit UI
def main():
    st.title("NLP Pipeline: Sentiment Analysis, Text Classification, and NER")
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded File Preview:")
        st.write(df.head())
        
        column_name = st.selectbox("Select Column with Feedback Text", df.columns)
        
        sentiment_type = st.selectbox("Select Sentiment Type to Filter", ['Positive', 'Negative', 'Neutral', 'All'])
        
        unique_classes = ['All'] + list(set(df[column_name].apply(predict_classification)))
        classification_category = st.selectbox("Select Classification Category to Filter", unique_classes)
        
        if st.button("Analyze NLP Pipeline"):
            if column_name in df.columns:
                df['Predicted Sentiment'] = df[column_name].apply(predict_sentiment)
                df['Predicted Category'] = df[column_name].apply(predict_classification)
                df['Named Entities'] = df[column_name].apply(lambda x: extract_entities(x))
                
                filtered_df = df.copy()
                if sentiment_type != 'All':
                    filtered_df = filtered_df[filtered_df['Predicted Sentiment'] == sentiment_type]
                if classification_category != 'All':
                    filtered_df = filtered_df[filtered_df['Predicted Category'] == classification_category]
                
                if len(filtered_df) > 0:
                    st.write(f"Results for Sentiment: {sentiment_type}, Classification: {classification_category}")
                    st.write(filtered_df[['Predicted Sentiment', 'Predicted Category', 'Named Entities', column_name]])
                else:
                    st.write(f"No results found for Sentiment: {sentiment_type} and Classification: {classification_category}.")
            else:
                st.error(f"The selected column '{column_name}' is not in the CSV file.")

if __name__ == "__main__":
    main()
