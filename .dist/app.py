import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Load model and vectorizer
model = joblib.load('D:\\Project\\ML\\paper\\.dist\\model.joblib')
vectorizer = joblib.load('D:\\Project\\ML\\paper\\.dist\\vectorizer.joblib')

# Function to predict sentiment
def predict_sentiment(text):
    # Vectorize the input text using the same vectorizer
    text_vectorized = vectorizer.transform([text])
    
    # Make prediction
    prediction = model.predict(text_vectorized)
    
    # Map the prediction to the sentiment
    if prediction == 1:
        return 'Positive'
    elif prediction == -1:
        return 'Negative'
    else:
        return 'Neutral'

# Streamlit UI
def main():
    st.title("Sentiment Analysis")
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(uploaded_file)
        
        # Display the uploaded file
        st.write("Uploaded File Preview:")
        st.write(df.head())
        
        # Column selection for feedback text
        column_name = st.selectbox("Select Column with Feedback Text", df.columns)
        
        # Sentiment dropdown selection
        sentiment_type = st.selectbox("Select Sentiment Type", ['Positive', 'Negative', 'Neutral'])
        
        # Analyze button
        if st.button("Analyze Sentiment"):
            if column_name in df.columns:
                # Predict sentiment for each feedback text
                df['Predicted Sentiment'] = df[column_name].apply(predict_sentiment)
                
                # Filter the dataframe based on the selected sentiment
                filtered_df = df[df['Predicted Sentiment'] == sentiment_type]
                
                if len(filtered_df) > 0:
                    st.write(f"Showing {sentiment_type} Feedback:")
                    st.write(filtered_df[['Predicted Sentiment', column_name]])
                else:
                    st.write(f"No {sentiment_type} feedback found.")
            else:
                st.error(f"The selected column '{column_name}' is not in the CSV file.")

if __name__ == "__main__":
    main()
