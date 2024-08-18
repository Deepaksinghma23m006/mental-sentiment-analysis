import streamlit as st
import joblib as j
import re as r
from nltk.stem import PorterStemmer

# Load the saved objects
vectorizer = j.load('vectorizer.pkl')
model = j.load('model.pkl')

# Initialize the Porter Stemmer
pt = PorterStemmer()

# Define the preprocessing function
def preprocessing(text):
    l = []
    text = r.sub('[^a-zA-Z0-9\s]', '', text.lower())
    for i in text.split():
        l.append(pt.stem(i.lower()))
    return " ".join(l)

# Define the prediction function
def predict(text):
    preprocessed_text = preprocessing(text)
    vector = vectorizer.transform([preprocessed_text])
    prediction = model.predict(vector)[0]
    return prediction

# Streamlit UI
st.title("Mental Health Sentiment Analysis")

input_text = st.text_area("Enter text for sentiment analysis:")

if st.button("Predict"):
    if input_text:
        result = predict(input_text)
        st.write(f"Predicted Sentiment: {result}")
    else:
        st.write("Please enter some text.")
