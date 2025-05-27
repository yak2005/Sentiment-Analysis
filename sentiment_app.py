import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Preprocess text
def preprocess_text(text):
    stop_words = set(stopwords.words("english"))
    tokens = word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return " ".join(filtered_tokens)

# Train the model
@st.cache
def train_model(data):
    data['cleaned_text'] = data['text'].apply(preprocess_text)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data['cleaned_text'])
    y = data['sentiment']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    return model, vectorizer

# Predict sentiment
def predict_sentiment(model, vectorizer, text):
    cleaned_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([cleaned_text])
    sentiment = model.predict(vectorized_text)[0]
    return sentiment

# Visualize sentiment distribution
def plot_sentiment_distribution(sentiments):
    sns.countplot(x=sentiments, palette="coolwarm")
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    st.pyplot()

# Streamlit app
def main():
    st.title("Sentiment Analysis App")
    st.write("Analyze the sentiment of text data (Positive, Negative, Neutral).")

    # Load dataset
    dataset_url = "https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv"
    data = pd.read_csv(dataset_url, usecols=["text", "label"])
    data.rename(columns={"label": "sentiment"}, inplace=True)

    # Train the model
    model, vectorizer = train_model(data)

    # User Input
    st.header("Single Text Analysis")
    user_input = st.text_input("Enter text:")
    if st.button("Analyze Sentiment"):
        if user_input:
            sentiment = predict_sentiment(model, vectorizer, user_input)
            st.success(f"Predicted Sentiment: {sentiment}")

    # Bulk Analysis
    st.header("Bulk Analysis")
    uploaded_file = st.file_uploader("Upload a CSV file with a 'text' column", type=["csv"])
    if uploaded_file:
        uploaded_data = pd.read_csv(uploaded_file)
        if "text" in uploaded_data.columns:
            uploaded_data['sentiment'] = uploaded_data['text'].apply(
                lambda x: predict_sentiment(model, vectorizer, x)
            )
            st.dataframe(uploaded_data)

            # Visualization
            st.subheader("Sentiment Distribution")
            plot_sentiment_distribution(uploaded_data['sentiment'])
        else:
            st.error("CSV file must contain a 'text' column.")

if __name__ == "__main__":
    main()
