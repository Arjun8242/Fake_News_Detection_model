import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import requests
from bs4 import BeautifulSoup
from newspaper import Article

# 2. NLTK DOWNLOADS (Run once)
nltk.download('stopwords')
nltk.download('wordnet')

# 3. LOAD MODELS (Put this after imports)
model = joblib.load('fake_news_model .joblib')
tfidf = joblib.load('tfidf_vectorizer .joblib')

# 4. TEXT PROCESSING FUNCTIONS
def clean_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    text = re.sub(r'[^a-zA-Z]', ' ', str(text))
    text = text.lower()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

def extract_text_from_url(url):
    """Extract text from URL using newspaper3k with fallback to BeautifulSoup"""
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except:
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            return ' '.join(p.get_text() for p in soup.find_all('p'))
        except Exception as e:
            st.error(f"URL Error: {str(e)}")
            return None

# 5. STREAMLIT UI
def main():
    st.title("🔍 Fake News Detector")
    
    input_type = st.radio("Input type:", ("Text", "URL"))

    if input_type == "Text":
        text = st.text_area("Enter news text:", height=200)
        if st.button("Analyze"):
            if text:
                cleaned = clean_text(text)
                features = tfidf.transform([cleaned])
                pred = model.predict(features)[0]
                proba = model.predict_proba(features)[0][pred]
                st.success(f"Result: {'Fake' if pred == 1 else 'Real'} (Confidence: {proba:.1%})")
            else:
                st.warning("Please enter text")
    else:
        url = st.text_input("Enter URL:")
        if st.button("Check URL"):
            if url:
                text = extract_text_from_url(url)
                if text:
                    cleaned = clean_text(text)
                    features = tfidf.transform([cleaned])
                    pred = model.predict(features)[0]
                    proba = model.predict_proba(features)[0][pred]
                    st.success(f"Result: {'Fake' if pred == 1 else 'Real'} (Confidence: {proba:.1%})")
                    st.text_area("Extracted text preview:", value=text[:500] + "...", height=150)
                else:
                    st.error("Failed to extract text from URL")
            else:
                st.warning("Please enter a URL")

if __name__ == "__main__":
    main()
