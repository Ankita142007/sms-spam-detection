import streamlit as st
import pickle
import nltk
import string

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ==========================
# NLTK Downloads
# ==========================

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# ==========================
# Page Configuration
# ==========================

st.set_page_config(
    page_title="SMS Spam Detection",
    page_icon="📩",
    layout="centered"
)

# ==========================
# Load Model & Vectorizer
# ==========================

@st.cache_resource
def load_models():
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_models()

# ==========================
# NLP Setup
# ==========================

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# ==========================
# Text Processing Function
# ==========================

def transform_text(text):

    text = text.lower()

    text = nltk.word_tokenize(text)

    y = []

    # Keep only alphanumeric words
    for word in text:
        if word.isalnum():
            y.append(word)

    text = y[:]
    y.clear()

    # Remove stopwords and punctuation
    for word in text:
        if word not in stop_words and word not in string.punctuation:
            y.append(word)

    text = y[:]
    y.clear()

    # Stemming
    for word in text:
        y.append(ps.stem(word))

    return " ".join(y)

# ==========================
# Header
# ==========================

st.title("📩 SMS Spam Detection System")

st.markdown("""
### NLP + Machine Learning Web Application

This application classifies SMS messages as:

- ✅ Legitimate Message (Ham)
- 🚨 Spam Message

**Model Used:** Multinomial Naive Bayes  
**Feature Extraction:** TF-IDF Vectorization
""")

st.divider()

# ==========================
# Sidebar
# ==========================

st.sidebar.title("📊 Model Information")

st.sidebar.write("Algorithm: Multinomial Naive Bayes")
st.sidebar.write("Vectorizer: TF-IDF")
st.sidebar.write("NLP: Tokenization + Stemming")
st.sidebar.write("Deployment: Streamlit Cloud")

# ==========================
# Input Section
# ==========================

sms = st.text_area(
    "Enter SMS Message",
    height=150,
    placeholder="Type or paste an SMS message here..."
)

# ==========================
# Prediction Section
# ==========================

if st.button("Predict"):

    if sms.strip() == "":
        st.warning("Please enter a message.")
    else:

        transformed_sms = transform_text(sms)

        vector_input = vectorizer.transform([transformed_sms])

        prediction = model.predict(vector_input)[0]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("🚨 Spam Message Detected")
        else:
            st.success("✅ Legitimate Message")

# ==========================
# Footer
# ==========================

st.divider()

st.subheader("Project Workflow")

st.markdown("""
SMS Message  
⬇  
Text Cleaning  
⬇  
Tokenization  
⬇  
Stopword Removal  
⬇  
Stemming  
⬇  
TF-IDF Vectorization  
⬇  
Multinomial Naive Bayes  
⬇  
Prediction
""")

st.subheader("Technologies Used")

st.markdown("""
- Python
- Streamlit
- Scikit-Learn
- NLTK
- Pandas
- NumPy
- TF-IDF
- Multinomial Naive Bayes
""")

st.subheader("Author")

st.write("Ankita Banerjee")