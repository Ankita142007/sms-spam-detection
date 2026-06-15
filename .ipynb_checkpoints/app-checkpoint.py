import streamlit as st
import pickle
import nltk
import string

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Downloads
nltk.download('punkt')
nltk.download('stopwords')

# Load Model
model = pickle.load(open("model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))

ps = PorterStemmer()

# Text Processing
def transform_text(text):

    text = text.lower()

    text = nltk.word_tokenize(text)

    y=[]

    for i in text:
        if i.isalnum():
            y.append(i)

    text=y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text=y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# -------------------
# PAGE CONFIG
# -------------------

st.set_page_config(
    page_title="SMS Spam Detection",
    page_icon="📩",
    layout="wide"
)

# -------------------
# HEADER
# -------------------

st.title("📩 SMS Spam Detection System")

st.markdown("""
This application uses **Natural Language Processing (NLP)** and
**Machine Learning** to classify SMS messages as:

- ✅ Legitimate Message (Ham)
- 🚨 Spam Message

**Model:** Multinomial Naive Bayes  
**Feature Extraction:** TF-IDF Vectorization
""")

st.divider()

# -------------------
# INPUT
# -------------------

sms = st.text_area(
    "Enter SMS Message",
    height=150
)

# -------------------
# PREDICT
# -------------------

if st.button("Predict"):

    transformed_sms = transform_text(sms)

    vector_input = vectorizer.transform([transformed_sms])

    prediction = model.predict(vector_input)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("🚨 Spam Message Detected")
    else:
        st.success("✅ Legitimate Message")

st.divider()

# -------------------
# PROJECT INFO
# -------------------

st.subheader("Project Information")

st.markdown("""
### Workflow

SMS Message → Text Cleaning → Tokenization → Stopword Removal →
Stemming → TF-IDF → Naive Bayes Model → Prediction

### Technologies Used

- Python
- Scikit-Learn
- NLTK
- Streamlit
- TF-IDF
- Multinomial Naive Bayes

### Author

Ankita Banerjee
""")