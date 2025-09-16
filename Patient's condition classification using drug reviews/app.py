import streamlit as st
import pickle
import html
import re
import string
import nltk
import base64
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Convert background image to base64
def get_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg_image = get_base64("doctors-day-seamless-pattern-design-with-medical-equipment-template-hand-drawn-illustration_2175-24378.avif")

# Apply full-page CSS with improved visibility and boxed headers
custom_css = f"""
<style>
.stApp {{
    background-image: url("data:image/avif;base64,{bg_image}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
    color: black !important;
}}
h1, h2, h3, h4, h5, h6, p, label, .markdown-text-container, .stText, .stMarkdown, .st-bx, .css-10trblm {{
    color: black !important;
}}
textarea, .stTextArea textarea {{
    background-color: rgba(255, 255, 255, 0.9);
    color: black;
    font-size: 16px;
}}
.stButton>button {{
    background-color: #00897B;
    color: white;
    font-weight: bold;
    border-radius: 5px;
    padding: 0.5rem 1rem;
}}
.box {{
    background-color: rgba(255, 255, 255, 0.85);
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
}}
footer {{
    visibility: hidden;
}}
</style>
"""
st.set_page_config(page_title="Medical Condition Predictor", layout="centered")
st.markdown(custom_css, unsafe_allow_html=True)

# Text cleaning function
def preprocess_review(text):
    text = html.unescape(text)
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub('[\.*?/]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('w*\d\w*', '', text)
    text = re.sub(r"[\'\"“”‘’,.]", "", text)
    words = nltk.word_tokenize(text)
    words = [word for word in words if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    cleaned_text = ' '.join(lemmatized_words)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

# Mapping labels to conditions
condition_map = {
    0: "Depression",
    1: "Diabetes Type 2",
    2: "High Blood Pressure"
}

# UI Layout
st.markdown("<div class='box'><h1>🩺 Medical Condition Predictor</h1></div>", unsafe_allow_html=True)
st.markdown("<div class='box'><h4>Enter a review or symptom description to predict the possible condition:</h4></div>", unsafe_allow_html=True)

user_input = st.text_area("Patient Review", height=150, placeholder="e.g., I feel tired, weak, and can't focus...")

if st.button("🔍 Predict Condition"):
    if not user_input.strip():
        st.warning("⚠️ Please enter a valid review.")
    else:
        cleaned_input = preprocess_review(user_input)
        transformed_input = vectorizer.transform([cleaned_input])
        prediction = model.predict(transformed_input)[0]
        condition = condition_map.get(prediction, "Unknown")

        # Result box styling
        if condition == "Depression":
            bg_color = "#f3e5f5"
            emoji = "💭"
            font_color = "#6A1B9A"
        elif condition == "Diabetes Type 2":
            bg_color = "#e1f5fe"
            emoji = "🍬"
            font_color = "#0277BD"
        else:
            bg_color = "#ffebee"
            emoji = "❤️"
            font_color = "#C62828"

        st.markdown(
            f"""
            <div style='background-color:{bg_color};padding:20px;border-radius:10px;margin-top:20px'>
                <h3 style='color:{font_color};'>{emoji} Predicted Condition: {condition}</h3>
            </div>
            """,
            unsafe_allow_html=True
        )
