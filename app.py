import streamlit as st
import json
import pandas as pd
import re
from sentence_transformers import SentenceTransformer, util

# LOAD DATA
with open("chatbot-ibu-hamil.json") as f:
    data = json.load(f)

rows = []
for intent in data['intents']:
    for pattern in intent['patterns']:
        rows.append({
            "question": pattern,
            "answer": intent.get('responses', [""])[0],
        })

df = pd.DataFrame(rows)

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['processed_question'] = df['question'].apply(preprocess)

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

@st.cache_data
def get_embeddings():
    return model.encode(df['processed_question'].tolist(), convert_to_tensor=False)

embeddings = get_embeddings()

def chatbot(user_input):
    user_input_clean = preprocess(user_input)
    user_emb = model.encode(user_input_clean, convert_to_tensor=False)

    similarity = util.cos_sim(user_emb, embeddings)

    index = similarity.argmax().item()
    score = similarity.max().item()

    if score < 0.4:
        return "😅 Maaf, aku belum menemukan jawaban"

    return df.iloc[index]['answer']

# UI
st.set_page_config(page_title="Pregnancy Chatbot", page_icon="🤰")

st.title("🤰 Pregnancy Chatbot")
st.write("Chatbot untuk membantu pertanyaan kehamilan 💖")

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.chat_input("Tanya sesuatu...")

if user_input:
    response = chatbot(user_input)
    st.session_state.history.append(("user", user_input))
    st.session_state.history.append(("bot", response))

for role, msg in st.session_state.history:
    if role == "user":
        st.chat_message("user").write(msg)
    else:
        st.chat_message("assistant").write(msg)

# sidebar
st.sidebar.title("📚 Sumber Data")
st.sidebar.write(f"Jumlah data: {len(df)}")
st.sidebar.dataframe(df.head())