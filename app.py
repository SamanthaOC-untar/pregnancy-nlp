import streamlit as st
import json
import pandas as pd
import re
from sentence_transformers import SentenceTransformer, util

# ================= LOAD DATA =================
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

# ================= PREPROCESS =================
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['processed_question'] = df['question'].apply(preprocess)

# ================= MODEL =================
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

@st.cache_data
def get_embeddings():
    return model.encode(df['processed_question'].tolist(), convert_to_tensor=False)

embeddings = get_embeddings()

# ================= RULE =================
def check_rules(text):
    if re.search(r'(sesak|pingsan)', text.lower()):
        return "🚨 DARURAT! Segera ke IGD!"
    if re.search(r'\b(halo|hai|hi)\b', text.lower()):
        return "👶 Halo bunda! Ada yang bisa saya bantu?"
    return None

# ================= CHATBOT =================
def chatbot(user_input):

    rule = check_rules(user_input)
    if rule:
        return rule

    user_input_clean = preprocess(user_input)
    user_emb = model.encode(user_input_clean, convert_to_tensor=False)

    similarity = util.cos_sim(user_emb, embeddings)

    index = similarity.argmax().item()
    score = similarity.max().item()

    if score < 0.4:
        return "😅 Maaf, aku belum menemukan jawaban"

    return df.iloc[index]['answer']

# ================= UI CONFIG =================
st.set_page_config(page_title="Pregnancy Chatbot", page_icon="🤰", layout="wide")

# ================= CSS =================
st.markdown("""
<style>

/* BACKGROUND */
.main {
    background-color: #fff0f5;
}

/* HEADER STICKY */
.header {
    position: sticky;
    top: 0;
    background-color: #fff0f5;
    padding: 15px;
    z-index: 999;
    border-bottom: 2px solid #ffb6c1;
}

/* USER CHAT */
.chat-bubble-user {
    background-color: #ff4d88;
    color: white;
    padding: 10px 15px;
    border-radius: 15px;
    margin: 5px 0;
    text-align: right;
}

/* BOT CHAT */
.chat-bubble-bot {
    background-color: #ffffff;
    color: black;
    padding: 10px 15px;
    border-radius: 15px;
    margin: 5px 0;
}

/* SCROLL AREA */
.chat-container {
    margin-top: 10px;
}

</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown("""
<div class="header">
    <h2 style="color:#ff4d88;">🤰 Pregnancy Chatbot</h2>
    <p>Asisten untuk membantu pertanyaan seputar kehamilan 💖</p>
</div>
""", unsafe_allow_html=True)

# ================= SESSION =================
if "history" not in st.session_state:
    st.session_state.history = []

# ================= INPUT =================
user_input = st.chat_input("Tanyakan sesuatu...")

if user_input:
    with st.spinner("🤖 Sedang berpikir..."):
        response = chatbot(user_input)
    st.session_state.history.append(("user", user_input))
    st.session_state.history.append(("bot", response))

# ================= CHAT =================
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

for role, msg in st.session_state.history:
    if role == "user":
        st.markdown(f"<div class='chat-bubble-user'>{msg}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-bubble-bot'>{msg}</div>", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ================= SIDEBAR =================
st.sidebar.markdown("## 📚 Sumber Data")
st.sidebar.write(f"Jumlah data: {len(df)}")

st.sidebar.markdown("### 💡 Cara Kerja")
st.sidebar.write("""
- Pertanyaan diubah jadi embedding
- Dicari yang paling mirip
- Jawaban diambil dari dataset
""")

st.sidebar.markdown("### 🔍 Contoh Data")
st.sidebar.dataframe(df.head())