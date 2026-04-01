import streamlit as st
import json
import pandas as pd
import re
from sentence_transformers import SentenceTransformer, util

# ================= CONFIG =================
st.set_page_config(page_title="Pregnancy Chatbot", page_icon="🤰", layout="wide")

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

# ================= STYLE =================
st.markdown("""
<style>

/* background */
.main {
    background-color: #fff0f5;
}

/* container */
.chat-container {
    max-width: 700px;
    margin: auto;
}

/* user bubble */
.user-msg {
    display: flex;
    justify-content: flex-end;
}
.user-bubble {
    background-color: #ff4d88;
    color: white;
    padding: 10px 15px;
    border-radius: 18px;
    margin: 5px 0;
    max-width: 70%;
}

/* bot bubble */
.bot-msg {
    display: flex;
    justify-content: flex-start;
}
.bot-bubble {
    background-color: white;
    color: black;
    padding: 10px 15px;
    border-radius: 18px;
    margin: 5px 0;
    max-width: 70%;
    box-shadow: 0 2px 5px rgba(0,0,0,0.05);
}

/* header */
.header {
    text-align: center;
    padding: 20px;
}

/* input */
.stChatInput {
    position: fixed;
    bottom: 20px;
    left: 50%;
    transform: translateX(-50%);
    max-width: 700px;
    width: 100%;
}

</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown("""
<div class="header">
    <h1 style="color:#ff4d88;">🤰 Pregnancy Chatbot</h1>
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
        st.markdown(f"""
        <div class="user-msg">
            <div class="user-bubble">{msg}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="bot-msg">
            <div class="bot-bubble">{msg}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ================= SIDEBAR =================
st.sidebar.markdown("## 📚 Sumber Data")
st.sidebar.write(f"Jumlah data: {len(df)}")

# 🔥 COLLAPSIBLE
with st.sidebar.expander("💡 Cara Kerja"):
    st.write("""
    - Pertanyaan diubah jadi embedding  
    - Dicari yang paling mirip  
    - Jawaban diambil dari dataset  
    """)

with st.sidebar.expander("🔍 Contoh Data"):
    st.dataframe(df.head())