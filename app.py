import gradio as gr
import json
import pandas as pd
import re
import random
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

# ================= LOAD DATA =================
with open("chatbot-ibu-hamil.json") as f:
    data = json.load(f)

rows = []
for intent in data['intents']:
    for pattern in intent['patterns']:
        rows.append({
            "question": pattern,
            "answer": intent.get('answer', {}).get('text', intent.get('responses', [""])[0]),
            "category": intent.get('category', {}).get('primary', intent.get('tag', 'unknown')),
            "intent": intent.get('intent', 'general'),
            "risk_level": intent.get('risk_level', 'unknown'),
            "source": intent.get('source', '-')
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
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

embeddings = model.encode(
    df['processed_question'].tolist(),
    convert_to_tensor=True
)

# ================= RULE-BASED =================
rules = {
    "emergency": {
        "patterns": [r'(sesak.*berat|nyeri dada.*berat|pingsan)'],
        "responses": ["🚨 DARURAT! Segera ke IGD!"]
    },
    "greeting": {
        "patterns": [r'\b(halo|hai|hi|hello)\b'],
        "responses": ["👋 Halo! Ada yang bisa saya bantu?"]
    }
}

def check_rules(text):
    for intent, data_rule in rules.items():
        for pattern in data_rule['patterns']:
            if re.search(pattern, text.lower()):
                return random.choice(data_rule['responses'])
    return None

# ================= CHATBOT =================
THRESHOLD = 0.4

def chatbot(user_input):

    rule = check_rules(user_input)
    if rule:
        return {"answer": rule}

    user_input_clean = preprocess(user_input)
    user_emb = model.encode(user_input_clean, convert_to_tensor=True)

    similarity = cos_sim(user_emb, embeddings)

    index = similarity.argmax().item()
    score = similarity.max().item()

    if score < THRESHOLD:
        return {"answer": "Tidak ditemukan jawaban"}

    row = df.iloc[index]

    return {
        "answer": row['answer'],
        "question": row['question'],   
        "category": row.get('category', '-'),
        "confidence": score
    }

# ================= RESPONSE (FIX GRADIO) =================
def respond(message, history):
    result = chatbot(message)

    answer = result["answer"]

    if answer == "Tidak ditemukan jawaban":
        reply = "😅 Maaf, aku belum menemukan jawaban yang sesuai"
    else:
        reply = f"💖 {answer}"

    # 🔥 FORMAT BARU (WAJIB)
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": reply})

    return history, ""

# ================= THEME =================
theme = gr.themes.Soft(
    primary_hue="pink",
    secondary_hue="rose",
    neutral_hue="gray"
)

# ================= UI =================
with gr.Blocks(theme=theme, css="""
body {
    background-color: #fff0f5;
}
.gradio-container {
    max-width: 1000px !important;
    margin: auto;
}
footer {display:none}
""") as demo:

    gr.Markdown("# 🤰 535240102 Pregnancy Chatbot")
    gr.Markdown("Chatbot berbasis data kehamilan untuk membantu menjawab pertanyaan ibu hamil 💖")

    with gr.Row():

        # CHAT
        with gr.Column(scale=3):
            chatbot_ui = gr.Chatbot(height=500, type="messages")  # 🔥 FIX
            msg = gr.Textbox(placeholder="Tanyakan seputar kehamilan...", label="")

        # SUMBER DATA
        with gr.Column(scale=2):

            gr.Markdown("## 📚 Sumber Data")

            gr.Markdown(f"""
Dataset yang digunakan:
- Pregnancy FAQ Dataset (Kaggle)

📊 Jumlah data: **{len(df)}**
📂 Kolom: **{', '.join(df.columns)}**

💡 Cara kerja:
- Pertanyaan diubah jadi embedding
- Dicari yang paling mirip di dataset
- Jawaban diambil dari FAQ terdekat
""")

            gr.Markdown("### 🔍 Contoh Data")
            gr.Dataframe(df.head(5))

    msg.submit(respond, [msg, chatbot_ui], [chatbot_ui, msg])

# ================= RUN =================
if __name__ == "__main__":
    demo.launch()