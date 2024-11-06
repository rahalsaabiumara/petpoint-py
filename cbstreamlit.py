import streamlit as st
import pandas as pd
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process

# Load datasets
disease_df = pd.read_csv('disease_classification.csv')
conversation_df = pd.read_csv('general_conversation.csv')

# Placeholder TensorFlow model (can be replaced with an actual trained model)
# For the sake of this example, we'll use a dummy model
class SimpleModel(tf.keras.Model):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(3, activation='softmax')  # assuming 3 output classes

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

# Function for chatbot !emergency
def emergency_chatbot(hewan, gejala_inputs):
    gejala_inputs = [gejala.lower() for gejala in gejala_inputs]
    filtered_df = disease_df[disease_df['Nama Hewan'].str.lower() == hewan.lower()]
    if filtered_df.empty:
        return f"Tidak ada informasi penyakit untuk {hewan}."
    gejala_list = filtered_df['Gejala'].tolist()
    penyakit_list = filtered_df['Nama Penyakit'].tolist()
    penanganan_list = filtered_df['Penanganan Pertama'].tolist()
    matching_scores = []
    for input_symptom in gejala_inputs:
        closest_match, score = process.extractOne(input_symptom, gejala_list)
        if score >= 60:
            matching_scores.append((closest_match, score))
    if not matching_scores:
        return "Gejala yang Anda sebutkan tidak ditemukan dalam database kami."
    closest_match_idx = max(range(len(gejala_list)), key=lambda i: sum(score for match, score in matching_scores if match in gejala_list[i]))
    penyakit_terkait = penyakit_list[closest_match_idx]
    penanganan_terkait = penanganan_list[closest_match_idx]
    return f"Berdasarkan gejala yang Anda sebutkan, {hewan} Anda mungkin mengalami {penyakit_terkait}. Penanganan pertama: {penanganan_terkait}"

# Function for chatbot !consultation
def consultation_chatbot(input_user):
    input_list = conversation_df['Input'].tolist()
    output_list = conversation_df['Output'].tolist()

    # Dummy model for the sake of example
    model = SimpleModel()

    # Preprocess the input for the model
    vectorizer = CountVectorizer().fit_transform(input_list + [input_user])
    vectors = vectorizer.toarray()

    # Use the TensorFlow model (this is just a placeholder for an actual implementation)
    input_tensor = tf.convert_to_tensor([vectors[-1]], dtype=tf.float32)
    predictions = model(input_tensor)
    closest_match_idx = tf.argmax(predictions, axis=1).numpy()[0]

    return output_list[closest_match_idx]

# Main Streamlit app
def main():
    st.title("Chatbot App")
    # CSS to style chat bubbles
    st.markdown("""
    <style>
    .chat-container {
        height: 400px;
        overflow-y: auto;
        border: 1px solid #ccc;
        padding: 10px;
    }
    .user-msg {
        background-color: #DCF8C6;
        border-radius: 10px;
        padding: 10px;
        margin: 5px;
        text-align: right;
        float: right;
        clear: both;
        display: inline-block;
    }
    .bot-msg {
        background-color: #F1F0F0;
        border-radius: 10px;
        padding: 10px;
        margin: 5px;
        text-align: left;
        float: left;
        clear: both;
        display: inline-block;
    }
    .clearfix {
        clear: both;
    }
    </style>
    """, unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = [
            ("bot", "Halo! Saya di sini untuk membantu Anda untuk mengatasi masalah pada hewan yang anda temui. Anda dapat menggunakan:\n"
                    "- !emergency untuk meminta bantuan mengenai pertolongan pertama terkait gejala penyakit pada hewan peliharaan Anda.\n"
                    "- !consultation untuk konsultasi umum.\n"
                    "- exit untuk keluar dari percakapan ini.\n"
                    "Silakan masukkan perintah Anda.")
        ]

    chat_placeholder = st.empty()

    def display_chat(messages):
        chat_html = '<div class="chat-container">'
        for sender, msg in messages:
            if sender == "user":
                chat_html += f'<div class="user-msg">{msg}</div><div class="clearfix"></div>'
            else:
                chat_html += f'<div class="bot-msg">{msg}</div><div class="clearfix"></div>'
        chat_html += '</div>'
        chat_placeholder.markdown(chat_html, unsafe_allow_html=True)

    if "current_step" not in st.session_state:
        st.session_state.current_step = None
        st.session_state.hewan = None
        st.session_state.gejala_inputs = []

    user_input = st.text_input("Anda: ")

    if st.button("Kirim") and user_input:
        st.session_state.messages.append(("user", user_input))

        if user_input.lower() in ['kucing', 'anjing']:
            st.session_state.current_step = "gejala"
            st.session_state.hewan = user_input.lower()
            response = f"Silakan sebutkan gejala yang dialami oleh {st.session_state.hewan} Anda (atau ketik 'tidak ada' jika selesai):"
        elif st.session_state.current_step == "gejala":
            if user_input.lower() == "tidak ada":
                response = emergency_chatbot(st.session_state.hewan, st.session_state.gejala_inputs)
                st.session_state.current_step = None
                st.session_state.hewan = None
                st.session_state.gejala_inputs = []
            else:
                st.session_state.gejala_inputs.append(user_input.lower())
                response = "Apakah ada gejala lainnya? Jika ada, beritahu saya apa gejalanya. Jika dirasa tidak, ketik 'tidak ada'"
        elif user_input.lower().startswith('!emergency'):
            response = "Silakan masukkan nama hewan (kucing/anjing):"
        elif user_input.lower().startswith('!consultation'):
            pertanyaan = user_input.split(' ', 1)[1] if ' ' in user_input else ""
            response = consultation_chatbot(pertanyaan) if pertanyaan else "Tolong ajukan pertanyaan Anda."
        else:
            response = "Kata kunci tidak dikenali. Gunakan !emergency atau !consultation."

        st.session_state.messages.append(("bot", response))
        display_chat(st.session_state.messages)

    display_chat(st.session_state.messages)

if __name__ == "__main__":
    main()
