import streamlit as st
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tf2crf import ModelWithCRFLoss
from nltk.tokenize import word_tokenize
import zipfile
import os

# Import fungsi dari utils
from utils import preprocess_text, predict_intent, predict_entities

# Memuat model dan file JSON
@st.cache_resource
def load_models():
    # Memuat tokenizer
    with open("models/tokenizer.json", "r", encoding="utf-8") as f:
        tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(json.load(f))

    # Memuat label encoder
    with open("models/label_encoder.json", "r", encoding="utf-8") as f:
        label_encoder = LabelEncoder()
        label_encoder.classes_ = np.array(json.load(f))

    # Memuat tag2idx dan idx2tag
    with open("models/tag2idx.json", "r", encoding="utf-8") as f:
        tag2idx = json.load(f)
    with open("models/idx2tag.json", "r", encoding="utf-8") as f:
        idx2tag = json.load(f)

    # Memuat model intent
    model_intent = tf.keras.models.load_model("models/model_intent")

    # Memuat model NER dengan CRF
    base_model_ner = tf.keras.models.load_model("models/model_ner_with_crf", compile=False)
    model_ner = ModelWithCRFLoss(base_model_ner)
    model_ner.compile()

    return tokenizer, label_encoder, tag2idx, idx2tag, model_intent, model_ner

# Memuat model dan tokenizer
tokenizer, label_encoder, tag2idx, idx2tag, model_intent, model_ner = load_models()

# Fungsi untuk memprediksi respons chatbot
def get_chatbot_response(user_input):
    # Preprocess input
    preprocessed_text = preprocess_text(user_input)
    # Prediksi intent
    intent = predict_intent(preprocessed_text, tokenizer, model_intent, label_encoder)
    # Prediksi entitas
    entities = predict_entities(preprocessed_text, tokenizer, model_ner, idx2tag)
    
    # Membuat respons berdasarkan intent dan entitas
    response = f"Intent terdeteksi: {intent}\n"
    if entities:
        response += "Entitas yang ditemukan:\n"
        for token, tag in entities:
            response += f" - {token}: {tag}\n"
    else:
        response += "Tidak ada entitas yang ditemukan."

    return response

# Streamlit UI
st.title("Chatbot Kesehatan Hewan Peliharaan")
st.write("Masukkan pertanyaan Anda tentang kesehatan hewan peliharaan.")

user_input = st.text_input("Input Anda:")

if st.button("Kirim"):
    if user_input:
        response = get_chatbot_response(user_input)
        st.text_area("Respons Chatbot:", value=response, height=200)
    else:
        st.warning("Silakan masukkan pertanyaan.")

