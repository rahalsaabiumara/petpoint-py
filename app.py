# app.py

import os
import streamlit as st
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import nltk
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from tf2crf import CRF, ModelWithCRFLoss
import random
import pandas as pd

# Menambahkan path ke nltk_data
nltk_data_path = os.path.join(os.path.dirname(__file__), 'dataset', 'nltk_data')
nltk.data.path.append(nltk_data_path)

# Memastikan resource NLTK sudah ada, jika belum, unduh secara otomatis
try:
    nltk.corpus.stopwords.words('indonesian')
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_path)

# Inisialisasi Streamlit
st.title("Chatbot Medis Hewan Anjing dan Kucing")

# Fungsi Preprocessing
def preprocess_text(text, slang_dict, stemmer, custom_stopwords):
    # Lowercase
    text = text.lower()
    # Remove URLs, mentions, and hashtags
    text = re.sub(r'http\S+|www\S+|@\S+|#\S+', '', text)
    # Remove or handle emoticons
    text = re.sub(r'[:;]-?[)D]', '', text)  # Simplified emoticon removal
    # Remove punctuation and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenize
    tokens = nltk.word_tokenize(text)
    # Normalize slang
    tokens = [slang_dict.get(token, token) for token in tokens]
    # Remove stopwords
    tokens = [word for word in tokens if word not in custom_stopwords]
    # Stemming
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

# Fungsi Load Resources
@st.cache_resource
def load_resources():
    # Load slang dictionary
    with open('dataset/combined_slang_words.txt', 'r', encoding='utf-8') as f:
        slang_dict = json.load(f)

    # Load Intent dataset
    with open('dataset/intents_dataset.json', 'r', encoding='utf-8') as f:
        intent_data = json.load(f)

    # Initialize stemmer dan stopwords
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    stopwords_indonesia = set(nltk.corpus.stopwords.words('indonesian'))
    custom_stopwords = stopwords_indonesia - {'anjing', 'kucing', 'sakit', 'gejala'}

    return slang_dict, intent_data, stemmer, custom_stopwords

# Fungsi Load Tokenizer
@st.cache_resource
def load_tokenizer():
    with open('models/tokenizer.json', 'r', encoding='utf-8') as f:
        tokenizer_json = json.load(f)
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)

    with open('models/tag2idx.json', 'r', encoding='utf-8') as f:
        tag2idx = json.load(f)
    with open('models/idx2tag.json', 'r', encoding='utf-8') as f:
        idx2tag = json.load(f)

    return tokenizer, tag2idx, idx2tag

# Fungsi Load Label Encoder
@st.cache_resource
def load_label_encoder():
    with open('models/label_encoder.json', 'r', encoding='utf-8') as f:
        label_encoder_classes = json.load(f)
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(label_encoder_classes)
    return label_encoder

# Fungsi Load Models
@st.cache_resource
def load_models():
    # Load model Intent
    model_intent = tf.keras.models.load_model('models/model_intent')

    # Load model NER dengan CRF
    model_ner = tf.keras.models.load_model('models/model_ner_with_crf', custom_objects={'CRF': CRF, 'ModelWithCRFLoss': ModelWithCRFLoss})

    return model_intent, model_ner

# Load semua resources
slang_dict, intent_data, stemmer, custom_stopwords = load_resources()
tokenizer, tag2idx, idx2tag = load_tokenizer()
label_encoder = load_label_encoder()
model_intent, model_ner = load_models()

# Mendefinisikan max_len berdasarkan model
max_len = model_intent.input_shape[1]
# Karena model_ner adalah ModelWithCRFLoss, yang tidak memiliki input_shape, tetapkan max_len_ner secara manual
max_len_ner = max_len  # Pastikan ini sesuai dengan yang digunakan saat pelatihan

# Fungsi Mengubah Indeks ke Tag
def sequences_to_tags(sequences, idx2tag):
    return [[idx2tag.get(str(idx), 'O') for idx in sequence] for sequence in sequences]

# Fungsi Prediksi Intent
def predict_intent(text):
    preprocessed_text = preprocess_text(text, slang_dict, stemmer, custom_stopwords)
    seq = tokenizer.texts_to_sequences([preprocessed_text])
    seq_padded = pad_sequences(seq, maxlen=max_len, padding='post')
    pred = model_intent.predict(seq_padded)
    intent_idx = np.argmax(pred, axis=1)[0]
    intent = label_encoder.inverse_transform([intent_idx])[0]
    return intent

# Fungsi Prediksi Entitas
def predict_entities(text):
    preprocessed_text = preprocess_text(text, slang_dict, stemmer, custom_stopwords)
    tokens = nltk.word_tokenize(preprocessed_text)
    seq = tokenizer.texts_to_sequences([preprocessed_text])
    seq_padded = pad_sequences(seq, maxlen=max_len_ner, padding='post')
    pred = model_ner.predict(seq_padded)
    ner_preds_labels = pred[0]

    # Pastikan ner_preds_labels adalah list of sequences
    if isinstance(ner_preds_labels, (list, np.ndarray)):
        if isinstance(ner_preds_labels[0], (int, np.integer)):
            ner_preds_labels = [ner_preds_labels]
        elif isinstance(ner_preds_labels[0], (list, np.ndarray)):
            pass  # Sudah benar
        else:
            raise ValueError("Unexpected element type in ner_preds_labels")
    else:
        raise ValueError("Unexpected type for ner_preds_labels")

    pred_tags = sequences_to_tags(ner_preds_labels, idx2tag)

    # Menghapus padding berdasarkan teks asli
    if len(pred_tags[0]) > len(tokens):
        pred_tags_clean = [pred_tags[0][:len(tokens)]]
    else:
        pred_tags_clean = pred_tags

    entities = []
    for token, tag in zip(tokens, pred_tags_clean[0]):
        if tag != 'O':
            entities.append((token, tag))
    return entities

# Fungsi untuk Mendapatkan Respon Chatbot
def get_chatbot_response(user_input):
    # Predict intent
    intent = predict_intent(user_input)
    # Predict entities
    entities = predict_entities(user_input)
    entities_dict = {}
    for token, tag in entities:
        label = tag.split('-')[-1]
        if label in entities_dict:
            entities_dict[label].append(token)
        else:
            entities_dict[label] = [token]
    # Find the appropriate response
    intent_data_item = next((item for item in intent_data['intents'] if item['intent'] == intent), None)
    if intent_data_item:
        response_template = random.choice(intent_data_item['responses'])
        # Replace placeholders in the response
        for entity_label in ['animal', 'condition', 'symptom', 'treatment']:
            placeholder = '{' + entity_label + '}'
            if placeholder in response_template:
                value = ', '.join(entities_dict.get(entity_label, ['tidak disebutkan']))
                response_template = response_template.replace(placeholder, value)
        return response_template
    else:
        return "Maaf, saya tidak mengerti. Bisa dijelaskan lebih lanjut?"

# Fungsi untuk Menampilkan Entitas dalam Tabel
def display_entities(entities):
    if entities:
        df = pd.DataFrame(entities, columns=["Token", "Tag"])
        st.table(df)
    else:
        st.write("Tidak ada entitas yang diprediksi.")

# Streamlit UI
user_input = st.text_input("Anda:", "")

if st.button("Kirim"):
    if user_input.strip() == "":
        st.write("Silakan masukkan pesan.")
    else:
        response = get_chatbot_response(user_input)
        st.markdown(f"**Chatbot:** {response}")

        # Menampilkan entitas yang diprediksi
        entities = predict_entities(user_input)
        if entities:
            st.subheader("Entitas yang Diprediksi:")
            display_entities(entities)
        else:
            st.write("Tidak ada entitas yang diprediksi.")
