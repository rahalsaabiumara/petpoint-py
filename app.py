import json
import numpy as np
import tensorflow as tf
import nltk
import re
import streamlit as st
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from tensorflow_addons.text.crf import crf_decode
from tensorflow.keras.models import load_model

# Load NLTK data
nltk.data.path.append('dataset/nltk_data')
nltk.download('punkt', download_dir='dataset/nltk_data')
nltk.download('stopwords', download_dir='dataset/nltk_data')

# Load tokenizer, label encoder, and mappings
with open('models/tokenizer.json', 'r', encoding='utf-8') as f:
    tokenizer_data = json.load(f)
tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_data)

with open('models/label_encoder.json', 'r', encoding='utf-8') as f:
    label_classes = json.load(f)
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(label_classes)

with open('models/tag2idx.json', 'r', encoding='utf-8') as f:
    tag2idx = json.load(f)
with open('models/idx2tag.json', 'r', encoding='utf-8') as f:
    idx2tag = json.load(f)

# Load models
model_intent = load_model('models/model_intent')
model_ner = load_model('models/model_ner_with_crf', compile=False)

# Load datasets
with open('dataset/revised_intents_dataset.json', 'r', encoding='utf-8') as f:
    intent_data = json.load(f)

factory = StemmerFactory()
stemmer = factory.create_stemmer()
stopwords_indonesia = set(nltk.corpus.stopwords.words('indonesian'))

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|@\S+|#\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords_indonesia]
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

# Prediction functions
def predict_intent(text):
    preprocessed_text = preprocess_text(text)
    seq = tokenizer.texts_to_sequences([preprocessed_text])
    seq_padded = pad_sequences(seq, maxlen=128, padding='post')
    pred = model_intent.predict(seq_padded)
    intent_idx = np.argmax(pred, axis=1)[0]
    intent = label_encoder.inverse_transform([intent_idx])[0]
    return intent

def sequences_to_tags(sequences):
    result = []
    for sequence in sequences:
        tag_sequence = [idx2tag.get(str(int(idx)), 'O') for idx in sequence]
        result.append(tag_sequence)
    return result

def predict_entities(text):
    preprocessed_text = preprocess_text(text)
    tokens = nltk.word_tokenize(preprocessed_text)
    seq = tokenizer.texts_to_sequences([preprocessed_text])
    seq_padded = pad_sequences(seq, maxlen=128, padding='post')
    pred = model_ner.predict(seq_padded)
    ner_preds_labels = pred[0]
    pred_tags = sequences_to_tags(ner_preds_labels)

    entities = []
    for token, tag in zip(tokens, pred_tags[0]):
        if tag != 'O':
            entities.append((token, tag))
    return entities

def get_chatbot_response(user_input):
    intent = predict_intent(user_input)
    entities = predict_entities(user_input)

    entities_dict = {}
    for token, tag in entities:
        label = tag.split('-')[-1]
        if label in entities_dict:
            entities_dict[label].append(token)
        else:
            entities_dict[label] = [token]

    intent_data_list = intent_data.get("intents", [])
    intent_data_item = next((item for item in intent_data_list if "conditions" in item and item["conditions"][0] == intent), None)

    if intent_data_item:
        response = intent_data_item.get("responses", ["Maaf, tidak ada saran yang tersedia."])
        response_text = f"Hewan Anda mungkin mengalami penyakit '{intent}'. Saran tindakan: {response[0]}."

        if 'symptom' in entities_dict:
            symptoms = ', '.join(entities_dict.get('symptom', []))
            response_text += f" Gejala yang terdeteksi: {symptoms}."
        if 'animal' in entities_dict:
            animal = ', '.join(entities_dict.get('animal', []))
            response_text += f" Hewan: {animal}."
        return response_text
    else:
        return "Maaf, saya tidak mengerti. Bisa dijelaskan lebih lanjut?"

# Streamlit UI
st.title("Chatbot Kesehatan Hewan")
st.write("Selamat datang di asisten kesehatan hewan! Silakan masukkan gejala yang dialami hewan Anda.")

user_input = st.text_input("Anda:")
if user_input:
    response = get_chatbot_response(user_input)
    st.write(f"Chatbot: {response}")
