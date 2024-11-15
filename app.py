import os
import json
import numpy as np
import tensorflow as tf
import nltk
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from tensorflow.keras.models import load_model
from tensorflow_addons.layers import CRF

# Fungsi Setup NLTK
def setup_nltk():
    nltk_data_path = os.path.join(os.getcwd(), 'dataset', 'nltk_data')
    nltk.data.path.append(nltk_data_path)

    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', download_dir=nltk_data_path)

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', download_dir=nltk_data_path)

# Inisialisasi NLTK
setup_nltk()

# Load tokenizer, label encoder, dan mappings
def load_resources():
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

    return tokenizer, label_encoder, tag2idx, idx2tag

# Load models
def load_models():
    model_intent = load_model('models/model_intent')
    model_ner = load_model('models/model_ner_with_crf', custom_objects={'CRF': CRF})
    return model_intent, model_ner

# Preprocessing teks
factory = StemmerFactory()
stemmer = factory.create_stemmer()
stopwords_indonesia = set(nltk.corpus.stopwords.words('indonesian'))
custom_stopwords = stopwords_indonesia - {'anjing', 'kucing', 'sakit', 'gejala'}

def preprocess_text(text, slang_dict=None):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|@\S+|#\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    if slang_dict:
        tokens = [slang_dict.get(token, token) for token in tokens]
    tokens = [word for word in tokens if word not in custom_stopwords]
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

def is_valid_input(text):
    return bool(re.match(r'^[a-zA-Z0-9\s,!?]+$', text))

def sequences_to_tags(sequences, idx2tag):
    if isinstance(sequences, np.ndarray) and sequences.ndim == 1:
        sequences = [sequences]
    return [[idx2tag.get(str(idx), 'O') for idx in sequence] for sequence in sequences]

def predict_intent(text, tokenizer, model_intent, label_encoder, max_len):
    preprocessed_text = preprocess_text(text)
    seq = tokenizer.texts_to_sequences([preprocessed_text])
    seq_padded = pad_sequences(seq, maxlen=max_len, padding='post')
    pred = model_intent.predict(seq_padded)
    intent_idx = np.argmax(pred, axis=1)[0]
    return label_encoder.inverse_transform([intent_idx])[0]

def predict_entities(text, tokenizer, model_ner, idx2tag, max_len_ner):
    preprocessed_text = preprocess_text(text)
    tokens = nltk.word_tokenize(preprocessed_text)
    seq = tokenizer.texts_to_sequences([preprocessed_text])
    seq_padded = pad_sequences(seq, maxlen=max_len_ner, padding='post')
    pred = model_ner.predict(seq_padded)
    pred_tags = sequences_to_tags(pred[0], idx2tag)
    return [(token, tag) for token, tag in zip(tokens, pred_tags[0]) if tag != 'O']

# Fungsi respons chatbot dengan penyimpanan konteks
def get_chatbot_response(user_input, tokenizer, model_intent, model_ner, label_encoder, idx2tag, max_len, max_len_ner, intent_data):
    from flask import session

    if not is_valid_input(user_input):
        return "Maaf, input tidak valid."

    user_input = preprocess_text(user_input)

    # Pengecekan manual untuk jenis hewan
    if 'anjing' in user_input:
        session['animal'] = 'anjing'
    elif 'kucing' in user_input:
        session['animal'] = 'kucing'

    # Cek apakah jenis hewan sudah tersimpan di session
    animal = session.get('animal', None)
    if not animal:
        return "Gejala terdeteksi. Mohon konfirmasi apakah hewan Anda adalah anjing atau kucing?"

    # Prediksi intent
    intent = predict_intent(user_input, tokenizer, model_intent, label_encoder, max_len)
    if intent == "unknown":
        return "Maaf, saya masih dalam tahap belajar."

    # Prediksi entitas
    entities = predict_entities(user_input, tokenizer, model_ner, idx2tag, max_len_ner)
    entities_dict = {tag.split('-')[-1]: [] for _, tag in entities}
    for token, tag in entities:
        label = tag.split('-')[-1]
        entities_dict[label].append(token)

    # Jika entitas `animal` tidak ditemukan, gunakan konteks dari session
    if not entities_dict.get('animal'):
        entities_dict['animal'] = [animal]

    # Cek apakah jenis hewan valid
    if animal not in ['anjing', 'kucing']:
        return f"Saat ini, saya hanya dapat membantu anjing dan kucing. Anda menyebutkan: {animal}."

    # Temukan intent dan respons berdasarkan data intent
    intent_item = next((item for item in intent_data.get("intents", []) if item["conditions"][0] == intent), None)
    if intent_item:
        response = np.random.choice(intent_item.get("responses", ["Saya tidak memiliki saran saat ini."]))
        return f"{animal} Anda mengalami penyakit {intent}. {response}"

    return "Maaf, saya tidak mengerti."
