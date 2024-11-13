import json
import numpy as np
import tensorflow as tf
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk

# Unduh dataset NLTK yang diperlukan
nltk.download('punkt')
nltk.download('stopwords')

# Fungsi untuk memuat kamus slang dari file teks
def load_slang_dict(filepath):
    slang_dict = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 2:
                slang_dict[parts[0].strip()] = parts[1].strip()
    return slang_dict

# Memuat kamus slang dari file teks
slang_dict = load_slang_dict('models/combined_slang_words.txt')

# Inisialisasi Stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()
stopwords_indonesia = set(nltk.corpus.stopwords.words('indonesian'))
custom_stopwords = stopwords_indonesia - {'anjing', 'kucing', 'sakit', 'gejala'}

# Fungsi preprocessing teks
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|@\S+|#\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [slang_dict.get(token, token) for token in tokens]
    tokens = [word for word in tokens if word not in custom_stopwords]
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

def predict_intent(text, tokenizer, model, label_encoder):
    seq = tokenizer.texts_to_sequences([text])
    seq_padded = pad_sequences(seq, maxlen=33, padding='post')
    pred = model.predict(seq_padded)
    intent_idx = np.argmax(pred, axis=1)[0]
    intent = label_encoder.inverse_transform([intent_idx])[0]
    return intent

def predict_entities(text, tokenizer, model, idx2tag):
    preprocessed_text = preprocess_text(text)
    tokens = word_tokenize(preprocessed_text)
    seq = tokenizer.texts_to_sequences([preprocessed_text])
    seq_padded = pad_sequences(seq, maxlen=33, padding='post')
    pred = model.predict(seq_padded)
    pred_indices = np.argmax(pred[0], axis=-1)
    if isinstance(pred_indices, np.integer):
        pred_indices = [pred_indices]
    pred_tags = [idx2tag.get(idx, 'O') for idx in pred_indices]

    entities = []
    for token, tag in zip(tokens, pred_tags):
        if tag != 'O':
            entities.append((token, tag))
    return entities
