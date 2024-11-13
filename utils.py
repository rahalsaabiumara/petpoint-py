import json
import numpy as np
import tensorflow as tf
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|@\S+|#\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    return ' '.join(tokens)

# Predict intent
def predict_intent(text, tokenizer, model, label_encoder):
    seq = tokenizer.texts_to_sequences([text])
    seq_padded = pad_sequences(seq, maxlen=33, padding='post')
    pred = model.predict(seq_padded)
    intent_idx = np.argmax(pred, axis=1)[0]
    intent = label_encoder.inverse_transform([intent_idx])[0]
    return intent

# Predict entities
def predict_entities(text, tokenizer, model, idx2tag):
    tokens = word_tokenize(text)
    seq = tokenizer.texts_to_sequences([text])
    seq_padded = pad_sequences(seq, maxlen=33, padding='post')
    pred = model.predict(seq_padded)
    pred_indices = np.argmax(pred[0], axis=-1)
    if isinstance(pred_indices, np.integer):
        pred_indices = [pred_indices]  # Convert to a list if it's a single integer
    pred_tags = [idx2tag.get(idx, 'O') for idx in pred_indices]


    entities = []
    for token, tag in zip(tokens, pred_tags):
        if tag != 'O':
            entities.append((token, tag))
    return entities
