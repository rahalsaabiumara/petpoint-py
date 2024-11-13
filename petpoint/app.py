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

# Inisialisasi
st.title("Chatbot NER dan Intent Classification")

# Load NLTK data
nltk.download('punkt')
nltk.download('stopwords')

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

# Fungsi Load Data dan Tokenizer
@st.cache(allow_output_mutation=True)
def load_resources():
    # Load slang dictionary
    with open('combined_slang_words.txt', 'r', encoding='utf-8') as f:
        slang_dict = json.load(f)
    
    # Load Intent dataset
    with open('ambiguintents3.json', 'r', encoding='utf-8') as f:
        intent_data = json.load(f)
    
    # Initialize stemmer and stopwords
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    stopwords_indonesia = set(nltk.corpus.stopwords.words('indonesian'))
    # Adjust the stopword list by excluding domain-specific words
    custom_stopwords = stopwords_indonesia - {'anjing', 'kucing', 'sakit', 'gejala'}
    
    return slang_dict, intent_data, stemmer, custom_stopwords

# Fungsi Load dan Siapkan Tokenizer
@st.cache(allow_output_mutation=True)
def load_tokenizer():
    # Load data untuk membangun tokenizer
    with open('ambiguner3.json', 'r', encoding='utf-8') as f:
        ner_data = json.load(f)
    with open('ambiguintents3.json', 'r', encoding='utf-8') as f:
        intent_data = json.load(f)
    utterances = []
    intents = []
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    stopwords_indonesia = set(nltk.corpus.stopwords.words('indonesian'))
    custom_stopwords = stopwords_indonesia - {'anjing', 'kucing', 'sakit', 'gejala'}
    slang_dict = {}
    with open('combined_slang_words.txt', 'r', encoding='utf-8') as f:
        slang_dict = json.load(f)
    def preprocess_text(text):
        # Same preprocessing as above
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|@\S+|#\S+', '', text)
        text = re.sub(r'[:;]-?[)D]', '', text)
        text = re.sub(r'[^a-z\s]', '', text)
        tokens = nltk.word_tokenize(text)
        tokens = [slang_dict.get(token, token) for token in tokens]
        tokens = [word for word in tokens if word not in custom_stopwords]
        tokens = [stemmer.stem(word) for word in tokens]
        return ' '.join(tokens)
    for intent_item in intent_data['intents']:
        intent = intent_item['intent']
        for utterance in intent_item['utterances']:
            preprocessed_utterance = preprocess_text(utterance)
            utterances.append(preprocessed_utterance)
            intents.append(intent)
    texts_ner = []
    labels_ner = []
    def align_tokens(original_tokens, preprocessed_tokens):
        alignment = []
        preprocessed_index = 0
        for orig_token in original_tokens:
            orig_token_processed = stemmer.stem(slang_dict.get(orig_token.lower(), orig_token.lower()))
            if preprocessed_index < len(preprocessed_tokens) and orig_token_processed == preprocessed_tokens[preprocessed_index]:
                alignment.append(preprocessed_index)
                preprocessed_index += 1
            else:
                alignment.append(None)
        return alignment
    for item in ner_data:
        text = item['text']
        entities = item['entities']
        preprocessed_text = preprocess_text(text)
        texts_ner.append(preprocessed_text)
    all_texts = utterances + texts_ner
    tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token='<OOV>')
    tokenizer.fit_on_texts(all_texts)
    return tokenizer

# Load resources
slang_dict, intent_data, stemmer, custom_stopwords = load_resources()
tokenizer = load_tokenizer()

# Encode intents
label_encoder = LabelEncoder()
intents = []
for intent_item in intent_data['intents']:
    intent = intent_item['intent']
    for utterance in intent_item['utterances']:
        intents.append(intent)
intent_labels = label_encoder.fit_transform(intents)
num_intent_classes = len(label_encoder.classes_)

# Load models
@st.cache(allow_output_mutation=True)
def load_models():
    # Load model Intent
    model_intent = tf.keras.models.load_model('model_intent.h5')
    
    # Load model NER dengan CRF
    model_ner = tf.keras.models.load_model('model_ner_with_crf.h5', custom_objects={'CRF': CRF, 'ModelWithCRFLoss': ModelWithCRFLoss})
    
    return model_intent, model_ner

model_intent, model_ner = load_models()

# Load tag mappings
@st.cache(allow_output_mutation=True)
def load_tag_mapping():
    # Assuming tag2idx and idx2tag are saved as JSON files
    # If not, recreate them as in the training script
    ner_tags = ['O', 'B-animal', 'I-animal', 'B-condition', 'I-condition', 
                'B-symptom', 'I-symptom', 'B-treatment', 'I-treatment',
                'B-greeting', 'I-greeting', 'B-farewell', 'I-farewell',
                'B-general_conversation', 'I-general_conversation']
    tag2idx = {tag: idx for idx, tag in enumerate(ner_tags)}
    idx2tag = {idx: tag for tag, idx in tag2idx.items()}
    return tag2idx, idx2tag

tag2idx, idx2tag = load_tag_mapping()

# Fungsi untuk Mengubah Indeks ke Tag
def sequences_to_tags(sequences, idx2tag):
    return [[idx2tag.get(idx, 'O') for idx in sequence] for sequence in sequences]

# Fungsi untuk Menghapus Padding
def remove_pad_sequences(sequences, sequences_raw):
    sequences_clean = []
    for seq, seq_raw in zip(sequences, sequences_raw):
        seq_len = len(seq_raw)
        sequences_clean.append(seq[:seq_len])
    return sequences_clean

# Fungsi Prediksi Intent
def predict_intent(text):
    preprocessed_text = preprocess_text(text, slang_dict, stemmer, custom_stopwords)
    seq = tokenizer.texts_to_sequences([preprocessed_text])
    seq_padded = pad_sequences(seq, maxlen=tokenizer.sequences_to_matrix(seq, mode='count').shape[1], padding='post')
    pred = model_intent.predict(seq_padded)
    intent_idx = np.argmax(pred, axis=1)[0]
    intent = label_encoder.inverse_transform([intent_idx])[0]
    return intent

# Fungsi Prediksi Entities
def predict_entities(text):
    preprocessed_text = preprocess_text(text, slang_dict, stemmer, custom_stopwords)
    tokens = nltk.word_tokenize(preprocessed_text)
    seq = tokenizer.texts_to_sequences([preprocessed_text])
    seq_padded = pad_sequences(seq, maxlen=model_ner.input_shape[1], padding='post')
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
    
    # Menghapus padding
    pred_tags_clean = remove_pad_sequences(pred_tags, [tokens])
    
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

# Streamlit UI
st.markdown("""
<style>
.big-font {
    font-size:20px !important;
}
</style>
""", unsafe_allow_html=True)

user_input = st.text_input("Anda:", "")

if st.button("Kirim"):
    if user_input.strip() == "":
        st.write("Silakan masukkan pesan.")
    else:
        response = get_chatbot_response(user_input)
        st.markdown(f"<p class='big-font'><b>Chatbot:</b> {response}</p>", unsafe_allow_html=True)
