import pickle
import random
import re

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.layers import Layer
from tensorflow.keras.preprocessing.sequence import pad_sequences

intent_animal_mapping = {
    "Melaporkan Hewan Terlantar": ["kucing", "anjing"],
    "Mendiagnosis Gejala": ["kucing", "anjing"],
    "Rekomendasi Penanganan Awal": ["kucing", "anjing"],
    "Konfirmasi Laporan": ["kucing", "anjing"],
    "Tindak Lanjut Laporan": ["kucing", "anjing"],
    "Rekomendasi Tindakan": ["kucing", "anjing"],
}


@tf.keras.utils.register_keras_serializable(package="Custom")
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="att_bias", shape=(1,), initializer="zeros", trainable=True
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        e = tf.squeeze(tf.tensordot(x, self.W, axes=1), axis=-1) + self.b
        alpha = tf.nn.softmax(e)
        alpha = tf.expand_dims(alpha, axis=-1)
        context = x * alpha
        return tf.reduce_sum(context, axis=1)


model_intent = None
model_ner_crf = None
tokenizer = None
label_encoder = None
ner_label_encoder = None
transition_params = None
ner_label_decoder = None
vectorizer = None
tfidf_matrix = None
df_utterances = None

max_seq_length = 9


def load_resources():
    global model_intent, model_ner_crf, tokenizer, label_encoder, ner_label_encoder, transition_params, ner_label_decoder, vectorizer, tfidf_matrix, df_utterances

    model_intent = tf.keras.models.load_model(
        "app/models/model_intent.keras",
        compile=False,
        custom_objects={"Custom>AttentionLayer": AttentionLayer},
    )

    model_ner_crf = tf.keras.models.load_model(
        "app/models/model_ner_crf.keras",
        compile=False,
        custom_objects={"Custom>AttentionLayer": AttentionLayer},
    )

    with open("app/encoders/tokenizer.pickle", "rb") as handle:
        tokenizer = pickle.load(handle)
    with open("app/encoders/label_encoder.pickle", "rb") as handle:
        label_encoder = pickle.load(handle)
    with open("app/encoders/ner_label_encoder.pickle", "rb") as handle:
        ner_label_encoder = pickle.load(handle)
        ner_label_decoder = {v: k for k, v in ner_label_encoder.items()}

    with open("app/encoders/transition_params.pickle", "rb") as handle:
        transition_params_np = pickle.load(handle)
        transition_params = tf.Variable(transition_params_np, dtype=tf.float32)

    with open("app/data/vectorizer.pickle", "rb") as handle:
        vectorizer = pickle.load(handle)
    with open("app/data/tfidf_matrix.pickle", "rb") as handle:
        tfidf_matrix = pickle.load(handle)

    df_utterances = pd.read_pickle("app/data/df_utterances.pkl")

    globals()["ner_label_decoder"] = ner_label_decoder


load_resources()

stop_words = set()
with open("app/data/stopword_list_tala.txt", "r", encoding="utf-8") as f:
    for line in f:
        stop_words.add(line.strip().lower())

factory = StemmerFactory()
stemmer = factory.create_stemmer()


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def preprocess_text(text):
    text = clean_text(text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [stemmer.stem(w) for w in tokens]
    return " ".join(tokens)


def predict_intent_fn(text, threshold=0.5):
    text_clean = preprocess_text(text)
    seq = tokenizer.texts_to_sequences([text_clean])
    padded = pad_sequences(seq, maxlen=max_seq_length, padding="post")
    pred = model_intent.predict(padded)
    # pred adalah array shape (1, num_classes)
    max_prob = np.max(pred)
    predicted_label = np.argmax(pred, axis=1)[0]

    if max_prob < threshold:
        # Intent tidak yakin
        return None
    else:
        return label_encoder.inverse_transform([predicted_label])[0]


def predict_entities_fn(text):
    text_clean = preprocess_text(text)
    seq = tokenizer.texts_to_sequences([text_clean])
    padded = pad_sequences(seq, maxlen=max_seq_length, padding="post")
    logits = model_ner_crf.predict(padded)
    viterbi_seq, _ = tfa.text.crf_decode(
        logits, transition_params, tf.fill([tf.shape(logits)[0]], max_seq_length)
    )
    tokens = text_clean.split()
    entities = []
    for idx, label_id in enumerate(viterbi_seq[0][: len(tokens)]):
        label = ner_label_decoder[label_id.numpy()]
        if label != "O":
            entity_type = label.split("-")[1].lower()
            entities.append({"entity": entity_type, "value": tokens[idx]})
    return entities


def adjust_intent(intent, entities):
    predicted_animals = intent_animal_mapping.get(intent, None)
    entity_animals = [
        ent["value"].lower() for ent in entities if ent["entity"] == "animal"
    ]
    if entity_animals and predicted_animals:
        user_animal = entity_animals[0]
        if user_animal not in predicted_animals:
            for i_name, animals in intent_animal_mapping.items():
                if user_animal in animals:
                    intent = i_name
                    break
            else:
                intent = None
    return intent


def get_default_response():
    default_responses = [
        "Maaf, saya belum bisa menjawab pertanyaan Anda.",
        "Mohon diperjelas, saya belum mengerti konteksnya.",
        "Silakan berikan informasi lebih detail.",
        "Maaf, saya hanya diprogram untuk menjawab mengenai kucing dan anjing.",
    ]
    return random.choice(default_responses)


def get_best_response(user_input):
    user_query_clean = preprocess_text(user_input)
    q_vec = vectorizer.transform([user_query_clean])
    sims = cosine_similarity(q_vec, tfidf_matrix)
    best_idx = sims[0].argmax()
    best_score = sims[0][best_idx]
    if best_score < 0.5:
        return None
    else:
        return df_utterances.iloc[best_idx]["bot_response"]


def get_response(user_input, intent=None, entities=None):
    # Jika intent adalah None, artinya model tidak yakin
    if intent is None:
        # Coba retrieval
        best_resp = get_best_response(user_input)
        if best_resp:
            return best_resp
        else:
            return get_default_response()
    else:
        # Jika intent dikenali
        if intent == "Mendiagnosis Gejala":
            return "Berdasarkan gejalanya, kemungkinan ada masalah kesehatan. Saya sarankan anda menggunakan fitur emergency untuk menemukan komunitas terdekat."
        elif intent == "Rekomendasi Penanganan Awal":
            return "Sebaiknya Anda menggunakan fitur emergency agar menemukan komunitas terdekat untuk membantu Anda menangani ini."
        else:
            # Intent dikenal tapi bukan dua intent khusus di atas.
            # Gunakan retrieval sebagai fallback.
            best_resp = get_best_response(user_input)
            if best_resp:
                return best_resp
            else:
                return get_default_response()


def chatbot_response(user_input):
    # Coba prediksi intent dengan threshold
    intent = predict_intent_fn(user_input, threshold=0.5)
    entities = predict_entities_fn(user_input)
    adjusted_intent = None
    if intent is not None:
        adjusted_intent = adjust_intent(intent, entities)
    # Jika adjusted_intent None atau intent None (model tidak yakin), go fallback
    return get_response(user_input, adjusted_intent, entities)
