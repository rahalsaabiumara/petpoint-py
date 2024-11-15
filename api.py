from flask import Flask, request, jsonify, session
from flask_cors import CORS
from app import (
    setup_nltk, load_resources, load_models,
    get_chatbot_response
)
import os
import json
import nltk

app = Flask(__name__)
app.secret_key = 'supersecretkey'
CORS(app, supports_credentials=True)

setup_nltk()
tokenizer, label_encoder, tag2idx, idx2tag = load_resources()
model_intent, model_ner = load_models()
max_len = model_intent.input_shape[1]
max_len_ner = 33

with open('dataset/revised_intents_dataset.json', 'r', encoding='utf-8') as f:
    intent_data = json.load(f)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get("message", "")
    if not user_input or not isinstance(user_input, str):
        return jsonify({"error": "Input tidak valid."}), 400

    response_text = get_chatbot_response(
        user_input, tokenizer, model_intent, model_ner,
        label_encoder, idx2tag, max_len, max_len_ner, intent_data
    )

    return jsonify({"response": response_text})

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "API is running"}), 200

@app.route('/reset', methods=['GET'])
def reset_session():
    session.clear()
    return jsonify({"status": "Context reset"}), 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
