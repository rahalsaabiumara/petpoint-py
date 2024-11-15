# petpoint-py

```bash
petpoint-py/
├── app.py                     # Logika utama chatbot dan fungsi Flask
├── api.py                     # API Flask untuk mengakses chatbot
├── dataset/
│   ├── nltk_data/             # Data NLTK seperti stopwords
│   ├── revised_intents_dataset.json # Dataset intent chatbot
├── models/
│   ├── model_intent/          # Model TensorFlow untuk intent classification
│   ├── model_ner_with_crf/    # Model TensorFlow untuk NER dengan CRF
│   ├── tokenizer.json         # Tokenizer untuk preprocessing teks
│   ├── label_encoder.json     # Label encoder untuk intent
│   ├── idx2tag.json           # Mapping indeks ke tag NER
│   └── tag2idx.json           # Mapping tag NER ke indeks
├── requirements.txt           # Daftar dependensi Python
