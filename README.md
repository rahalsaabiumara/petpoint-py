# Petpoint Chatbot
PetPoint Chatbot API adalah aplikasi chatbot yang dikembangkan menggunakan Flask dan TensorFlow. Chatbot ini dirancang untuk membantu pengguna dalam mengidentifikasi gejala penyakit pada hewan peliharaan, terutama anjing dan kucing, melalui analisis teks.

# Fitur Utama
1.**Prediksi Intent: Mengidentifikasi maksud dari input pengguna, seperti deteksi gejala.**
2.**Named Entity Recognition (NER): Mengenali entitas penting dalam teks, seperti jenis hewan dan gejala.**
3.**Konteks Percakapan: Mempertahankan konteks percakapan menggunakan session Flask.**
4.**RESTful API: Endpoint yang mudah digunakan dan diintegrasikan dengan aplikasi web atau mobile.**

#Teknologi yang Digunakan
1.**Flask: Framework web untuk API.**
2.**TensorFlow: Framework deep learning untuk model intent dan NER.**
3.**NLTK: Library NLP untuk preprocessing teks.**
4.**Sastrawi: Library untuk stemming bahasa Indonesia.**

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![API Status](https://img.shields.io/badge/API-available-blue)
![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

#Struktur Proyek
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
```
#Instalasi
1. **Clone Repository**
```bash
git clone https://github.com/username/petpoint-py.git
cd petpoint-py
```
2. **Buat Virtual Environment dan Install Dependensi**
```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

#Cara Menjalankan API
Jalankan perintah berikut untuk memulai API Flask:

```bash
python api.py
```
API akan tersedia di http://127.0.0.1:5000.

#Penggunaan API
1. Health Check
Endpoint: /health
Method: GET
Deskripsi: Mengecek status API.
Contoh:
```bash
curl http://127.0.0.1:5000/health
```
Respons:
```bash
{
  "status": "API is running"
}
```
2. Reset Session
Endpoint: /reset
Method: GET
Deskripsi: Menghapus konteks percakapan untuk memulai percakapan baru.
Contoh:
```bash
curl http://127.0.0.1:5000/reset
```
Respons:
```
{
  "status": "Context reset"
}
```
3. Chat Endpoint
Endpoint: /chat
Method: POST
Deskripsi: Mengirim pesan teks ke chatbot.
Contoh:
```bash
curl -X POST http://127.0.0.1:5000/chat \
-H "Content-Type: application/json" \
-d '{"message": "Anjing saya muntah dan tidak mau makan"}'
```
Respons:
```
{
  "response": "Anjing Anda mengalami penyakit Gastroenteritis. Terapi cairan, Antiemetik, Antibiotik"
}
```
#Tips Penggunaan
Gunakan endpoint /reset sebelum memulai percakapan baru untuk menghapus konteks yang tersimpan.
