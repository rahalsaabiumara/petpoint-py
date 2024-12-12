# 🐾 Panduan Menjalankan Chatbot Hewan dengan Docker Compose

## 📋 Daftar Isi
- [🔧 Prasyarat](#prasyarat)
- [📥 Mengkloning Repository](#mengkloning-repository)
- [🚀 Menjalankan Aplikasi dengan Docker Compose](#menjalankan-aplikasi-dengan-docker-compose)
- [🔍 Memeriksa Status Kontainer](#memeriksa-status-kontainer)
- [📡 Mengakses API](#mengakses-api)
- [🧹 Menghentikan dan Menghapus Kontainer](#menghentikan-dan-menghapus-kontainer)
- [📂 Struktur Proyek](#struktur-proyek)

## 🔧 Prasyarat

Sebelum memulai, pastikan Anda telah menginstal hal-hal berikut di sistem Anda:

- **Docker**: Platform untuk menjalankan aplikasi dalam kontainer. Panduan instalasi dapat ditemukan di [dokumentasi resmi Docker](https://docs.docker.com/get-docker/).
- **Docker Compose**: Alat untuk menjalankan aplikasi multi-kontainer. Panduan instalasi dapat ditemukan di [dokumentasi resmi Docker Compose](https://docs.docker.com/compose/install/).

## 📥 Mengkloning Repository

1. **Kloning Repository**:

   Buka terminal atau command prompt dan jalankan perintah berikut untuk mengkloning repository proyek Anda:

   ```bash
   git clone git@github.com:rahalsaabiumara/petpoint-py.git
   ```

2. **Navigasi ke Direktori Proyek**:

   ```bash
   cd petpoint-py
   ```

## 🚀 Menjalankan Aplikasi dengan Docker Compose

1. **Membangun dan Menjalankan Kontainer**:

   Jalankan perintah berikut di root direktori proyek Anda untuk membangun image Docker dan menjalankan kontainer:

   ```bash
   docker-compose up --build
   ```

   Catatan:
   - `--build`: Membuat ulang image Docker jika ada perubahan pada Dockerfile atau dependensi.

   Jika Anda ingin menjalankan kontainer di latar belakang (detached mode), gunakan flag `-d`:

   ```bash
   docker-compose up --build -d
   ```

2. **Memastikan Kontainer Berjalan**:

   Setelah menjalankan perintah di atas, periksa apakah semua kontainer berjalan dengan baik:

   ```bash
   docker-compose ps
   ```

## 🔍 Memeriksa Status Kontainer

1. **Melihat Log Kontainer Chatbot**:

   Untuk melihat log dari kontainer chatbot, gunakan perintah berikut:

   ```bash
   docker-compose logs chatbot
   ```

2. **Mengikuti Log Secara Real-Time**:

   Jika Anda ingin melihat log secara real-time, gunakan flag `-f`:

   ```bash
   docker-compose logs -f chatbot
   ```

   Catatan: Tekan `Ctrl + C` untuk menghentikan tampilan log real-time.

## 📡 Mengakses API

Aplikasi chatbot Anda sekarang berjalan pada port 5000. Berikut adalah cara mengakses endpoint yang tersedia:

1. **Health Check**

   - **URL**: `http://localhost:5000/health/`
   - **Method**: `GET`
   - **Deskripsi**: Memeriksa status API.

   Contoh Request menggunakan curl:

   ```bash
   curl http://localhost:5000/health/
   ```

   Contoh Response:

   ```json
   {
     "message": "Chatbot API is running.",
     "status": "OK"
   }
   ```

2. **Chat Endpoint**

   - **URL**: `http://localhost:5000/chat/`
   - **Method**: `POST`
   - **Deskripsi**: Menerima input teks pengguna dan memberikan respons chatbot.

   Contoh Request menggunakan curl:

   ```bash
   curl -X POST http://localhost:5000/chat/        -H "Content-Type: application/json"        -d '{"text": "Halo, saya melihat seekor kucing sakit di depan rumah.", "session_id": "12345"}'
   ```

   Contoh Response:

   ```json
   {
     "response": "Terima kasih atas laporannya. Apakah kucing tersebut menunjukkan gejala seperti demam atau muntah?"
   }
   ```

## 🧹 Menghentikan dan Menghapus Kontainer

1. **Menghentikan Kontainer**:

   Untuk menghentikan semua kontainer yang sedang berjalan tanpa menghapusnya:

   ```bash
   docker-compose stop
   ```

2. **Menghapus Kontainer, Network, dan Volume**:

   Jika Anda ingin menghentikan dan menghapus semua kontainer, network, dan volume yang dibuat oleh Docker Compose:

   ```bash
   docker-compose down -v
   ```

   Peringatan: Flag `-v` akan menghapus volume yang terkait, yang dapat mengakibatkan kehilangan data yang disimpan di volume tersebut.

## 📂 Struktur Proyek

```
├── Dockerfile
├── README.md
├── app
│   ├── __init__.py
│   ├── data
│   │   ├── data2.json
│   │   ├── df_utterances.pkl
│   │   ├── stopword_list_tala.txt
│   │   ├── tfidf_matrix.pickle
│   │   └── vectorizer.pickle
│   ├── encoders
│   │   ├── label_encoder.pickle
│   │   ├── ner_label_encoder.pickle
│   │   ├── tokenizer.pickle
│   │   └── transition_params.pickle
│   ├── models
│   │   ├── model_intent.keras
│   │   └── model_ner_crf.keras
│   ├── routes.py
│   └── utils.py
├── config.py
├── docker-compose.yml
├── requirements.txt
└── run.py
```

---

🎉 **Selesai!**

Dengan mengikuti langkah-langkah di atas, Anda sekarang telah berhasil menjalankan aplikasi Chatbot Hewan menggunakan Docker Compose. Jika Anda mengalami masalah atau memiliki pertanyaan, silakan merujuk ke [dokumentasi resmi Docker](https://docs.docker.com/).
