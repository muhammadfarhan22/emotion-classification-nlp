# 🎭 Emotion Classification & Granular Sentiment Analysis

> Pipeline NLP untuk klasifikasi emosi teks berbasis Machine Learning dan analisis sentimen granular tingkat kalimat. Mendukung input Bahasa Indonesia dan Inggris.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red?logo=streamlit&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-3.8%2B-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 🔍 Tentang Proyek

Proyek ini dikembangkan sebagai tugas Ujian Akhir Semester mata kuliah **Natural Language Processing**. Sistem ini membangun pipeline lengkap untuk:

1. **Mengklasifikasikan emosi** dari teks ke dalam 6 kelas (sadness, joy, love, anger, fear, surprise) menggunakan pendekatan Machine Learning berbasis TF-IDF
2. **Menganalisis sentimen** polaritas (positif, negatif, netral) menggunakan VADER lexicon
3. **Memecah kalimat majemuk** menjadi segmen-segmen yang dianalisis secara terpisah, sehingga satu kalimat bisa menghasilkan lebih dari satu emosi

Dataset yang digunakan adalah [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) dari Hugging Face — 20.000 teks Twitter berbahasa Inggris berlabel 6 emosi.

---

## 🎬 Demo

Jalankan aplikasi web Streamlit secara lokal:

```bash
streamlit run app_streamlit.py
```

Buka browser ke `http://localhost:8501`, masukkan teks, dan lihat hasil analisis granular secara real-time.

**Contoh output untuk kalimat majemuk:**

| Segmen                                  | Emosi      | Sentimen | Skor |
| --------------------------------------- | ---------- | -------- | ---- |
| i am happy with my new job              | 😄 Joy     | Positive | 87%  |
| but                                     | 😐 Neutral | Neutral  | 50%  |
| i am scared of the new responsibilities | 😨 Fear    | Negative | 79%  |

---

## ✨ Fitur Unggulan

- **Perbandingan 3 model** — Naive Bayes, Logistic Regression, Linear SVM dengan GridSearchCV otomatis
- **Analisis granular** — kalimat majemuk dipecah per segmen, tiap segmen dianalisis emosinya terpisah
- **Dua lapisan analisis** — emosi (ML model) dan sentimen polaritas (VADER) berjalan paralel dan saling melengkapi
- **Save/load model** — model tersimpan otomatis setelah training, run berikutnya langsung load tanpa training ulang
- **Dual interface** — tersedia sebagai CLI interaktif (`pipeline_core.py`) dan web app (`app_streamlit.py`)
- **Robust error handling** — translasi gagal, input kosong, dan model error ditangani dengan graceful degradation
- **Dukungan Bahasa Indonesia** — teks otomatis diterjemahkan via Google Translate sebelum dianalisis

---

## 🏗️ Arsitektur Sistem

Sistem dibangun dengan desain **modular** — 4 komponen dengan tanggung jawab yang jelas dan terpisah:

```
Input Teks (ID/EN)
       │
       ▼
┌─────────────────┐
│  deep-translator │  ← translasi otomatis ke Bahasa Inggris (fallback: teks asli)
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│   TextPreprocessor   │  ← lowercase, hapus URL/mention, stopwords, lemmatization
│       (NLTK)         │
└────────┬────────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐ ┌──────────┐
│TF-IDF  │ │  VADER   │  ← dua lapisan analisis berjalan paralel
│+ SVM   │ │ Lexicon  │
└───┬────┘ └────┬─────┘
    │            │
    ▼            ▼
┌────────┐ ┌──────────┐
│6 Kelas │ │Polaritas │
│ Emosi  │ │Sentimen  │
└───┬────┘ └────┬─────┘
    └─────┬─────┘
          │
          ▼
   Output terstruktur
  (overall + breakdown
     per segmen)
```

**Komponen utama:**

| Komponen              | Tanggung Jawab                                                     |
| --------------------- | ------------------------------------------------------------------ |
| `Config`              | Parameter, path file, dan label emosi terpusat di satu tempat      |
| `TextPreprocessor`    | Pembersihan dan normalisasi teks mentah                            |
| `EmotionModelTrainer` | Load data, training GridSearchCV, evaluasi, save/load model        |
| `GranularAnalyzer`    | Translasi, segmentasi kalimat, deteksi emosi + sentimen per segmen |

---

## 🛠️ Tech Stack

| Library                  | Versi  | Fungsi                                                                        |
| ------------------------ | ------ | ----------------------------------------------------------------------------- |
| `scikit-learn`           | ≥ 1.3  | TF-IDF Vectorizer, Naive Bayes, Logistic Regression, Linear SVM, GridSearchCV |
| `nltk`                   | ≥ 3.8  | Stopword removal, lemmatization, VADER sentiment analysis                     |
| `datasets`               | ≥ 2.16 | Load dataset dair-ai/emotion dari Hugging Face                                |
| `deep-translator`        | ≥ 1.11 | Translasi otomatis Bahasa Indonesia → Inggris                                 |
| `streamlit`              | ≥ 1.32 | Antarmuka web interaktif                                                      |
| `joblib`                 | ≥ 1.3  | Simpan dan load model terlatih ke disk                                        |
| `pandas`                 | ≥ 2.0  | Manipulasi data dan ekspor CSV                                                |
| `matplotlib` + `seaborn` | —      | Confusion matrix visualization                                                |

---

## 🚀 Instalasi

### Prasyarat

- Python 3.9 atau lebih baru
- Koneksi internet (untuk download dataset dan resource NLTK saat pertama kali)

### Langkah instalasi

```bash
# 1. Clone repository ini
git clone https://github.com/muhammadfarhan22/emotion-classification-nlp.git
cd emotion-classification-nlp

# 2. (Opsional tapi disarankan) Buat virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 3. Install semua dependency
pip install -r requirements.txt
```

---

## 💻 Cara Penggunaan

### Opsi A — Web App (Streamlit) ⭐ Disarankan

```bash
streamlit run app_streamlit.py
```

Browser otomatis terbuka ke `http://localhost:8501`.

- **Pertama kali:** training otomatis berjalan (estimasi 5–15 menit tergantung spesifikasi komputer)
- **Berikutnya:** model langsung dimuat dari file, instan
- Tombol **Retrain Model** di sidebar tersedia jika ingin training ulang dari awal

### Opsi B — CLI Interaktif

```bash
python pipeline_core.py
```

Flag yang tersedia:

```bash
python pipeline_core.py                  # training jika belum ada model, lalu masuk CLI
python pipeline_core.py --skip-training  # langsung load model tersimpan → CLI
python pipeline_core.py --force-retrain  # paksa training ulang meskipun model sudah ada
python pipeline_core.py --no-cli         # jalankan training saja tanpa membuka CLI
```

### Contoh interaksi CLI

```
Input Text (ID/EN) > aku senang dengan pekerjaan baru tapi aku takut dengan tanggung jawabnya

[Overall Analysis]
----------------------------------------
Status           : Sentiment analysis complete.
Overall Emotion  : Joy
Overall Sentiment: Positive
Overall Score    : 0.81
----------------------------------------

[Sentence-Level Breakdown]
Sentence                            | Emotion      | Sentiment  | Score
--------------------------------------------------------------------------------
i am happy with my new job          | joy          | positive   | 0.87
but                                 | neutral      | neutral    | 0.50
i am scared of new responsibilities | fear         | negative   | 0.79
```

---

## 📊 Hasil Evaluasi

Evaluasi dilakukan pada **test set dair-ai/emotion** (2.000 sampel yang tidak disentuh selama training):

### Perbandingan akurasi model

| Model               | Akurasi    | Best Params                           |
| ------------------- | ---------- | ------------------------------------- |
| Naive Bayes         | 83.50%     | max_features=10000, ngram_range=(1,2) |
| Logistic Regression | 87.80%     | max_features=10000, C=2.0             |
| **Linear SVM** ⭐   | **89.05%** | max_features=10000, C=1.0             |

### Recall per kelas (Linear SVM)

| Kelas Emosi | Support | Recall |
| ----------- | ------- | ------ |
| Joy         | 695     | 92.66% |
| Sadness     | 581     | 92.43% |
| Anger       | 275     | 89.09% |
| Fear        | 224     | 84.38% |
| Love        | 159     | 76.73% |
| Surprise    | 66      | 66.67% |

> **Catatan:** Perbedaan recall antar kelas disebabkan oleh _class imbalance_ pada dataset — kelas dengan sampel lebih sedikit (surprise, love) cenderung memiliki recall lebih rendah.

### Pola kesalahan utama

- **Love ↔ Joy** (61 kasus) — kedua emosi positif berbagi kosakata serupa
- **Sadness ↔ Anger** (33 kasus) — dua emosi negatif yang sulit dibedakan tanpa konteks panjang
- **Fear → Sadness** (15 kasus) — nada kekhawatiran dan kesedihan sering tumpang tindih

---

## 📚 Referensi

- Saravia, E., et al. (2018). _CARER: Contextualized Affect Representations for Emotion Recognition_. EMNLP 2018.
- Hutto, C. J., & Gilbert, E. (2014). _VADER: A Parsimonious Rule-Based Model for Sentiment Analysis of Social Media Text_. ICWSM-14.
- Joachims, T. (1998). _Text Categorization with Support Vector Machines_. ECML-98.
- [Hugging Face — dair-ai/emotion dataset](https://huggingface.co/datasets/dair-ai/emotion)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [NLTK Documentation](https://www.nltk.org/)

---

<div align="center">
  <sub>Dibuat untuk Ujian Akhir Semester — Natural Language Processing · 2026</sub>
</div>
