"""
Emotion Classification & Granular Sentiment Analysis Pipeline
================================================================
Pipeline NLP untuk:
  1. Melatih & mengevaluasi beberapa model klasifikasi emosi (Naive Bayes,
     Logistic Regression, Linear SVM) pada dataset dair-ai/emotion.
  2. Menyediakan mode analisis granular interaktif (CLI) yang memecah
     kalimat menjadi segmen dan mendeteksi emosi (model ML) + sentimen
     (VADER) per segmen.

Cara pakai:
    python emotion_classification_sentanalysis.py            # training + CLI
    python emotion_classification_sentanalysis.py --skip-training
                                                               # langsung pakai model tersimpan
    python emotion_classification_sentanalysis.py --force-retrain
                                                               # paksa training ulang

Model yang sudah dilatih disimpan ke disk (joblib) sehingga run berikutnya
tidak perlu training ulang dari nol.
"""

from __future__ import annotations

import argparse
import logging
import re
import string
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

# ===============================================================
# Konfigurasi global (ubah di sini, bukan di tengah-tengah logic)
# ===============================================================

@dataclass(frozen=True)
class Config:
    # Lokasi penyimpanan artefak
    artifact_dir: Path = Path("artifacts")
    model_path: Path = field(init=False)
    results_csv_path: Path = field(init=False)
    misclassified_csv_path: Path = field(init=False)
    confusion_matrix_path: Path = field(init=False)
    log_path: Path = field(init=False)

    # Label emosi dataset dair-ai/emotion
    label_map: dict = field(default_factory=lambda: {
        0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"
    })

    # Hyperparameter grid per model (dipakai GridSearchCV)
    cv_folds: int = 3
    random_state: int = 42

    min_token_len: int = 3  # token < panjang ini dibuang saat preprocessing

    def __post_init__(self):
        object.__setattr__(self, "model_path", self.artifact_dir / "best_emotion_model.joblib")
        object.__setattr__(self, "results_csv_path", self.artifact_dir / "hasil_perbandingan_model.csv")
        object.__setattr__(self, "misclassified_csv_path", self.artifact_dir / "analisis_kesalahan_model.csv")
        object.__setattr__(self, "confusion_matrix_path", self.artifact_dir / "confusion_matrix_best_model.png")
        object.__setattr__(self, "log_path", self.artifact_dir / "pipeline_nlp.log")


CONFIG = Config()
CONFIG.artifact_dir.mkdir(parents=True, exist_ok=True)


def setup_logging(log_path: Path) -> logging.Logger:
    """Konfigurasi logger terpusat (dipanggil sekali di main)."""
    logger = logging.getLogger("emotion_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


logger = setup_logging(CONFIG.log_path)


# ===============================================================
# 1. Text Preprocessing
# ===============================================================

class TextPreprocessor:
    """Membersihkan & menormalisasi teks mentah menjadi token siap-vectorize.

    Di-encapsulate sebagai class (bukan fungsi bebas) supaya resource NLTK
    (stopwords, lemmatizer) hanya di-load sekali dan bisa di-reuse di mana saja
    termasuk saat inference, tanpa perlu pass parameter berulang.
    """

    URL_PATTERN = re.compile(r"http\S+|www\S+|https\S+")
    MENTION_PATTERN = re.compile(r"@\w+")
    DIGIT_PATTERN = re.compile(r"\d+")
    WHITESPACE_PATTERN = re.compile(r"\s+")
    PUNCT_TABLE = str.maketrans("", "", string.punctuation)

    def __init__(self, min_token_len: int = 3):
        self.min_token_len = min_token_len
        self._stop_words = None
        self._lemmatizer = None
        self._ensure_nltk_resources()

    def _ensure_nltk_resources(self) -> None:
        """Download resource NLTK yang dibutuhkan, idempotent & aman dipanggil berkali-kali."""
        import nltk

        required = ["stopwords", "wordnet", "omw-1.4"]
        for resource in required:
            try:
                nltk.download(resource, quiet=True)
            except Exception as e:
                logger.warning(f"Gagal mengunduh resource NLTK '{resource}': {e}")

        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer

        self._stop_words = set(stopwords.words("english"))
        self._lemmatizer = WordNetLemmatizer()

    def clean(self, text: str) -> str:
        """Bersihkan satu string teks. Selalu mengembalikan string (bisa kosong)."""
        if text is None:
            return ""

        text = str(text).lower()
        text = self.URL_PATTERN.sub("", text)
        text = self.MENTION_PATTERN.sub("", text)
        text = text.replace("#", "")
        text = self.DIGIT_PATTERN.sub("", text)
        text = text.translate(self.PUNCT_TABLE)
        text = self.WHITESPACE_PATTERN.sub(" ", text).strip()

        if not text:
            return ""

        tokens = [
            self._lemmatizer.lemmatize(tok)
            for tok in text.split()
            if tok not in self._stop_words and len(tok) > self.min_token_len - 1
        ]
        return " ".join(tokens)

    def clean_series(self, series: pd.Series) -> pd.Series:
        """Versi vektor untuk DataFrame column, dengan progress log per batch."""
        total = len(series)
        logger.info(f"Membersihkan {total} baris teks...")
        return series.apply(self.clean)


# ===============================================================
# 2. Training & Evaluasi Model
# ===============================================================

class EmotionModelTrainer:
    """Mengelola siklus penuh: load data -> preprocessing -> training ->
    evaluasi -> simpan artefak (model, hasil perbandingan, confusion matrix,
    analisis kesalahan).
    """

    def __init__(self, config: Config, preprocessor: TextPreprocessor):
        self.config = config
        self.preprocessor = preprocessor
        self.best_model = None
        self.best_model_name: Optional[str] = None
        self.best_accuracy: float = 0.0
        self.results_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None

    # ---------- Data loading ----------

    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        from datasets import load_dataset

        logger.info("Memuat dataset dair-ai/emotion...")
        try:
            dataset = load_dataset("dair-ai/emotion")
        except Exception as e:
            logger.error(f"Gagal memuat dataset: {e}")
            raise

        train_df = pd.DataFrame(dataset["train"])
        val_df = pd.DataFrame(dataset["validation"])
        test_df = pd.DataFrame(dataset["test"])

        for df in (train_df, val_df, test_df):
            df["label_name"] = df["label"].map(self.config.label_map)

        logger.info(
            f"Dimensi Data - Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}"
        )
        return train_df, val_df, test_df

    # ---------- Model definitions ----------

    def _build_models_config(self) -> dict:
        """Definisi model + grid hyperparameter.

        Catatan: grid sengaja dipangkas dibanding versi awal (kombinasi C
        untuk Linear SVM dikurangi, n_jobs dipakai konsisten) supaya waktu
        training tetap wajar tanpa mengorbankan kualitas pencarian terbaik.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.pipeline import Pipeline
        from sklearn.svm import LinearSVC

        return {
            "Naive Bayes": {
                "model": Pipeline([("tfidf", TfidfVectorizer()), ("clf", MultinomialNB())]),
                "params": {
                    "tfidf__max_features": [5000, 10000],
                    "tfidf__ngram_range": [(1, 1), (1, 2)],
                },
            },
            "Logistic Regression": {
                "model": Pipeline([
                    ("tfidf", TfidfVectorizer()),
                    ("clf", LogisticRegression(max_iter=500, random_state=self.config.random_state)),
                ]),
                "params": {
                    "tfidf__max_features": [5000, 10000],
                    "clf__C": [0.5, 1.0, 2.0],
                },
            },
            "Linear SVM": {
                "model": Pipeline([
                    ("tfidf", TfidfVectorizer()),
                    ("clf", LinearSVC(dual=False, random_state=self.config.random_state)),
                ]),
                "params": {
                    "tfidf__max_features": [5000, 10000],
                    "clf__C": [0.1, 1.0],
                },
            },
        }

    # ---------- Training ----------

    def train_and_evaluate(self) -> None:
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import GridSearchCV

        train_df, val_df, test_df = self.load_data()

        logger.info("Memulai proses text preprocessing...")
        for df in (train_df, val_df, test_df):
            df["clean_text"] = self.preprocessor.clean_series(df["text"])

        X_train_full = pd.concat([train_df["clean_text"], val_df["clean_text"]], axis=0)
        y_train_full = pd.concat([train_df["label"], val_df["label"]], axis=0)
        X_test = test_df["clean_text"]
        y_test = test_df["label"]

        models_config = self._build_models_config()
        results = []

        logger.info("Memulai proses training dan hyperparameter tuning...")
        for name, cfg in models_config.items():
            logger.info(f"Training {name} dengan GridSearchCV...")
            try:
                grid_search = GridSearchCV(
                    cfg["model"], cfg["params"],
                    cv=self.config.cv_folds, scoring="accuracy", n_jobs=-1,
                )
                grid_search.fit(X_train_full, y_train_full)
            except Exception as e:
                logger.error(f"Training '{name}' gagal, dilewati: {e}")
                continue

            best_estimator = grid_search.best_estimator_
            y_pred = best_estimator.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            logger.info(f"[{name}] Best Params: {grid_search.best_params_} | Accuracy: {acc:.4f}")
            results.append({
                "Model": name,
                "Accuracy": acc,
                "Best Params": str(grid_search.best_params_),
            })

            if acc > self.best_accuracy:
                self.best_accuracy = acc
                self.best_model_name = name
                self.best_model = best_estimator

        if self.best_model is None:
            raise RuntimeError("Semua model gagal dilatih — tidak ada model terbaik yang dihasilkan.")

        self.results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
        self.test_df = test_df
        logger.info(f"Model terbaik diraih oleh {self.best_model_name} dengan akurasi {self.best_accuracy:.4f}")

        self._evaluate_errors(X_test, y_test)
        self._save_artifacts()

    # ---------- Evaluasi lanjutan ----------

    def _evaluate_errors(self, X_test: pd.Series, y_test: pd.Series) -> None:
        import matplotlib
        matplotlib.use("Agg")  # backend non-interaktif, aman untuk batch run
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix

        y_pred_best = self.best_model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred_best)
        label_names = list(self.config.label_map.values())

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=label_names, yticklabels=label_names)
        plt.title(f"Confusion Matrix - {self.best_model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(self.config.confusion_matrix_path)
        plt.close()
        logger.info(f"Confusion matrix disimpan di '{self.config.confusion_matrix_path}'.")

        self.test_df["predicted_label"] = y_pred_best
        self.test_df["predicted_name"] = self.test_df["predicted_label"].map(self.config.label_map)

        misclassified_df = self.test_df[self.test_df["label"] != self.test_df["predicted_label"]].copy()
        misclassified_df.to_csv(self.config.misclassified_csv_path, index=False)

        logger.info(
            f"Total sampel salah klasifikasi: {len(misclassified_df)} dari {len(self.test_df)}"
        )

    def _save_artifacts(self) -> None:
        self.results_df.to_csv(self.config.results_csv_path, index=False)
        logger.info(f"Hasil perbandingan model disimpan di '{self.config.results_csv_path}'.")

        joblib.dump(
            {
                "model": self.best_model,
                "model_name": self.best_model_name,
                "accuracy": self.best_accuracy,
                "label_map": self.config.label_map,
            },
            self.config.model_path,
        )
        logger.info(f"Model terbaik ('{self.best_model_name}') disimpan di '{self.config.model_path}'.")

    # ---------- Load model tersimpan ----------

    def load_saved_model(self) -> bool:
        """Coba load model dari disk. Return True jika berhasil."""
        if not self.config.model_path.exists():
            return False
        try:
            payload = joblib.load(self.config.model_path)
            self.best_model = payload["model"]
            self.best_model_name = payload["model_name"]
            self.best_accuracy = payload["accuracy"]
            logger.info(
                f"Model tersimpan dimuat: {self.best_model_name} (akurasi training terakhir: {self.best_accuracy:.4f})"
            )
            return True
        except Exception as e:
            logger.warning(f"Gagal memuat model tersimpan, akan training ulang: {e}")
            return False


# ===============================================================
# 3. Analisis Granular (Emotion + Sentiment) — untuk CLI interaktif
# ===============================================================

class GranularAnalyzer:
    """Membungkus model emosi terlatih + VADER + translator untuk dipakai
    secara interaktif. Semua titik gagal eksternal (translator, model
    kosong/None) ditangani secara eksplisit supaya CLI tidak crash.
    """

    CONNECTORS = ("but", "and", "or", "because", "although")
    SEGMENT_PATTERN = re.compile(r"(?i)\b(but|and|or|because|although)\b|[.,;!]+")

    def __init__(self, model, label_map: dict, preprocessor: TextPreprocessor):
        if model is None:
            raise ValueError("GranularAnalyzer membutuhkan model yang sudah dilatih (tidak boleh None).")

        self.model = model
        self.label_map = label_map
        self.preprocessor = preprocessor

        self._sia = self._init_vader()
        self._translator = self._init_translator()

        classifier = model.named_steps.get("clf")
        self.has_proba = hasattr(classifier, "predict_proba")
        self.has_decision = hasattr(classifier, "decision_function")

    @staticmethod
    def _init_vader():
        import nltk
        try:
            nltk.download("vader_lexicon", quiet=True)
        except Exception as e:
            logger.warning(f"Gagal mengunduh vader_lexicon: {e}")
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer()

    @staticmethod
    def _init_translator():
        from deep_translator import GoogleTranslator
        return GoogleTranslator(source="auto", target="en")

    def translate_safe(self, text: str) -> str:
        """Translate dengan fallback: kalau translator gagal (network/rate-limit),
        kembalikan teks asli alih-alih melempar exception ke CLI.
        """
        try:
            translated = self._translator.translate(text)
            return translated.lower() if translated else text.lower()
        except Exception as e:
            logger.warning(f"Translasi gagal ('{e}'), menggunakan teks asli tanpa translasi.")
            return text.lower()

    def get_vader_sentiment(self, text: str) -> str:
        if not text.strip():
            return "neutral"
        score = self._sia.polarity_scores(text)["compound"]
        if score >= 0.05:
            return "positive"
        if score <= -0.05:
            return "negative"
        return "neutral"

    def get_emotion_and_score(self, text: str) -> tuple[str, float]:
        clean_txt = self.preprocessor.clean(text)

        if not clean_txt:
            return "neutral", 0.50

        try:
            pred = self.model.predict([clean_txt])[0]
            emotion = self.label_map.get(pred, "unknown")

            if self.has_proba:
                proba = self.model.predict_proba([clean_txt])[0]
                score = float(np.max(proba))
            elif self.has_decision:
                decision_scores = self.model.decision_function([clean_txt])[0]
                exp_scores = np.exp(decision_scores - np.max(decision_scores))
                proba = exp_scores / exp_scores.sum()
                score = float(np.max(proba))
            else:
                score = 1.00

            return emotion, score
        except Exception as e:
            logger.warning(f"Prediksi emosi gagal untuk teks '{text[:50]}...': {e}")
            return "unknown", 0.0

    def segment_sentence(self, text: str) -> list[str]:
        segments = self.SEGMENT_PATTERN.split(text)
        return [s.strip() for s in segments if s and s.strip()]

    def analyze(self, raw_text: str) -> dict:
        """Analisis penuh satu input: overall + breakdown per segmen.
        Mengembalikan dict terstruktur (mudah dipakai ulang non-CLI, mis. API).
        """
        translated_text = self.translate_safe(raw_text)

        overall_emotion, overall_score = self.get_emotion_and_score(translated_text)
        overall_sentiment = self.get_vader_sentiment(translated_text)

        breakdown = []
        for segment in self.segment_sentence(translated_text):
            if segment in self.CONNECTORS:
                breakdown.append({
                    "segment": segment, "emotion": "neutral",
                    "sentiment": "neutral", "score": 0.50,
                })
            else:
                emo, score = self.get_emotion_and_score(segment)
                sent = self.get_vader_sentiment(segment)
                breakdown.append({
                    "segment": segment, "emotion": emo,
                    "sentiment": sent, "score": score,
                })

        return {
            "translated_text": translated_text,
            "overall": {
                "emotion": overall_emotion,
                "sentiment": overall_sentiment,
                "score": overall_score,
            },
            "breakdown": breakdown,
        }


# ===============================================================
# 4. CLI Interaktif
# ===============================================================

def print_analysis_result(result: dict) -> None:
    print("\n[Overall Analysis]")
    print("-" * 40)
    print("Status           : Sentiment analysis complete.")
    print(f"Overall Emotion  : {result['overall']['emotion'].capitalize()}")
    print(f"Overall Sentiment: {result['overall']['sentiment'].capitalize()}")
    print(f"Overall Score    : {result['overall']['score']:.2f}")
    print("-" * 40)

    print("\n[Sentence-Level Breakdown]")
    print(f"{'Sentence':<35} | {'Emotion':<12} | {'Sentiment':<10} | {'Score'}")
    print("-" * 80)
    for row in result["breakdown"]:
        print(f"{row['segment']:<35} | {row['emotion']:<12} | {row['sentiment']:<10} | {row['score']:.2f}")
    print("=" * 85)


def run_interactive_cli(analyzer: GranularAnalyzer) -> None:
    print("\n" + "=" * 85)
    print("MODE ANALISIS GRANULAR (OVERALL & SENTENCE-LEVEL BREAKDOWN)")
    print("Sistem memisahkan deteksi Emosi (TF-IDF/ML) dan Sentimen (VADER Lexicon).")
    print("Ketik 'exit' atau 'keluar' untuk keluar.")
    print("=" * 85)

    while True:
        try:
            user_input = input("\nInput Text (ID/EN) > ")
        except (EOFError, KeyboardInterrupt):
            print("\nSesi dihentikan oleh pengguna.")
            break

        if user_input.strip().lower() in ("exit", "keluar"):
            print("Keluar dari mode analisis. Sampai jumpa!")
            break
        if not user_input.strip():
            continue

        try:
            result = analyzer.analyze(user_input)
            print_analysis_result(result)
        except Exception as e:
            logger.error(f"Terjadi error saat menganalisis input: {e}")
            print(f"Error: {e}. Silakan coba input lain.")


# ===============================================================
# 5. Orkestrasi (main)
# ===============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emotion Classification & Granular Sentiment Analysis Pipeline")
    parser.add_argument(
        "--skip-training", action="store_true",
        help="Lewati training jika model tersimpan sudah ada, langsung ke CLI.",
    )
    parser.add_argument(
        "--force-retrain", action="store_true",
        help="Paksa training ulang meskipun model tersimpan sudah ada.",
    )
    parser.add_argument(
        "--no-cli", action="store_true",
        help="Hanya jalankan training/evaluasi, tanpa membuka mode CLI interaktif.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preprocessor = TextPreprocessor(min_token_len=CONFIG.min_token_len)
    trainer = EmotionModelTrainer(CONFIG, preprocessor)

    model_loaded = False
    if not args.force_retrain:
        model_loaded = trainer.load_saved_model()

    if not model_loaded:
        if args.skip_training:
            logger.error("Tidak ada model tersimpan dan --skip-training diaktifkan. Berhenti.")
            sys.exit(1)
        trainer.train_and_evaluate()

    if args.no_cli:
        logger.info("Selesai (--no-cli aktif, mode interaktif dilewati).")
        return

    try:
        analyzer = GranularAnalyzer(trainer.best_model, CONFIG.label_map, preprocessor)
    except Exception as e:
        logger.error(f"Gagal menginisialisasi analyzer granular: {e}")
        sys.exit(1)

    run_interactive_cli(analyzer)


if __name__ == "__main__":
    main()