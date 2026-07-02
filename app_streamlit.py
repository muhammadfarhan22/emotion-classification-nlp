"""
Streamlit Web App — Emotion Classification & Granular Sentiment Analysis
===========================================================================
Antarmuka web untuk pipeline NLP yang didefinisikan di `pipeline_core.py`.

Alur:
  1. Saat app pertama kali dibuka, cek apakah model tersimpan sudah ada
     (artifacts/best_emotion_model.joblib).
  2. Jika BELUM ada -> training otomatis berjalan (dengan progress indicator),
     karena ini bisa memakan waktu beberapa menit (GridSearchCV x 3 model).
  3. Jika SUDAH ada -> model langsung di-load (instan), tidak training ulang.
  4. Setelah model siap, pengguna bisa mengetik teks dan mendapatkan analisis
     granular (emosi per segmen kalimat + sentimen VADER) secara interaktif.

Cara menjalankan:
    streamlit run app_streamlit.py
"""

from __future__ import annotations

import time
import traceback

import pandas as pd
import streamlit as st

from pipeline_core import (
    CONFIG,
    EmotionModelTrainer,
    GranularAnalyzer,
    TextPreprocessor,
    logger,
)

# ===============================================================
# Konfigurasi halaman
# ===============================================================

st.set_page_config(
    page_title="Emotion & Sentiment Analyzer",
    page_icon="🎭",
    layout="centered",
    initial_sidebar_state="expanded",
)

EMOTION_COLORS = {
    "sadness": "#3B82F6",
    "joy": "#F2B705",
    "love": "#EC4899",
    "anger": "#DC2626",
    "fear": "#7C3AED",
    "surprise": "#0EA5A0",
    "neutral": "#94A3B8",
    "unknown": "#94A3B8",
}

SENTIMENT_COLORS = {
    "positive": "#16A34A",
    "negative": "#DC2626",
    "neutral": "#64748B",
}

EMOTION_EMOJI = {
    "sadness": "😢", "joy": "😄", "love": "❤️",
    "anger": "😠", "fear": "😨", "surprise": "😲",
    "neutral": "😐", "unknown": "❔",
}


# ===============================================================
# Resource caching — preprocessor & trainer hanya dibuat sekali
# per sesi server, bukan setiap kali pengguna berinteraksi.
# ===============================================================

@st.cache_resource(show_spinner=False)
def get_preprocessor() -> TextPreprocessor:
    return TextPreprocessor(min_token_len=CONFIG.min_token_len)


@st.cache_resource(show_spinner=False)
def get_analyzer(_preprocessor: TextPreprocessor, _model, label_map: dict) -> GranularAnalyzer:
    """Underscore prefix pada parameter memberi tahu Streamlit untuk tidak
    mencoba meng-hash objek tersebut (model sklearn tidak hashable secara wajar).
    """
    return GranularAnalyzer(_model, label_map, _preprocessor)


def ensure_model_ready(preprocessor: TextPreprocessor) -> EmotionModelTrainer:
    """Load model tersimpan jika ada; jika tidak, training otomatis dengan
    progress indicator. Hasil disimpan di session_state agar tidak training
    ulang setiap kali Streamlit rerun script (yang terjadi di setiap interaksi).
    """
    if "trainer" in st.session_state:
        return st.session_state["trainer"]

    trainer = EmotionModelTrainer(CONFIG, preprocessor)
    model_loaded = trainer.load_saved_model()

    if not model_loaded:
        st.warning(
            "Model tersimpan belum ditemukan. Training otomatis akan dimulai — "
            "proses ini bisa memakan waktu beberapa menit (3 model x GridSearchCV)."
        )
        progress_placeholder = st.empty()
        status_placeholder = st.empty()

        with progress_placeholder.container():
            progress_bar = st.progress(0, text="Mempersiapkan dataset...")

        try:
            status_placeholder.info("Memuat dataset dair-ai/emotion dari Hugging Face...")
            progress_bar.progress(10, text="Memuat dataset...")

            status_placeholder.info("Melatih dan mengevaluasi 3 model (Naive Bayes, Logistic Regression, Linear SVM)...")
            progress_bar.progress(35, text="Training & hyperparameter tuning (mohon tunggu)...")

            trainer.train_and_evaluate()

            progress_bar.progress(100, text="Training selesai!")
            status_placeholder.success(
                f"Training selesai. Model terbaik: **{trainer.best_model_name}** "
                f"dengan akurasi **{trainer.best_accuracy:.2%}**."
            )
            time.sleep(1.2)
        except Exception as e:
            logger.error(f"Training gagal di Streamlit app: {e}\n{traceback.format_exc()}")
            status_placeholder.error(
                f"Training gagal: {e}\n\n"
                "Periksa koneksi internet (untuk mengunduh dataset/resource NLTK) "
                "dan lihat log untuk detail lebih lanjut."
            )
            st.stop()
        finally:
            progress_placeholder.empty()
            status_placeholder.empty()

    st.session_state["trainer"] = trainer
    return trainer


# ===============================================================
# Komponen UI
# ===============================================================

def render_sidebar(trainer: EmotionModelTrainer) -> None:
    with st.sidebar:
        st.markdown("### ℹ️ Informasi Model")
        st.markdown(f"**Model aktif:** {trainer.best_model_name}")
        st.markdown(f"**Akurasi (test set):** {trainer.best_accuracy:.2%}")
        st.markdown("---")
        st.markdown("### 🏷️ Kelas Emosi")
        for name in CONFIG.label_map.values():
            st.markdown(f"{EMOTION_EMOJI.get(name, '•')} {name.capitalize()}")
        st.markdown("---")
        st.caption(
            "Sistem memisahkan deteksi **Emosi** (model ML berbasis TF-IDF) "
            "dan **Sentimen** polaritas (VADER lexicon)."
        )
        if st.button("🔄 Retrain Model", help="Paksa training ulang dari awal. Akan menghapus model tersimpan saat ini."):
            st.session_state.pop("trainer", None)
            get_analyzer.clear()
            CONFIG.model_path.unlink(missing_ok=True)
            st.rerun()


def render_result(result: dict) -> None:
    overall = result["overall"]
    emotion = overall["emotion"]
    sentiment = overall["sentiment"]

    st.markdown("#### Hasil Analisis Keseluruhan")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label="Emosi",
            value=f"{EMOTION_EMOJI.get(emotion, '')} {emotion.capitalize()}",
        )
    with col2:
        st.metric(
            label="Sentimen",
            value=sentiment.capitalize(),
        )
    with col3:
        st.metric(
            label="Skor Keyakinan",
            value=f"{overall['score']:.0%}",
        )

    if result["translated_text"]:
        with st.expander("Lihat teks setelah translasi (jika input bukan Bahasa Inggris)"):
            st.code(result["translated_text"], language=None)

    st.markdown("#### Rincian per Segmen Kalimat")
    breakdown_df = pd.DataFrame(result["breakdown"])
    breakdown_df = breakdown_df.rename(columns={
        "segment": "Segmen", "emotion": "Emosi",
        "sentiment": "Sentimen", "score": "Skor",
    })
    breakdown_df["Emosi"] = breakdown_df["Emosi"].apply(lambda e: f"{EMOTION_EMOJI.get(e, '')} {e.capitalize()}")
    breakdown_df["Sentimen"] = breakdown_df["Sentimen"].str.capitalize()
    breakdown_df["Skor"] = breakdown_df["Skor"].apply(lambda s: f"{s:.0%}")

    st.dataframe(breakdown_df, use_container_width=True, hide_index=True)

    if len(result["breakdown"]) > 1:
        unique_emotions = {row["emotion"] for row in result["breakdown"] if row["emotion"] not in ("neutral", "unknown")}
        if len(unique_emotions) > 1:
            st.info(
                f"🔀 Kalimat ini mengandung **{len(unique_emotions)} emosi berbeda**: "
                f"{', '.join(e.capitalize() for e in unique_emotions)}. "
                "Ini menunjukkan kelebihan analisis granular dibanding klasifikasi single-label biasa."
            )


# ===============================================================
# Main app
# ===============================================================

def main() -> None:
    st.title("🎭 Emotion & Granular Sentiment Analyzer")
    st.caption(
        "Deteksi emosi (Machine Learning) dan sentimen (VADER) dari teks, "
        "dipecah per segmen kalimat. Mendukung input Bahasa Indonesia & Inggris."
    )

    preprocessor = get_preprocessor()
    trainer = ensure_model_ready(preprocessor)

    try:
        analyzer = get_analyzer(preprocessor, trainer.best_model, CONFIG.label_map)
    except Exception as e:
        st.error(f"Gagal menginisialisasi analyzer: {e}")
        st.stop()

    render_sidebar(trainer)

    st.markdown("---")
    user_input = st.text_area(
        "Masukkan teks untuk dianalisis",
    )

    analyze_clicked = st.button("🔍 Analisis", type="primary", use_container_width=True)

    if analyze_clicked:
        if not user_input.strip():
            st.warning("Mohon masukkan teks terlebih dahulu.")
        else:
            with st.spinner("Menganalisis..."):
                try:
                    result = analyzer.analyze(user_input)
                    st.session_state["last_result"] = result
                except Exception as e:
                    logger.error(f"Analisis gagal: {e}")
                    st.error(f"Terjadi kesalahan saat menganalisis: {e}")

    if "last_result" in st.session_state:
        st.markdown("---")
        render_result(st.session_state["last_result"])


if __name__ == "__main__":
    main()