from __future__ import annotations

import io
import json
import os
import re

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from wordcloud import WordCloud


# ============================================================
# KONFIGURASI
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE, DATA_FILE, STEM_FILE = (
    "model_nb_sentimen.joblib",
    "data_bersih_berlabel.csv",
    "stem_cache.json",
)
SEED, TEST_SIZE = 42, 0.2
COLORS = {
    "Positif": "#10B981",
    "Netral": "#F59E0B",
    "Negatif": "#EF4444",
}

st.set_page_config(
    page_title="Sentiment Mining · Ruangguru",
    page_icon="💬",
    layout="wide",
)

st.markdown(
    """
    <style>
    .hero {
        background: linear-gradient(120deg, #4F46E5 0%, #7C3AED 55%, #EC4899 100%);
        padding: 26px 32px;
        border-radius: 18px;
        color: white;
        margin-bottom: 16px;
    }
    .hero h1 { margin: 0; font-size: 1.9rem; color: white; }
    .hero p { margin: 6px 0 0; opacity: .92; }
    .kpi {
        background: var(--secondary-background-color);
        border-radius: 14px;
        padding: 14px 18px;
        border-left: 6px solid #7C3AED;
    }
    .kpi .v { font-size: 1.65rem; font-weight: 700; }
    .kpi .l {
        font-size: .78rem;
        opacity: .7;
        text-transform: uppercase;
        letter-spacing: .05em;
    }
    .insight {
        background: var(--secondary-background-color);
        border-radius: 12px;
        padding: 10px 16px;
        margin: 6px 0;
        border-left: 4px solid #EC4899;
    }
    .stTabs [data-baseweb="tab"] { font-weight: 600; }
    </style>
    <div class="hero">
      <h1>💬 Sentiment Mining Dashboard · Ruangguru</h1>
      <p>Analisis notebook dengan <b>Multinomial Naive Bayes</b> · TF-IDF ·
      SMOTE · Influential Features · Analisis Aspek</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def kpi(col, label, value, color="#7C3AED"):
    col.markdown(
        f'<div class="kpi" style="border-left-color:{color}">'
        f'<div class="l">{label}</div><div class="v">{value}</div></div>',
        unsafe_allow_html=True,
    )


def _find(name):
    path = os.path.join(BASE_DIR, name)
    return path if os.path.exists(path) else None


# ============================================================
# LOAD MODEL DAN DATA
# ============================================================
with st.sidebar:
    st.header("Hasil Notebook")
    up_model = st.file_uploader(
        "Model (.joblib) – opsional",
        type=["joblib"],
        help="Unggah jika file model tidak ada di folder aplikasi.",
    )
    up_data = st.file_uploader(
        "Data bersih (.csv) – opsional",
        type=["csv"],
    )

model_src = up_model.getvalue() if up_model else _find(MODEL_FILE)
data_src = up_data.getvalue() if up_data else _find(DATA_FILE)

if model_src is None or data_src is None:
    st.error("File hasil notebook belum ditemukan.")
    st.markdown(
        f"""
        Letakkan file berikut satu folder dengan `app_streamlit.py`
        (folder saat ini: `{BASE_DIR}`), atau unggah lewat sidebar:
        - `{MODEL_FILE}`
        - `{DATA_FILE}`

        Jalankan notebook sampai bagian penyimpanan model selesai.
        """
    )
    st.stop()


@st.cache_resource(show_spinner="Memuat model...")
def load_bundle(src):
    if isinstance(src, bytes):
        return joblib.load(io.BytesIO(src))
    return joblib.load(src)


@st.cache_data(show_spinner="Memuat data...")
def load_data(src):
    if isinstance(src, bytes):
        data = pd.read_csv(io.BytesIO(src))
    else:
        data = pd.read_csv(src)

    required = {"clean_content", "label", "content", "score"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(
            "Kolom wajib tidak ditemukan di CSV: " + ", ".join(sorted(missing))
        )

    data["clean_content"] = data["clean_content"].fillna("").astype(str)
    data["content"] = data["content"].fillna("").astype(str)
    data["label"] = data["label"].astype(str)
    data["score"] = pd.to_numeric(data["score"], errors="coerce")

    if "at" in data.columns:
        data["at"] = pd.to_datetime(data["at"], errors="coerce")
    else:
        data["at"] = pd.NaT

    return data


try:
    bundle = load_bundle(model_src)
    df = load_data(data_src)
except Exception as exc:
    st.error(f"Gagal memuat model atau data: {exc}")
    st.stop()

if not isinstance(bundle, dict) or "vectorizer" not in bundle or "model" not in bundle:
    st.error(
        "Format file model tidak sesuai. File .joblib harus berisi dictionary "
        "dengan kunci 'vectorizer' dan 'model'."
    )
    st.stop()

vec, model = bundle["vectorizer"], bundle["model"]
classes = list(model.classes_)
bp = bundle.get("best_params", {})
cmap = {c: COLORS.get(c, "#6366F1") for c in classes}
fitur = np.asarray(vec.get_feature_names_out())
flp = model.feature_log_prob_


# ============================================================
# EVALUASI PADA SPLIT YANG SAMA
# ============================================================
@st.cache_data(show_spinner="Menghitung evaluasi pada data uji...")
def build_eval(_key, data, _vec, _model):
    x_train, x_test, y_train, y_test = train_test_split(
        data["clean_content"],
        data["label"],
        test_size=TEST_SIZE,
        stratify=data["label"],
        random_state=SEED,
    )

    cls = list(_model.classes_)
    x_test_vec = _vec.transform(x_test)
    y_pred = _model.predict(x_test_vec)
    prob = _model.predict_proba(x_test_vec)

    roc = {}
    if len(cls) == 2:
        y_binary = (y_test == cls[1]).astype(int)
        fpr, tpr, _ = roc_curve(y_binary, prob[:, 1])
        auc_value = roc_auc_score(y_binary, prob[:, 1])
        roc[cls[1]] = (fpr, tpr, auc_value)
    else:
        y_binary = label_binarize(y_test, classes=cls)
        for i, class_name in enumerate(cls):
            # ROC-AUC tidak terdefinisi jika kelas aktual tidak memiliki
            # variasi positif/negatif pada split uji.
            if len(np.unique(y_binary[:, i])) < 2:
                continue
            fpr, tpr, _ = roc_curve(y_binary[:, i], prob[:, i])
            auc_value = roc_auc_score(y_binary[:, i], prob[:, i])
            roc[class_name] = (fpr, tpr, auc_value)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision (macro)": precision_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "Recall (macro)": recall_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "F1-Score (macro)": f1_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "F1-Score (weighted)": f1_score(
            y_test, y_pred, average="weighted", zero_division=0
        ),
        "ROC-AUC (macro)": (
            float(np.mean([item[2] for item in roc.values()]))
            if roc else float("nan")
        ),
    }

    report = pd.DataFrame(
        classification_report(
            y_test,
            y_pred,
            labels=cls,
            output_dict=True,
            zero_division=0,
        )
    ).T

    errors = pd.DataFrame(
        {
            "ulasan": x_test.to_numpy(),
            "aktual": y_test.to_numpy(),
            "prediksi": y_pred,
            "keyakinan": prob.max(axis=1),
        }
    )
    errors = errors[errors["aktual"] != errors["prediksi"]].sort_values(
        "keyakinan", ascending=False
    )

    return {
        "classes": cls,
        "metrics": metrics,
        "report": report,
        "cm": confusion_matrix(y_test, y_pred, labels=cls),
        "roc": roc,
        "errors": errors,
        "X_tr": x_train,
        "y_tr": y_train,
        "n_tr": len(x_train),
        "n_te": len(x_test),
        "y_te": y_test,
        "y_pred": y_pred,
    }


eval_key = (
    f"{len(df)}-{len(fitur)}-"
    f"{bundle.get('metrics', {}).get('Accuracy', 0):.6f}"
)
try:
    ev = build_eval(eval_key, df, vec, model)
except Exception as exc:
    st.error(
        "Gagal membentuk evaluasi. Pastikan data dan model berasal dari "
        f"proses notebook yang sama. Detail: {exc}"
    )
    st.stop()

M = ev["metrics"]
saved_acc = bundle.get("metrics", {}).get("Accuracy")
in_sync = saved_acc is None or abs(saved_acc - M["Accuracy"]) < 1e-9

with st.sidebar:
    st.divider()
    st.subheader("Model")
    st.markdown(
        f"""
        - Algoritma: **Multinomial Naive Bayes**
        - alpha: `{bp.get('nb__alpha', getattr(model, 'alpha', 'N/A'))}`
        - n-gram: `{bp.get('tfidf__ngram_range', getattr(vec, 'ngram_range', 'N/A'))}`
        - max_features: `{bp.get('tfidf__max_features', getattr(vec, 'max_features', 'N/A'))}`
        - min_df: `{bp.get('tfidf__min_df', getattr(vec, 'min_df', 'N/A'))}`
        - Fitur TF-IDF: `{len(fitur):,}`
        - Data latih / uji: `{ev['n_tr']}` / `{ev['n_te']}`
        """
    )
    if in_sync:
        st.success("Akurasi sesuai dengan metrik yang tersimpan.")
    else:
        st.warning(
            "Akurasi berbeda dari notebook. Pastikan model dan data berasal "
            "dari proses yang sama."
        )


# ============================================================
# PREPROCESSING UNTUK TEKS BARU
# ============================================================
STOPWORDS = {
    "di", "dari", "ke", "pada", "dengan", "untuk", "oleh", "tentang",
    "kepada", "yang", "dan", "atau", "tetapi", "tapi", "karena",
    "meskipun", "jika", "agar", "saya", "aku", "kamu", "dia", "kita",
    "mereka", "ini", "itu", "lah", "tuh", "nya", "dong", "aja", "gitu",
    "deh", "emang", "ya", "sih", "kok", "lho", "nih", "si", "pun",
    "juga", "sudah", "akan", "lagi", "pernah", "hanya", "adalah",
    "dalam", "ada", "para", "sebagai", "saat", "ketika", "kalau",
    "kalo", "jadi", "sama", "mau",
}
NEGATIONS = {
    "tidak", "bukan", "belum", "jangan", "kurang", "gak", "ga",
    "nggak", "ngga", "tak", "gk", "enggak",
}
DOMAIN_STOPWORDS = {"ruangguru", "rg", "ruang", "kak"}
SLANG = {
    "gak": "tidak", "ga": "tidak", "nggak": "tidak", "ngga": "tidak",
    "gk": "tidak", "enggak": "tidak", "tdk": "tidak", "tak": "tidak",
    "bgt": "banget", "bgus": "bagus", "bagu": "bagus", "bagust": "bagus",
    "mantul": "mantap", "udh": "sudah", "udah": "sudah", "sdh": "sudah",
    "blm": "belum", "yg": "yang", "dgn": "dengan", "krn": "karena",
    "karna": "karena", "utk": "untuk", "jd": "jadi", "tp": "tapi",
    "tpi": "tapi", "klo": "kalau", "kalo": "kalau", "sy": "saya",
    "sm": "sama", "sma": "sama", "bs": "bisa", "bsa": "bisa",
    "gabisa": "tidak bisa", "lemot": "lambat", "lag": "lambat",
    "ngelag": "lambat", "eror": "error", "erorr": "error", "apk": "aplikasi",
    "app": "aplikasi", "aplikasinya": "aplikasi", "thx": "terima kasih",
    "makasih": "terima kasih", "terimakasih": "terima kasih",
    "trims": "terima kasih", "oke": "ok", "okey": "ok",
    "recomended": "recommended", "rekomen": "recommended",
    "kereen": "keren", "jelek": "buruk", "jlek": "buruk",
    "pake": "pakai", "pke": "pakai", "jg": "juga", "dr": "dari",
    "dl": "dulu", "bgmn": "bagaimana", "gmn": "bagaimana",
    "knp": "kenapa", "kyk": "seperti", "bnyk": "banyak",
    "bebrpa": "beberapa",
}

STEM_PATH = os.path.join(BASE_DIR, STEM_FILE)
_stem_cache = {}
if os.path.exists(STEM_PATH):
    try:
        with open(STEM_PATH, "r", encoding="utf-8") as file:
            _stem_cache = json.load(file)
    except (OSError, ValueError):
        _stem_cache = {}


@st.cache_resource(show_spinner="Memuat stemmer Sastrawi...")
def get_stemmer():
    try:
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
        return StemmerFactory().create_stemmer()
    except ImportError:
        return None


def stem_kata(word):
    if word not in _stem_cache:
        stemmer = get_stemmer()
        _stem_cache[word] = stemmer.stem(word) if stemmer else word
    return _stem_cache[word]


def bersihkan_teks(teks):
    text = str(teks).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\bruang\s+guru\b", "ruangguru", text)
    text = re.sub(r"(.)\1{2,}", r"\1", text)
    text = re.sub(r"[^a-z\s]", " ", text)

    tokens = " ".join(SLANG.get(word, word) for word in text.split()).split()
    tokens = [
        word for word in tokens
        if (word not in STOPWORDS or word in NEGATIONS)
        and word not in DOMAIN_STOPWORDS
        and len(word) > 1
    ]
    return " ".join(stem_kata(word) for word in tokens)


def simpan_stem_cache():
    try:
        with open(STEM_PATH, "w", encoding="utf-8") as file:
            json.dump(_stem_cache, file, ensure_ascii=False)
    except OSError:
        pass


ASPEK = {
    "Mentor / Guru": r"guru|mentor|pengajar|tutor|kakak|penjelasan|menjelaskan",
    "Materi & Soal": r"materi|soal|latihan|rangkuman|pembahasan|kuis|pelajaran|kurikulum|bab",
    "Video & Animasi": r"video|animasi|tayang|nonton|subtitle|suara",
    "Harga & Langganan": r"harga|mahal|langganan|berlangganan|paket|bayar|biaya|promo|diskon|gratis|saldo|refund",
    "Teknis & Performa": r"error|bug|crash|lemot|lambat|loading|download|kuota|login|update|hang|ngelag|tidak bisa|gak bisa|susah",
    "Fitur & Tampilan": r"fitur|tampilan|menu|interface|desain|navigasi|opsi",
    "Layanan & Iklan": r"spam|iklan|telepon|sales|cs|customer|layanan|notifikasi|tawar",
}


@st.cache_data(show_spinner=False)
def aspect_table(data, key):
    low = data["content"].astype(str).str.lower()
    rows = []
    for name, pattern in ASPEK.items():
        mask = low.str.contains(pattern, regex=True, na=False)
        subset = data[mask]
        rows.append(
            {
                "Aspek": name,
                "Jumlah Ulasan": int(mask.sum()),
                "Porsi (%)": float(mask.mean() * 100) if len(mask) else 0.0,
                "Rata-rata Rating": subset["score"].mean(),
                "% Negatif": (subset["label"] == "Negatif").mean() * 100 if len(subset) else 0.0,
                "% Netral": (subset["label"] == "Netral").mean() * 100 if len(subset) else 0.0,
                "% Positif": (subset["label"] == "Positif").mean() * 100 if len(subset) else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(
        "Jumlah Ulasan", ascending=False
    ).reset_index(drop=True)


def influential(top_n=15):
    rows = []
    for i, class_name in enumerate(classes):
        other_mean = np.delete(flp, i, axis=0).mean(axis=0) if len(classes) > 1 else flp[i]
        diff = flp[i] - other_mean
        for rank, j in enumerate(np.argsort(diff)[::-1][:top_n], 1):
            rows.append(
                {
                    "Kelas": class_name,
                    "Rank": rank,
                    "Fitur": fitur[j],
                    "Skor Pengaruh": diff[j],
                    "Rasio Odds": float(np.exp(np.clip(diff[j], -700, 700))),
                    "log P(kata|kelas)": flp[i, j],
                }
            )
    return pd.DataFrame(rows)


def prediksi(teks, top_k=8):
    bersih = bersihkan_teks(teks)
    if not bersih.strip():
        return None

    x = vec.transform([bersih])
    probabilities = model.predict_proba(x)[0]
    order = np.argsort(probabilities)[::-1]
    predicted = classes[order[0]]
    reference = classes[order[1]] if len(order) > 1 else predicted

    contributions = []
    for j in x.nonzero()[1]:
        contributions.append(
            {
                "Kata": fitur[j],
                "TF-IDF": float(x[0, j]),
                f"Dorongan ke {predicted}": float(
                    (
                        flp[classes.index(predicted), j]
                        - flp[classes.index(reference), j]
                    ) * x[0, j]
                ),
            }
        )

    contribution_df = pd.DataFrame(contributions)
    if not contribution_df.empty:
        contribution_df = contribution_df.sort_values(
            contribution_df.columns[-1], ascending=False
        ).head(top_k).reset_index(drop=True)

    return {
        "clean": bersih,
        "pred": predicted,
        "proba": dict(zip(classes, probabilities)),
        "contrib": contribution_df,
    }


# ============================================================
# TABS DASHBOARD
# ============================================================
tabs = st.tabs(
    [
        "🏠 Ringkasan",
        "📊 Eksplorasi Data",
        "☁ Analisis Teks",
        "🤖 Evaluasi Model",
        "⭐ Influential Features",
        "🧩 Aspek & Anomali",
        "🔮 Prediksi",
        "📘 Metodologi",
    ]
)

# ------------------------------------------------------------
# 1. RINGKASAN
# ------------------------------------------------------------
with tabs[0]:
    metric_colors = ["#4F46E5", "#10B981", "#F59E0B", "#EC4899", "#06B6D4"]
    cols = st.columns(5)
    summary_keys = [
        "Accuracy", "Precision (macro)", "Recall (macro)",
        "F1-Score (macro)", "ROC-AUC (macro)",
    ]
    for col, key, color in zip(cols, summary_keys, metric_colors):
        value = M[key]
        display = f"{value:.3f}" if "AUC" in key else f"{value * 100:.1f}%"
        kpi(col, key, display, color)

    st.write("")
    cols = st.columns(4)
    kpi(cols[0], "Total Ulasan", f"{len(df):,}")
    kpi(cols[1], "Rata-rata Rating", f"{df['score'].mean():.2f} ⭐", "#F59E0B")
    kpi(cols[2], "% Positif", f"{(df['label'] == 'Positif').mean() * 100:.1f}%", "#10B981")
    kpi(cols[3], "% Negatif", f"{(df['label'] == 'Negatif').mean() * 100:.1f}%", "#EF4444")

    st.subheader("💡 Temuan Utama")
    report_classes = ev["report"].reindex(classes)
    if not report_classes.empty:
        best_class = report_classes["f1-score"].idxmax()
        worst_class = report_classes["f1-score"].idxmin()
        insights = [
            f"Model paling baik mengenali kelas **{best_class}** "
            f"(F1 = {report_classes.loc[best_class, 'f1-score']:.2f}) dan paling "
            f"lemah pada kelas **{worst_class}** "
            f"(F1 = {report_classes.loc[worst_class, 'f1-score']:.2f})."
        ]
    else:
        insights = []

    asp = aspect_table(df, len(df))
    if not asp.empty:
        negative_aspect = asp.sort_values("% Negatif", ascending=False).iloc[0]
        top_aspect = asp.iloc[0]
        insights.append(
            f"Aspek yang paling sering dibahas adalah **{top_aspect['Aspek']}** "
            f"({top_aspect['Porsi (%)']:.0f}% ulasan), sedangkan aspek dengan "
            f"proporsi negatif tertinggi adalah **{negative_aspect['Aspek']}** "
            f"({negative_aspect['% Negatif']:.0f}%)."
        )

    if df["at"].notna().any():
        dated = df.dropna(subset=["at"]).copy()
        dated["Tahun"] = dated["at"].dt.year
        yearly = dated.groupby("Tahun")["score"].agg(["mean", "size"])
        yearly = yearly[yearly["size"] >= 20]
        if not yearly.empty:
            insights.append(
                f"Rata-rata rating tertinggi tercatat pada tahun "
                f"**{int(yearly['mean'].idxmax())}** ({yearly['mean'].max():.2f}) "
                f"dan terendah pada **{int(yearly['mean'].idxmin())}** "
                f"({yearly['mean'].min():.2f})."
            )

    imbalance = df["label"].value_counts(normalize=True)
    if not imbalance.empty:
        insights.append(
            f"Kelas **{imbalance.idxmax()}** mendominasi data "
            f"({imbalance.max() * 100:.0f}%). F1-macro membantu melihat performa "
            f"antar kelas, bukan hanya akurasi keseluruhan."
        )

    for insight in insights:
        st.markdown(
            f'<div class="insight">{insight}</div>',
            unsafe_allow_html=True,
        )

    a, b = st.columns(2)
    label_counts = df["label"].value_counts().rename_axis("Sentimen").reset_index(name="Jumlah")
    a.plotly_chart(
        px.pie(
            label_counts,
            names="Sentimen",
            values="Jumlah",
            hole=0.55,
            color="Sentimen",
            color_discrete_map=cmap,
            title="Distribusi Sentimen",
        ),
        width="stretch",
    )

    rp = report_classes[["precision", "recall", "f1-score"]].reset_index()
    rp = rp.melt("index", var_name="Metrik", value_name="Nilai")
    b.plotly_chart(
        px.bar(
            rp,
            x="index",
            y="Nilai",
            color="Metrik",
            barmode="group",
            range_y=[0, 1],
            title="Kinerja per Kelas (data uji)",
            labels={"index": "Kelas"},
        ),
        width="stretch",
    )


# ------------------------------------------------------------
# 2. EKSPLORASI DATA
# ------------------------------------------------------------
with tabs[1]:
    data_view = df.copy()
    filter_a, filter_b = st.columns(2)

    if data_view["at"].notna().any():
        years = sorted(data_view["at"].dropna().dt.year.unique())
        year_range = filter_a.select_slider(
            "Rentang tahun",
            options=years,
            value=(years[0], years[-1]),
        )
        data_view = data_view[
            data_view["at"].dt.year.between(*year_range) | data_view["at"].isna()
        ]

    selected_labels = filter_b.multiselect(
        "Sentimen", classes, default=classes
    )
    data_view = data_view[data_view["label"].isin(selected_labels)]
    st.caption(f"{len(data_view):,} ulasan terpilih")

    a, b = st.columns(2)
    score_counts = data_view["score"].value_counts().sort_index().reset_index()
    score_counts.columns = ["Rating", "Jumlah"]
    a.plotly_chart(
        px.bar(
            score_counts,
            x="Rating",
            y="Jumlah",
            text="Jumlah",
            color="Rating",
            color_continuous_scale="Viridis",
            title="Distribusi Rating",
        ),
        width="stretch",
    )

    data_view = data_view.copy()
    data_view["panjang_kata"] = data_view["content"].str.split().str.len()
    b.plotly_chart(
        px.histogram(
            data_view,
            x="panjang_kata",
            color="label",
            nbins=30,
            barmode="overlay",
            opacity=0.7,
            color_discrete_map=cmap,
            title="Panjang Ulasan (kata)",
        ),
        width="stretch",
    )

    if data_view["at"].notna().any():
        dated = data_view.dropna(subset=["at"]).copy()
        dated["Tahun"] = dated["at"].dt.year
        a, b = st.columns(2)

        yearly_labels = dated.groupby(["Tahun", "label"]).size().reset_index(name="Jumlah")
        a.plotly_chart(
            px.bar(
                yearly_labels,
                x="Tahun",
                y="Jumlah",
                color="label",
                color_discrete_map=cmap,
                title="Jumlah Ulasan per Tahun",
            ),
            width="stretch",
        )

        yearly_rating = dated.groupby("Tahun")["score"].agg(["mean", "size"]).reset_index()
        yearly_rating.columns = ["Tahun", "Rating", "Jumlah"]
        fig = px.line(
            yearly_rating,
            x="Tahun",
            y="Rating",
            markers=True,
            hover_data=["Jumlah"],
            title="Rata-rata Rating per Tahun",
        )
        fig.update_traces(line_color="#7C3AED")
        fig.update_yaxes(range=[1, 5])
        b.plotly_chart(fig, width="stretch")

    st.subheader("Tabel Ulasan")
    query = st.text_input("Cari kata di ulasan")
    table_view = (
        data_view[data_view["content"].str.contains(query, case=False, regex=False, na=False)]
        if query else data_view
    )
    display_columns = [col for col in ["at", "score", "label", "content"] if col in table_view.columns]
    st.dataframe(table_view[display_columns].head(300), width="stretch")


# ------------------------------------------------------------
# 3. ANALISIS TEKS
# ------------------------------------------------------------
with tabs[2]:
    picked_class = st.selectbox("WordCloud untuk", ["Semua"] + classes)
    selected_text = (
        df["clean_content"]
        if picked_class == "Semua"
        else df.loc[df["label"] == picked_class, "clean_content"]
    )
    word_text = " ".join(selected_text.astype(str))
    if word_text.strip():
        map_name = {
            "Negatif": "Reds",
            "Netral": "Oranges",
            "Positif": "Greens",
        }.get(picked_class, "viridis")
        cloud = WordCloud(
            width=1000,
            height=420,
            background_color="white",
            colormap=map_name,
            max_words=100,
            random_state=SEED,
        ).generate(word_text)
        fig, ax = plt.subplots(figsize=(11, 4.6))
        ax.imshow(cloud, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Tidak ada teks untuk membuat WordCloud.")

    columns = st.columns(len(classes))
    for col, class_name in zip(columns, classes):
        words = pd.Series(
            " ".join(df.loc[df["label"] == class_name, "clean_content"]).split()
        ).value_counts().head(12)[::-1]
        col.plotly_chart(
            px.bar(
                x=words.values,
                y=words.index,
                orientation="h",
                title=f"Kata teratas – {class_name}",
                color_discrete_sequence=[cmap[class_name]],
                labels={"x": "Frekuensi", "y": ""},
            ),
            width="stretch",
        )

    st.divider()
    st.subheader("🔍 Penelusur Kata")
    keyword = st.text_input("Ketik kata/frasa (mis. error, mahal, guru, video)")
    if keyword:
        mask = df["content"].str.contains(
            keyword, case=False, regex=False, na=False
        )
        average_rating = df.loc[mask, "score"].mean()
        rating_display = f"{average_rating:.2f}" if pd.notna(average_rating) else "N/A"
        st.write(
            f"**{int(mask.sum())}** ulasan menyebut “{keyword}” · "
            f"rata-rata rating **{rating_display}**"
        )
        if mask.any():
            distribution = (
                df.loc[mask, "label"].value_counts()
                .rename_axis("Sentimen").reset_index(name="Jumlah")
            )
            a, b = st.columns([1, 2])
            a.plotly_chart(
                px.pie(
                    distribution,
                    names="Sentimen",
                    values="Jumlah",
                    hole=0.5,
                    color="Sentimen",
                    color_discrete_map=cmap,
                ),
                width="stretch",
            )
            b.dataframe(
                df.loc[mask, ["score", "label", "content"]].head(50),
                width="stretch",
            )


# ------------------------------------------------------------
# 4. EVALUASI MODEL
# ------------------------------------------------------------
with tabs[3]:
    cols = st.columns(6)
    metric_colors = ["#4F46E5", "#10B981", "#F59E0B", "#EC4899", "#8B5CF6", "#06B6D4"]
    for col, (key, value), color in zip(cols, M.items(), metric_colors):
        display = "N/A" if pd.isna(value) else (f"{value:.3f}" if "AUC" in key else f"{value * 100:.2f}%")
        kpi(col, key, display, color)

    st.write("")
    a, b = st.columns(2)
    cm = ev["cm"]
    row_totals = cm.sum(axis=1, keepdims=True)
    cm_pct = np.divide(
        cm,
        row_totals,
        out=np.zeros_like(cm, dtype=float),
        where=row_totals != 0,
    ) * 100

    mode = a.radio(
        "Tampilan confusion matrix",
        ["Jumlah", "% per kelas aktual (recall)"],
        horizontal=True,
    )
    z, fmt = (cm, "d") if mode == "Jumlah" else (cm_pct, ".1f")
    a.plotly_chart(
        px.imshow(
            z,
            x=classes,
            y=classes,
            text_auto=fmt,
            color_continuous_scale="Purples",
            labels={"x": "Prediksi", "y": "Aktual", "color": ""},
            title="Confusion Matrix",
        ),
        width="stretch",
    )

    roc_fig = go.Figure()
    for class_name, (fpr, tpr, auc_value) in ev["roc"].items():
        roc_fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"{class_name} (AUC={auc_value:.3f})",
                line=dict(width=3, color=cmap.get(class_name)),
            )
        )
    roc_fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(dash="dash", color="gray"),
            showlegend=False,
        )
    )
    roc_fig.update_layout(
        title="ROC Curve (One-vs-Rest)",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
    )
    b.plotly_chart(roc_fig, width="stretch")

    st.subheader("Classification Report")
    st.dataframe(
        ev["report"].style.format("{:.3f}").background_gradient(
            cmap="Purples",
            subset=["precision", "recall", "f1-score"],
        ),
        width="stretch",
    )

    st.subheader("Cross-Validation (opsional)")
    st.caption(
        "Menghitung ulang TF-IDF + oversampling + Naive Bayes pada data latih. "
        "Hasil ini adalah evaluasi tambahan dan bisa berbeda dari notebook."
    )
    folds = st.slider("Jumlah fold", 3, 10, 5)

    if st.button("Jalankan Cross-Validation"):
        try:
            from imblearn.over_sampling import RandomOverSampler, SMOTE
            from imblearn.pipeline import Pipeline as ImbPipeline
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.model_selection import StratifiedKFold, cross_validate
            from sklearn.naive_bayes import MultinomialNB

            min_class_count = int(ev["y_tr"].value_counts().min())
            min_train_fold = min_class_count - int(np.ceil(min_class_count / folds))
            if min_train_fold < 1:
                st.error("Data per kelas terlalu sedikit untuk jumlah fold ini.")
            else:
                smote_neighbors = min(5, min_train_fold - 1)
                sampler = (
                    SMOTE(k_neighbors=smote_neighbors, random_state=SEED)
                    if smote_neighbors >= 1
                    else RandomOverSampler(random_state=SEED)
                )

                pipeline = ImbPipeline(
                    [
                        (
                            "tfidf",
                            TfidfVectorizer(
                                ngram_range=vec.ngram_range,
                                max_features=vec.max_features,
                                min_df=vec.min_df,
                                sublinear_tf=getattr(vec, "sublinear_tf", True),
                            ),
                        ),
                        ("sampler", sampler),
                        ("nb", type(model)(alpha=model.alpha)),
                    ]
                )

                with st.spinner("Menghitung cross-validation..."):
                    cv_result = cross_validate(
                        pipeline,
                        ev["X_tr"],
                        ev["y_tr"],
                        cv=StratifiedKFold(
                            n_splits=folds, shuffle=True, random_state=SEED
                        ),
                        scoring=["accuracy", "f1_macro"],
                        error_score="raise",
                    )

                cv_data = pd.DataFrame(
                    {
                        "Fold": [f"Fold {i + 1}" for i in range(folds)],
                        "Accuracy": cv_result["test_accuracy"],
                        "F1-macro": cv_result["test_f1_macro"],
                    }
                )
                st.session_state["cv"] = cv_data
        except ImportError:
            st.error(
                "Paket imbalanced-learn belum terpasang. Jalankan: "
                "python -m pip install imbalanced-learn"
            )
        except Exception as exc:
            st.error(f"Cross-validation gagal: {exc}")

    if "cv" in st.session_state:
        cv_data = st.session_state["cv"]
        st.dataframe(cv_data, width="stretch")
        st.plotly_chart(
            px.bar(
                cv_data.melt("Fold", var_name="Metrik", value_name="Nilai"),
                x="Fold",
                y="Nilai",
                color="Metrik",
                barmode="group",
                range_y=[0, 1],
                title=(
                    f"CV: Accuracy {cv_data['Accuracy'].mean():.3f} ± "
                    f"{cv_data['Accuracy'].std():.3f} · F1-macro "
                    f"{cv_data['F1-macro'].mean():.3f} ± "
                    f"{cv_data['F1-macro'].std():.3f}"
                ),
            ),
            width="stretch",
        )

    with st.expander("❌ Analisis kesalahan prediksi"):
        errors = ev["errors"]
        st.write(f"{len(errors)} salah prediksi dari {ev['n_te']} data uji")
        pairs = (
            errors.groupby(["aktual", "prediksi"])
            .size()
            .reset_index(name="Jumlah")
            .sort_values("Jumlah", ascending=False)
        )
        st.dataframe(pairs, width="stretch")
        st.dataframe(
            errors.head(50).style.format({"keyakinan": "{:.2f}"}),
            width="stretch",
        )


# ------------------------------------------------------------
# 5. INFLUENTIAL FEATURES
# ------------------------------------------------------------
with tabs[4]:
    st.subheader("⭐ Influential Features")
    st.caption(
        "Skor pengaruh = log P(kata | kelas) dikurangi rata-rata log "
        "P(kata | kelas lain). Nilai besar menunjukkan kata yang khas "
        "untuk kelas tersebut. Rasio odds = exp(skor)."
    )

    a, b = st.columns([3, 1])
    top_n = a.slider("Jumlah fitur teratas", 5, 30, 15)
    feature_type = b.radio(
        "Jenis fitur", ["Semua", "Kata (unigram)", "Frasa (bigram+)"]
    )

    inf = influential(max(300, top_n))
    if feature_type == "Kata (unigram)":
        inf = inf[~inf["Fitur"].str.contains(" ", regex=False)]
    elif feature_type == "Frasa (bigram+)":
        inf = inf[inf["Fitur"].str.contains(" ", regex=False)]

    inf = pd.concat(
        [
            inf[inf["Kelas"] == class_name]
            .sort_values("Skor Pengaruh", ascending=False)
            .head(top_n)
            for class_name in classes
        ],
        ignore_index=True,
    )

    cols = st.columns(len(classes))
    for col, class_name in zip(cols, classes):
        selected = inf[inf["Kelas"] == class_name].sort_values("Skor Pengaruh")
        fig = px.bar(
            selected,
            x="Skor Pengaruh",
            y="Fitur",
            orientation="h",
            title=class_name,
            color_discrete_sequence=[cmap[class_name]],
            hover_data=["Rasio Odds"],
        )
        fig.update_layout(
            height=max(380, top_n * 27),
            margin=dict(l=0, r=0, t=40, b=0),
        )
        col.plotly_chart(fig, width="stretch")

    st.subheader("Heatmap log-probabilitas (kata × kelas)")
    selected_indices = []
    for i in range(len(classes)):
        other_mean = np.delete(flp, i, axis=0).mean(axis=0) if len(classes) > 1 else flp[i]
        selected_indices.extend(
            list(np.argsort(flp[i] - other_mean)[::-1][:8])
        )
    selected_indices = list(dict.fromkeys(selected_indices))
    heatmap = pd.DataFrame(
        flp[:, selected_indices].T,
        index=fitur[selected_indices],
        columns=classes,
    )
    st.plotly_chart(
        px.imshow(
            heatmap,
            aspect="auto",
            color_continuous_scale="RdPu",
            text_auto=".1f",
            height=max(420, len(heatmap) * 22),
        ),
        width="stretch",
    )

    st.subheader("🔎 Cek pengaruh sebuah kata")
    word_query = st.text_input(
        "Kata/frasa (hasil stemming, mis. error, mahal, bantu)"
    ).strip().lower()
    if word_query:
        indices = [j for j, feature in enumerate(fitur) if word_query in feature]
        if not indices:
            st.info(
                "Kata tidak ada di kosakata model (mungkin terlalu jarang "
                "atau berbeda bentuk stemming)."
            )
        else:
            rows = []
            for j in indices[:40]:
                row = {"Fitur": fitur[j]}
                for i, class_name in enumerate(classes):
                    other_values = np.delete(flp[:, j], i)
                    row[f"Skor {class_name}"] = (
                        flp[i, j] - other_values.mean()
                        if len(other_values) else 0.0
                    )
                row["Condong ke"] = max(
                    classes,
                    key=lambda name: row.get(f"Skor {name}", float("-inf")),
                )
                rows.append(row)
            score_columns = [f"Skor {class_name}" for class_name in classes]
            st.dataframe(
                pd.DataFrame(rows).style.format(
                    {column: "{:.2f}" for column in score_columns}
                ),
                width="stretch",
            )

    st.download_button(
        "⬇ Unduh Influential Features (CSV)",
        influential(30).to_csv(index=False).encode("utf-8"),
        "influential_features.csv",
        "text/csv",
    )


# ------------------------------------------------------------
# 6. ASPEK & ANOMALI
# ------------------------------------------------------------
with tabs[5]:
    st.subheader("🧩 Analisis Aspek")
    aspect_data = aspect_table(df, len(df))
    a, b = st.columns([3, 2])

    long_data = aspect_data.melt(
        "Aspek",
        ["% Negatif", "% Netral", "% Positif"],
        var_name="Sentimen",
        value_name="Persen",
    )
    long_data["Sentimen"] = long_data["Sentimen"].str.replace(
        "% ", "", regex=False
    )
    a.plotly_chart(
        px.bar(
            long_data,
            y="Aspek",
            x="Persen",
            color="Sentimen",
            orientation="h",
            barmode="stack",
            color_discrete_map=cmap,
            title="Komposisi Sentimen per Aspek (%)",
            category_orders={
                "Aspek": aspect_data.sort_values("% Negatif")["Aspek"].tolist()
            },
        ),
        width="stretch",
    )
    b.plotly_chart(
        px.bar(
            aspect_data.sort_values("Jumlah Ulasan"),
            x="Jumlah Ulasan",
            y="Aspek",
            orientation="h",
            color="Rata-rata Rating",
            color_continuous_scale="RdYlGn",
            range_color=[2, 5],
            title="Seberapa Sering Aspek Disebut",
        ),
        width="stretch",
    )
    st.dataframe(
        aspect_data.style.format(
            {
                "Porsi (%)": "{:.1f}",
                "Rata-rata Rating": "{:.2f}",
                "% Negatif": "{:.1f}",
                "% Netral": "{:.1f}",
                "% Positif": "{:.1f}",
            }
        ).background_gradient(cmap="Reds", subset=["% Negatif"]),
        width="stretch",
    )

    st.subheader("Telusuri Ulasan per Aspek")
    c1, c2 = st.columns(2)
    selected_aspect = c1.selectbox("Aspek", list(ASPEK))
    default_sentiments = [name for name in classes if name == "Negatif"] or classes
    selected_sentiments = c2.multiselect(
        "Sentimen", classes, default=default_sentiments
    )
    aspect_mask = (
        df["content"].str.lower().str.contains(
            ASPEK[selected_aspect], regex=True, na=False
        )
        & df["label"].isin(selected_sentiments)
    )
    st.write(f"{int(aspect_mask.sum())} ulasan")
    aspect_columns = [
        col for col in ["at", "score", "label", "content"] if col in df.columns
    ]
    st.dataframe(
        df.loc[aspect_mask, aspect_columns].head(100),
        width="stretch",
    )

    st.divider()
    st.subheader("⚠ Ulasan Anomali: Teks Berlawanan dengan Rating")
    st.caption(
        "Menampilkan ulasan saat label pada dataset berbeda dari prediksi teks "
        "model dengan keyakinan tinggi. Label dataset berasal dari rating."
    )
    confidence = st.slider("Minimal keyakinan model", 0.5, 0.99, 0.80, 0.01)
    probabilities = model.predict_proba(vec.transform(df["clean_content"]))
    anomalies = df[["score", "label", "content"]].copy()
    anomalies["prediksi_teks"] = model.classes_[probabilities.argmax(axis=1)]
    anomalies["keyakinan"] = probabilities.max(axis=1)
    anomalies = anomalies[
        (anomalies["prediksi_teks"] != anomalies["label"])
        & (anomalies["keyakinan"] >= confidence)
    ].sort_values("keyakinan", ascending=False)
    st.write(f"{len(anomalies)} ulasan anomali")
    st.dataframe(
        anomalies.head(100).style.format({"keyakinan": "{:.2f}"}),
        width="stretch",
    )


# ------------------------------------------------------------
# 7. PREDIKSI
# ------------------------------------------------------------
with tabs[6]:
    st.subheader("🔮 Prediksi Sentimen Real-time")
    example = st.selectbox(
        "Contoh cepat",
        [
            "(ketik sendiri)",
            "Aplikasinya bagus banget, materi lengkap dan gurunya jelas!",
            "Sering error, tidak bisa login, kecewa sekali",
            "Lumayan lah tapi masih banyak yang perlu diperbaiki",
        ],
    )
    review_text = st.text_area(
        "Masukkan ulasan",
        "" if example == "(ketik sendiri)" else example,
        height=110,
    )

    if st.button("Analisis Sentimen", type="primary"):
        if not review_text.strip():
            st.warning("Masukkan ulasan terlebih dahulu.")
        else:
            with st.spinner("Memproses..."):
                output = prediksi(review_text)
                simpan_stem_cache()

            if output is None:
                st.warning(
                    "Teks kosong setelah dibersihkan. Coba kalimat lain."
                )
            else:
                emoji = {
                    "Positif": "😄",
                    "Netral": "😐",
                    "Negatif": "😡",
                }.get(output["pred"], "🤔")
                a, b = st.columns([1, 2])
                a.markdown(
                    f'<div class="kpi" style="border-left-color:{cmap[output["pred"]]}">'
                    f'<div class="l">Prediksi</div>'
                    f'<div class="v">{emoji} {output["pred"]}</div></div>',
                    unsafe_allow_html=True,
                )
                a.caption(f"Teks bersih: `{output['clean']}`")

                probabilities_df = pd.DataFrame(
                    {
                        "Kelas": list(output["proba"]),
                        "Probabilitas": list(output["proba"].values()),
                    }
                )
                b.plotly_chart(
                    px.bar(
                        probabilities_df,
                        x="Probabilitas",
                        y="Kelas",
                        orientation="h",
                        color="Kelas",
                        color_discrete_map=cmap,
                        range_x=[0, 1],
                        text=probabilities_df["Probabilitas"].map(
                            "{:.1%}".format
                        ),
                    ),
                    width="stretch",
                )

                if not output["contrib"].empty:
                    st.markdown("**Kata yang paling mendorong keputusan ini:**")
                    st.dataframe(output["contrib"], width="stretch")

    st.divider()
    st.subheader("📂 Prediksi Batch (CSV)")
    uploaded_batch = st.file_uploader(
        "CSV dengan kolom 'content'",
        type=["csv"],
        key="batch",
    )
    if uploaded_batch is not None:
        try:
            batch_df = pd.read_csv(uploaded_batch)
            if "content" not in batch_df.columns:
                st.error("Kolom 'content' tidak ditemukan.")
            else:
                with st.spinner("Memproses ulasan..."):
                    cleaned = batch_df["content"].fillna("").astype(str).apply(
                        bersihkan_teks
                    )
                    simpan_stem_cache()

                batch_probabilities = model.predict_proba(vec.transform(cleaned))
                batch_df["prediksi"] = model.classes_[
                    batch_probabilities.argmax(axis=1)
                ]
                batch_df["keyakinan"] = batch_probabilities.max(axis=1).round(3)
                st.dataframe(batch_df.head(100), width="stretch")
                st.plotly_chart(
                    px.pie(
                        batch_df,
                        names="prediksi",
                        hole=0.5,
                        color="prediksi",
                        color_discrete_map=cmap,
                    ),
                    width="stretch",
                )
                st.download_button(
                    "⬇ Unduh hasil",
                    batch_df.to_csv(index=False).encode("utf-8"),
                    "hasil_prediksi.csv",
                    "text/csv",
                )
        except Exception as exc:
            st.error(f"Gagal memproses CSV batch: {exc}")


# ------------------------------------------------------------
# 8. METODOLOGI
# ------------------------------------------------------------
with tabs[7]:
    st.subheader("📘 Alur Proses")
    st.markdown(
        """
        Proses berikut dijalankan di **notebook**; aplikasi ini menampilkan
        hasilnya dan menyediakan prediksi teks baru.

        1. **Load data** ulasan Ruangguru dari Google Play Store.
        2. **Pelabelan** berdasarkan rating: 1–2 Negatif, 3 Netral, 4–5 Positif.
        3. **Pra-pemrosesan**: case folding, penghapusan URL/simbol, normalisasi
           slang, stopword (negasi dipertahankan), dan stemming Sastrawi.
        4. **Split stratified** 80/20 dengan `random_state=42`; TF-IDF dan
           SMOTE seharusnya diterapkan hanya pada data latih.
        5. **Tuning** GridSearchCV menggunakan F1-macro dan pelatihan
           Multinomial Naive Bayes.
        6. **Evaluasi**: Accuracy, Precision, Recall, F1, confusion matrix,
           ROC-AUC One-vs-Rest, serta cross-validation.
        7. **Influential Features** berdasarkan `feature_log_prob_` dan
           **analisis aspek** berbasis kata kunci.

        **Catatan:** hasil evaluasi akan sesuai dengan notebook hanya jika data,
        preprocessing, urutan kelas, vektorisasi, dan pembagian data sama.
        Label yang berasal dari rating mengukur kemampuan model memprediksi
        label berbasis rating dari teks, bukan sentimen hasil anotasi manual.
        """
    )
    st.download_button(
        "⬇ Unduh data bersih + label (CSV)",
        df.to_csv(index=False).encode("utf-8"),
        "data_bersih.csv",
        "text/csv",
    )
