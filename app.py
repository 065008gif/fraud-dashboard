import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from model import train_and_evaluate
from sklearn.metrics import confusion_matrix
import pandas as pd
import time

df = pd.read_csv("creditcard.csv")

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="💳",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

/* ===== Main App ===== */
.stApp {
    background: linear-gradient(135deg, #0F172A, #020617);
    color: white;
}

/* ===== Text ===== */
body, p {
    color: white;
}

h1, h2, h3 {
    color: white;
}

/* ===== Sidebar ===== */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617, #0F172A);
    border-right: 1px solid rgba(255,255,255,0.05);
}

/* Sidebar labels & headers ONLY */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: white !important;
    font-size: 15px;
}

/* ===== FIX SELECTBOX ===== */

/* Selectbox container */
section[data-testid="stSidebar"] div[data-baseweb="select"] {
    background-color: white !important;
    border-radius: 10px !important;
}

/* Selected value text */
section[data-testid="stSidebar"] div[data-baseweb="select"] span {
    color: black !important;
    font-weight: 500;
}

/* Dropdown menu */
div[role="listbox"] {
    background-color: white !important;
}

/* Dropdown options */
div[role="option"] span {
    color: black !important;
}

/* Hover effect */
div[role="option"]:hover {
    background-color: #f3f4f6 !important;
}

/* ===== Cards ===== */
.glass-card {
    background: rgba(255,255,255,0.05);
    border-radius: 16px;
    padding: 25px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.08);
}

/* ===== Metrics Polish ===== */
.metric-value {
    font-size: 38px !important;
    font-weight: 800 !important;
    color: #FF4B4B !important;
}

/* ===== Buttons ===== */
div.stButton > button {
    background: linear-gradient(135deg, #FF4B4B, #ff2e2e);
    color: white;
    border-radius: 12px;
    font-size: 16px;
    font-weight: 600;
    border: none;
}

/* ===== Tabs ===== */
button[data-baseweb="tab"] {
    color: white !important;
    font-size: 16px;
}

/* ===== Section Box ===== */
.section-box {
    background: rgba(255,255,255,0.04);
    padding: 25px;
    border-radius: 18px;
    margin-bottom: 25px;
    border: 1px solid rgba(255,255,255,0.05);
}

/* ===================================================== */
/* ===== Selectbox Fix (STEP 3 – Learning Rate) ===== */
/* ===================================================== */

/* Selected value */
section[data-testid="stSidebar"] div[data-baseweb="select"] span {
    color: black !important;
}

/* Dropdown background */
div[role="listbox"] {
    background-color: white !important;
}

/* Dropdown text */
div[role="listbox"] span {
    color: black !important;
}

/* Hover effect */
div[role="option"]:hover {
    background-color: #f1f1f1 !important;
}

/* Input text */
section[data-testid="stSidebar"] input {
    color: black !important;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<h1 style='text-align: center;'>💳 Credit Card Fraud Detection Dashboard</h1>
<p style='text-align: center; font-size:18px;'>
Artificial Neural Network for Fraud Risk Classification
</p>
""", unsafe_allow_html=True)

# ================= DATASET INSIGHTS =================
st.markdown("<div class='section-box'>", unsafe_allow_html=True)
st.markdown("## 📊 Dataset Insights")

fraud_count = df['Class'].sum()
total = len(df)
fraud_pct = (fraud_count / total) * 100

c1, c2, c3 = st.columns(3)
c1.metric("Total Transactions", f"{total:,}")
c2.metric("Fraudulent Transactions", f"{int(fraud_count):,}")
c3.metric("Fraud Rate", f"{fraud_pct:.4f}%")

st.warning(f"⚠️ Severe Class Imbalance: Only {fraud_pct:.4f}% fraud cases")

st.markdown("</div>", unsafe_allow_html=True)

# ================= SIDEBAR =================
st.sidebar.markdown("## ⚙️ Model Controls")

hidden_layers = st.sidebar.slider("Hidden Layers", 1, 5, 2)
neurons = st.sidebar.slider("Neurons per Layer", 8, 128, 32)
learning_rate = st.sidebar.selectbox("Learning Rate", [0.1, 0.01, 0.001])
epochs = st.sidebar.slider("Epochs", 5, 50, 10)

# ================= TABS (TRAINING FIRST) =================
tab_train, tab_overview, tab_diag = st.tabs(
    ["🚀 Training", "📊 Overview", "🧪 Diagnostics"]
)

# ---------------- 🚀 TRAINING ----------------
with tab_train:
    st.markdown("## 🚀 Train Model")

    if st.button("Start Training"):

        with st.spinner("Training ANN Model..."):
            accuracy, recall, precision, f1, history, y_true, y_pred = train_and_evaluate(
                hidden_layers,
                neurons,
                learning_rate,
                epochs
            )

        st.success("✅ Training Complete")

        # ---- Metrics Row ----
        st.markdown("### 📊 Model Performance")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{accuracy:.4f}")
        c2.metric("Recall (Fraud Detection)", f"{recall:.4f}")
        c3.metric("Precision", f"{precision:.4f}")
        c4.metric("F1 Score", f"{f1:.4f}")

        # ---- Loss Curve ----
        st.markdown("### 📉 Training Loss Curve")

        fig, ax = plt.subplots()
        ax.plot(history.history['loss'], label="Training Loss")
        ax.plot(history.history['val_loss'], label="Validation Loss")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.legend()

        st.pyplot(fig)

# ---------------- 📊 OVERVIEW ----------------
with tab_overview:
    st.markdown("## 🔧 Current Configuration")

    col1, col2, col3, col4 = st.columns(4)

    configs = [
        (hidden_layers, "Hidden Layers"),
        (neurons, "Neurons"),
        (learning_rate, "Learning Rate"),
        (epochs, "Epochs")
    ]

    for col, (val, label) in zip([col1, col2, col3, col4], configs):
        with col:
            st.markdown(
                f"""
                <div class='glass-card'>
                    <h2>{val}</h2>
                    <p>{label}</p>
                </div>
                """,
                unsafe_allow_html=True
            )


## ---------------- 🧪 DIAGNOSTICS ----------------
with tab_diag:
    st.markdown("## 🧪 Diagnostics")

    if st.button("Generate Confusion Matrix"):

        with st.spinner("Running diagnostics..."):
            accuracy, recall, precision, f1, history, y_true, y_pred = train_and_evaluate(
                hidden_layers,
                neurons,
                learning_rate,
                epochs
            )

        st.success("✅ Diagnostics Ready")

        # ---- Confusion Matrix ----
        cm = confusion_matrix(y_true, y_pred)

        st.markdown("### 🔍 Confusion Matrix")

        fig2, ax2 = plt.subplots()
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Reds",
            ax=ax2
        )

        ax2.set_xlabel("Predicted")
        ax2.set_ylabel("Actual")

        st.pyplot(fig2)

        # ---- Metrics ----
        st.markdown("### 📊 Evaluation Metrics")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{accuracy:.4f}")
        c2.metric("Recall", f"{recall:.4f}")
        c3.metric("Precision", f"{precision:.4f}")
        c4.metric("F1 Score", f"{f1:.4f}")
