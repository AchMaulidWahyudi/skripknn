import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

@st.cache_resource
def load_model():
    return joblib.load("knn_model_26fitur_k5_manhattan.pkl")

model = load_model()

BEST_FEATURES = [
    'having_IP_Address', 'URL_Length', 'Shortining_Service', 'having_At_Symbol',
    'double_slash_redirecting', 'Prefix_Suffix', 'having_Sub_Domain', 'SSLfinal_State',
    'Domain_registeration_length', 'Favicon', 'port', 'HTTPS_token', 'Request_URL',
    'URL_of_Anchor', 'Links_in_tags', 'SFH', 'Submitting_to_email', 'Abnormal_URL',
    'Redirect', 'on_mouseover', 'RightClick', 'popUpWidnow', 'Iframe', 'age_of_domain',
    'DNSRecord', 'web_traffic'
]

st.set_page_config(page_title="KNN Phishing Website Detector", layout="wide")
st.title("Deteksi Phishing Website menggunakan KNN")

menu = st.sidebar.radio("Navigasi", ["📊 Ringkasan Hasil Penelitian", "🧪 Uji Coba Data Baru"])

if menu == "📊 Ringkasan Hasil Penelitian":
    st.header("📊 Ringkasan Eksperimen & Model Terbaik")
    st.markdown("""
    Model terbaik:
    - 26 fitur (hasil seleksi dengan Information Gain)
    - k=5, weights='distance', metric='manhattan'
    - F1 Score (80:20 split): 0.9742
    - F1 Score (10-fold CV): 0.9758
    """)
    st.subheader("📌 Daftar 26 Fitur Terpilih")
    st.write(pd.DataFrame(BEST_FEATURES, columns=["Fitur"]).reset_index(drop=True))

elif menu == "🧪 Uji Coba Data Baru":
    st.header("🧪 Uji Model terhadap Data Baru")
    uploaded_file = st.file_uploader("Unggah file CSV (delimiter ;)", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, delimiter=';')
            missing_cols = [col for col in BEST_FEATURES if col not in df.columns]
            if missing_cols:
                st.error(f"Kolom berikut tidak ditemukan: {missing_cols}")
            else:
                X_test = df[BEST_FEATURES]
                y_pred = model.predict(X_test)
                df['Prediksi'] = ['Phishing' if p == -1 else 'Legitimate' for p in y_pred]

                st.success("Prediksi berhasil dilakukan.")
                st.dataframe(df.head(10))

                if 'Result' in df.columns:
                    st.subheader("📈 Evaluasi terhadap Label Aktual")
                    y_true = df['Result']
                    report = classification_report(y_true, y_pred, output_dict=True)
                    st.dataframe(pd.DataFrame(report).transpose())

                    st.subheader("📉 Confusion Matrix")
                    cm = confusion_matrix(y_true, y_pred, labels=[1, -1])
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                                xticklabels=["Legitimate", "Phishing"],
                                yticklabels=["Legitimate", "Phishing"])
                    ax.set_xlabel("Prediksi")
                    ax.set_ylabel("Kelas Sebenarnya")
                    st.pyplot(fig)
                else:
                    st.info("Kolom 'Result' tidak ditemukan. Menampilkan hasil prediksi saja.")
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")
