import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score

# Konfigurasi halaman
st.set_page_config(page_title="🧠 Aplikasi Data Mining", layout="centered")

# Header
st.markdown("""
    <meta name="google" content="notranslate">
    <div style='text-align: center;'>
        <h1 style='margin-bottom: 0;'>🧠 APLIKASI DATA MINING </h1>
        <p style='font-size: 0.9rem; color: gray; margin-top: 0;'>
            Data Bukan Sekadar Angka — Ia Adalah Cerita Yang Menunggu Untuk Ditemukan.
        </p>
    </div>
""", unsafe_allow_html=True)

# Upload file
uploaded_file = st.file_uploader("📁 Upload file CSV dataset kamu", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("🔍 Preview Dataset")
    st.dataframe(df.head())

    st.subheader("📌 Ringkasan Dataset")
    col1, col2 = st.columns(2)
    col1.metric("Jumlah Baris", df.shape[0])
    col2.metric("Jumlah Kolom", df.shape[1])

    st.subheader("🌐 Informasi Dataset")
    info_df = pd.DataFrame({
        "Kolom": df.columns,
        "Tipe Data": df.dtypes.astype(str).values,
        "Non-Null": df.notnull().sum().values,
        "Null": df.isnull().sum().values
    })
    st.dataframe(info_df)

    st.subheader("📊 Statistik Deskriptif")
    st.dataframe(df.describe(include='all'))

    # Visualisasi
    fitur_num = df.select_dtypes(include=np.number).columns.tolist()
    fitur_cat = df.select_dtypes(include='object').columns.tolist()

    st.subheader("📈 Visualisasi Numerik")
    if fitur_num:
        selected_num = st.selectbox("Pilih fitur numerik", fitur_num)
        fig, ax = plt.subplots()
        sns.histplot(df[selected_num], kde=True, ax=ax, color="skyblue")
        ax.set_title(f'Distribusi {selected_num}')
        st.pyplot(fig)
    else:
        st.warning("Tidak ada fitur numerik.")

    st.subheader("🧾 Visualisasi Kategorikal")
    if fitur_cat:
        selected_cat = st.selectbox("Pilih fitur kategorikal", fitur_cat)
        fig, ax = plt.subplots()
        sns.countplot(x=selected_cat, data=df, palette='pastel', ax=ax)
        ax.set_title(f'Frekuensi Kategori {selected_cat}')
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)
    else:
        st.warning("Tidak ada fitur kategorikal.")

    st.subheader("📉 Korelasi Fitur Numerik")
    if len(fitur_num) >= 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df[fitur_num].corr(), annot=True, cmap='coolwarm', ax=ax)
        ax.set_title("Matriks Korelasi")
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("⚙️ Pemodelan Machine Learning")

    algo = st.selectbox("Pilih Algoritma", [
        "Regresi Linier", "Regresi Logistik", "Naive Bayes",
        "SVM", "KNN", "Decision Tree", "K-Means"
    ])

    def is_classification(y):
        return y.dtype == 'object' or y.nunique() <= 15

    if algo != "K-Means":
        target = st.selectbox("Pilih Kolom Target", df.columns)
        X = pd.get_dummies(df.drop(columns=[target]))
        y = df[target]
        problem = "klasifikasi" if is_classification(y) else "regresi"

        if algo in ["Regresi Logistik", "Naive Bayes", "SVM", "KNN", "Decision Tree"] and problem != "klasifikasi":
            st.error("❌ Target harus berupa label diskrit.")
            st.stop()
        if algo == "Regresi Linier" and problem != "regresi":
            st.error("❌ Target harus berupa nilai numerik.")
            st.stop()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        X = df.select_dtypes(include=np.number)

    if st.button("🚀 Jalankan Model"):
        st.markdown("---")
        st.subheader(f"Hasil: {algo}")

        if algo == "Regresi Linier":
            model = LinearRegression().fit(X_train, y_train)
            y_pred = model.predict(X_test)
            st.metric("R² Score", f"{r2_score(y_test, y_pred):.2f}")
            st.metric("MSE", f"{mean_squared_error(y_test, y_pred):.2f}")
            fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual', 'y': 'Predicted'})
            st.plotly_chart(fig)

        elif algo == "Regresi Logistik":
            model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
            y_pred = model.predict(X_test)
            st.metric("Akurasi", f"{accuracy_score(y_test, y_pred):.2f}")
            st.text(classification_report(y_test, y_pred))
            fig = px.imshow(confusion_matrix(y_test, y_pred), text_auto=True)
            st.plotly_chart(fig)

        elif algo == "Naive Bayes":
            model = GaussianNB().fit(X_train, y_train)
            y_pred = model.predict(X_test)
            st.metric("Akurasi", f"{accuracy_score(y_test, y_pred):.2f}")
            st.text(classification_report(y_test, y_pred))
            fig = px.imshow(confusion_matrix(y_test, y_pred), text_auto=True)
            st.plotly_chart(fig)

        elif algo == "SVM":
            model = SVC().fit(X_train, y_train)
            y_pred = model.predict(X_test)
            st.metric("Akurasi", f"{accuracy_score(y_test, y_pred):.2f}")
            st.text(classification_report(y_test, y_pred))
            fig = px.imshow(confusion_matrix(y_test, y_pred), text_auto=True)
            st.plotly_chart(fig)

        elif algo == "KNN":
            model = KNeighborsClassifier().fit(X_train, y_train)
            y_pred = model.predict(X_test)
            st.metric("Akurasi", f"{accuracy_score(y_test, y_pred):.2f}")
            st.text(classification_report(y_test, y_pred))
            fig = px.imshow(confusion_matrix(y_test, y_pred), text_auto=True)
            st.plotly_chart(fig)

        elif algo == "Decision Tree":
            model = DecisionTreeClassifier().fit(X_train, y_train)
            y_pred = model.predict(X_test)
            st.metric("Akurasi", f"{accuracy_score(y_test, y_pred):.2f}")
            st.text(classification_report(y_test, y_pred))
            fig = px.imshow(confusion_matrix(y_test, y_pred), text_auto=True)
            st.plotly_chart(fig)

        elif algo == "K-Means":
            k = st.slider("Jumlah Cluster", 2, 10, 3)
            model = KMeans(n_clusters=k).fit(X)
            df['Cluster'] = model.labels_
            st.dataframe(df['Cluster'].value_counts())
            fig = px.scatter(df, x=X.columns[0], y=X.columns[1], color='Cluster')
            st.plotly_chart(fig)

        def download_csv(df):
            buffer = BytesIO()
            df.to_csv(buffer, index=False)
            return buffer.getvalue()

        hasil = (
            X_test.copy().assign(Actual=y_test, Predicted=y_pred)
            if algo != "K-Means" else df
        )

        st.download_button("📥 Download Hasil CSV", data=download_csv(hasil),
                           file_name="hasil_model.csv", mime="text/csv")

else:
    st.info("📌 Silakan Upload File CSV Untuk Memulai Analisis.")
