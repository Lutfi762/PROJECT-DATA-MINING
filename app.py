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
from sklearn.utils.multiclass import type_of_target

# Konfigurasi halaman
st.set_page_config(page_title="🧠 Aplikasi Analisis & Prediksi CSV", layout="centered")

# Header
st.markdown("""
    <meta name="google" content="notranslate">
    <div style='text-align: center;'>
        <h1 style='margin-bottom: 0;'>🧠 Aplikasi Analisis & Prediksi CSV </h1>
        <p style='font-size: 0.9rem; color: gray; margin-top: 0;'>
            Data bukan sekadar angka — ia adalah cerita yang menunggu untuk ditemukan.
        </p>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
### 📘 Tentang Aplikasi
Aplikasi ini mampu memprediksi data dari berbagai kasus hanya dengan mengunggah dataset CSV. Algoritma Machine Learning akan otomatis disesuaikan berdasarkan jenis data target yang kamu pilih.

**Jenis Algoritma & Kegunaannya:**
| Algoritma             | Jenis Target          | Contoh Kasus                                 |
|----------------------|------------------------|----------------------------------------------|
| Regresi Linier        | Numerik                | Prediksi harga, suhu                         |
| Regresi Logistik      | Kategorikal            | Kelulusan, penyakit (Ya/Tidak)               |
| Naive Bayes           | Kategorikal            | Filtering spam                               |
| SVM                   | Kategorikal            | Deteksi emosi/sentimen                       |
| KNN                   | Kategorikal            | Segmentasi pelanggan                         |
| Decision Tree         | Kategorikal            | Diagnosis penyakit                           |
| K-Means               | Tidak ada target       | Segmentasi tanpa label (unsupervised)        |
""", unsafe_allow_html=True)

# Upload CSV
uploaded_file = st.file_uploader("📁 Upload file CSV kamu", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File berhasil diunggah!")
    st.dataframe(df.head())

    st.subheader("🔍 Preview Dataset")
    st.dataframe(df.head())

    st.subheader("📌 Ringkasan Dataset")
    col1, col2 = st.columns(2)
    col1.metric("Jumlah Baris", df.shape[0])
    col2.metric("Jumlah Kolom", df.shape[1])

    st.subheader("🌐 Informasi Dataset")
    info_df = pd.DataFrame({
        "Kolom": df.columns,
        "Tipe Data": df.dtypes.astype(str),
        "Non-Null": df.notnull().sum(),
        "Null": df.isnull().sum()
    }).reset_index(drop=True)
    st.dataframe(info_df)
    st.markdown("""
**Keterangan:**
- Kolom: Nama fitur.
- Tipe Data: Jenis datanya (numerik/kategorikal).
- Non-Null: Jumlah data valid.
- Null: Jumlah data kosong yang perlu diproses.
""")

    st.subheader("📊 Statistik Deskriptif")
    st.dataframe(df.describe(include='all'))
    st.markdown("""
**📘 Penjelasan Statistik Deskriptif:**
- **count**: Jumlah data tidak kosong.
- **mean**: Nilai rata-rata.
- **std**: Penyimpangan baku dari rata-rata (semakin besar berarti data tersebar).
- **min/max**: Nilai terkecil dan terbesar.
- **25%, 50%, 75%**: Kuartil, membantu memahami distribusi data (misal 50% adalah median).
""")

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
        st.markdown(f"""
**📘 Penjelasan Grafik Histogram:**
- Grafik memperlihatkan **penyebaran data** pada kolom `{selected_num}`.
- Garis lengkung (KDE) memperlihatkan estimasi distribusi: apakah data simetris (normal) atau tidak.
- Jika grafik **miring ke kiri/kanan (skew)**, artinya data tidak seimbang.
- **Puncak tinggi** menandakan banyak data terkumpul pada nilai tersebut.
""")
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
        st.markdown(f"""
**📘 Penjelasan Grafik Kategori `{selected_cat}`:**
- Setiap batang menunjukkan **jumlah data** dalam masing-masing kategori.
- Jika satu batang jauh lebih tinggi dari yang lain, artinya data **tidak seimbang**.
- Hal ini penting diketahui karena ketidakseimbangan bisa mempengaruhi kinerja model klasifikasi.
""")
    else:
        st.warning("Tidak ada fitur kategorikal.")
    
    st.subheader("📉 Korelasi Fitur Numerik")
    if len(fitur_num) >= 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df[fitur_num].corr(), annot=True, cmap='coolwarm', ax=ax)
        ax.set_title("Matriks Korelasi")
        st.pyplot(fig)
        st.markdown("""
**📘 Penjelasan Korelasi:**
- **Nilai korelasi** berkisar dari -1 hingga 1.
    - **1** artinya hubungan positif sempurna: saat satu naik, yang lain ikut naik.
    - **-1** artinya negatif sempurna: saat satu naik, yang lain turun.
    - **0** artinya tidak ada hubungan.
- Warna **biru** = korelasi positif.  
- Warna **merah** = korelasi negatif.
- Ini membantu memilih fitur penting untuk model.
""")

    st.markdown("---")
    st.subheader("⚙️ Pemodelan Machine Learning")
    algo = st.selectbox("Pilih Algoritma", [
        "Regresi Linier", "Regresi Logistik", "Naive Bayes",
        "SVM", "KNN", "Decision Tree", "K-Means"
    ])

    if algo == "K-Means":
        n_clusters = st.slider("Pilih Jumlah Cluster", min_value=2, max_value=10, value=3)
        X = df.select_dtypes(include=np.number)
    else:
        target = st.selectbox("Pilih Kolom Target", df.columns)
        X = pd.get_dummies(df.drop(columns=[target]))
        y = df[target]

        def is_classification(y):
            return y.dtype == 'object' or y.nunique() <= 15

        problem = "klasifikasi" if is_classification(y) else "regresi"

        if algo == "Regresi Linier" and problem != "regresi":
            st.error("❌ Regresi Linier hanya untuk target numerik.")
            st.stop()

        if algo != "Regresi Linier" and problem != "klasifikasi":
            st.error("❌ Algoritma ini hanya untuk target kategorikal.")
            st.stop()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Tombol tetap di luar
    if st.button("🚀 Jalankan Model"):
        st.markdown("## 🧪 Hasil Model")
        # lanjutkan model training & evaluasi di sini

        if algo == "Regresi Linier":
            model = LinearRegression().fit(X_train, y_train)
            y_pred = model.predict(X_test)
            st.metric("R² Score", f"{r2_score(y_test, y_pred):.2f}")
            st.metric("MSE", f"{mean_squared_error(y_test, y_pred):.2f}")
            st.plotly_chart(px.scatter(x=y_test, y=y_pred, labels={'x': 'Aktual', 'y': 'Prediksi'}))

        elif algo == "Regresi Logistik":
            model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
            y_pred = model.predict(X_test)
            st.metric("Akurasi", f"{accuracy_score(y_test, y_pred):.2f}")
            st.text(classification_report(y_test, y_pred))
            st.plotly_chart(px.imshow(confusion_matrix(y_test, y_pred), text_auto=True))

        elif algo == "Naive Bayes":
            try:
                model = GaussianNB().fit(X_train, y_train)
                y_pred = model.predict(X_test)
                st.metric("Akurasi", f"{accuracy_score(y_test, y_pred):.2f}")
                st.text(classification_report(y_test, y_pred))
            except ValueError as e:
                st.error("❌ Target harus berupa label kategorikal untuk Naive Bayes (bukan numerik kontinyu).")
                model = None

        elif algo == "SVM":
            model = SVC().fit(X_train, y_train)
            y_pred = model.predict(X_test)
            st.metric("Akurasi", f"{accuracy_score(y_test, y_pred):.2f}")
            st.text(classification_report(y_test, y_pred))
            st.plotly_chart(px.imshow(confusion_matrix(y_test, y_pred), text_auto=True))

        elif algo == "KNN":
            model = KNeighborsClassifier().fit(X_train, y_train)
            y_pred = model.predict(X_test)
            st.metric("Akurasi", f"{accuracy_score(y_test, y_pred):.2f}")
            st.text(classification_report(y_test, y_pred))
            st.plotly_chart(px.imshow(confusion_matrix(y_test, y_pred), text_auto=True))

        elif algo == "Decision Tree":
            model = DecisionTreeClassifier().fit(X_train, y_train)
            y_pred = model.predict(X_test)
            st.metric("Akurasi", f"{accuracy_score(y_test, y_pred):.2f}")
            st.text(classification_report(y_test, y_pred))
            st.plotly_chart(px.imshow(confusion_matrix(y_test, y_pred), text_auto=True))

        elif algo == "K-Means":
            X = df.select_dtypes(include=np.number)  # Gunakan fitur numerik saja
            model = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
            df['Cluster'] = model.labels_

            st.success(f"✅ K-Means selesai dengan {n_clusters} cluster.")
            st.write("Hasil Cluster:")
            st.dataframe(df[['Cluster'] + X.columns.tolist()])

            # Visualisasi cluster (jika 2D saja)
            if X.shape[1] >= 2:
                fig = px.scatter(df, x=X.columns[0], y=X.columns[1], color=df['Cluster'].astype(str),
                                 title="Visualisasi Cluster (2D)", labels={'color': 'Cluster'})
                st.plotly_chart(fig)

        # === Prediksi Manual ===
        if 'model' in locals() and model is not None:
            st.subheader("🧠 Prediksi Manual")
            st.markdown("Masukkan data untuk memprediksi hasil menggunakan model yang telah dilatih.")

            try:
                input_data = {}
                manual_input_cols = X.columns.tolist()

                for col in manual_input_cols:
                    input_data[col] = st.number_input(f"{col}", value=float(X[col].mean()))

                input_df = pd.DataFrame([input_data])
                manual_pred = model.predict(input_df)[0]
                st.success(f"📌 Hasil Prediksi Manual: **{manual_pred}**")

            except Exception as e:
                st.error(f"Gagal menampilkan form input: {e}")

            # Analisis Pengaruh Fitur
            if algo == "Naive Bayes" and 'sex' in X.columns:
                st.markdown("### 📍 Pengaruh Fitur: `sex` terhadap Prediksi")
                mean_by_class = df.groupby(y)['sex'].mean()
                st.write("Rata-rata nilai `sex` per kelas target:")
                st.write(mean_by_class)
                st.markdown(f"""
**Interpretasi:**
- Nilai rata-rata `sex` pada kelas target menunjukkan pengaruhnya terhadap prediksi.
- Semakin tinggi perbedaan rata-rata antar kelas, semakin kuat pengaruh fitur tersebut.
""")

else:
    st.info("📌 Silakan Upload File CSV untuk mulai analisis.")
