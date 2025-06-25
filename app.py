import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score

# Inisialisasi session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'input_data' not in st.session_state:
    st.session_state.input_data = {}
if 'X' not in st.session_state:
    st.session_state.X = None
if 'target' not in st.session_state:
    st.session_state.target = None
if 'algo' not in st.session_state:
    st.session_state.algo = None
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None

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
    st.session_state.algo = algo  # Simpan algoritma yang dipilih di session state

    if algo == "K-Means":
        n_clusters = st.slider("Pilih Jumlah Cluster", min_value=2, max_value=10, value=3)
        X = df.select_dtypes(include=np.number)
        st.session_state.X = X.columns.tolist()  # Simpan daftar fitur
    else:
        target = st.selectbox("Pilih Kolom Target", df.columns)
        st.session_state.target = target  # Simpan target di session state
        X = pd.get_dummies(df.drop(columns=[target]))
        st.session_state.X = X.columns.tolist()  # Simpan daftar fitur
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
            
        if algo == "Regresi Linier":
            model = LinearRegression().fit(X_train, y_train)
            y_pred = model.predict(X_test)
            st.session_state.model = model  # Simpan model di session state
            st.metric("R² Score", f"{r2_score(y_test, y_pred):.2f}")
            st.metric("MSE", f"{mean_squared_error(y_test, y_pred):.2f}")
            st.plotly_chart(px.scatter(x=y_test, y=y_pred, labels={'x': 'Aktual', 'y': 'Prediksi'}))

        elif algo == "Regresi Logistik":
            model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
            y_pred = model.predict(X_test)
            st.session_state.model = model  # Simpan model di session state
            st.metric("Akurasi", f"{accuracy_score(y_test, y_pred):.2f}")
            st.text(classification_report(y_test, y_pred))
            st.plotly_chart(px.imshow(confusion_matrix(y_test, y_pred), text_auto=True))

        elif algo == "Naive Bayes":
            try:
                model = GaussianNB().fit(X_train, y_train)
                y_pred = model.predict(X_test)
                st.session_state.model = model  # Simpan model di session state
                st.metric("Akurasi", f"{accuracy_score(y_test, y_pred):.2f}")
                st.text(classification_report(y_test, y_pred))
            except ValueError as e:
                st.error("❌ Target harus berupa label kategorikal untuk Naive Bayes (bukan numerik kontinyu).")
                model = None

        elif algo == "SVM":
            if is_classification(y):
                model = SVC().fit(X_train, y_train)
                y_pred = model.predict(X_test)
                st.session_state.model = model  # Simpan model di session state
                st.metric("Akurasi", f"{accuracy_score(y_test, y_pred):.2f}")
                st.text(classification_report(y_test, y_pred))
                st.plotly_chart(px.imshow(confusion_matrix(y_test, y_pred), text_auto=True))
            else:
                st.error("❌ SVM hanya dapat digunakan untuk masalah klasifikasi.")

        elif algo == "KNN":
            model = KNeighborsClassifier().fit(X_train, y_train)
            y_pred = model.predict(X_test)
            st.session_state.model = model  # Simpan model di session state
            st.metric("Akurasi", f"{accuracy_score(y_test, y_pred):.2f}")
            st.text(classification_report(y_test, y_pred))
            st.plotly_chart(px.imshow(confusion_matrix(y_test, y_pred), text_auto=True))

        elif algo == "Decision Tree":
            model = DecisionTreeClassifier().fit(X_train, y_train)
            y_pred = model.predict(X_test)
            st.session_state.model = model  # Simpan model di session state
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
if st.session_state.model is not None:
    st.subheader("🧠 Prediksi Manual")
    st.markdown("Masukkan nilai untuk setiap fitur untuk mendapatkan prediksi:")
    
    with st.form(key='prediction_form'):
        input_data = {}
        
        # Pastikan st.session_state.X tidak None
        if st.session_state.X is not None:
            # Buat input untuk setiap fitur
            for col in st.session_state.X:
                # Handle kolom numerik
                if pd.api.types.is_numeric_dtype(df[col]):
                    min_val = float(df[col].min())
                    max_val = float(df[col].max())
                    default_val = float(df[col].mean())
                    
                    # Pastikan step sesuai dengan tipe data
                    if df[col].dtype == 'float64':
                        step = 0.1  # float step
                    else:
                        step = 1  # int step
                    
                    input_data[col] = st.number_input(
                        f"{col} (min: {min_val:.2f}, max: {max_val:.2f})",
                        min_value=min_val,
                        max_value=max_val,
                        value=default_val,
                        step=float(step)  # Pastikan step adalah float
                    )
                # Handle kolom kategorikal (jika ada)
                else:
                    unique_vals = df[col].unique()
                    input_data[col] = st.selectbox(f"{col}", unique_vals)
        else:
            st.error("❌ Tidak ada fitur yang tersedia untuk prediksi. Pastikan model telah dilatih dengan benar.")

        # Tambahkan tombol kirim
        submitted = st.form_submit_button("🔮 Prediksi")
        
        if submitted:
            try:
                # Buat DataFrame dari input
                input_df = pd.DataFrame([input_data])
                
                # Handle one-hot encoding jika ada
                input_df = pd.get_dummies(input_df)
                
                # Pastikan kolom input sama dengan yang digunakan saat training
                missing_cols = set(X.columns) - set(input_df.columns)
                for col in missing_cols:
                    input_df[col] = 0
                input_df = input_df[X.columns]
                
                # Dapatkan prediksi
                prediction = st.session_state.model.predict(input_df)[0]
                
                # Tampilkan hasil prediksi
                st.success(f"📌 Hasil Prediksi Manual: **{prediction}**")

                # Penjelasan hasil prediksi
                st.markdown("### 📊 Interpretasi Hasil")
                
                if st.session_state.algo == "Regresi Linier":
                    target_min = df[st.session_state.target].min()
                    target_max = df[st.session_state.target].max()
                    target_range = target_max - target_min
                    normalized = (prediction - target_min) / target_range
                    
                    st.markdown(f"""
                    - **Nilai Prediksi:** {prediction:.2f}
                    - **Rentang Target dalam Data:** {target_min:.2f} sampai {target_max:.2f}
                    """)
                    
                    if prediction < target_min:
                        st.warning("Prediksi berada DI BAWAH rentang data training")
                    elif prediction > target_max:
                        st.warning("Prediksi berada DI ATAS rentang data training")
                    else:
                        st.info(f"Prediksi berada di {normalized*100:.1f}% dari rentang data")
                        
                    # Tampilkan faktor penting
                    if hasattr(st.session_state.model, 'coef_'):
                        st.markdown("### 🔍 Faktor Paling Berpengaruh")
                        coef_df = pd.DataFrame({
                            'Fitur': X.columns,
                            'Koefisien': st.session_state.model.coef_
                        }).sort_values('Koefisien', key=abs, ascending=False)
                        
                        top_features = coef_df.head(3)
                        for _, row in top_features.iterrows():
                            effect = "meningkatkan" if row['Koefisien'] > 0 else "menurunkan"
                            st.write(f"- **{row['Fitur']}**: {effect} hasil prediksi")
                
                elif st.session_state.algo in ["Regresi Logistik", "Naive Bayes", "SVM", "KNN", "Decision Tree"]:
                    st.markdown(f"**Kelas Prediksi:** {prediction}")
                    
                    # Tampilkan probabilitas jika ada
                    if hasattr(st.session_state.model, "predict_proba"):
                        proba = st.session_state.model.predict_proba(input_df)[0]
                        proba_df = pd.DataFrame({
                            'Kelas': st.session_state.model.classes_,
                            'Probabilitas': proba
                        }).sort_values('Probabilitas', ascending=False)
                        
                        st.markdown("**Probabilitas Kelas:**")
                        st.dataframe(proba_df)

            except Exception as e:
                st.error(f"Terjadi error saat memprediksi: {str(e)}")

else:
    st.info("📌 Silakan Upload File CSV untuk mulai analisis.")
