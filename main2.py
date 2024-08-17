import streamlit as st
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from streamlit_option_menu import option_menu
from sklearn.compose import ColumnTransformer
import joblib

# Nama Website
st.set_page_config(page_title="SISTEM KLASIFIKASI KELAYAKAN BANTUAN SEMBAKO")

class MainClass:

    def __init__(self):
        self.data = Data()
        self.preprocessing = Preprocessing()
        self.data_mining = DataMining()
        self.prediction = Prediction()

    def run(self):
        st.markdown(
            "<h2><center>SISTEM KLASIFIKASI STATUS KELAYAKAN PENERIMA BANTUAN DI KELURAHAN CIBEUREUM</h2></center>",
            unsafe_allow_html=True,
        )
        with st.sidebar:
            selected = option_menu(
                "Fitur",
                ["Data", "Preprocessing Data", "Data Mining & Visualisasi", "Prediksi", "Logout"],
                default_index=0,
            )

        if selected == "Data":
            self.data.menu_data()

        elif selected == "Preprocessing Data":
            self.preprocessing.menu_preprocessing()

        elif selected == "Data Mining & Visualisasi":
            self.data_mining.menu_data_mining()

        elif selected == "Prediksi":
            self.prediction.menu_prediction()

        elif selected == "Logout":
            st.write("Logout")

class Data:

    def __init__(self):
        self.file_names = {"bpnt": None, "demografis": None, "rastrada": None}

    def menu_data(self):
        self.upload_files()

    def upload_files(self):
        st.header("Upload Data BPNT, Data Demografis, dan Data Rastrada")

        # File uploaders for each category
        self.file_names["bpnt"] = st.file_uploader("Upload Data BPNT", type=["xlsx"])
        self.file_names["demografis"] = st.file_uploader("Upload Data Demografis", type=["xlsx"])
        self.file_names["rastrada"] = st.file_uploader("Upload Data Rastrada", type=["xlsx"])

        # Check if all files are uploaded
        if all(self.file_names.values()):
            st.write(f"**File BPNT**: {self.file_names['bpnt'].name}")
            st.write(f"**File Demografis**: {self.file_names['demografis'].name}")
            st.write(f"**File Rastrada**: {self.file_names['rastrada'].name}")

            # Display dataframes for each file
            df_bpnt = pd.read_excel(self.file_names["bpnt"])
            df_demografis = pd.read_excel(self.file_names["demografis"])
            df_rastrada = pd.read_excel(self.file_names["rastrada"])

            st.dataframe(df_bpnt)
            st.dataframe(df_demografis)
            st.dataframe(df_rastrada)

            st.success("File telah berhasil di upload")

            # Simpan data ke session state
            st.session_state.uploaded_files = {
                "bpnt": self.file_names["bpnt"],
                "demografis": self.file_names["demografis"],
                "rastrada": self.file_names["rastrada"],
            }
        else:
            st.warning("Silahkan upload data : Data BPNT, Data Demografis, dan Data Rastrada")

class Preprocessing:

    def menu_preprocessing(self):
        self.preprocess_transform_data()

    def preprocess_transform_data(self):
        if "uploaded_files" in st.session_state:
            uploaded_files = st.session_state.uploaded_files
            data_bpnt = pd.read_excel(uploaded_files["bpnt"])
            data_demografis = pd.read_excel(uploaded_files["demografis"])
            data_rastrada = pd.read_excel(uploaded_files["rastrada"])

            # Menggabungkan data berdasarkan 'Id_Training'
            merged_df1 = pd.merge(data_bpnt, data_demografis, on=["Id_Training"], suffixes=("", "_demografis"))
            merged_df = pd.merge(merged_df1, data_rastrada, on=["Id_Training"], suffixes=("", "_rastrada"))
            st.subheader("Penggabungan Data")
            st.write(merged_df)

            # Pemilihan Atribut yang akan digunakan
            selected_columns = [
                "Id_Training",
                "Nama_KRT",
                "Alamat",
                "Pekerjaan",
                "Usia",
                "Status Perkawinan",
                "Status_Bangunan",
                "Tanggungan",
                "Pendapatan",
                "Kondisi_Dinding",
                "Kesehatan",
                "Status_Kelayakan",
            ]
            selected_columns = [col for col in selected_columns if col in merged_df.columns]

            merged_df = merged_df[selected_columns]

            st.subheader("Pemilihan Atribut")
            st.write(merged_df)

            # Cek nilai null setelah pemilihan atribut
            st.subheader("Pengecekan Missing Value Setelah Pemilihan Atribut")
            null_values = merged_df.isnull().sum()
            if null_values.sum() == 0:
                st.success("Tidak ditemukan missing value dalam data.")
            else:
                st.warning("Terdapat ditemukan missing value dalam data.")
                st.write("Atribut yang memiliki missing value beserta jumlahnya:")
                for col in merged_df.columns:
                    st.write(f"{col}: {null_values[col]} missing value")

            # Cek duplikat Data setelah pemilihan atribut
            st.subheader("Pengecekan Duplikasi Data Setelah Pemilihan Atribut")
            duplicate_rows = merged_df[merged_df.duplicated(subset=["Id_Training"], keep=False)]
            num_duplicate_rows = len(duplicate_rows)
            if num_duplicate_rows > 0:
                st.warning(f"Baris duplikat terdeteksi pada data gabungan: {num_duplicate_rows} baris duplikat.")
                st.write("Baris duplikat:")
                st.write(duplicate_rows)
            else:
                st.success("Tidak ditemukan baris yang mempunyai duplikat.")
            merged_df = merged_df.drop_duplicates(subset=["Id_Training"]).reset_index(drop=True)
            
            # Split data menjadi training dan testing
            try:
                train_df, test_df = train_test_split(merged_df, test_size=0.2, random_state=42)
                st.session_state.train_df = train_df
                st.session_state.test_df = test_df
            except Exception as e:
                st.error(f"Error splitting data: {e}")
                return

            st.subheader("Data Training (Data Latih)")
            st.write(train_df)

            st.subheader("Data Testing (Data Uji)")
            st.write(test_df)

class DataMining:

    def menu_data_mining(self):
        if "train_df" in st.session_state:
            # Subheader dan menampilkan data latih
            st.subheader("Data Training (Data Latih)")
            train_df = st.session_state.train_df
            st.write(train_df)

            # Encoding categorical features
            categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()
            st.session_state.column_transformer = ColumnTransformer(
                transformers=[
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
                ],
                remainder='passthrough'
            )
            train_data_encoded = st.session_state.column_transformer.fit_transform(train_df)

            # Training Naive Bayes model
            model = MultinomialNB()
            X_train = train_data_encoded
            y_train = train_df['Status_Kelayakan']
            model.fit(X_train, y_train)

            st.session_state.model = model
            st.session_state.preprocessed_data = train_df

            st.success("Model berhasil dilatih.")

            # Saving the model to a .pkl file
            model_filename = 'trained_model.pkl'
            joblib.dump(model, model_filename)
            st.success(f"Model berhasil disimpan sebagai {model_filename}.")

            # Daftar pekerjaan yang akan dihitung probabilitas kelayakannya
            pekerjaan_list = ['PEDAGANG', 'BURUH HARIAN LEPAS', 'WIRASWASTA', 'KARYAWAN SWASTA', 'TIDAK BEKERJA']

            # Menghitung probabilitas kelayakan untuk setiap pekerjaan
            for pekerjaan in pekerjaan_list:
                pekerjaan_df = train_df[train_df['Pekerjaan'] == pekerjaan]
                jumlah_pekerjaan_layak = pekerjaan_df[pekerjaan_df['Status_Kelayakan'] == 'Layak'].shape[0]
                total_jumlah_pekerjaan = pekerjaan_df.shape[0]

                if total_jumlah_pekerjaan > 0:
                    probabilitas_pekerjaan_layak = jumlah_pekerjaan_layak / total_jumlah_pekerjaan
                else:
                    probabilitas_pekerjaan_layak = 0

                st.write(f"Probabilitas {pekerjaan.lower()} yang layak: {probabilitas_pekerjaan_layak:.2f}")

            # Daftar penyakit yang akan dihitung probabilitas kelayakannya
            penyakit_list = ['TIDAK ADA', 'HIPERTENSI', 'JANTUNG', 'PARU-PARU', 'DIABETES']

            # Menghitung probabilitas kelayakan untuk setiap penyakit
            for penyakit in penyakit_list:
                penyakit_df = train_df[train_df['Kesehatan'] == penyakit]
                jumlah_penyakit_layak = penyakit_df[penyakit_df['Status_Kelayakan'] == 'Layak'].shape[0]
                total_jumlah_penyakit = penyakit_df.shape[0]

                if total_jumlah_penyakit > 0:
                    probabilitas_penyakit_layak = jumlah_penyakit_layak / total_jumlah_penyakit
                else:
                    probabilitas_penyakit_layak = 0

                st.write(f"Probabilitas dengan penyakit {penyakit.lower()} yang layak: {probabilitas_penyakit_layak:.2f}")
            # Daftar pendapatan yang akan dihitung probabilitas kelayakannya
            pendapatan_list = ['Rendah', 'Sangat Rendah', 'Menengah', 'Tinggi', 'Sangat Tinggi']

            # Menghitung probabilitas kelayakan untuk setiap pendapatan
            for pendapatan in pendapatan_list:
                pendapatan_df = train_df[train_df['Pendapatan'] == pendapatan]
                jumlah_pendapatan_layak = pendapatan_df[pendapatan_df['Status_Kelayakan'] == 'Layak'].shape[0]
                total_jumlah_pendapatan = pendapatan_df.shape[0]

                if total_jumlah_pendapatan > 0:
                    probabilitas_pendapatan_layak = jumlah_pendapatan_layak / total_jumlah_pendapatan
                else:
                    probabilitas_pendapatan_layak = 0

                st.write(f"Probabilitas dengan pendapatan {pendapatan.lower()} yang layak: {probabilitas_pendapatan_layak:.2f}")

            # Daftar status bangunan yang akan dihitung probabilitas kelayakannya
            status_bangunan_list = ['Milik Sendiri', 'Kontrak/Sewa', 'Milik Orang Lain', 'Bantuan Ritalahu']

            # Menghitung probabilitas kelayakan untuk setiap status bangunan
            for status_bangunan in status_bangunan_list:
                status_bangunan_df = train_df[train_df['Status_Bangunan'] == status_bangunan]
                jumlah_status_bangunan_layak = status_bangunan_df[status_bangunan_df['Status_Kelayakan'] == 'Layak'].shape[0]
                total_jumlah_status_bangunan = status_bangunan_df.shape[0]

                if total_jumlah_status_bangunan > 0:
                    probabilitas_status_bangunan_layak = jumlah_status_bangunan_layak / total_jumlah_status_bangunan
                else:
                    probabilitas_status_bangunan_layak = 0

                st.write(f"Probabilitas dengan status bangunan {status_bangunan.lower()} yang layak: {probabilitas_status_bangunan_layak:.2f}")

            # Daftar kondisi dinding yang akan dihitung probabilitas kelayakannya
            kondisi_dinding_list = ['Jelek/Kualitas rendah', 'Bagus/kualitas tinggi']

            # Menghitung probabilitas kelayakan untuk setiap kondisi dinding
            for kondisi_dinding in kondisi_dinding_list:
                kondisi_dinding_df = train_df[train_df['Kondisi_Dinding'] == kondisi_dinding]
                jumlah_kondisi_dinding_layak = kondisi_dinding_df[kondisi_dinding_df['Status_Kelayakan'] == 'Layak'].shape[0]
                total_jumlah_kondisi_dinding = kondisi_dinding_df.shape[0]

                if total_jumlah_kondisi_dinding > 0:
                    probabilitas_kondisi_dinding_layak = jumlah_kondisi_dinding_layak / total_jumlah_kondisi_dinding
                else:
                    probabilitas_kondisi_dinding_layak = 0

                st.write(f"Probabilitas dengan kondisi dinding {kondisi_dinding.lower()} yang layak: {probabilitas_kondisi_dinding_layak:.2f}")
        else:
            st.warning("Data training belum tersedia. Lakukan preprocessing terlebih dahulu.")

class Prediction:

    def menu_prediction(self):
        self.make_prediction()

    def make_prediction(self):
        if "preprocessed_data" in st.session_state and "model" in st.session_state:
            uploaded_file = st.file_uploader("Upload Data untuk Prediksi", type=["xlsx"])
            if uploaded_file:
                data = pd.read_excel(uploaded_file)
                st.write(data)
                
                # Ensure the correct columns are present
                if set(data.columns).issubset(set(st.session_state.preprocessed_data.columns)):
                    data_transformed = st.session_state.column_transformer.transform(data)
                    predictions = st.session_state.model.predict(data_transformed)
                    st.write("Predictions:", predictions)
                else:
                    st.error("Data yang diupload tidak sesuai dengan format yang diharapkan.")
        else:
            st.warning("Model atau data preprocessing belum tersedia.")

if __name__ == "__main__":
    app = MainClass()
    app.run()