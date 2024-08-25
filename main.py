import streamlit as st
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from streamlit_option_menu import option_menu

# Nama Website
st.set_page_config(page_title="SISTEM KLASIFIKASI KELAYAKAN BANTUAN SEMBAKO")

class MainClass:

    def __init__(self):
        self.data = Data()
        self.preprocessing = Preprocessing()
        self.data_mining = DataMining()
        self.prediction = Prediction()

    def run(self):
        # Cek apakah pengguna sudah login
        if "logged_in" not in st.session_state:
            st.session_state["logged_in"] = False

        if st.session_state["logged_in"]:
            # Jika sudah login, tampilkan menu aplikasi
            st.markdown(
                "<h2><center>SISTEM KLASIFIKASI STATUS KELAYAKAN PENERIMA BANTUAN SEMBAKO DI KELURAHAN CIBEUREUM</h2></center>",
                unsafe_allow_html=True,
            )
            with st.sidebar:
                selected = option_menu(
                    "Fitur",
                    ["Data", "Preprocessing Data", "Data Mining", "Prediksi", "Logout"],
                    default_index=0,
                )

            if selected == "Data":
                self.data.menu_data()

            elif selected == "Preprocessing Data":
                self.preprocessing.menu_preprocessing()

            elif selected == "Data Mining":
                self.data_mining.menu_data_mining()

            elif selected == "Prediksi":
                self.prediction.menu_prediction()

            elif selected == "Logout":
                self.logout()
        else:
            # Jika belum login, tampilkan halaman login
            self.show_login()

    def show_login(self):
        st.markdown("<h2><center>Login</h2></center>", unsafe_allow_html=True)
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            self.login(username, password)

    def login(self, username, password):
        """Validasi login sederhana."""
        if username == "admin" and password == "admin":
            st.session_state["logged_in"] = True
            st.success("Login berhasil!")
        else:
            st.error("Username atau password salah.")

    def logout(self):
        """Logout pengguna."""
        st.session_state["logged_in"] = False
        st.success("Anda telah logout.")


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
        self.load_and_prepare_data()

    def load_and_prepare_data(self):
        if "uploaded_files" in st.session_state:
            uploaded_files = st.session_state.uploaded_files
            data_bpnt = pd.read_excel(uploaded_files["bpnt"])
            data_demografis = pd.read_excel(uploaded_files["demografis"])
            data_rastrada = pd.read_excel(uploaded_files["rastrada"])

            # Menggabungkan data berdasarkan 'Id_Penduduk'
            merged_df1 = pd.merge(data_bpnt, data_demografis, on=["Id_Penduduk"], suffixes=("", "_demografis"))
            merged_df = pd.merge(merged_df1, data_rastrada, on=["Id_Penduduk"], suffixes=("", "_rastrada"))
            st.subheader("Penggabungan Data")
            st.write(merged_df)

            # Pemilihan Atribut yang akan digunakan
            selected_columns = [
                "Id_Penduduk",
                "Nama_KRT",
                "Alamat",
                "Pekerjaan",
                "Usia",
                "Status_Perkawinan",
                "Status_Bangunan",
                "Pendapatan",
                "Tanggungan",
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
                st.warning("Terdapat missing value dalam data.")
                st.write("Atribut yang memiliki missing value beserta jumlahnya:")
                for col in merged_df.columns:
                    st.write(f"{col}: {null_values[col]} missing value")

            # Cek duplikat Data setelah pemilihan atribut
            st.subheader("Pengecekan Duplikasi Data Setelah Pemilihan Atribut")
            duplicate_rows = merged_df[merged_df.duplicated(subset=["Id_Penduduk"], keep=False)]
            num_duplicate_rows = len(duplicate_rows)
            if num_duplicate_rows > 0:
                st.warning(f"Baris duplikat terdeteksi pada data gabungan: {num_duplicate_rows} baris duplikat.")
                st.write("Baris duplikat:")
                st.write(duplicate_rows)
            else:
                st.success("Tidak ditemukan baris yang mempunyai duplikat.")
            merged_df = merged_df.drop_duplicates(subset=["Id_Penduduk"]).reset_index(drop=True)

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
            st.subheader("Data Training (Data Latih)")
            train_df = st.session_state.train_df
            st.write(train_df)

            # Menghitung probabilitas prior (kelas)
            frekuensi_kelas = train_df['Status_Kelayakan'].value_counts()
            total_data = len(train_df)
            prior_probabilities = frekuensi_kelas / total_data

            st.subheader("Probabilitas Prior (Kelas)")
            st.write(f"Probabilitas Layak: {prior_probabilities['Layak']:.3f}")
            st.write(f"Probabilitas Tidak Layak: {prior_probabilities['Tidak Layak']:.3f}")

            # Definisikan kolom kategorikal
            categorical_cols = ['Pekerjaan', 'Status_Perkawinan', 'Status_Bangunan', 'Pendapatan', 'Kondisi_Dinding', 'Kesehatan']
            available_categorical_cols = [col for col in categorical_cols if col in train_df.columns]

            # Menghitung probabilitas kondisional untuk setiap atribut kategorikal
            conditional_probabilities = {}
            for col in available_categorical_cols:
                conditional_probabilities[col] = train_df.groupby(['Status_Kelayakan', col]).size().unstack().fillna(0)
                # Menormalkan probabilitas kondisional
                conditional_probabilities[col] = conditional_probabilities[col].div(frekuensi_kelas, axis=0)

            # Tampilkan hasil probabilitas kondisional dengan 3 angka di belakang koma
            st.subheader("Probabilitas Kondisional (Atribut Kategorikal)")
            for col in available_categorical_cols:
                st.write(f"Probabilitas Kondisional untuk {col}:")
                # Menggunakan format string untuk menampilkan dengan 3 angka di belakang koma
                formatted_probabilities = conditional_probabilities[col].applymap(lambda x: f"{x:.3f}")
                st.write(formatted_probabilities)

            # Menghitung probabilitas kondisional untuk atribut kontinu (Usia dan Tanggungan)
            numerical_cols = ['Usia', 'Tanggungan']
            numerical_stats = {}
            for col in numerical_cols:
                numerical_stats[col] = train_df.groupby('Status_Kelayakan').agg(
                    mean=(col, 'mean'),
                    variance=(col, lambda x: np.var(x, ddof=1)),
                    stddev=(col, lambda x: np.std(x, ddof=1))
                )

            st.subheader("Statistik (Mean, Variance, StdDev) untuk Atribut Kontinu")
            for col in numerical_cols:
                st.write(f"Statistik untuk {col}:")
                st.write(numerical_stats[col])

            # Menghitung probabilitas kondisional menggunakan distribusi Gaussian
            def gaussian_probability(x, mean, stddev):
                exponent = np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))
                return (1 / (stddev * np.sqrt(2 * np.pi))) * exponent

            st.subheader("Probabilitas Kondisional (Atribut Kontinu) Menggunakan Distribusi Gaussian")
            for col in numerical_cols:
                st.write(f"Probabilitas Kondisional untuk {col}:")
                for status in ['Layak', 'Tidak Layak']:
                    mean = numerical_stats[col].loc[status, 'mean']
                    stddev = numerical_stats[col].loc[status, 'stddev']
                    conditional_probs = train_df[col].apply(lambda x: gaussian_probability(x, mean, stddev))
                    
                    # Membuat dataframe untuk menampilkan nilai usia/tanggungan beserta probabilitasnya
                    result_df = pd.DataFrame({
                        col: train_df[col],
                        'Probabilitas': conditional_probs.apply(lambda x: f"{x:.3f}")
                    })

                    st.write(f"Probabilitas untuk '{col}' dengan status '{status}':")
                    st.write(result_df)

            # Simpan probabilitas ke dalam session_state
            st.session_state.prior_probabilities = prior_probabilities
            st.session_state.conditional_probabilities = conditional_probabilities
            st.session_state.numerical_stats = numerical_stats

class Prediction:
    def __init__(self):
        pass

    def gaussian_probability(self, x, mean, stddev):
        exponent = np.exp(-((x - mean) ** 2 / (2 * stddev ** 2)))
        return (1 / (stddev * np.sqrt(2 * np.pi))) * exponent

    def menu_prediction(self):
        st.header("Prediksi Status Kelayakan Masyarakat Penerima Bantuan Sembako")
        
        # Menambahkan opsi untuk mengunggah data uji
        uploaded_test_file = st.file_uploader("Upload Data BPNT", type=["xlsx"])

        if uploaded_test_file is not None:
            test_df = pd.read_excel(uploaded_test_file)
            st.subheader("Data BPNT")
            st.write(test_df)
            
            if "train_df" in st.session_state:
                train_df = st.session_state.train_df

                # Mengambil probabilitas prior dan probabilitas kondisional yang sudah dihitung
                prior_probabilities = st.session_state.prior_probabilities
                conditional_probabilities = st.session_state.conditional_probabilities
                numerical_stats = st.session_state.numerical_stats

                categorical_cols = ['Pekerjaan', 'Status_Perkawinan', 'Status_Bangunan', 'Pendapatan', 'Kondisi_Dinding', 'Kesehatan']
                numerical_cols = ['Usia', 'Tanggungan']

                # Mengonversi kolom numerik ke tipe data float
                for col in numerical_cols:
                    test_df[col] = pd.to_numeric(test_df[col], errors='coerce')

                # Mengisi nilai NaN dengan mean dari kolom tersebut
                test_df = test_df.fillna(test_df.mean())

                # Fungsi untuk menghitung probabilitas untuk setiap kelas berdasarkan data uji
                def calculate_class_probabilities(row):
                    class_probabilities = {}
                    for status in prior_probabilities.index:
                        # Mulai dengan probabilitas prior
                        class_probabilities[status] = prior_probabilities[status]

                        # Kalikan dengan probabilitas kondisional untuk setiap atribut kategorikal
                        for col in categorical_cols:
                            if col in row:
                                class_probabilities[status] *= conditional_probabilities[col].get(row[col], {}).get(status, 0)

                        # Kalikan dengan probabilitas kondisional untuk setiap atribut kontinu menggunakan Gaussian
                        for col in numerical_cols:
                            mean = numerical_stats[col].loc[status, 'mean']
                            stddev = numerical_stats[col].loc[status, 'stddev']
                            class_probabilities[status] *= self.gaussian_probability(row[col], mean, stddev)

                    return class_probabilities

                # Melakukan prediksi untuk setiap baris di data uji
                predictions = []
                for index, row in test_df.iterrows():
                    class_probabilities = calculate_class_probabilities(row)
                    # Ambil kelas dengan probabilitas tertinggi
                    best_class = max(class_probabilities, key=class_probabilities.get)
                    predictions.append(best_class)

                y_pred = np.array(predictions)

                # Menampilkan hasil prediksi
                st.subheader("Hasil Prediksi Status Kelayakan Masyarakat Penerima Bantuan Sembako")
                test_df['Prediksi Status Kelayakan'] = y_pred
                st.write(test_df[['Nama_KRT', 'Alamat', 'Prediksi Status Kelayakan']])

                # Menghitung jumlah masyarakat yang layak dan tidak layak
                layak_count = sum(y_pred == "Layak")
                tidak_layak_count = sum(y_pred == "Tidak Layak")
                st.subheader("Jumlah Masyarakat Berdasarkan Status Kelayakan")
                st.write(f"Jumlah Masyarakat yang Layak Menerima Bantuan: {layak_count}")
                st.write(f"Jumlah Masyarakat yang Tidak Layak Menerima Bantuan: {tidak_layak_count}")

                # Menampilkan alamat yang paling banyak layak menerima bantuan
                st.subheader("Alamat dengan Penerima Bantuan Layak Terbanyak")
                if 'Alamat' in test_df.columns:
                    alamat_terbanyak = test_df[test_df['Prediksi Status Kelayakan'] == 'Layak']['Alamat'].value_counts().idxmax()
                    jumlah_terbanyak = test_df[test_df['Prediksi Status Kelayakan'] == 'Layak']['Alamat'].value_counts().max()
                    st.write(f"Alamat: {str(alamat_terbanyak)}")
                    st.write(f"Jumlah Penerima Bantuan Layak: {str(jumlah_terbanyak)}")
                else:
                    st.warning("Kolom 'Alamat' tidak ditemukan dalam data uji.")

                # Evaluasi model (hanya jika kolom 'Status_Kelayakan' ada di data uji)
                if 'Status_Kelayakan' in test_df.columns:
                    y_test = test_df['Status_Kelayakan'].values
                    st.subheader("Evaluasi Model")
                    
                    # Menampilkan Confusion Matrix sebagai gambar
                    st.write("Confusion Matrix:")
                    cm = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                    ax.set_xlabel('Predicted Labels')
                    ax.set_ylabel('True Labels')
                    st.pyplot(fig)
                    
                    # Menampilkan metrik lainnya
                    st.write("Accuracy Score:", accuracy_score(y_test, y_pred))
                    st.write("Classification Report:")
                    st.write(classification_report(y_test, y_pred))
            else:
                st.warning("Data training belum tersedia. Silakan lakukan pelatihan model terlebih dahulu di bagian Data Mining & Visualisasi.")
        else:
            st.info("Silakan upload Data BPNT untuk melakukan prediksi.")

if __name__ == "__main__":
    main = MainClass()
    main.run()
