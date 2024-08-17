import streamlit as st
import pandas as pd
import numpy as np
import math
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
                    [
                        "Data",
                        "Preprocessing Data",
                        "Data Mining & Visualisasi",
                        "Prediksi",
                        "Logout",
                    ],
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
        self.file_names["demografis"] = st.file_uploader(
            "Upload Data Demografis", type=["xlsx"]
        )
        self.file_names["rastrada"] = st.file_uploader(
            "Upload Data Rastrada", type=["xlsx"]
        )

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
            st.warning(
                "Silahkan upload data : Data BPNT, Data Demografis, dan Data Rastrada"
            )


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
            merged_df1 = pd.merge(
                data_bpnt,
                data_demografis,
                on=["Id_Training"],
                suffixes=("", "_demografis"),
            )
            merged_df = pd.merge(
                merged_df1,
                data_rastrada,
                on=["Id_Training"],
                suffixes=("", "_rastrada"),
            )
            st.subheader("Penggabungan Data")
            st.write(merged_df)

            # Pemilihan Atribut yang akan digunakan
            selected_columns = [
                "Id_Training",
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
            selected_columns = [
                col for col in selected_columns if col in merged_df.columns
            ]

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
            duplicate_rows = merged_df[
                merged_df.duplicated(subset=["Id_Training"], keep=False)
            ]
            num_duplicate_rows = len(duplicate_rows)
            if num_duplicate_rows > 0:
                st.warning(
                    f"Baris duplikat terdeteksi pada data gabungan: {num_duplicate_rows} baris duplikat."
                )
                st.write("Baris duplikat:")
                st.write(duplicate_rows)
            else:
                st.success("Tidak ditemukan baris yang mempunyai duplikat.")
            merged_df = merged_df.drop_duplicates(subset=["Id_Training"]).reset_index(
                drop=True
            )

            # Split data menjadi training dan testing
            try:
                train_df, test_df = train_test_split(
                    merged_df, test_size=0.2, random_state=42
                )
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
            frekuensi_kelas = train_df["Status_Kelayakan"].value_counts()
            total_data = len(train_df)
            prior_probabilities = frekuensi_kelas / total_data

            st.subheader("Probabilitas Prior (Kelas)")
            st.write(f"Probabilitas Layak: {prior_probabilities['Layak']:.3f}")
            st.write(
                f"Probabilitas Tidak Layak: {prior_probabilities['Tidak Layak']:.3f}"
            )

            # Definisikan kolom kategorikal
            categorical_cols = [
                "Pekerjaan",
                "Status_Perkawinan",
                "Status_Bangunan",
                "Kondisi_Dinding",
                "Kesehatan",
            ]
            available_categorical_cols = [
                col for col in categorical_cols if col in train_df.columns
            ]

            # Menghitung probabilitas kondisional untuk setiap atribut kategorikal
            conditional_probabilities = {}
            for col in available_categorical_cols:
                conditional_probabilities[col] = (
                    train_df.groupby(["Status_Kelayakan", col])
                    .size()
                    .unstack()
                    .fillna(0)
                )
                # Menormalkan probabilitas kondisional
                conditional_probabilities[col] = conditional_probabilities[col].div(
                    frekuensi_kelas, axis=0
                )

            # Tampilkan hasil probabilitas kondisional dengan 3 angka di belakang koma
            st.subheader("Probabilitas Kondisional (Atribut Kategorikal)")
            for col in available_categorical_cols:
                st.write(f"Probabilitas Kondisional untuk {col}:")
                # Menggunakan format string untuk menampilkan dengan 3 angka di belakang koma
                formatted_probabilities = conditional_probabilities[col].applymap(
                    lambda x: f"{x:.3f}"
                )
                st.write(formatted_probabilities)

            # Menghitung probabilitas kondisional untuk atribut kontinu (Usia dan Tanggungan)
            numerical_cols = ["Usia", "Tanggungan"]
            numerical_stats = {}
            for col in numerical_cols:
                numerical_stats[col] = train_df.groupby("Status_Kelayakan").agg(
                    mean=(col, "mean"),
                    variance=(col, lambda x: np.var(x, ddof=1)),
                    stddev=(col, lambda x: np.std(x, ddof=1)),
                )

            st.subheader("Statistik (Mean, Variance, StdDev) untuk Atribut Kontinu")
            for col in numerical_cols:
                st.write(f"Statistik untuk {col}:")
                st.write(numerical_stats[col])

            # Menghitung probabilitas kondisional menggunakan distribusi Gaussian
            def gaussian_probability(x, mean, stddev):
                exponent = np.exp(-((x - mean) ** 2) / (2 * stddev**2))
                return (1 / (stddev * np.sqrt(2 * np.pi))) * exponent

            st.subheader(
                "Probabilitas Kondisional (Atribut Kontinu) Menggunakan Distribusi Gaussian"
            )
            for col in numerical_cols:
                st.write(f"Probabilitas Kondisional untuk {col}:")
                for status in ["Layak", "Tidak Layak"]:
                    mean = numerical_stats[col].loc[status, "mean"]
                    stddev = numerical_stats[col].loc[status, "stddev"]
                    conditional_probs = train_df[col].apply(
                        lambda x: gaussian_probability(x, mean, stddev)
                    )

                    # Membuat dataframe untuk menampilkan nilai usia/tanggungan beserta probabilitasnya
                    result_df = pd.DataFrame(
                        {
                            col: train_df[col],
                            "Probabilitas": conditional_probs.apply(
                                lambda x: f"{x:.3f}"
                            ),
                        }
                    )

                    st.write(f"Probabilitas untuk '{col}' dengan status '{status}':")
                    st.write(result_df)

            # Simpan probabilitas ke dalam session_state
            st.session_state.prior_probabilities = prior_probabilities
            st.session_state.conditional_probabilities = conditional_probabilities
            st.session_state.numerical_stats = numerical_stats


import seaborn as sns
import matplotlib.pyplot as plt


class Prediction:
    def __init__(self):
        pass

    def gaussian_probability(self, x, mean, stddev):
        exponent = np.exp(-((x - mean) ** 2 / (2 * stddev**2)))
        return (1 / (stddev * np.sqrt(2 * np.pi))) * exponent

    def menu_prediction(self):
        st.header("Prediksi Status Kelayakan")

        # Menambahkan opsi untuk mengunggah data uji
        uploaded_test_file = st.file_uploader("Upload Data Uji", type=["xlsx"])

        if uploaded_test_file is not None:
            test_df = pd.read_excel(uploaded_test_file)
            st.subheader("Data Testing (Data Uji)")
            st.write(test_df)

            if "train_df" in st.session_state:
                train_df = st.session_state.train_df

                # Mengambil probabilitas prior dan probabilitas kondisional yang sudah dihitung
                prior_probabilities = st.session_state.prior_probabilities
                conditional_probabilities = st.session_state.conditional_probabilities
                numerical_stats = st.session_state.numerical_stats

                categorical_cols = [
                    "Pekerjaan",
                    "Status_Perkawinan",
                    "Status_Bangunan",
                    "Kondisi_Dinding",
                    "Kesehatan",
                ]
                numerical_cols = ["Usia", "Tanggungan"]

                y_true = test_df["Status_Kelayakan"]
                y_pred = []

                for index, row in test_df.iterrows():
                    prob_layak = prior_probabilities["Layak"]
                    prob_tidak_layak = prior_probabilities["Tidak Layak"]

                    # Hitung probabilitas untuk atribut kategorikal
                    for col in categorical_cols:
                        if col in row:
                            prob_layak *= conditional_probabilities[col].loc[
                                "Layak", row[col]
                            ]
                            prob_tidak_layak *= conditional_probabilities[col].loc[
                                "Tidak Layak", row[col]
                            ]

                    # Hitung probabilitas untuk atribut kontinu menggunakan distribusi Gaussian
                    for col in numerical_cols:
                        if col in row:
                            mean_layak = numerical_stats[col].loc["Layak", "mean"]
                            stddev_layak = numerical_stats[col].loc["Layak", "stddev"]
                            mean_tidak_layak = numerical_stats[col].loc[
                                "Tidak Layak", "mean"
                            ]
                            stddev_tidak_layak = numerical_stats[col].loc[
                                "Tidak Layak", "stddev"
                            ]

                            prob_layak *= self.gaussian_probability(
                                row[col], mean_layak, stddev_layak
                            )
                            prob_tidak_layak *= self.gaussian_probability(
                                row[col], mean_tidak_layak, stddev_tidak_layak
                            )

                    # Tentukan kelas dengan probabilitas tertinggi
                    if prob_layak > prob_tidak_layak:
                        y_pred.append("Layak")
                    else:
                        y_pred.append("Tidak Layak")

                # Tampilkan hasil prediksi
                st.subheader("Hasil Prediksi")
                test_df["Hasil Prediksi Status Kelayakan"] = y_pred
                st.write(test_df)

                # Tampilkan confusion matrix menggunakan seaborn
                cm = confusion_matrix(y_true, y_pred, labels=["Layak", "Tidak Layak"])
                cm_df = pd.DataFrame(
                    cm,
                    index=["Actual Layak", "Actual Tidak Layak"],
                    columns=["Predicted Layak", "Predicted Tidak Layak"],
                )

                st.subheader("Confusion Matrix")

                fig, ax = plt.subplots()
                sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", ax=ax)
                ax.set_ylabel("Actual")
                ax.set_xlabel("Predicted")
                st.pyplot(fig)

                # Tampilkan metrik evaluasi lainnya
                st.subheader("Accuracy Score")
                accuracy = accuracy_score(y_true, y_pred)
                st.write(f"Accuracy: {accuracy:.2f}")

                st.subheader("Classification Report")
                report = classification_report(
                    y_true, y_pred, target_names=["Layak", "Tidak Layak"]
                )
                st.text(report)


if __name__ == "__main__":
    main = MainClass()
    main.run()
