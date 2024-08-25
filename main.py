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
import io

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

            # Button to download combined data as Excel
            if st.button("Download Data sebagai Excel"):
                self.download_excel(df_bpnt, df_demografis, df_rastrada)

        else:
            st.warning("Silahkan upload data : Data BPNT, Data Demografis, dan Data Rastrada")

    def download_excel(self, df_bpnt, df_demografis, df_rastrada):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_bpnt.to_excel(writer, sheet_name='Data BPNT', index=False)
            df_demografis.to_excel(writer, sheet_name='Data Demografis', index=False)
            df_rastrada.to_excel(writer, sheet_name='Data Rastrada', index=False)
        writer.save()
        processed_data = output.getvalue()

        st.download_button(
            label="Download Data sebagai Excel",
            data=processed_data,
            file_name="combined_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )


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
        else:
            st.warning("Data belum siap, silahkan lakukan Preprocessing terlebih dahulu.")


class Prediction:

    def menu_prediction(self):
        st.header("Prediksi Status Kelayakan Penerima Bantuan")
        if "test_df" not in st.session_state:
            st.warning("Data untuk prediksi belum tersedia. Silakan lakukan Preprocessing terlebih dahulu.")
            return

        train_df = st.session_state.train_df
        test_df = st.session_state.test_df

        st.subheader("Data Testing (Data Uji)")
        st.write(test_df)

        # Prediksi dengan Gaussian Naive Bayes
        st.subheader("Prediksi dengan Gaussian Naive Bayes")
        try:
            le = LabelEncoder()
            X_train = train_df.drop(columns=["Status_Kelayakan"])
            y_train = le.fit_transform(train_df["Status_Kelayakan"])

            model = GaussianNB()
            model.fit(X_train, y_train)

            # Prediksi pada data uji
            X_test = test_df.drop(columns=["Status_Kelayakan"], errors="ignore")
            y_test = test_df.get("Status_Kelayakan", None)
            predictions = model.predict(X_test)

            if y_test is not None:
                y_test = le.transform(y_test)
                st.subheader("Hasil Prediksi")
                st.write(predictions)

                # Confusion Matrix
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, predictions)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)

                # Evaluation Metrics
                st.subheader("Evaluation Metrics")
                st.write(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")
                st.text(classification_report(y_test, predictions, target_names=le.classes_))
            else:
                st.subheader("Prediksi Status Kelayakan Penerima Bantuan")
                st.write(predictions)
        except Exception as e:
            st.error(f"Error during prediction: {e}")


if __name__ == "__main__":
    app = MainClass()
    app.run()
