import streamlit as st
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from streamlit_option_menu import option_menu

# Nama Website
st.set_page_config(page_title="SISTEM KLASIFIKASI KELURAHAN")

class MainClass:

    def _init_(self):
        self.data = Data()
        self.preprocessing = Preprocessing()
        # self.data_mining = DataMining()
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

    def _init_(self):
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
            st.write(f"*File BPNT*: {self.file_names['bpnt'].name}")
            st.write(f"*File Demografis*: {self.file_names['demografis'].name}")
            st.write(f"*File Rastrada*: {self.file_names['rastrada'].name}")

            # Display dataframes for each file
            df_bpnt = pd.read_excel(self.file_names["bpnt"])
            df_demografis = pd.read_excel(self.file_names["demografis"])
            df_rastrada = pd.read_excel(self.file_names["rastrada"])

            st.dataframe(df_bpnt)
            st.dataframe(df_demografis)
            st.dataframe(df_rastrada)

            st.success("File telah berhasil di upload")

            # Save data to session state
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
            try:
                data_bpnt = pd.read_excel(uploaded_files["bpnt"])
                data_demografis = pd.read_excel(uploaded_files["demografis"])
                data_rastrada = pd.read_excel(uploaded_files["rastrada"])
            except Exception as e:
                st.error(f"Error loading files: {e}")
                return

            # Cek nilai null
            st.subheader("Pencarian Nilai Null")
            if data_bpnt.isnull().sum().sum() == 0 and data_demografis.isnull().sum().sum() == 0 and data_rastrada.isnull().sum().sum() == 0:
                st.success("Tidak ada nilai null dalam data.")
            else:
                st.warning("Terdapat nilai null dalam data.")

            # Cek duplikat Data
            st.subheader("Pencarian Duplikasi Data")
            duplicate_bpnt = data_bpnt.duplicated().sum()
            duplicate_demografis = data_demografis.duplicated().sum()
            duplicate_rastrada = data_rastrada.duplicated().sum()

            data_bpnt = data_bpnt.drop_duplicates(subset=["Id_Training"]).reset_index(drop=True)
            data_demografis = data_demografis.drop_duplicates(subset=["Id_Training"]).reset_index(drop=True)
            data_rastrada = data_rastrada.drop_duplicates(subset=["Id_Training"]).reset_index(drop=True)

            if duplicate_bpnt > 0 or duplicate_demografis > 0 or duplicate_rastrada > 0:
                st.warning(
                    f"Baris duplikat terdeteksi pada data: {duplicate_bpnt} pada data bpnt, {duplicate_demografis} pada data demografis, {duplicate_rastrada} pada data rastrada."
                )
            else:
                st.success("Tidak ada baris yang mempunyai duplikat.")

            # Menggabungkan data berdasarkan 'Id_Training'
            try:
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
            except KeyError as e:
                st.error(f"KeyError during merge: {e}")
                return

            # Menghapus kolom 'Id_Training' dari merged_df
            cols_kriminalitas = [
                col
                for col in merged_df.columns
                if "_demografis" not in col
                and "_rastrada" not in col
                and col not in ["Id_Training"]
            ]
            cols_waktu = [
                col
                for col in merged_df.columns
                if "_demografis" in col or "_rastrada" in col
            ]
            final_cols = cols_kriminalitas + cols_waktu
            merged_df = merged_df[final_cols]

            st.subheader("Penggabungan Data")
            st.write(merged_df)

            # Pemilihan Atribut
            expected_columns = [
                "Nama_KRT", "Alamat", "Pekerjaan", "Usia",
                "Status_Perkawinan", "Status_Bangunan", "Tanggungan",
                "Pendapatan", "Kondisi_Dinding", "Kesehatan", "Status_Kelayakan"
            ]

            # Periksa kolom yang ada dalam merged_df
            available_columns = [col for col in expected_columns if col in merged_df.columns]
            
            if len(available_columns) < len(expected_columns):
                missing_columns = set(expected_columns) - set(available_columns)
                st.warning(f"Beberapa kolom yang diharapkan tidak ada dalam data: {missing_columns}")

            merged_df = merged_df[available_columns]

            st.subheader("Pemilihan Atribut")
            st.write(merged_df)

            # Split data menjadi training dan testing
            try:
                train_df, test_df = train_test_split(merged_df, test_size=0.2, random_state=42)
            except Exception as e:
                st.error(f"Error splitting data: {e}")
                return

            st.subheader("Training Data")
            st.write(train_df)

            st.subheader("Testing Data")
            st.write(test_df)

            # Identifikasi fitur kategorikal dan numerikal
            cat_features = [col for col in train_df.columns if train_df[col].dtype == 'object' and col != 'Status_Kelayakan']
            num_features = [col for col in train_df.columns if train_df[col].dtype != 'object' and col != 'Status_Kelayakan']
            
            st.write("Fitur Kategorikal:", cat_features)
            st.write("Fitur Numerikal:", num_features)

            # Buat preprocessor
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', SimpleImputer(strategy='mean'), num_features),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
                ]
            )

            X_train = preprocessor.fit_transform(train_df.drop('Status_Kelayakan', axis=1))
            y_train = train_df['Status_Kelayakan']
            X_test = preprocessor.transform(test_df.drop('Status_Kelayakan', axis=1))
            y_test = test_df['Status_Kelayakan']

            st.session_state.preprocessor = preprocessor
            st.session_state.X_train = X_train
            st.session_state.y_train = y_train
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test

            st.success("Data preprocessing berhasil!")


class Prediction:
    def menu_prediction(self):
        self.run_prediction()

    def run_prediction(self):
        st.write("Prediksi Status Kelayakan")

        # Tambahkan opsi untuk melatih model
        if st.button("Latih Model"):
            if 'X_train' not in st.session_state or 'y_train' not in st.session_state:
                st.error("Data training belum tersedia. Pastikan Anda sudah melakukan preprocessing data.")
                return

            X_train = st.session_state.X_train
            y_train = st.session_state.y_train

            # Melatih model Naive Bayes
            model = MultinomialNB()
            model.fit(X_train, y_train)
            st.session_state.model = model

            st.success("Model berhasil dilatih.")

        # File uploader untuk prediksi
        uploaded_file = st.file_uploader("Upload File Data untuk Prediksi", type=["xlsx"])

        if uploaded_file:
            if "preprocessor" not in st.session_state:
                st.error("Preprocessing belum dilakukan.")
                return

            data_for_prediction = pd.read_excel(uploaded_file)

            if 'Id_Training' in data_for_prediction.columns:
                data_for_prediction = data_for_prediction.drop('Id_Training', axis=1)

            # Apply the preprocessing to the prediction data
            preprocessor = st.session_state.preprocessor
            X_pred = preprocessor.transform(data_for_prediction)

            # Load or train Naive Bayes model here
            model_option = st.selectbox("Pilih Model", ["Naive Bayes Manual", "MultinomialNB"])
            
            if model_option == "Naive Bayes Manual":
                if "model" not in st.session_state:
                    st.error("Model belum dilatih.")
                    return
                model = st.session_state.model
            elif model_option == "MultinomialNB":
                if 'model' not in st.session_state:
                    st.error("Model belum dilatih.")
                    return
                model = st.session_state.model

            # Perform prediction
            predictions = model.predict(X_pred)

            # Add predictions to the original dataframe
            data_for_prediction['Status_Kelayakan'] = predictions

            st.subheader("Hasil Prediksi dengan Status Kelayakan")
            st.write(data_for_prediction)

            # Calculate accuracy, confusion matrix, and classification report if test data is available
            if 'X_test' in st.session_state and 'y_test' in st.session_state:
                X_test = st.session_state.X_test
                y_test = st.session_state.y_test

                y_pred = model.predict(X_test)

                accuracy = accuracy_score(y_test, y_pred)
                conf_matrix = confusion_matrix(y_test, y_pred)
                class_report = classification_report(y_test, y_pred)

                st.subheader("Akurasi Model")
                st.write(f"Akurasi: {accuracy:.2f}")

                st.subheader("Confusion Matrix")
                st.write(conf_matrix)

                st.subheader("Classification Report")
                st.text(class_report)
            else:
                st.warning("Data testing belum tersedia untuk menghitung akurasi, confusion matrix, dan classification report.")


if __name__ == "_main_":
    main = MainClass()
    main.run()