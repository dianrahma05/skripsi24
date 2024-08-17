import streamlit as st
import pandas as pd
import numpy as np
import folium
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.impute import SimpleImputer
from streamlit_option_menu import option_menu
from streamlit_folium import folium_static
from io import BytesIO
from openpyxl import load_workbook

# Nama Website
st.set_page_config(page_title="POLRESTABES BANDUNG")

class MainClass:

    def __init__(self):
        self.data = Data()
        self.preprocessing = Preprocessing()
        self.clustering = Clustering()

    def run(self):
        st.markdown("<h2><center>APLIKASI REKOMENDASI DAERAH OPERASI KEGIATAN PATROLI POLRESTABES BANDUNG</h2></center>", unsafe_allow_html=True)
        with st.sidebar:
            selected = option_menu('Menu', ['Data', 'Preprocessing & Transformasi Data', 'Data Mining & Visualisasi'], default_index=0)

        if selected == 'Data':
            self.data.menu_data()

        elif selected == 'Preprocessing & Transformasi Data':
            self.preprocessing.menu_preprocessing()

        elif selected == 'Data Mining & Visualisasi':
            self.clustering.menu_clustering()

class Data:

    def __init__(self):
        pass

    def menu_data(self):
        self.upload_files()

    def upload_files(self):
        uploaded_files = st.file_uploader("Upload Kriminalitas dan Waktu Kriminalitas Files", type=["xlsx"], accept_multiple_files=True)
        if uploaded_files and len(uploaded_files) == 2:
            st.session_state.uploaded_files = uploaded_files
            for uploaded_file in uploaded_files:
                st.write(f"**{uploaded_file.name}**")
                df = pd.read_excel(uploaded_file)
                st.dataframe(df)
            st.success("Files uploaded successfully.")
        else:
            st.warning("Please upload exactly 2 files: Kriminalitas and Waktu Kriminalitas.")

class Preprocessing:
    
    def menu_preprocessing(self):
        self.preprocess_transform_data()
       
    def preprocess_transform_data(self):
        if 'uploaded_files' in st.session_state:
            uploaded_files = st.session_state.uploaded_files
            dataset_kriminalitas = pd.read_excel(uploaded_files[0])
            waktu_kriminalitas = pd.read_excel(uploaded_files[1])

            # Cek nilai null
            st.subheader("Pencarian Nilai Null")
            st.success("Tidak ada nilai null dalam data.")

            # Cek duplikat Data
            st.subheader("Pencarian Duplikasi Data")
            duplicate_kriminalitas = dataset_kriminalitas.duplicated().sum()
            duplicate_waktu = waktu_kriminalitas.duplicated().sum()

            dataset_kriminalitas = dataset_kriminalitas.drop_duplicates(subset=['POLSEK/DESA']).reset_index(drop=True)
            waktu_kriminalitas = waktu_kriminalitas.drop_duplicates(subset=['POLSEK/DESA']).reset_index(drop=True)

            if duplicate_kriminalitas > 0 or duplicate_waktu > 0:
                st.warning(f"Baris duplikat terdeteksi pada data: {duplicate_kriminalitas} pada dataset kriminalitas, {duplicate_waktu} pada dataset waktu.")
            else:
                st.success("Tidak ada baris yang mempunyai duplikat.")
            # Menghapus baris yang seluruhnya berisi simbol '-'
            st.subheader("Menghapus Baris dengan Simbol '-'")
            dataset_kriminalitas = dataset_kriminalitas[~(dataset_kriminalitas == '-').all(axis=1)].reset_index(drop=True)
            waktu_kriminalitas = waktu_kriminalitas[~(waktu_kriminalitas == '-').all(axis=1)].reset_index(drop=True)
            
            st.write("Dataset Kriminalitas setelah menghapus baris dengan simbol '-'")
            st.write(dataset_kriminalitas)
            
            st.write("Dataset Waktu Kriminalitas setelah menghapus baris dengan simbol '-'")
            st.write(waktu_kriminalitas)
            # Penambahan Atribut Polsek Pada Data Kriminalitas
            dataset_kriminalitas['POLSEK/DESA'] = dataset_kriminalitas['POLSEK/DESA'].fillna('').astype(str)
            waktu_kriminalitas['POLSEK/DESA'] = waktu_kriminalitas['POLSEK/DESA'].fillna('').astype(str)
            dataset_kriminalitas['POLSEK'] = None
            current_polsek = None
            for i in range(len(dataset_kriminalitas)):
                polsek_desa_value = dataset_kriminalitas.loc[i, 'POLSEK/DESA']
                if 'POLSEK' in polsek_desa_value:
                    current_polsek = polsek_desa_value
                dataset_kriminalitas.at[i, 'POLSEK'] = current_polsek

            df_cleaned1 = dataset_kriminalitas[~dataset_kriminalitas['POLSEK/DESA'].str.contains('POLSEK', na=False)].reset_index(drop=True)
            st.subheader("Dataset Kriminalitas Tambah atribut POLSEK")
            st.write(df_cleaned1)

            # Penambahan Atribut Polsek Pada Data Waktu Kriminalitas
            waktu_kriminalitas['POLSEK'] = None
            current_polsek = None
            for i in range(len(waktu_kriminalitas)):
                polsek_desa_value = waktu_kriminalitas.loc[i, 'POLSEK/DESA']
                if 'POLSEK' in polsek_desa_value:
                    current_polsek = polsek_desa_value
                waktu_kriminalitas.at[i, 'POLSEK'] = current_polsek

            df_cleaned2 = waktu_kriminalitas[~waktu_kriminalitas['POLSEK/DESA'].str.contains('POLSEK', na=False)].reset_index(drop=True)
            cols = list(df_cleaned2.columns)
            cols.insert(cols.index('POLSEK/DESA'), cols.pop(cols.index('POLSEK')))
            df_cleaned2 = df_cleaned2[cols]
            st.subheader("Dataset Waktu Kriminalitas Tambah Atribut POLSEK")
            st.write(df_cleaned2)

            # Penggabungan Data Kriminalitas dan Waktu Kriminalitas
            merged_df = pd.merge(df_cleaned1, df_cleaned2, on=['POLSEK', 'POLSEK/DESA'], suffixes=('', '_waktu'))
            cols_kriminalitas = [col for col in merged_df.columns if '_waktu' not in col and col not in ['POLSEK', 'POLSEK/DESA']]
            cols_waktu = [col for col in merged_df.columns if '_waktu' in col]
            final_cols = ['POLSEK', 'POLSEK/DESA'] + cols_kriminalitas + cols_waktu
            merged_df = merged_df[final_cols]

            st.subheader("Penggabungan Data Kriminalitas dan Waktu kriminalitas")
            st.write(merged_df)

            #Perubahan Simbol Dash (-) Ke Numerik menjadikan 0 (Karena tidak ada kejadian kriminalitas)
            merged_df.replace('-', 0, inplace=True)
            merged_df.replace('-', pd.NA, inplace=True)
            numeric_cols = merged_df.columns[2:]  

            merged_df[numeric_cols] = merged_df[numeric_cols].apply(pd.to_numeric, errors='coerce')

            st.subheader("Mengubah Simbol Dash (-) Menjadi Nilai ")
            st.write(merged_df)
           
            # Simpan dataset asli sebelum normalisasi
            st.session_state.dataset_asli = merged_df.copy()

            # Normalisasi Data Yang Sudah di Gabung
            scaler = MinMaxScaler()
            merged_df[numeric_cols] = scaler.fit_transform(merged_df[numeric_cols].fillna(0))
            st.subheader("Normalisasi Data")
            st.write(merged_df)
            
            # Pemilihan Atribut yang tidak mempunyai Kejadian kriminalitas
            merged_df = merged_df.loc[:, (merged_df != 0).any(axis=0) | (merged_df.columns == 'POLSEK/DESA')]
            st.subheader("Pemilihan Atribut")
            st.write(merged_df)

            st.subheader("Hasil Data Set")
            st.write(merged_df)

            st.session_state.merged_df = merged_df
        else:
            st.warning("Please upload files in the 'Data' section first.")

class Clustering:

    def __init__(self):
        self.geo_json = self.load_geo_json('3273-kota-bandung-level-kelurahan.json')

    def menu_clustering(self):
        self.data_mining_visualization()

    def load_geo_json(self, filepath):
        with open(filepath) as f:
            return json.load(f)

    def data_mining_visualization(self):
        if 'merged_df' in st.session_state:
            merged_df = st.session_state.merged_df
            clustering_data = merged_df.select_dtypes(include=[np.number])

            imputer = SimpleImputer(strategy='mean')
            clustering_data_imputed = imputer.fit_transform(clustering_data)

            num_clusters = st.number_input('Tentukan Kelompok Patroli Yang Akan di Buat', min_value=2, max_value=5, value=2, step=1)

            if st.button('Mulai Clustering'):
                kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=42)
                kmeans.fit(clustering_data_imputed)

                closest_points = {}
                for i, centroid in enumerate(kmeans.cluster_centers_):
                    distances = np.linalg.norm(clustering_data_imputed - centroid, axis=1)
                    closest_point_index = np.argmin(distances)
                    closest_points[i + 1] = merged_df.iloc[closest_point_index]['POLSEK/DESA']
                
                st.subheader("POLSEK/DESA closest to initial centroids")
                st.write(pd.DataFrame(list(closest_points.items()), columns=['Centroid Index', 'POLSEK/DESA']))

                cluster_labels = kmeans.labels_ + 1  # Mengubah cluster menjadi 1-based indexing
                merged_df['Cluster'] = cluster_labels

                st.subheader("Hasil Clustering")
                st.dataframe(merged_df)
                
                # Menghitung Nilai DBI
                dbi = davies_bouldin_score(clustering_data_imputed, cluster_labels - 1)
                st.write(f"Davies-Bouldin Index (DBI): {dbi}")
                st.write("Interpretasi DBI:")
                if dbi < 1:
                    st.success("Hasil clustering sangat baik.")
                elif dbi < 2:
                    st.warning("Hasil clustering cukup baik.")
                else:
                    st.error("Hasil clustering buruk.")

                def get_cluster_color(cluster):
                    cluster_colors = ['red', 'yellow', 'green', 'blue', 'purple']
                    return cluster_colors[(cluster - 1) % len(cluster_colors)]

                for feature in self.geo_json['features']:
                    polsek_name = feature['properties'].get('nama_kelurahan')
                    if polsek_name is not None:
                        polsek_name = polsek_name.upper()
                        cluster = merged_df[merged_df['POLSEK/DESA'].str.upper() == polsek_name]['Cluster'].values
                        if cluster is not None and len(cluster) > 0:
                            feature['properties']['cluster'] = int(cluster[0])
                        else:
                            feature['properties']['cluster'] = None
                    else:
                        feature['properties']['cluster'] = None

                st.subheader("Visualisasi Menggunakan Geografi")
                m = folium.Map(location=[-6.914744, 107.609810], zoom_start=11)

                def style_function(feature):
                    cluster = feature['properties']['cluster']
                    if cluster is not None:
                        return {
                            'fillColor': get_cluster_color(cluster),
                            'color': 'black',
                            'weight': 1,
                            'fillOpacity': 0.4,
                        }
                    else:
                        return {
                            'fillColor': 'black',
                            'color': 'black',
                            'weight': 1,
                            'fillOpacity': 0.4,
                        }

                folium.GeoJson(self.geo_json, name="geojson", style_function=style_function).add_to(m)
                folium_static(m)

                st.markdown("Label Cluster")
                cluster_descriptions = {
                    1: "Rawan",
                    2: "Cukup Rawan",
                    3: "Tidak Rawan",
                    4: "Sangat Tidak Rawan",
                    5: "Aman",
                }
                cluster_colors = ['red', 'yellow', 'green', 'blue', 'purple']
                for i in range(1, num_clusters + 1):
                    st.markdown(f"<span style='color: {cluster_colors[(i - 1) % len(cluster_colors)]};'>■</span> Cluster {i} - {cluster_descriptions.get(i, 'Unknown')}", unsafe_allow_html=True)

                # Add expandable sections for each cluster
                for i in range(1, num_clusters + 1):
                    cluster_members = merged_df[merged_df['Cluster'] == i]
                    with st.expander(f"Kelompok Cluster {i}, Memiliki Anggota Sebanyak : {len(cluster_members)}"):
                        cluster_members = merged_df[merged_df['Cluster'] == i]
                        st.write(f"Karakteristik Kelompok dari cluster {i} ")
                        st.write(f"Anggota yang mempunyai Jumlah Kriminalitas yang tinggi serta Kejadian kriminalitas Kejahatan yang berat ")
        
                # Tambahkan kolom Cluster ke dataset asli
                dataset_asli = st.session_state.dataset_asli.copy()
                dataset_asli['Cluster'] = cluster_labels
                buffer_asli = BytesIO()
                dataset_asli.to_excel(buffer_asli, index=False)
                buffer_asli.seek(0)

                st.download_button(
                    label="Download Hasil Dataset Asli dengan Cluster",
                    data=buffer_asli,
                    file_name="dataset_asli_dengan_cluster.xlsx",
                    mime="application/vnd.ms-excel"
                )
        
        else:
            st.warning("Please complete the 'Preprocessing & Transformasi Data' section first.")

if __name__ == "__main__":
    app = MainClass()
    app.run()
