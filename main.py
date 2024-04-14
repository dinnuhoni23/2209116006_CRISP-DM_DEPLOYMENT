import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from streamlit_option_menu import *
import pickle


df = pd.read_csv('https://raw.githubusercontent.com/dinnuhoni23/Mini-Project-Data-Mining/main/Data%20Cleaned.csv')
df2 = pd.read_csv('before_mapping.csv')
df3 = pd.read_csv('Data Cleaned (2).csv')

with st.sidebar :
    selected = option_menu('Luxury Watch',['Introducing','Data Distribution','Relation','Composition & Comparison','Predict','Clustering'],default_index=0)

def distribusi_brand_models():
    summary_data = df2.groupby('Crystal Material')['Price (USD)'].median().reset_index()

    # Plot bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(summary_data['Crystal Material'], summary_data['Price (USD)'], color='skyblue')
    plt.xlabel('Crystal Material')
    plt.ylabel('Median Price (USD)')
    plt.title('Median Price (USD) by Crystal Material')
    st.pyplot(plt)

def relations():
    numeric_cols = df3.select_dtypes(include=['int', 'float'])
    correlation_matrix = numeric_cols.corr()

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Heatmap of Column Correlations')
    plt.show()
    st.pyplot(plt)

def composition():
    df = df3.select_dtypes(include=['int', 'float64'])
    price_category_composition = df.groupby('Price Category').mean()

# Plot komposisi kategori harga
    plt.figure(figsize=(10, 6))
    sns.heatmap(price_category_composition.T, annot=True,fmt='g' , cmap='YlGnBu')
    plt.title('Komposisi untuk setiap kategori harga')
    plt.xlabel('Kategori Harga')
    plt.ylabel('Fitur')
    st.pyplot(plt)

def predict(df):
    case_material = st.selectbox('Select Case Material',[i for i in df['Case Material'].unique()])
    strap_material = st.selectbox('Select Strap Material',[i for i in df['Strap Material'].unique()])
    movemen_type = st.selectbox('Select MOvement Type',[i for i in df['Movement Type'].unique()])
    dial_color = st.selectbox('Select Dial Color',[i for i in df['Dial Color'].unique()])
    Crysal_materal = st.selectbox('Select Crystal Material',[i for i in df['Crystal Material'].unique()])
    price = st.number_input('Ínput Price',0,40000)
    water = st.number_input('Ínput water resistance',30,2000)

    data = pd.DataFrame({
        'Case Material' : [np.where(df['Case Material'].unique() == case_material)[0][0]],
        'Strap Material' : [np.where(df['Strap Material'].unique() == strap_material)[0][0]],
        'Movement Type' : [np.where(df['Movement Type'].unique() == movemen_type)[0][0]],
        'Dial Color' : [np.where(df['Dial Color'].unique() == dial_color)[0][0]],
        'Crystal Material' : [np.where(df['Crystal Material'].unique() == Crysal_materal)[0][0]],
        'Price (USD)' : [price],
        'Water Resistance (m)' : [water]
    })
    st.write(data)
    button = st.button('predict')
    if button:
        with open('gnb.pkl','rb') as file:
            loaded = pickle.load(file)
        
        predict = loaded.predict(data)
        if predict == 0:
            st.write('cheap')
        elif predict == 1:
            st.write('middle')
        elif predict == 2:
            st.write('expensive')
        else :
            st.write('not defined')

def clustering(df):
    case_material = st.selectbox('Select Case Material',[i for i in df['Case Material'].unique()])
    strap_material = st.selectbox('Select Strap Material',[i for i in df['Strap Material'].unique()])
    movement_type = st.selectbox('Select Movement Type',[i for i in df['Movement Type'].unique()])
    dial_color = st.selectbox('Select Dial Color',[i for i in df['Dial Color'].unique()])
    crystal_material = st.selectbox('Select Crystal Material',[i for i in df['Crystal Material'].unique()])
    price = st.number_input('Input Price',0,40000)
    price_category = st.selectbox('Select Price Category',[i for i in df['Movement Type'].unique()])
    water = st.number_input('Input Water Resistance',30,2000)
    button = st.button('Clustering!')

    if button:
        # Membuat DataFrame baru dari input pengguna
        data = pd.DataFrame({
            'Case Material': [case_material],
            'Strap Material': [strap_material],
            'Movement Type': [movement_type],
            'Dial Color': [dial_color],
            'Crystal Material': [crystal_material],
            'Price (USD)': [price],
            'Price Category': [price_category],
            'Water Resistance (m)': [water]
        })

        # Memuat model KMeans
        with open('kmeans.pkl', 'rb') as file:
            loaded_model = pickle.load(file)
        
        # Melakukan prediksi klaster
        cluster_prediction = loaded_model.predict(data)

        st.write(f'Predicted Cluster: {cluster_prediction[0]}')


if (selected == 'Introducing'):
    st.title('Analisis Penjualan Jam Tangan Mewah')
    st.subheader('Tabel Data Jam Mewah')
    st.dataframe(df2)
    st.write('Tabel diatas merupakan dataset dari data jam mewah. Dimana dataset diatas memiliki 507 baris dan 14 kolom yang berbeda. Berikut adalah penjelasannya:')
    st.markdown("""
            1. **Brand** : Merek jam tangan.
            2. **Merek** : Model spesifik dari jam tangan yang dimiliki oleh merek tersebut.
            3. **Case Material** : Bahan yang digunakan untuk bagian luar dari jam tangan.
            4. **Strap Material** : Bahan yang digunakan untuk tali jam.
            5. **Movement Type** : Jenis mekanisme yang digunakan untuk menjalankan jam tangan.
            6. **Water Resistance** : Menunjukkan seberapa tahan air jam tangan tersebut.
            7. **Case Diameter (mm)** : Diameter dari bagian luar jam tangan.
            8. **Case Thickness (mm)** : Ketebalan bagian luar jam tangan.
            9. **Band Width (mm)** : Lebar dari tali jam.
            10. **Dial Color** : Warna latar belakang di mana angka dan jarum jam ditempatkan.
            11. **Crystal Material** : Bahan yang digunakan untuk kaca penutup jam tangan.
            12. **Complications** : Mengacu pada fungsi tambahan yang dimiliki oleh jam tangan selain dari waktu dasar.
            13. **Power Reserve** : Menunjukkan berapa lama jam tangan akan tetap berfungsi setelah sepenuhnya diisi.
            14. **Price (USD)** : Harga jam tangan dalam dolar Amerika Serikat.
            """)

if (selected == 'Data Distribution'):
    st.header("Data Distribution")
    distribusi_brand_models()
    st.markdown("""
            Grafik diatas menunjukkan median harga (dalam USD) dari berbagai jenis Crystal Material. Semakin tinggi harga, semakin mahal jenis bahan kristalnya. Semakin rendah harga, semakin murah jenis bahan kristalnya. Jadi dapat disimpulkan dari grafik diatas bahwa:
             - **Hardlex** adalah bahan kristal yang paling murah, dengan median harga $1.000 USD.
             - **Hesalite** adalah bahan kristal termahal kedua, dengan median harga $5.500 USD.
             - **Mineral** adalah bahan kristal paling murah sama seperti Hardlex, dengan median harga $1.000 USD.
             - **Sapphire** adalah bahan kristal yang paling mahal, dengan median harga $9.000 USD.""")
    
    
if (selected == 'Relation'):
    st.title('Relations')
    relations()
    st.write('Gambar diatas merupakan heatmap yang menunjukkan korelasi antara semua kolom dalam dataset ini. Semakin merah warnanya, semakin kuat korelasi positifnya. Sebaliknya, semakin biru warnanya, semakin kuat korelasi negatifnya.')
    

if (selected == 'Composition & Comparison'):
    st.title('Composition')
    composition()
    st.write('Dari gambar diatas menunjukkan komposisi kelas yang diambil dari median setiap fitur yang ada yang telah melakukan proses mapping dan ada perbandingan dari setiap fitur yang digunakan.')
    

if (selected == 'Predict'):
    st.title('Let\'s Predict!')
    predict(df2)
    

if (selected == 'Clustering'):
    st.title('Clustering!')
    clustering(df3)