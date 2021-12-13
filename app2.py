import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# me-non aktifkan peringatan pada python
import warnings
warnings.filterwarnings('ignore')

from model_method import predict
classes = {0:'Berat',1:'Ringan',2:'Sedang'}
class_labels = list(classes.values())
st.title("Klasifikasi Banjir")
st.markdown('Model ini dapat mengkategorikan banjir berdasarkan : **ringan, sedang, berat** ')

def predict_class():
    data = list(map(float,[rata_rata_ketinggian_air,jumlah_terdampak_kk, frekuensi_kejadian]))
    result, probs = predict(data)
    st.write("Daerah tersebut merupakan daerah banjir kategori:  ",result)
    
rata_rata_ketinggian_air = st.text_input('Masukkan rata-rata ketinggian air', '')
jumlah_terdampak_kk = st.text_input('Masukkan jumlah terdampak kk', '')
frekuensi_kejadian = st.text_input('Masukkan frekuensi kejadian', '')

if st.button("Prediksi"):
    predict_class()