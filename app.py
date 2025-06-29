# app.py
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import pandas as pd
import os
from datetime import datetime

# Load model dan label
model = tf.keras.models.load_model('klasifikasi_batik.h5')
class_names = ['Awan', 'Bangkai', 'Kujang', 'Kijang', 'Talas']

def predict_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)[0]
    return class_names[np.argmax(pred)], np.max(pred)

# UI Streamlit
st.title("Klasifikasi Motif Batik")
uploaded_file = st.file_uploader("Upload Gambar Batik", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar Diunggah", use_column_width=True)
    
    label, conf = predict_image(img)
    st.success(f"Prediksi: {label} ({conf:.2f})")

    # Simpan ke katalog (lokal dulu)
    os.makedirs("hasil", exist_ok=True)
    img_path = f"hasil/{label}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
    img.save(img_path)

    # Simpan ke CSV
    csv_path = "hasil/katalog.csv"
    new_data = pd.DataFrame([[os.path.basename(img_path), label, round(conf, 4)]],
                            columns=["nama_file", "motif_prediksi", "confidence"])
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df = pd.concat([df, new_data], ignore_index=True)
    else:
        df = new_data
    df.to_csv(csv_path, index=False)
    st.info("✅ Hasil disimpan ke katalog.")

