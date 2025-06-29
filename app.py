import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import os
import pandas as pd

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('klasifikasi_batik.h5')
    return model

model = load_model()
class_names = ['Awan', 'Bangkai', 'Kujang', 'Kijang', 'Talas']

# Fungsi untuk memproses gambar
def preprocess_image(image_file):
    image = Image.open(image_file).convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0), image

st.title("🧵 Klasifikasi Motif Batik")

uploaded_file = st.file_uploader("Unggah gambar batik", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Gambar diunggah', use_column_width=True)

    img_array, display_image = preprocess_image(uploaded_file)

    if st.button("Prediksi Motif"):
        pred = model.predict(img_array)[0]
        kelas = class_names[np.argmax(pred)]
        confidence = np.max(pred)

        st.image(display_image, caption=f"Prediksi: {kelas} ({confidence:.2f})", use_column_width=True)

        st.subheader("Detail Probabilitas")
        for cls, prob in zip(class_names, pred):
            st.write(f"{cls}: {prob:.4f}")

        # Simpan ke CSV jika diperlukan
        if st.checkbox("Simpan ke katalog"):
            df = pd.DataFrame([[uploaded_file.name, kelas, round(confidence, 4)]],
                              columns=["nama_file", "motif_prediksi", "confidence"])
            if os.path.exists("katalog_batik.csv"):
                df_existing = pd.read_csv("katalog_batik.csv")
                df = pd.concat([df_existing, df], ignore_index=True)
            df.to_csv("katalog_batik.csv", index=False)
            st.success("✅ Disimpan ke katalog_batik.csv")
