import streamlit as st
import numpy as np
from PIL import Image
from utils.cnn_inference import load_cnn_anfis_assets, predict_cnn_anfis
from utils.cnn_inference import predict_cnn_anfis

from utils.manual_anfis_inference import (
    load_manual_anfis_model,
    predict_manual_anfis
)



st.set_page_config(
    page_title="UAV Waste Severity Mapping",
    layout="wide"
)

st.title("🚁 UAV Waste Severity Mapping")
st.write("Perbandingan model Manual Feature Engineering + ANFIS dan CNN + ANFIS.")

MODEL_PATH = "models/model_manual_feature.pth"
CNN_PATH = "models/finetuned_efficientnet_b0_grid256_classifier.pth"
CNN_ANFIS_PATH = "models/model_efficientnet_b0_finetuned_original-hflip-vflip_grid256_anfis.pth"
CNN_PCA_PATH = "models/pca_efficientnet_b0_finetuned_original-hflip-vflip_grid256_best.pkl"
CNN_SCALER_PATH = "models/scaler_efficientnet_b0_finetuned_original-hflip-vflip_grid256_best.pkl"


@st.cache_resource
def cached_load_anfis_model():
    return load_manual_anfis_model(MODEL_PATH)

@st.cache_resource
def cached_load_cnn_anfis():
    return load_cnn_anfis_assets(
        CNN_PATH,
        CNN_ANFIS_PATH,
        CNN_PCA_PATH,
        CNN_SCALER_PATH
    )

st.sidebar.header("Pengaturan")

uploaded_file = st.sidebar.file_uploader(
    "Upload citra UAV",
    type=["jpg", "jpeg", "png"]
)

image_scale = st.sidebar.select_slider(
    "Skala Resolusi Input",
    options=[1.0, 1.25, 1.5, 2.0],
    value=1.0
)

grid_size = st.sidebar.select_slider(
    "Ukuran Grid Heatmap",
    options=[32, 64, 128, 256, 384, 512],
    value=128
)

model_choice = st.sidebar.selectbox(
    "Pilih Model",
    options=[
        "Manual Feature Engineering + ANFIS",
        "CNN + ANFIS"
    ]
)

predict_button = st.sidebar.button("Prediksi")


if uploaded_file is None:
    st.info("Silakan upload citra UAV terlebih dahulu.")

else:
    image = Image.open(uploaded_file).convert("RGB")

    if image_scale != 1.0:
        new_size = (
            int(image.width * image_scale),
            int(image.height * image_scale)
        )
        image = image.resize(new_size, Image.Resampling.BICUBIC)

    img_rgb = np.array(image)

    st.subheader("Citra Asli")
    st.image(img_rgb, use_container_width=True)

    if predict_button:
        with st.spinner("Sedang melakukan prediksi..."):
                
            if model_choice == "Manual Feature Engineering + ANFIS":
                model, scaler, feature_columns = cached_load_anfis_model()

                heatmap, overlay, prediction_map, df_result, class_percent = predict_manual_anfis(
                    img_rgb=img_rgb,
                    model=model,
                    scaler=scaler,
                    feature_columns=feature_columns,
                    grid_size=grid_size
                )

                st.success("Prediksi selesai.")

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Heatmap Prediksi")
                    st.image(heatmap, use_container_width=True)

                with col2:
                    st.subheader("Overlay Hasil")
                    st.image(overlay, use_container_width=True)

                st.subheader("Persentase Kelas")

                c1, c2, c3 = st.columns(3)

                c1.metric("Aman", f"{class_percent.get('Aman', 0)}%")
                c2.metric("Tersebar", f"{class_percent.get('Tersebar', 0)}%")
                c3.metric("Kritis", f"{class_percent.get('Kritis', 0)}%")

                st.subheader("Tabel Hasil Grid")
                st.dataframe(df_result, use_container_width=True)

            elif model_choice == "CNN + ANFIS":
                feature_extractor, preprocess, cnn_anfis_model, pca, cnn_scaler, checkpoint = cached_load_cnn_anfis()

                st.info("Model CNN + ANFIS menggunakan grid 256 sesuai training.")

                heatmap, overlay, df_result, class_percent = predict_cnn_anfis(
                    img_rgb=img_rgb,
                    feature_extractor=feature_extractor,
                    preprocess=preprocess,
                    anfis_model=cnn_anfis_model,
                    pca=pca,
                    scaler=cnn_scaler,
                    grid_size=256
                )

                st.success("Prediksi CNN + ANFIS selesai.")

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Heatmap Prediksi")
                    st.image(heatmap, use_container_width=True)

                with col2:
                    st.subheader("Overlay Hasil")
                    st.image(overlay, use_container_width=True)

                st.subheader("Persentase Kelas")

                c1, c2, c3 = st.columns(3)
                c1.metric("Aman", f"{class_percent.get('Aman', 0)}%")
                c2.metric("Tersebar", f"{class_percent.get('Tersebar', 0)}%")
                c3.metric("Kritis", f"{class_percent.get('Kritis', 0)}%")

                st.subheader("Tabel Hasil Grid")
                st.dataframe(df_result, use_container_width=True)
