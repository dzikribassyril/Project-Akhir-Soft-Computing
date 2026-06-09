import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from utils.feature_extraction import (
    ekstrak_fitur_glcm,
    hitung_indeks_exr_exgr,
    hitung_indeks_vegetasi_exg,
)
from utils.manual_anfis_model import ANFIS

FEATURE_COLUMNS = [
    "X1_Entropy",
    "X2_Contrast",
    "X3_Homogeneity",
    "X4_Energy",
    "X5_Correlation",
    "X6_Indeks_ExG",
    "X7_Indeks_ExR",
    "X8_Indeks_ExGR",
]

CLASS_NAMES = {0: "Aman", 1: "Tersebar", 2: "Kritis"}


def load_manual_anfis_model(model_path):
    checkpoint = torch.load(model_path, map_location="cpu")

    model = ANFIS(
        num_inputs=checkpoint["num_inputs"],
        num_rules=checkpoint["num_rules"],
        num_classes=checkpoint["num_classes"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    scaler = checkpoint["scaler_params"]
    feature_columns = checkpoint["feature_columns"]

    return model, scaler, feature_columns


def normalize_features(features_dict, scaler):
    normalized = {}

    for key, value in features_dict.items():
        if key in scaler:
            min_val = scaler[key]["min"]
            max_val = scaler[key]["max"]

            if max_val != min_val:
                normalized[key] = (value - min_val) / (max_val - min_val)
            else:
                normalized[key] = 0.0
        else:
            normalized[key] = value

    return normalized


def _pad_to_grid(img_rgb, grid_size):
    """Tambah padding merata di semua sisi agar grid mulai dari tengah.
    Mengembalikan (padded_img, pad_top, pad_left)."""
    tinggi, lebar, _ = img_rgb.shape

    sisa_h = tinggi % grid_size
    sisa_w = lebar % grid_size

    # Berapa piksel padding yang dibutuhkan agar habis dibagi grid_size
    pad_h = (grid_size - sisa_h) % grid_size
    pad_w = (grid_size - sisa_w) % grid_size

    # Bagi rata atas-bawah dan kiri-kanan
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    padded = cv2.copyMakeBorder(
        img_rgb,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_REFLECT_101,
    )
    return padded, pad_top, pad_left


def _ensure_min_grids(img_rgb, grid_size, min_grids):
    tinggi, lebar, _ = img_rgb.shape
    min_size = int(grid_size * min_grids)

    if tinggi >= min_size and lebar >= min_size:
        return img_rgb

    scale = max(min_size / tinggi, min_size / lebar)
    new_width = int(np.ceil(lebar * scale))
    new_height = int(np.ceil(tinggi * scale))

    return cv2.resize(img_rgb, (new_width, new_height), interpolation=cv2.INTER_CUBIC)


def predict_manual_anfis(
    img_rgb, model, scaler, feature_columns, grid_size=128, min_grids=4
):
    img_rgb = _ensure_min_grids(img_rgb, grid_size, min_grids)

    # Padding agar grid simetris dan tidak ada sisa di tepi
    padded, pad_top, pad_left = _pad_to_grid(img_rgb, grid_size)
    tinggi_pad, lebar_pad, _ = padded.shape
    tinggi_ori, lebar_ori, _ = img_rgb.shape

    overlay_pad = padded.copy()
    heat_map_pad = np.zeros_like(padded, dtype=np.uint8)

    n_rows = tinggi_pad // grid_size
    n_cols = lebar_pad // grid_size

    prediction_map = np.full((n_rows, n_cols), -1, dtype=int)

    results = []
    model.eval()

    for y_idx in range(n_rows):
        for x_idx in range(n_cols):
            y = y_idx * grid_size
            x = x_idx * grid_size

            grid_img = padded[y : y + grid_size, x : x + grid_size]

            glcm_feats = ekstrak_fitur_glcm(grid_img)
            exg_val = hitung_indeks_vegetasi_exg(grid_img)
            exr_val, exgr_val = hitung_indeks_exr_exgr(grid_img)

            raw_features = {
                **glcm_feats,
                "X6_Indeks_ExG": exg_val,
                "X7_Indeks_ExR": exr_val,
                "X8_Indeks_ExGR": exgr_val,
            }

            norm_features = normalize_features(raw_features, scaler)

            input_tensor = torch.tensor(
                [[norm_features[col] for col in feature_columns]], dtype=torch.float32
            )

            with torch.no_grad():
                logits, _ = model(input_tensor)
                probabilities = F.softmax(logits, dim=1)
                predicted_class = torch.argmax(logits, dim=1).item()

            prediction_map[y_idx, x_idx] = predicted_class

            prob_aman = probabilities[0, 0].item()
            prob_tersebar = probabilities[0, 1].item()
            prob_kritis = probabilities[0, 2].item()

            heat_score = prob_tersebar * 0.5 + prob_kritis * 1.0
            heat_value = int(np.clip(heat_score, 0.0, 1.0) * 255)

            color = cv2.applyColorMap(
                np.array([[heat_value]], dtype=np.uint8), cv2.COLORMAP_JET
            )[0, 0]
            color_rgb = color[::-1]

            heat_map_pad[y : y + grid_size, x : x + grid_size] = color_rgb

            cv2.rectangle(
                overlay_pad,
                (x, y),
                (x + grid_size, y + grid_size),
                tuple(int(c) for c in color_rgb),
                1,
            )

            # Koordinat relatif terhadap gambar asli (sebelum padding)
            x_ori = x - pad_left
            y_ori = y - pad_top

            results.append(
                {
                    "x": x_ori,
                    "y": y_ori,
                    "predicted_class": predicted_class,
                    "label": ["Aman", "Tersebar", "Kritis"][predicted_class],
                    "prob_aman": prob_aman,
                    "prob_tersebar": prob_tersebar,
                    "prob_kritis": prob_kritis,
                    "heat_score": heat_score,
                }
            )

    img_heatmap_pad = cv2.addWeighted(overlay_pad, 0.35, heat_map_pad, 0.65, 0)

    # Crop kembali ke ukuran gambar asli (buang padding)
    h_end = pad_top + tinggi_ori
    w_end = pad_left + lebar_ori
    heat_map = heat_map_pad[pad_top:h_end, pad_left:w_end]
    img_heatmap = img_heatmap_pad[pad_top:h_end, pad_left:w_end]

    df_result = pd.DataFrame(results)

    class_percent = (
        df_result["label"].value_counts(normalize=True).mul(100).round(2).to_dict()
    )

    return heat_map, img_heatmap, prediction_map, df_result, class_percent
