import cv2
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F

from utils.manual_anfis_model import ANFIS
from utils.feature_extraction import (
    ekstrak_fitur_glcm,
    hitung_indeks_vegetasi_exg,
    hitung_indeks_exr_exgr
)


FEATURE_COLUMNS = [
    "X1_Entropy",
    "X2_Contrast",
    "X3_Homogeneity",
    "X4_Energy",
    "X5_Correlation",
    "X6_Indeks_ExG",
    "X7_Indeks_ExR",
    "X8_Indeks_ExGR"
]

CLASS_NAMES = {
    0: "Aman",
    1: "Tersebar",
    2: "Kritis"
}


def load_manual_anfis_model(model_path):
    checkpoint = torch.load(model_path, map_location="cpu")

    model = ANFIS(
        num_inputs=checkpoint["num_inputs"],
        num_rules=checkpoint["num_rules"],
        num_classes=checkpoint["num_classes"]
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


def predict_manual_anfis(img_rgb, model, scaler, feature_columns, grid_size=128):
    tinggi, lebar, _ = img_rgb.shape

    overlay = img_rgb.copy()
    heat_map = np.zeros_like(img_rgb, dtype=np.uint8)

    prediction_map = np.full(
        (tinggi // grid_size + 1, lebar // grid_size + 1),
        -1,
        dtype=int
    )

    results = []

    model.eval()

    for y_idx, y in enumerate(range(0, tinggi, grid_size)):
        for x_idx, x in enumerate(range(0, lebar, grid_size)):

            grid_img = img_rgb[y:y + grid_size, x:x + grid_size]

            if grid_img.shape[0] != grid_size or grid_img.shape[1] != grid_size:
                continue

            glcm_feats = ekstrak_fitur_glcm(grid_img)
            exg_val = hitung_indeks_vegetasi_exg(grid_img)
            exr_val, exgr_val = hitung_indeks_exr_exgr(grid_img)

            raw_features = {
                **glcm_feats,
                "X6_Indeks_ExG": exg_val,
                "X7_Indeks_ExR": exr_val,
                "X8_Indeks_ExGR": exgr_val
            }

            norm_features = normalize_features(raw_features, scaler)

            input_tensor = torch.tensor(
                [[norm_features[col] for col in feature_columns]],
                dtype=torch.float32
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
                np.array([[heat_value]], dtype=np.uint8),
                cv2.COLORMAP_JET
            )[0, 0]

            color_rgb = color[::-1]

            heat_map[y:y + grid_size, x:x + grid_size] = color_rgb

            cv2.rectangle(
                overlay,
                (x, y),
                (x + grid_size, y + grid_size),
                tuple(int(c) for c in color_rgb),
                1
            )

            results.append({
                "x": x,
                "y": y,
                "predicted_class": predicted_class,
                "label": ["Aman", "Tersebar", "Kritis"][predicted_class],
                "prob_aman": prob_aman,
                "prob_tersebar": prob_tersebar,
                "prob_kritis": prob_kritis,
                "heat_score": heat_score
            })

    img_heatmap = cv2.addWeighted(overlay, 0.35, heat_map, 0.65, 0)

    df_result = pd.DataFrame(results)

    class_percent = (
        df_result["label"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
        .to_dict()
    )

    return heat_map, img_heatmap, prediction_map, df_result, class_percent