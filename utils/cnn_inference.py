import cv2
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

CLASS_NAMES = {0: "Aman", 1: "Tersebar", 2: "Kritis"}


class EfficientNetFeatureExtractor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.features = model.features
        self.avgpool = model.avgpool

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


class ANFIS(nn.Module):
    def __init__(self, num_inputs, num_rules, num_classes):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_rules = num_rules
        self.num_classes = num_classes

        self.mu = nn.Parameter(torch.randn(num_rules, num_inputs))
        self.sigma_raw = nn.Parameter(torch.ones(num_rules, num_inputs) * 0.5)

        self.consequent_weights = nn.Parameter(
            torch.randn(num_rules, num_inputs) * 0.05
        )
        self.consequent_bias = nn.Parameter(torch.zeros(num_rules))
        self.classifier = nn.Linear(num_rules, num_classes)

    @property
    def sigma(self):
        return F.softplus(self.sigma_raw) + 1e-6

    def forward(self, x):
        x_expanded = x.unsqueeze(1).expand(-1, self.num_rules, -1)
        sigma = self.sigma

        log_mf = -0.5 * ((x_expanded - self.mu) / sigma) ** 2
        log_firing = torch.sum(log_mf, dim=2)
        normalized_firing = torch.softmax(log_firing, dim=1)

        consequent = (
            torch.sum(x_expanded * self.consequent_weights, dim=2)
            + self.consequent_bias
        )
        weighted_output = normalized_firing * consequent
        logits = self.classifier(weighted_output)

        return logits, normalized_firing


class CocoMaskCNNANFIS(nn.Module):
    def __init__(self, num_inputs, num_rules, num_classes):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_rules = num_rules
        self.num_classes = num_classes

        self.mu = nn.Parameter(torch.randn(num_rules, num_inputs))
        self.sigma_raw = nn.Parameter(torch.ones(num_rules, num_inputs) * 0.5)
        self.consequent_weights = nn.Parameter(torch.randn(num_rules, num_inputs) * 0.05)
        self.consequent_bias = nn.Parameter(torch.zeros(num_rules))
        self.classifier = nn.Linear(num_rules, num_classes)

    def forward(self, x):
        sigma = F.softplus(self.sigma_raw) + 1e-6
        x_expanded = x.unsqueeze(1).expand(-1, self.num_rules, -1)
        membership = torch.exp(-0.5 * ((x_expanded - self.mu) / sigma) ** 2)

        firing = torch.prod(membership, dim=2)
        firing_norm = firing / (firing.sum(dim=1, keepdim=True) + 1e-8)

        consequent = (x_expanded * self.consequent_weights.unsqueeze(0)).sum(dim=2)
        consequent = consequent + self.consequent_bias.unsqueeze(0)
        rule_outputs = firing_norm * consequent

        logits = self.classifier(rule_outputs)
        return logits, firing_norm


def load_cnn_anfis_assets(cnn_path, anfis_path, pca_path, scaler_path):
    device = torch.device("cpu")

    weights = EfficientNet_B0_Weights.DEFAULT
    cnn_classifier = efficientnet_b0(weights=weights)

    in_features = cnn_classifier.classifier[1].in_features
    cnn_classifier.classifier[1] = nn.Linear(in_features, 3)

    cnn_classifier.load_state_dict(torch.load(cnn_path, map_location=device))
    cnn_classifier.eval()

    feature_extractor = EfficientNetFeatureExtractor(cnn_classifier)
    feature_extractor.eval()

    preprocess = weights.transforms()

    checkpoint = torch.load(anfis_path, map_location=device)

    anfis_model = ANFIS(
        num_inputs=checkpoint["num_inputs"],
        num_rules=checkpoint["num_rules"],
        num_classes=checkpoint["num_classes"],
    )

    anfis_model.load_state_dict(checkpoint["model_state_dict"])
    anfis_model.eval()

    pca = joblib.load(pca_path)
    scaler = joblib.load(scaler_path)

    return feature_extractor, preprocess, anfis_model, pca, scaler, checkpoint


def load_coco_mask_cnn_anfis_assets(
    anfis_path,
    pca_path,
    scaler_path,
    config_path=None,
    num_rules=12,
    num_classes=3,
):
    device = torch.device("cpu")

    weights = EfficientNet_B0_Weights.DEFAULT
    cnn_model = efficientnet_b0(weights=weights)
    feature_extractor = EfficientNetFeatureExtractor(cnn_model)
    feature_extractor.eval()

    preprocess = weights.transforms()
    pca = joblib.load(pca_path)
    scaler = joblib.load(scaler_path)

    config = {}
    if config_path is not None:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

    num_rules = int(config.get("anfis_rules", num_rules))
    num_classes = int(config.get("num_classes", num_classes))
    num_inputs = int(getattr(pca, "n_components_", 24))
    anfis_model = CocoMaskCNNANFIS(
        num_inputs=num_inputs,
        num_rules=num_rules,
        num_classes=num_classes,
    )
    anfis_model.load_state_dict(torch.load(anfis_path, map_location=device))
    anfis_model.eval()

    config = {
        **config,
        "grid_size": 128,
        "num_inputs": num_inputs,
        "num_rules": num_rules,
        "num_classes": num_classes,
        "class_names": CLASS_NAMES,
        "cnn_model": "efficientnet_b0_default_feature_extractor",
    }

    return feature_extractor, preprocess, anfis_model, pca, scaler, config


def extract_cnn_embedding(grid_img, feature_extractor, preprocess, use_augmentation=True):
    pil_img = Image.fromarray(grid_img)

    if use_augmentation:
        augmented_imgs = [pil_img, ImageOps.mirror(pil_img), ImageOps.flip(pil_img)]
    else:
        augmented_imgs = [pil_img]

    tensors = [preprocess(img) for img in augmented_imgs]
    batch = torch.stack(tensors, dim=0)

    with torch.no_grad():
        feats = feature_extractor(batch)
        feats = feats.float().mean(dim=0, keepdim=True)

    return feats.numpy()


def _pad_to_grid(img_rgb, grid_size):
    """Tambah padding merata di semua sisi agar grid mulai dari tengah.
    Mengembalikan (padded_img, pad_top, pad_left)."""
    tinggi, lebar, _ = img_rgb.shape

    sisa_h = tinggi % grid_size
    sisa_w = lebar % grid_size

    pad_h = (grid_size - sisa_h) % grid_size
    pad_w = (grid_size - sisa_w) % grid_size

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


def predict_cnn_anfis(
    img_rgb,
    feature_extractor,
    preprocess,
    anfis_model,
    pca,
    scaler,
    grid_size=256,
    min_grids=4,
    use_augmentation=True,
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

    results = []

    for y_idx in range(n_rows):
        for x_idx in range(n_cols):
            y = y_idx * grid_size
            x = x_idx * grid_size

            grid_img = padded[y : y + grid_size, x : x + grid_size]

            embedding = extract_cnn_embedding(
                grid_img,
                feature_extractor,
                preprocess,
                use_augmentation=use_augmentation,
            )

            pca_features = pca.transform(embedding)
            scaled_features = scaler.transform(pca_features).astype(np.float32)
            input_tensor = torch.tensor(scaled_features, dtype=torch.float32)

            with torch.no_grad():
                logits, _ = anfis_model(input_tensor)
                probabilities = F.softmax(logits, dim=1)
                predicted_class = torch.argmax(logits, dim=1).item()

            prob_aman = probabilities[0, 0].item()
            prob_tersebar = probabilities[0, 1].item()
            prob_kritis = probabilities[0, 2].item()
            heat_score = prob_tersebar * 0.5 + prob_kritis * 1.0

            if predicted_class == 0:
                color_rgb = np.array([0, 255, 0], dtype=np.uint8)
            elif predicted_class == 1:
                color_rgb = np.array([255, 255, 0], dtype=np.uint8)
            else:
                color_rgb = np.array([255, 0, 0], dtype=np.uint8)

            heat_map_pad[y : y + grid_size, x : x + grid_size] = color_rgb

            cv2.rectangle(
                overlay_pad,
                (x, y),
                (x + grid_size, y + grid_size),
                tuple(int(c) for c in color_rgb),
                2,
            )

            # Koordinat relatif terhadap gambar asli (sebelum padding)
            x_ori = x - pad_left
            y_ori = y - pad_top

            results.append(
                {
                    "x": x_ori,
                    "y": y_ori,
                    "predicted_class": predicted_class,
                    "label": CLASS_NAMES[predicted_class],
                    "prob_aman": prob_aman,
                    "prob_tersebar": prob_tersebar,
                    "prob_kritis": prob_kritis,
                    "heat_score": heat_score,
                }
            )

    overlay_pad_result = cv2.addWeighted(overlay_pad, 0.35, heat_map_pad, 0.65, 0)

    # Crop kembali ke ukuran gambar asli (buang padding)
    h_end = pad_top + tinggi_ori
    w_end = pad_left + lebar_ori
    heat_map = heat_map_pad[pad_top:h_end, pad_left:w_end]
    overlay_result = overlay_pad_result[pad_top:h_end, pad_left:w_end]

    df_result = pd.DataFrame(results)

    class_percent = (
        df_result["label"].value_counts(normalize=True).mul(100).round(2).to_dict()
    )

    return heat_map, overlay_result, df_result, class_percent
