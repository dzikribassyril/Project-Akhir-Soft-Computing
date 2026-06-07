import cv2
import joblib
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image, ImageOps
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


CLASS_NAMES = {
    0: "Aman",
    1: "Tersebar",
    2: "Kritis"
}


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

        self.consequent_weights = nn.Parameter(torch.randn(num_rules, num_inputs) * 0.05)
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

        consequent = torch.sum(x_expanded * self.consequent_weights, dim=2) + self.consequent_bias
        weighted_output = normalized_firing * consequent
        logits = self.classifier(weighted_output)

        return logits, normalized_firing


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
        num_classes=checkpoint["num_classes"]
    )

    anfis_model.load_state_dict(checkpoint["model_state_dict"])
    anfis_model.eval()

    pca = joblib.load(pca_path)
    scaler = joblib.load(scaler_path)

    return feature_extractor, preprocess, anfis_model, pca, scaler, checkpoint


def extract_cnn_embedding(grid_img, feature_extractor, preprocess):
    pil_img = Image.fromarray(grid_img)

    augmented_imgs = [
        pil_img,
        ImageOps.mirror(pil_img),
        ImageOps.flip(pil_img)
    ]

    tensors = [preprocess(img) for img in augmented_imgs]
    batch = torch.stack(tensors, dim=0)

    with torch.no_grad():
        feats = feature_extractor(batch)
        feats = feats.float().mean(dim=0, keepdim=True)

    return feats.numpy()


def predict_cnn_anfis(img_rgb, feature_extractor, preprocess, anfis_model, pca, scaler, grid_size=256):
    tinggi, lebar, _ = img_rgb.shape

    overlay = img_rgb.copy()
    heat_map = np.zeros_like(img_rgb, dtype=np.uint8)

    results = []

    for y in range(0, tinggi, grid_size):
        for x in range(0, lebar, grid_size):

            grid_img = img_rgb[y:y + grid_size, x:x + grid_size]

            if grid_img.shape[0] != grid_size or grid_img.shape[1] != grid_size:
                continue

            embedding = extract_cnn_embedding(
                grid_img,
                feature_extractor,
                preprocess
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

            if predicted_class == 0:      # Aman
                color_rgb = np.array([0, 255, 0], dtype=np.uint8)

            elif predicted_class == 1:    # Tersebar
                color_rgb = np.array([255, 255, 0], dtype=np.uint8)

            else:                         # Kritis
                color_rgb = np.array([255, 0, 0], dtype=np.uint8)

            heat_map[y:y + grid_size, x:x + grid_size] = color_rgb

            cv2.rectangle(
                overlay,
                (x, y),
                (x + grid_size, y + grid_size),
                tuple(int(c) for c in color_rgb),
                2
            )

            results.append({
                "x": x,
                "y": y,
                "predicted_class": predicted_class,
                "label": CLASS_NAMES[predicted_class],
                "prob_aman": prob_aman,
                "prob_tersebar": prob_tersebar,
                "prob_kritis": prob_kritis,
                "heat_score": heat_score
            })

    overlay_result = cv2.addWeighted(overlay, 0.35, heat_map, 0.65, 0)

    df_result = pd.DataFrame(results)

    class_percent = (
        df_result["label"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
        .to_dict()
    )

    return heat_map, overlay_result, df_result, class_percent