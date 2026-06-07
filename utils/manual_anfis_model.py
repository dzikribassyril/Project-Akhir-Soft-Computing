import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans


class ANFIS(nn.Module):
    """
    Adaptive Neuro-Fuzzy Inference System (ANFIS) - Arsitektur Takagi-Sugeno-Kang (TSK) Order 1.
    """
    def __init__(self, num_inputs, num_rules, num_classes):
        super(ANFIS, self).__init__()
        self.num_inputs = num_inputs
        self.num_rules = num_rules
        self.num_classes = num_classes

        # Layer 1: Gaussian MF
        self.mu = nn.Parameter(torch.randn(num_rules, num_inputs))
        self.sigma_raw = nn.Parameter(torch.ones(num_rules, num_inputs) * 0.5)

        # Layer 4: Konsekuen TSK
        self.consequent_weights = nn.Parameter(torch.randn(num_rules, num_inputs) * 0.1)
        self.consequent_bias = nn.Parameter(torch.zeros(num_rules))

        # Layer Klasifikasi
        self.classifier = nn.Linear(num_rules, num_classes)

    @property
    def sigma(self):
        """Sigma selalu positif via softplus transformation."""
        return F.softplus(self.sigma_raw) + 1e-6

    def init_from_kmeans(self, X_train):
        """
        Inisialisasi centroid (mu) dan spread (sigma) menggunakan KMeans clustering.
        Ini memberikan titik awal yang jauh lebih baik daripada random initialization.
        """
        kmeans = KMeans(n_clusters=self.num_rules, random_state=42, n_init=10)
        kmeans.fit(X_train)

        # Set mu & sigma
        self.mu.data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)

        labels = kmeans.labels_
        sigma_init = torch.ones(self.num_rules, self.num_inputs) * 0.3
        for i in range(self.num_rules):
            cluster_data = X_train[labels == i]
            if len(cluster_data) > 1:
                std_vals = np.std(cluster_data, axis=0)
                std_vals = np.maximum(std_vals, 0.05)
                sigma_init[i] = torch.tensor(std_vals, dtype=torch.float32)

        self.sigma_raw.data = torch.log(torch.exp(sigma_init) - 1 + 1e-6)

        print(f"MF diinisialisasi dari KMeans ({self.num_rules} clusters)")

    def forward(self, x):
        batch_size = x.shape[0]

        # Layer 1: Fuzzifikasi
        x_expanded = x.unsqueeze(1).expand(-1, self.num_rules, -1)
        sigma = self.sigma
        mf_out = torch.exp(-0.5 * ((x_expanded - self.mu) / sigma) ** 2)

        # Layer 2: Rule Firing
        firing_strength = torch.prod(mf_out, dim=2)

        # Layer 3: Normalisasi
        normalizer = torch.sum(firing_strength, dim=1, keepdim=True) + 1e-8
        normalized_firing = firing_strength / normalizer

        # Layer 4: Konsekuen TSK
        consequent = torch.sum(x_expanded * self.consequent_weights, dim=2) + self.consequent_bias

        # Layer 5: Defuzzifikasi & Klasifikasi
        weighted_output = normalized_firing * consequent 

        class_logits = self.classifier(weighted_output)

        return class_logits, normalized_firing
