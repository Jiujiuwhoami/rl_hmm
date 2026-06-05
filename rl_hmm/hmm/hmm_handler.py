import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class HMMHandler:
    """状态处理器 - 使用K-means作为简化的状态建模"""

    def __init__(self, n_components=3):
        self.n_components = n_components
        self.model = None
        self.scaler = None

    def fit(self, features: np.ndarray):
        """使用K-means进行状态聚类"""
        if len(features) < self.n_components:
            self.model = None
            return None
        
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features)
        
        self.model = KMeans(
            n_clusters=self.n_components,
            random_state=42,
            n_init=10
        )
        self.model.fit(features_scaled)
        return self.model

    def predict(self, feature_vector: np.ndarray):
        if self.model is None or self.scaler is None:
            return 0
        features_scaled = self.scaler.transform(feature_vector.reshape(1, -1))
        return int(self.model.predict(features_scaled)[0])

    def predict_proba(self, feature_vector: np.ndarray):
        if self.model is None or self.scaler is None:
            return np.ones(self.n_components) / self.n_components
        
        features_scaled = self.scaler.transform(feature_vector.reshape(1, -1))
        distances = self.model.transform(features_scaled)[0]
        # 转换为概率（距离越近概率越高）
        eps = 1e-8
        inv_distances = 1.0 / (distances + eps)
        return inv_distances / inv_distances.sum()