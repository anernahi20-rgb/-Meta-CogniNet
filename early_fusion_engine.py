import numpy as np
from typing import Dict, Any, List, Optional

class EarlyFusionEngine:
    """
    EARLY FUSION (Data-Level Multimodal Fusion):
    Concatenates raw features or initial embeddings from all modalities (Text, Vision, Audio/Sensor)
    at the input layer before feeding into a unified neural network.
    """

    def __init__(self, text_dim: int = 128, vision_dim: int = 128, sensor_dim: int = 64):
        self.text_dim = text_dim
        self.vision_dim = vision_dim
        self.sensor_dim = sensor_dim
        self.total_dim = text_dim + vision_dim + sensor_dim

    def encode_text_feature(self, text_str: str) -> np.ndarray:
        """Simulate dense semantic embedding vector for text."""
        np.random.seed(abs(hash(text_str)) % (2**32))
        return np.random.normal(loc=0.0, scale=1.0, size=(self.text_dim,)).astype(np.float32)

    def encode_vision_feature(self, visual_cue: str) -> np.ndarray:
        """Simulate visual patch embedding vector."""
        np.random.seed((abs(hash(visual_cue)) + 42) % (2**32))
        return np.random.normal(loc=0.5, scale=0.8, size=(self.vision_dim,)).astype(np.float32)

    def encode_sensor_feature(self, sensor_signal: str) -> np.ndarray:
        """Simulate sensor / audio spectrogram embedding vector."""
        np.random.seed((abs(hash(sensor_signal)) + 101) % (2**32))
        return np.random.normal(loc=-0.2, scale=0.5, size=(self.sensor_dim,)).astype(np.float32)

    def fuse_early(self, text_input: str, vision_input: str, sensor_input: str) -> Dict[str, Any]:
        """
        Execute early fusion by concatenating all raw feature vectors at input level.
        """
        t_vec = self.encode_text_feature(text_input)
        v_vec = self.encode_vision_feature(vision_input)
        s_vec = self.encode_sensor_feature(sensor_input)

        # Vector concatenation at input layer
        concatenated_vector = np.concatenate([t_vec, v_vec, s_vec], axis=0)

        # Normalize joint embedding
        norm = np.linalg.norm(concatenated_vector) + 1e-8
        joint_embedding = concatenated_vector / norm

        # Unified linear projection head simulation
        weights = np.sin(np.linspace(0, np.pi, self.total_dim, dtype=np.float32))
        raw_score = float(np.dot(joint_embedding, weights))
        probability = float(1.0 / (1.0 + np.exp(-raw_score)))

        return {
            "fusion_type": "EARLY_FUSION_DATA_LEVEL",
            "text_dim": self.text_dim,
            "vision_dim": self.vision_dim,
            "sensor_dim": self.sensor_dim,
            "concatenated_dim": int(self.total_dim),
            "joint_vector_norm": float(round(float(norm), 4)),
            "fusion_confidence_score": float(round(probability, 4)),
            "status": "UNIFIED_INPUT_CONCATENATED"
        }
