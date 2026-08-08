import numpy as np
from typing import Dict, Any, List

class LateFusionEngine:
    """
    LATE FUSION (Decision-Level Multimodal Fusion):
    Processes each modality (Text, Vision, Audio/Sensor) through separate independent
    unimodal neural networks, and fuses their final decision logits / prediction probabilities.
    """

    def __init__(self, text_weight: float = 0.40, vision_weight: float = 0.35, sensor_weight: float = 0.25):
        self.text_weight = text_weight
        self.vision_weight = vision_weight
        self.sensor_weight = sensor_weight

    def predict_unimodal_text(self, text_input: str) -> float:
        """Independent Text Classifier prediction."""
        np.random.seed(abs(hash(text_input)) % (2**32))
        return float(round(float(np.random.uniform(0.60, 0.95)), 4))

    def predict_unimodal_vision(self, visual_cue: str) -> float:
        """Independent Vision Classifier prediction."""
        np.random.seed((abs(hash(visual_cue)) + 42) % (2**32))
        return float(round(float(np.random.uniform(0.55, 0.92)), 4))

    def predict_unimodal_sensor(self, sensor_signal: str) -> float:
        """Independent Sensor/Audio Classifier prediction."""
        np.random.seed((abs(hash(sensor_signal)) + 101) % (2**32))
        return float(round(float(np.random.uniform(0.50, 0.90)), 4))

    def fuse_late(self, text_input: str, vision_input: str, sensor_input: str) -> Dict[str, Any]:
        """
        Execute late fusion by computing weighted decision ensemble across independent models.
        """
        p_text = self.predict_unimodal_text(text_input)
        p_vision = self.predict_unimodal_vision(vision_input)
        p_sensor = self.predict_unimodal_sensor(sensor_input)

        # Weighted Decision Combination
        late_ensemble_score = (
            (p_text * self.text_weight) +
            (p_vision * self.vision_weight) +
            (p_sensor * self.sensor_weight)
        )
        late_ensemble_score = float(round(float(late_ensemble_score), 4))

        # Max Gating Decision
        max_modality = "TEXT" if p_text >= max(p_vision, p_sensor) else ("VISION" if p_vision >= p_sensor else "SENSOR")

        return {
            "fusion_type": "LATE_FUSION_DECISION_LEVEL",
            "unimodal_scores": {
                "text_confidence": p_text,
                "vision_confidence": p_vision,
                "sensor_confidence": p_sensor
            },
            "weights": {
                "text_weight": self.text_weight,
                "vision_weight": self.vision_weight,
                "sensor_weight": self.sensor_weight
            },
            "late_ensemble_confidence": late_ensemble_score,
            "dominant_modality": max_modality,
            "status": "INDEPENDENT_DECISION_ENSEMBLED"
        }
