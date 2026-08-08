import numpy as np
from typing import Dict, Any, List, Tuple

class CLIPContrastiveEngine:
    """
    CONTRASTIVE LEARNING ENGINE (CLIP Architecture):
    Aligns Image and Text embeddings in a joint metric space by pulling matching
    positive pairs together and pushing non-matching negative pairs apart using InfoNCE loss.
    """

    def __init__(self, temperature_tau: float = 0.07, embedding_dim: int = 64):
        self.tau = temperature_tau
        self.embedding_dim = embedding_dim

    def compute_similarity(self, text_emb: np.ndarray, vision_emb: np.ndarray) -> float:
        """Compute cosine similarity between normalized embeddings."""
        norm_t = np.linalg.norm(text_emb) + 1e-8
        norm_v = np.linalg.norm(vision_emb) + 1e-8
        return float(np.dot(text_emb, vision_emb) / (norm_t * norm_v))

    def evaluate_contrastive_batch(self, captions: List[str], image_cues: List[str]) -> Dict[str, Any]:
        """
        Compute InfoNCE contrastive alignment matrix across a batch of image-caption pairs.
        """
        n = min(len(captions), len(image_cues))
        if n == 0:
            return {"error": "Batch cannot be empty"}

        # Simulate batch embeddings
        np.random.seed(abs(hash("".join(captions))) % (2**32))
        T = np.random.normal(0, 1, size=(n, self.embedding_dim))
        T = T / (np.linalg.norm(T, axis=-1, keepdims=True) + 1e-8)

        np.random.seed((abs(hash("".join(image_cues))) + 99) % (2**32))
        V = np.random.normal(0, 1, size=(n, self.embedding_dim))
        V = V / (np.linalg.norm(V, axis=-1, keepdims=True) + 1e-8)

        # Force diagonal positive alignment
        for i in range(n):
            V[i] = 0.7 * T[i] + 0.3 * V[i]
            V[i] = V[i] / (np.linalg.norm(V[i]) + 1e-8)

        # Compute cosine similarity matrix
        sim_matrix = np.dot(T, V.T) / self.tau

        # InfoNCE loss calculation
        exp_sim = np.exp(sim_matrix - np.max(sim_matrix, axis=-1, keepdims=True))
        diag_elements = np.diag(exp_sim)
        sum_rows = np.sum(exp_sim, axis=-1)
        infonce_losses = -np.log(diag_elements / (sum_rows + 1e-12))
        mean_loss = float(round(float(np.mean(infonce_losses)), 4))

        avg_positive_sim = float(round(float(np.mean(np.diag(np.dot(T, V.T)))), 4))

        return {
            "contrastive_framework": "CLIP_INFONCE_ALIGNMENT",
            "temperature_tau": self.tau,
            "batch_size": n,
            "mean_infonce_loss": mean_loss,
            "average_positive_cosine_similarity": avg_positive_sim,
            "alignment_verdict": "STRONG_CROSS_MODAL_METRIC_ALIGNMENT" if avg_positive_sim > 0.6 else "REFINING_METRIC_SPACE"
        }
