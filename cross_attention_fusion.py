import numpy as np
from typing import Dict, Any, List

class CrossAttentionFusionEngine:
    """
    INTERMEDIATE / CROSS-ATTENTION FUSION (Transformer Token-Level Interaction):
    Used by modern multimodal models (ViT, GPT-4o, Gemini, Flamingo).
    Enables bidirectional or cross-modal attention between text tokens and visual patches
    at intermediate transformer layers: CrossAttn(Q_text, K_vision, V_vision).
    """

    def __init__(self, num_heads: int = 8, embed_dim: int = 64):
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads

    def compute_cross_attention(self, text_tokens: List[str], vision_patches: List[str]) -> Dict[str, Any]:
        """
        Compute Cross-Attention Matrix between Text Queries (Q) and Vision Key/Values (K, V).
        """
        num_q = len(text_tokens)
        num_k = len(vision_patches)

        # Generate synthetic Query and Key projection matrices
        np.random.seed(abs(hash("".join(text_tokens))) % (2**32))
        Q = np.random.normal(0, 1, size=(num_q, self.embed_dim)).astype(np.float32)
        
        np.random.seed((abs(hash("".join(vision_patches))) + 77) % (2**32))
        K = np.random.normal(0, 1, size=(num_k, self.embed_dim)).astype(np.float32)

        # Scaled Dot-Product Attention: Softmax(Q * K^T / sqrt(d_k))
        scale = np.sqrt(self.head_dim)
        raw_attention_logits = np.dot(Q, K.T) / scale

        # Softmax normalization across visual patches
        exp_logits = np.exp(raw_attention_logits - np.max(raw_attention_logits, axis=-1, keepdims=True))
        attention_matrix = exp_logits / (np.sum(exp_logits, axis=-1, keepdims=True) + 1e-12)

        # Attention entropy as cross-modal alignment metric
        entropy = -np.sum(attention_matrix * np.log(attention_matrix + 1e-12), axis=-1)
        mean_entropy = float(round(float(np.mean(entropy)), 4))

        cross_modal_grounding_score = float(round(float(np.mean(np.max(attention_matrix, axis=-1))), 4))

        return {
            "fusion_type": "INTERMEDIATE_CROSS_ATTENTION",
            "num_attention_heads": self.num_heads,
            "embed_dim": self.embed_dim,
            "num_text_queries": num_q,
            "num_visual_patches": num_k,
            "attention_matrix_shape": [num_q, num_k],
            "sample_attention_weights": attention_matrix[:3, :3].round(3).tolist(),
            "mean_cross_attention_entropy": mean_entropy,
            "cross_modal_grounding_score": cross_modal_grounding_score,
            "status": "CROSS_TOKEN_INTERACTION_ACTIVE"
        }
