import unittest
import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from early_fusion_engine import EarlyFusionEngine
from late_fusion_engine import LateFusionEngine
from cross_attention_fusion import CrossAttentionFusionEngine
from clip_contrastive_engine import CLIPContrastiveEngine
from multimodal_orchestrator import OmniFusionOrchestrator

class TestOmniFusion(unittest.TestCase):

    def setUp(self):
        self.early = EarlyFusionEngine(text_dim=64, vision_dim=64, sensor_dim=32)
        self.late = LateFusionEngine()
        self.cross = CrossAttentionFusionEngine()
        self.clip = CLIPContrastiveEngine()

    def test_early_fusion_concatenation(self):
        res = self.early.fuse_early("text query", "visual image", "sensor stream")
        self.assertEqual(res["concatenated_dim"], 160)
        self.assertGreater(res["fusion_confidence_score"], 0.0)
        self.assertEqual(res["status"], "UNIFIED_INPUT_CONCATENATED")

    def test_late_fusion_ensemble(self):
        res = self.late.fuse_late("text query", "visual image", "sensor stream")
        self.assertIn("late_ensemble_confidence", res)
        self.assertIn(res["dominant_modality"], ["TEXT", "VISION", "SENSOR"])

    def test_cross_attention_computation(self):
        res = self.cross.compute_cross_attention(["patient", "chest"], ["patch_1", "patch_2", "patch_3"])
        self.assertIn("mean_cross_attention_entropy", res)
        self.assertGreater(res["cross_modal_grounding_score"], 0.0)

    def test_clip_contrastive_loss(self):
        res = self.clip.evaluate_contrastive_batch(["caption 1", "caption 2"], ["image 1", "image 2"])
        self.assertIn("mean_infonce_loss", res)
        self.assertIn("average_positive_cosine_similarity", res)

if __name__ == "__main__":
    unittest.main()
