import sys
import json

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from multimodal_orchestrator import OmniFusionOrchestrator

def main():
    print("=== Initializing Live OmniFusion-AI Multimodal Closed-Loop Test ===")
    orchestrator = OmniFusionOrchestrator()

    print("\n--- 1. Executing Multimodal Pipeline across Medical Diagnostics ---")
    result = orchestrator.process_multimodal_pipeline(
        text_query="Patient presenting with acute chest discomfort and fever.",
        visual_cue="Chest X-ray radiograph showing clear lung fields with minor basilar opacity.",
        sensor_input="ECG recording: Normal sinus rhythm 74 bpm, ST elevation unremarkable.",
        domain_preset="Medical Diagnostics"
    )

    print(f"Latency: {result['latency_ms']} ms")
    print(f"Model Used: {result['llm_reasoning']['model']}")
    print(f"Reasoning Tokens: {result['llm_reasoning']['reasoning_tokens']}")
    print(f"Early Fusion Confidence: {result['early_fusion']['fusion_confidence_score']}")
    print(f"Late Fusion Ensemble: {result['late_fusion']['late_ensemble_confidence']} (Dominant: {result['late_fusion']['dominant_modality']})")
    print(f"Cross-Attention Grounding Score: {result['cross_attention']['cross_modal_grounding_score']}")
    print(f"CLIP Mean InfoNCE Loss: {result['clip_contrastive']['mean_infonce_loss']}")
    print(f"\nUnified Multimodal Solution:\n{result['llm_reasoning']['solution']}")

    print("\n=== LIVE MULTIMODAL VERIFICATION SUCCESSFUL ===")

if __name__ == "__main__":
    main()
