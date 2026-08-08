import sys
import time
import os
import json

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

SLIDES = [
    {
        "title": "🌌 OmniFusion-AI: Advanced Multimodal Fusion Intelligence",
        "subtitle": "Executive Architecture Overview",
        "bullets": [
            "• Multimodal Learning transcends unimodal silos by fusing Text, Vision, Audio, and Sensors.",
            "• Early Fusion (Data-level): Raw feature & embedding concatenation.",
            "• Late Fusion (Decision-level): Independent model prediction ensembling.",
            "• Intermediate Cross-Attention: Transformer token interaction (GPT-4o, Gemini, Flamingo)."
        ]
    },
    {
        "title": "📦 1. Early Fusion (Data-Level Concat)",
        "subtitle": "Raw Input Representation Merging",
        "bullets": [
            "• Concatenates text vectors, visual patch embeddings, and sensor signals at layer 0.",
            "• Unified Neural Network processes joint high-dimensional input vector.",
            "• Pros: Full cross-modal correlation from start | Cons: High input dimension & sensitivity to missing data."
        ]
    },
    {
        "title": "⚖️ 2. Late Fusion (Decision-Level Ensemble)",
        "subtitle": "Independent Unimodal Classifiers",
        "bullets": [
            "• Modality-specific models train independently (Text Transformer, Vision ViT, Audio CNN).",
            "• Decision combination via weighted average, softmax gating, or learned MLP ensemble.",
            "• Pros: Highly modular & fault-tolerant | Cons: Misses low-level cross-modal synergies."
        ]
    },
    {
        "title": "⚡ 3. Intermediate / Cross-Attention Fusion",
        "subtitle": "Modern Transformer Token-Level Interaction",
        "bullets": [
            "• Employed by modern foundation models (ViT, GPT-4o, Gemini, Flamingo).",
            "• Text Query tokens (Q) attend directly to Visual/Sensor Key-Value tokens (K, V).",
            "• CrossAttn(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V across transformer layers."
        ]
    },
    {
        "title": "🎯 4. Contrastive Learning (CLIP Framework)",
        "subtitle": "InfoNCE Metric Space Alignment",
        "bullets": [
            "• Pulls matching positive image-caption pairs together (cosine similarity -> 1.0).",
            "• Pushes mismatched negative pairs far apart in joint embedding space.",
            "• Foundation for zero-shot image classification and cross-modal retrieval."
        ]
    },
    {
        "title": "🏥 5. Real-World Multimodal Applications",
        "subtitle": "Grounding Beyond Pure Text",
        "bullets": [
            "• Medical Diagnostics: Patient X-ray radiograph + Clinical history notes + ECG wave data.",
            "• Autonomous Vehicles: Camera RGB footage + LiDAR 3D point clouds + RADAR telemetry.",
            "• E-Commerce Search: Image query + Text modifier ('Find shoes like this image in red')."
        ]
    },
    {
        "title": "🚀 6. Technical System Structure & Files",
        "subtitle": "Production-Ready Python Framework",
        "bullets": [
            "• app.py                      : Interactive Streamlit Web Dashboard",
            "• early_fusion_engine.py      : Data-Level Feature Concatenator",
            "• late_fusion_engine.py       : Decision-Level Ensemble Classifier",
            "• cross_attention_fusion.py   : Multi-Head Intermediate Transformer Cross-Attention",
            "• clip_contrastive_engine.py  : InfoNCE Contrastive Loss Metric Space Aligner",
            "• multimodal_orchestrator.py  : Central Multimodal Intelligence Pipeline",
            "• openrouter_client.py        : OpenRouter Multimodal LLM Client"
        ]
    }
]

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def render_slide(index):
    clear_screen()
    slide = SLIDES[index]
    print("=" * 75)
    print(f" SLIDE [{index + 1}/{len(SLIDES)}] : {slide['title']}")
    print("=" * 75)
    print(f"\n  📌 {slide['subtitle']}\n")
    print("-" * 75)
    for bullet in slide["bullets"]:
        print(f"  {bullet}")
    print("-" * 75)
    print("\n  [N] Next Slide  |  [P] Previous Slide  |  [D] Demo Multimodal Pipeline  |  [Q] Quit Presentation")

def run_presentation():
    current_idx = 0
    while True:
        render_slide(current_idx)
        choice = input("\nSelect Action [N/P/D/Q]: ").strip().lower()
        if choice == 'n' or choice == '':
            if current_idx < len(SLIDES) - 1:
                current_idx += 1
        elif choice == 'p':
            if current_idx > 0:
                current_idx -= 1
        elif choice == 'd':
            clear_screen()
            print("=== Launching Live OmniFusion Multimodal Pipeline Demo ===")
            from multimodal_orchestrator import OmniFusionOrchestrator
            try:
                orch = OmniFusionOrchestrator()
                res = orch.process_multimodal_pipeline(
                    text_query="Patient presenting with acute chest discomfort.",
                    visual_cue="X-ray radiograph showing clear lung fields.",
                    sensor_input="ECG recording: Normal sinus rhythm 72 bpm.",
                    domain_preset="Medical Diagnostics"
                )
                print(f"\nLatency: {res['latency_ms']} ms")
                print(f"Early Fusion Score: {res['early_fusion']['fusion_confidence_score']}")
                print(f"Late Fusion Score: {res['late_fusion']['late_ensemble_confidence']}")
                print(f"Cross-Attention Grounding: {res['cross_attention']['cross_modal_grounding_score']}")
                print(f"CLIP InfoNCE Loss: {res['clip_contrastive']['mean_infonce_loss']}")
                print(f"\nLLM Reasoning Solution:\n{res['llm_reasoning']['solution']}")
            except Exception as e:
                print(f"Error launching demo: {e}")
            input("\nPress Enter to return to presentation...")
        elif choice == 'q':
            print("\nExiting Presentation. Thank you for exploring OmniFusion-AI!")
            break

if __name__ == "__main__":
    run_presentation()
