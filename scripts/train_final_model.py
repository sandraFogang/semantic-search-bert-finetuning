"""
train_final_model.py — Fine-tune CustomPooledSemanticEncoder et sauvegarde
les poids du Linear dans models/custom_pooler.pt (~3 MB).

Usage :
    python scripts/train_final_model.py

Durée : ~15 minutes sur CPU.
"""

import sys
from pathlib import Path

# Permet d'importer src/ depuis scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import BertTokenizer, BertModel

from src.semantic_search.encoders import CustomPooledSemanticEncoder
from src.semantic_search.training import finetune_encoder


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")

    print("Téléchargement de bert-base-uncased (~440 MB la première fois)...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_base = BertModel.from_pretrained('bert-base-uncased')

    q_enc = CustomPooledSemanticEncoder(bert_base)
    p_enc = CustomPooledSemanticEncoder(bert_base)

    print("Fine-tuning (~15 min sur CPU)...")
    q_enc, p_enc, loss_data = finetune_encoder(
        q_enc, p_enc, tokenizer,
        lr=1e-4, nsteps=100, batch_size=32, device=device,
    )

    # Sauvegarde compacte : uniquement les poids du Linear (~3 MB total)
    Path("models").mkdir(exist_ok=True)
    checkpoint = {
        "query_linear": q_enc.get_trainable_state_dict(),
        "passage_linear": p_enc.get_trainable_state_dict(),
        "loss_data": loss_data,
        "hyperparameters": {"lr": 1e-4, "nsteps": 100, "batch_size": 32},
    }
    torch.save(checkpoint, "models/custom_pooler.pt")
    print("Sauvegardé : models/custom_pooler.pt")

    # Optionnel : sauvegarder la courbe de loss pour le README
    try:
        import matplotlib.pyplot as plt
        Path("outputs/figures").mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(8, 5))
        plt.plot(*zip(*loss_data))
        plt.xlabel("Step")
        plt.ylabel("Cosine Embedding Loss")
        plt.title("Convergence du fine-tuning — CustomPooledSemanticEncoder")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("outputs/figures/convergence_loss.png", dpi=120)
        print("Courbe de loss sauvegardée : outputs/figures/convergence_loss.png")
    except ImportError:
        print("(matplotlib non installé — graphique de loss ignoré)")


if __name__ == "__main__":
    main()
