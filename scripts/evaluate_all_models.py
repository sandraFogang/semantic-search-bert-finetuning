"""
evaluate_all_models.py — Évaluation comparée des 4 modèles sur 100 requêtes
SQuAD validation. Produit le tableau de résultats du README.

Modèles évalués :
    1. BertBaseSemanticEncoder (CLS pooler, pas de fine-tuning)
    2. FinetunedSemanticEncoder (pooler fine-tuné)
    3. CustomPooledSemanticEncoder (notre meilleur)
    4. sentence-transformers/all-MiniLM-L6-v2 (SOTA pré-entraîné)

Usage :
    python scripts/evaluate_all_models.py

Durée : ~30-40 minutes (les 3 fine-tunings prennent ~15 min chacun).
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel

from src.semantic_search.encoders import (
    BertBaseSemanticEncoder,
    FinetunedSemanticEncoder,
    CustomPooledSemanticEncoder,
)
from src.semantic_search.data import load_squad_validation, make_collate_fn
from src.semantic_search.training import finetune_encoder
from src.semantic_search.evaluation import encode_passage_ensemble, evaluate_semantic_search
from src.semantic_search.baseline import evaluate_sentence_transformer


def evaluate_pytorch_encoder(encoder, val_data, val_dataloader, tokenizer, device):
    """Helper : encode passages + évalue top-k precision."""
    passage_emb = encode_passage_ensemble(
        val_data.all_titles_passages, encoder, tokenizer, device=device
    )
    return evaluate_semantic_search(passage_emb, val_dataloader, encoder, device=device)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}\n")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    val_data = load_squad_validation(max_size=100)
    val_dataloader = DataLoader(val_data, batch_size=32, collate_fn=make_collate_fn(tokenizer))

    results = {}

    # ---- Modèle 1 : BERT base (sans fine-tuning) ----
    print("=" * 60)
    print("Modèle 1 : BertBaseSemanticEncoder (sans fine-tuning)")
    print("=" * 60)
    bert_base = BertModel.from_pretrained('bert-base-uncased')
    encoder_1 = BertBaseSemanticEncoder(bert_base)
    results["1_bert_base"] = evaluate_pytorch_encoder(
        encoder_1, val_data, val_dataloader, tokenizer, device
    )
    print(results["1_bert_base"])

    # ---- Modèle 2 : Finetuned pooler ----
    print("\n" + "=" * 60)
    print("Modèle 2 : FinetunedSemanticEncoder (~15 min)")
    print("=" * 60)
    bert_base_2 = BertModel.from_pretrained('bert-base-uncased')
    q_enc_2 = FinetunedSemanticEncoder(bert_base_2)
    p_enc_2 = FinetunedSemanticEncoder(bert_base_2)
    q_enc_2, p_enc_2, _ = finetune_encoder(
        q_enc_2, p_enc_2, tokenizer, lr=1e-4, nsteps=100, batch_size=32, device=device
    )
    passage_emb = encode_passage_ensemble(
        val_data.all_titles_passages, p_enc_2, tokenizer, device=device
    )
    results["2_finetuned_pooler"] = evaluate_semantic_search(
        passage_emb, val_dataloader, q_enc_2, device=device
    )
    print(results["2_finetuned_pooler"])

    # ---- Modèle 3 : Custom pooled (notre meilleur) ----
    print("\n" + "=" * 60)
    print("Modèle 3 : CustomPooledSemanticEncoder (~15 min)")
    print("=" * 60)
    bert_base_3 = BertModel.from_pretrained('bert-base-uncased')
    q_enc_3 = CustomPooledSemanticEncoder(bert_base_3)
    p_enc_3 = CustomPooledSemanticEncoder(bert_base_3)
    q_enc_3, p_enc_3, _ = finetune_encoder(
        q_enc_3, p_enc_3, tokenizer, lr=1e-4, nsteps=100, batch_size=32, device=device
    )
    passage_emb = encode_passage_ensemble(
        val_data.all_titles_passages, p_enc_3, tokenizer, device=device
    )
    results["3_custom_pooler"] = evaluate_semantic_search(
        passage_emb, val_dataloader, q_enc_3, device=device
    )
    print(results["3_custom_pooler"])

    # ---- Modèle 4 : sentence-transformers SOTA ----
    print("\n" + "=" * 60)
    print("Modèle 4 : sentence-transformers/all-MiniLM-L6-v2 (SOTA pré-entraîné)")
    print("=" * 60)
    results["4_sentence_transformers_sota"] = evaluate_sentence_transformer(
        val_data, device=device
    )
    print(results["4_sentence_transformers_sota"])

    # ---- Sauvegarde ----
    Path("outputs").mkdir(exist_ok=True)
    with open("outputs/results_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("RÉCAPITULATIF")
    print("=" * 60)
    print(f"{'Modèle':<35} {'top-1':>8} {'top-5':>8} {'top-10':>8}")
    print("-" * 60)
    for name, r in results.items():
        print(
            f"{name:<35} "
            f"{r['mean_top1_precision']:>8.2%} "
            f"{r['mean_top5_precision']:>8.2%} "
            f"{r['mean_top10_precision']:>8.2%}"
        )
    print("\nRésultats sauvegardés dans outputs/results_comparison.json")


if __name__ == "__main__":
    main()
