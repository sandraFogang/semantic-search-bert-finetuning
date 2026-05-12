"""
build_index.py — Construit l'index FAISS sur le corpus SQuAD validation
(~10 000 passages Wikipedia uniques) en utilisant le modèle entraîné.

Usage :
    python scripts/build_index.py

Durée : ~5-10 minutes sur CPU.
Pré-requis : models/custom_pooler.pt doit exister (lancer train_final_model.py d'abord).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import BertTokenizer, BertModel

from src.semantic_search.encoders import CustomPooledSemanticEncoder
from src.semantic_search.search_engine import SemanticSearchEngine
from src.semantic_search.data import load_squad_full_corpus


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")

    checkpoint_path = Path("models/custom_pooler.pt")
    if not checkpoint_path.exists():
        print("ERREUR : models/custom_pooler.pt introuvable.")
        print("Lancer d'abord : python scripts/train_final_model.py")
        sys.exit(1)

    print("Chargement de bert-base-uncased...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_base = BertModel.from_pretrained('bert-base-uncased')

    print("Reconstruction des encodeurs et chargement des poids fine-tunés...")
    q_enc = CustomPooledSemanticEncoder(bert_base)
    p_enc = CustomPooledSemanticEncoder(bert_base)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    q_enc.load_trainable_state_dict(checkpoint["query_linear"])
    p_enc.load_trainable_state_dict(checkpoint["passage_linear"])

    print("Chargement du corpus SQuAD validation...")
    corpus = load_squad_full_corpus()
    print(f"Corpus unique : {len(corpus)} passages.")

    engine = SemanticSearchEngine(q_enc, p_enc, tokenizer, device=device)
    engine.build_index(corpus)

    Path("index").mkdir(exist_ok=True)
    engine.save("index/squad_val.faiss", "index/corpus.pkl")
    print("Sauvegardés : index/squad_val.faiss + index/corpus.pkl")

    # Test rapide
    print("\nTest : 'Who was the first president of the United States?'")
    results = engine.search("Who was the first president of the United States?", top_k=3)
    for r in results:
        print(f"  #{r['rank']} ({r['score']:.3f}) [{r['title']}] {r['passage'][:120]}...")


if __name__ == "__main__":
    main()
