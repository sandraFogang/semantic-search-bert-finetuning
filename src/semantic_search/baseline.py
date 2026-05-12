"""
baseline.py — Comparaison avec sentence-transformers (modèle SOTA).

sentence-transformers est la librairie de référence pour les embeddings
sémantiques en production. Le modèle 'all-MiniLM-L6-v2' :
    - 22M paramètres (vs 110M pour bert-base)
    - 384 dimensions (vs 768)
    - Entraîné spécifiquement sur 1 milliard de paires de phrases
    - Performance bien supérieure sur la similarité sémantique

But de cette comparaison : montrer qu'on sait se situer honnêtement par
rapport à l'état de l'art, ce qui crédibilise notre analyse.
"""

import torch
import numpy as np


class SentenceTransformerBaseline:
    """Wrapper minimal autour de sentence-transformers pour évaluation comparée.

    Compatible avec evaluate_semantic_search() en exposant la même interface
    qu'un encodeur PyTorch (forward avec input_ids/token_type_ids/attention_mask
    n'est PAS implémenté ici — utiliser plutôt evaluate_sentence_transformer ci-dessous).
    """

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)
        self.device = device
        self.model_name = model_name

    def encode(self, texts, batch_size=32, normalize=True):
        """Encode une liste de textes en embeddings numpy."""
        emb = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )
        return emb.astype('float32')


def evaluate_sentence_transformer(dataset, model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu"):
    """Évalue un modèle sentence-transformers sur DocRetrievalDataset.

    Args:
        dataset : DocRetrievalDataset.
        model_name : nom HuggingFace du modèle sentence-transformers.
        device : "cpu" ou "cuda".

    Returns:
        dict avec 'mean_top1_precision', 'mean_top5_precision', 'mean_top10_precision'.
    """
    baseline = SentenceTransformerBaseline(model_name=model_name, device=device)

    passages = [p for _, (_, p) in dataset.all_titles_passages]
    queries = dataset.queries
    labels = dataset.labels

    print(f"Encodage de {len(passages)} passages avec {model_name}...")
    passage_emb = baseline.encode(passages)
    print(f"Encodage de {len(queries)} queries...")
    query_emb = baseline.encode(queries)

    # Similarité cosinus = dot product car déjà normalisé
    scores = query_emb @ passage_emb.T  # (n_queries, n_passages)
    ranked = np.argsort(-scores, axis=1)

    labels_arr = np.array(labels)
    p1 = (ranked[:, :1] == labels_arr[:, None]).any(axis=1).mean()
    p5 = (ranked[:, :5] == labels_arr[:, None]).any(axis=1).mean()
    p10 = (ranked[:, :10] == labels_arr[:, None]).any(axis=1).mean()

    return {
        'mean_top1_precision': float(p1),
        'mean_top5_precision': float(p5),
        'mean_top10_precision': float(p10),
    }
