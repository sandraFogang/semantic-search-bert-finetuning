"""
evaluation.py — Évaluation de la recherche sémantique : top-k precision.

Pour chaque question du dataloader, on calcule la similarité cosinus entre
son embedding et tous les embeddings de passages. On classe les passages
et on vérifie si le passage correct apparaît dans le top-1, top-5, top-10.
"""

import torch
import torch.nn as nn


def encode_passage_ensemble(all_titles_passages, encoder, tokenizer, device="cpu"):
    """Calcule les embeddings de tous les passages.

    Args:
        all_titles_passages : list of (idx, (title, passage)).
        encoder : nn.Module passé en mode eval.
        tokenizer : BertTokenizer.
        device : "cpu" ou "cuda".

    Returns:
        torch.Tensor de shape (n_passages, 768)
    """
    encoder.to(device).eval()
    passages = [passage for _, (_, passage) in all_titles_passages]
    tokenized = tokenizer(passages, return_tensors='pt', padding=True, truncation=True)
    tokenized = {k: v.to(device) for k, v in tokenized.items()}
    with torch.no_grad():
        embeddings = encoder(**tokenized)
    return embeddings


def evaluate_semantic_search(all_passage_embeddings, dataloader, encoder, device="cpu"):
    """Calcule top-1/top-5/top-10 precision sur un dataloader de validation.

    Args:
        all_passage_embeddings : torch.Tensor (n_passages, 768).
        dataloader : DataLoader produisant (batched_queries, batched_labels, _).
        encoder : nn.Module passé en mode eval.
        device : "cpu" ou "cuda".

    Returns:
        dict avec keys 'mean_top1_precision', 'mean_top5_precision', 'mean_top10_precision'.
    """
    encoder.to(device).eval()
    cos_sim = nn.CosineSimilarity(dim=-1)

    p1, p5, p10, total = 0.0, 0.0, 0.0, 0
    for batched_queries, batched_labels, _ in dataloader:
        with torch.no_grad():
            q_emb = encoder(**{k: v.to(device) for k, v in batched_queries.items()}).squeeze(1)
        scores = cos_sim(q_emb.unsqueeze(1), all_passage_embeddings.to(device))
        ranked = torch.argsort(scores, dim=-1, descending=True)
        labels = batched_labels.to(device).unsqueeze(1)
        p1 += (ranked[:, :1] == labels).any(dim=1).sum().item()
        p5 += (ranked[:, :5] == labels).any(dim=1).sum().item()
        p10 += (ranked[:, :10] == labels).any(dim=1).sum().item()
        total += len(batched_labels)

    return {
        'mean_top1_precision': p1 / total,
        'mean_top5_precision': p5 / total,
        'mean_top10_precision': p10 / total,
    }
