"""
test_search_engine.py — Tests unitaires basiques.

Usage : pytest tests/

Note : ces tests utilisent un mini-corpus (3 passages) et un encoder trivial
pour valider la mécanique sans dépendre du téléchargement de BERT.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.semantic_search.search_engine import SemanticSearchEngine


class DummyEncoder(nn.Module):
    """Encodeur trivial pour tests : retourne un vecteur fixe par mot."""

    def __init__(self, dim=8):
        super().__init__()
        self.dim = dim

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        # Retourne un embedding aléatoire mais déterministe basé sur input_ids
        batch_size = input_ids.shape[0]
        torch.manual_seed(int(input_ids.sum().item()))
        return torch.randn(batch_size, self.dim)


class DummyTokenizer:
    """Tokenizer trivial : convertit du texte en torch.tensor d'IDs aléatoires."""

    def __call__(self, texts, return_tensors='pt', padding=True, truncation=True, max_length=None):
        if isinstance(texts, str):
            texts = [texts]
        ids = [[hash(t) % 1000 for t in text.split()[:8]] for text in texts]
        max_len = max(len(seq) for seq in ids) or 1
        padded = [seq + [0] * (max_len - len(seq)) for seq in ids]
        ids_tensor = torch.tensor(padded, dtype=torch.long)
        return {
            "input_ids": ids_tensor,
            "token_type_ids": torch.zeros_like(ids_tensor),
            "attention_mask": (ids_tensor != 0).long(),
        }


def test_build_index_and_search():
    """L'index se construit et la recherche retourne le bon nombre de résultats."""
    encoder = DummyEncoder()
    tokenizer = DummyTokenizer()
    engine = SemanticSearchEngine(encoder, encoder, tokenizer)

    corpus = [
        ("Title A", "First passage about cats and dogs."),
        ("Title B", "Second passage about Python programming."),
        ("Title C", "Third passage about machine learning."),
    ]
    engine.build_index(corpus, verbose=False)

    results = engine.search("any query", top_k=2)
    assert len(results) == 2
    assert "rank" in results[0]
    assert "score" in results[0]
    assert "title" in results[0]
    assert "passage" in results[0]
    assert results[0]["rank"] == 1
    assert results[1]["rank"] == 2


def test_search_without_index_raises():
    """Appeler search() sans index construit doit lever une erreur."""
    encoder = DummyEncoder()
    tokenizer = DummyTokenizer()
    engine = SemanticSearchEngine(encoder, encoder, tokenizer)
    with pytest.raises(RuntimeError):
        engine.search("query")


def test_save_and_load(tmp_path):
    """L'index peut être sauvegardé puis rechargé sans perte."""
    encoder = DummyEncoder()
    tokenizer = DummyTokenizer()
    engine = SemanticSearchEngine(encoder, encoder, tokenizer)

    corpus = [
        ("Title A", "Passage one."),
        ("Title B", "Passage two."),
    ]
    engine.build_index(corpus, verbose=False)

    index_path = str(tmp_path / "test.faiss")
    corpus_path = str(tmp_path / "corpus.pkl")
    engine.save(index_path, corpus_path)

    engine2 = SemanticSearchEngine(encoder, encoder, tokenizer)
    engine2.load(index_path, corpus_path)

    assert engine2.index.ntotal == 2
    assert engine2.titles == ["Title A", "Title B"]
