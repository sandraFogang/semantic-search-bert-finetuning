"""
search_engine.py — Moteur de recherche sémantique unifié.

Encapsule :
    - encodage de passages via passage_encoder
    - indexation FAISS (IndexFlatIP avec vecteurs normalisés = cosine similarity)
    - recherche top-k pour une requête
    - sauvegarde/chargement de l'index

Pourquoi FAISS plutôt qu'une boucle Python :
    - Recherche en O(n) optimisée C++ : ~1 ms sur 10 000 vecteurs
    - Standard industrie (Meta) pour la recherche par similarité
    - Permet de passer à l'échelle (IndexIVFFlat, IndexHNSW...) si besoin
"""

import pickle
import faiss
import numpy as np
import torch


class SemanticSearchEngine:
    """Moteur de recherche sémantique avec indexation FAISS.

    Usage typique :
        engine = SemanticSearchEngine(q_enc, p_enc, tokenizer)
        engine.build_index([(title1, passage1), (title2, passage2), ...])
        results = engine.search("Who wrote Hamlet?", top_k=5)
    """

    def __init__(self, query_encoder, passage_encoder, tokenizer, device="cpu"):
        self.query_encoder = query_encoder.to(device).eval()
        self.passage_encoder = passage_encoder.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device
        self.index = None
        self.passages = []
        self.titles = []

    def _encode_passages_batched(self, passages, batch_size=16):
        """Encode une liste de passages par batch pour économiser la RAM."""
        all_emb = []
        for i in range(0, len(passages), batch_size):
            batch = passages[i:i + batch_size]
            tok = self.tokenizer(
                batch, return_tensors='pt', padding=True, truncation=True, max_length=256
            )
            tok = {k: v.to(self.device) for k, v in tok.items()}
            with torch.no_grad():
                emb = self.passage_encoder(**tok).cpu().numpy()
            all_emb.append(emb)
        return np.vstack(all_emb).astype('float32')

    def build_index(self, passages_with_titles, batch_size=16, verbose=True):
        """Construit l'index FAISS à partir d'une liste de (title, passage).

        Utilise IndexFlatIP avec vecteurs L2-normalisés, ce qui équivaut
        à une recherche par similarité cosinus.
        """
        self.titles = [t for t, _ in passages_with_titles]
        self.passages = [p for _, p in passages_with_titles]
        if verbose:
            print(f"Encodage de {len(self.passages)} passages...")
        emb = self._encode_passages_batched(self.passages, batch_size)
        # Normaliser pour que inner product = cosine similarity
        faiss.normalize_L2(emb)
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb)
        if verbose:
            print(f"Index FAISS construit : {self.index.ntotal} vecteurs de dim {emb.shape[1]}.")

    def search(self, query, top_k=5):
        """Recherche les top_k passages les plus similaires à la requête.

        Returns:
            list of dict avec keys 'rank', 'score', 'title', 'passage'.
        """
        if self.index is None:
            raise RuntimeError("Index non construit. Appeler build_index() ou load_index().")
        tok = self.tokenizer(
            [query], return_tensors='pt', padding=True, truncation=True, max_length=64
        )
        tok = {k: v.to(self.device) for k, v in tok.items()}
        with torch.no_grad():
            q_emb = self.query_encoder(**tok).cpu().numpy().astype('float32')
        faiss.normalize_L2(q_emb)
        scores, indices = self.index.search(q_emb, top_k)
        return [
            {
                "rank": r + 1,
                "score": float(scores[0][r]),
                "title": self.titles[idx],
                "passage": self.passages[idx],
            }
            for r, idx in enumerate(indices[0])
        ]

    def save(self, index_path, corpus_path):
        """Sauvegarde l'index FAISS et le corpus associé."""
        if self.index is None:
            raise RuntimeError("Pas d'index à sauvegarder.")
        faiss.write_index(self.index, index_path)
        with open(corpus_path, "wb") as f:
            pickle.dump(list(zip(self.titles, self.passages)), f)

    def load(self, index_path, corpus_path):
        """Charge un index FAISS et le corpus depuis disque."""
        self.index = faiss.read_index(index_path)
        with open(corpus_path, "rb") as f:
            corpus = pickle.load(f)
        self.titles = [t for t, _ in corpus]
        self.passages = [p for _, p in corpus]
