"""
bm25_retriever.py — BM25 retrieval (recherche par mots-clés) + RRF fusion.

BM25 (Best Matching 25) est l'algorithme de recherche par mots-clés standard
depuis 1994 (utilisé par Elasticsearch, Lucene, Solr). Il combine :
    - TF (term frequency) : combien de fois le mot apparaît dans le passage
    - IDF (inverse document frequency) : rareté du mot dans le corpus
    - Normalisation par la longueur du passage

Pourquoi utiliser BM25 EN PLUS de notre BERT dense ?
    - Notre BERT est bon pour les paraphrases ("auteur de Hamlet" vs "wrote Hamlet")
    - BM25 est imbattable pour les mots-clés rares ("Super Bowl 50", "Carolina Panthers")
    - Hybrid retrieval (dense + BM25 + RRF) = standard industrie 2026

RRF (Reciprocal Rank Fusion) : méthode de fusion qui combine plusieurs rankings
sans avoir besoin de calibrer les scores. Formule : RRF_score(d) = sum_i 1/(k + rank_i(d))
où k=60 par convention. Robuste et largement utilisée.
"""

import re
import pickle
import numpy as np
from rank_bm25 import BM25Okapi


def _tokenize(text):
    """Tokenisation simple : minuscules + extraction de mots alphanumériques."""
    return re.findall(r"\w+", text.lower())


class BM25Retriever:
    """Retriever BM25 sur un corpus de passages textuels.

    Usage typique :
        bm25 = BM25Retriever()
        bm25.build_index(passages_with_titles)
        results = bm25.search("Carolina Panthers Super Bowl", top_k=20)
    """

    def __init__(self):
        self.bm25 = None
        self.passages = []
        self.titles = []

    def build_index(self, passages_with_titles, verbose=True):
        """Construit l'index BM25 à partir d'une liste de (title, passage)."""
        self.titles = [t for t, _ in passages_with_titles]
        self.passages = [p for _, p in passages_with_titles]

        if verbose:
            print(f"Tokenisation de {len(self.passages)} passages...")
        tokenized_corpus = [_tokenize(p) for p in self.passages]

        if verbose:
            print("Construction de l'index BM25...")
        self.bm25 = BM25Okapi(tokenized_corpus)

        if verbose:
            print(f"Index BM25 construit : {len(self.passages)} documents.")

    def search(self, query, top_k=20):
        """Recherche les top_k passages les plus pertinents.

        Returns:
            list de dicts avec 'rank', 'score', 'title', 'passage'.
        """
        if self.bm25 is None:
            raise RuntimeError("Index BM25 non construit. Appeler build_index().")

        tokens = _tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            {
                "rank": i + 1,
                "score": float(scores[idx]),
                "title": self.titles[idx],
                "passage": self.passages[idx],
            }
            for i, idx in enumerate(top_indices)
        ]

    def save(self, path):
        """Sauvegarde l'index BM25 et le corpus dans un fichier pickle."""
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "bm25": self.bm25,
                    "passages": self.passages,
                    "titles": self.titles,
                },
                f,
            )

    def load(self, path):
        """Charge un index BM25 depuis disque."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.passages = data["passages"]
        self.titles = data["titles"]


def reciprocal_rank_fusion(rankings_lists, k=60, top_n=None):
    """Fusion de plusieurs rankings via Reciprocal Rank Fusion.

    Formule : RRF_score(doc) = sum_i 1 / (k + rank_i(doc))
    où rank_i(doc) est le rang du document dans le i-ème ranking.

    Args:
        rankings_lists : list of lists of dicts. Chaque dict doit avoir une
            clé 'passage' (utilisée comme identifiant unique).
        k : constante de smoothing (60 par convention).
        top_n : limite le nombre de résultats. None = pas de limite.

    Returns:
        list de dicts fusionnés, triés par RRF score, avec 'rrf_score' et 'rank'.
    """
    scores = {}
    seen_passages = {}

    for rankings in rankings_lists:
        for rank, p in enumerate(rankings):
            # Clé unique : le texte du passage
            key = p["passage"]
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            if key not in seen_passages:
                seen_passages[key] = p

    # Trier par score RRF décroissant
    sorted_keys = sorted(scores.keys(), key=lambda x: -scores[x])

    # Construire le résultat final
    result = []
    selected = sorted_keys[:top_n] if top_n else sorted_keys
    for i, key in enumerate(selected):
        p = dict(seen_passages[key])
        p["rrf_score"] = scores[key]
        p["rank"] = i + 1
        result.append(p)

    return result
