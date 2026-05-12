"""
reranker.py — Cross-encoder reranking pour affiner le retrieval.

Idée : un bi-encoder (notre BERT custom) est rapide mais imprécis.
Un cross-encoder est lent mais très précis. Stratégie :
    1. Bi-encoder ramène top-20 candidats (rapide sur 20 000 passages)
    2. Cross-encoder reclasse ces 20 candidats (lent mais sur 20 paires seulement)

Modèle utilisé : cross-encoder/ms-marco-MiniLM-L-6-v2
    - ~80 MB
    - Entraîné sur MS MARCO (1M+ paires question/passage)
    - Standard industrie pour le reranking RAG en 2026
"""

from sentence_transformers import CrossEncoder


class Reranker:
    """Wrapper autour d'un cross-encoder pour reranker des passages.

    Le cross-encoder prend (question, passage) ensemble et retourne un score
    de pertinence. Beaucoup plus précis qu'une similarité cosinus entre
    embeddings séparés, mais plus lent.
    """

    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)
        self.model_name = model_name

    def rerank(self, query, passages, top_n=5):
        """Reclasse les passages par pertinence cross-encoder.

        Args:
            query : string.
            passages : liste de dicts avec au moins 'passage'.
            top_n : nombre de résultats à retourner.

        Returns:
            list de dicts triés par 'rerank_score' descendant, avec 'rank' mis à jour.
        """
        if not passages:
            return []

        pairs = [[query, p["passage"]] for p in passages]
        scores = self.model.predict(pairs, show_progress_bar=False)

        # Annoter chaque passage avec son score reranker
        annotated = []
        for p, s in zip(passages, scores):
            new_p = dict(p)
            new_p["rerank_score"] = float(s)
            annotated.append(new_p)

        # Trier par rerank_score décroissant
        annotated.sort(key=lambda x: x["rerank_score"], reverse=True)

        # Garder top_n et mettre à jour le rank
        result = annotated[:top_n]
        for i, p in enumerate(result):
            p["rank"] = i + 1
        return result
