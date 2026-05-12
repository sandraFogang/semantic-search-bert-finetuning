"""
hybrid_search.py — Pipeline RAG complet (Hybrid retrieval + Rerank + QA).

Architecture :
    1. Hybrid Retrieval : Dense (BERT) + BM25 -> RRF fusion -> top-N candidats
    2. Reranking : Cross-encoder reclasse les candidats -> top-K précis
    3. QA : DistilBERT extrait la réponse depuis les top-K passages

C'est l'architecture standard production RAG en 2026.
"""

from .bm25_retriever import reciprocal_rank_fusion
from .qa import answer_from_passages


class HybridRAGSystem:
    """Système RAG complet combinant dense retrieval, BM25, reranking et QA.

    Composants attendus en argument :
        - dense_engine : SemanticSearchEngine (notre BERT custom + FAISS)
        - bm25_retriever : BM25Retriever
        - reranker : Reranker (cross-encoder)
        - qa_pipeline : pipeline QA (dict retourné par load_qa_pipeline)
    """

    def __init__(self, dense_engine, bm25_retriever, reranker, qa_pipeline):
        self.dense = dense_engine
        self.bm25 = bm25_retriever
        self.reranker = reranker
        self.qa = qa_pipeline

    def search_and_answer(
        self,
        query,
        top_k_dense=20,
        top_k_bm25=20,
        top_k_fused=20,
        top_k_rerank=5,
        min_qa_confidence=0.05,
    ):
        """Pipeline complet : retrieval hybride + reranking + QA.

        Returns:
            dict détaillé avec les résultats de chaque étape, utile pour le
            debug et l'affichage transparent dans l'UI :
                - 'dense' : résultats du dense retrieval
                - 'bm25' : résultats BM25
                - 'fused' : résultats fusionnés via RRF
                - 'reranked' : résultats reclassés par cross-encoder
                - 'qa' : résultat final du QA extractif
        """
        # Étape 1.a — Dense retrieval (notre BERT custom)
        dense_results = self.dense.search(query, top_k=top_k_dense)

        # Étape 1.b — BM25 retrieval
        bm25_results = self.bm25.search(query, top_k=top_k_bm25)

        # Étape 1.c — Fusion RRF
        fused = reciprocal_rank_fusion(
            [dense_results, bm25_results],
            top_n=top_k_fused,
        )

        # Étape 2 — Cross-encoder reranking
        reranked = self.reranker.rerank(query, fused, top_n=top_k_rerank)

        # Étape 3 — QA extractif
        qa_result = answer_from_passages(
            query, reranked, self.qa, min_confidence=min_qa_confidence
        )

        return {
            "dense": dense_results,
            "bm25": bm25_results,
            "fused": fused,
            "reranked": reranked,
            "qa": qa_result,
        }
