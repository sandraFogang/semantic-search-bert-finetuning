"""
semantic_search — Système RAG hybride basé sur BERT.

Architecture :
    Hybrid Retrieval (Dense + BM25 -> RRF) -> Cross-encoder Rerank -> DistilBERT QA

Modules :
    encoders : 3 architectures d'encodeurs sémantiques BERT
    data : Dataset PyTorch et chargement SQuAD (val ou train+val étendu)
    training : Fonction de fine-tuning avec CosineEmbeddingLoss
    evaluation : Métriques top-k precision
    search_engine : Dense retrieval avec FAISS
    bm25_retriever : BM25 retrieval + fusion RRF
    reranker : Cross-encoder reranking
    qa : Question Answering extractif avec DistilBERT
    hybrid_search : Pipeline RAG unifié (orchestration de tous les composants)
    hf_loader : Téléchargement des artefacts depuis HuggingFace Hub
    baseline : Comparaison avec sentence-transformers (SOTA)
"""

__version__ = "2.0.0"
__author__ = "Sandra Desmair Fogang Lontouo"
