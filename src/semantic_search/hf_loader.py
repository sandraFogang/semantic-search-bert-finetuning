"""
hf_loader.py — Téléchargement des artefacts depuis HuggingFace Hub.

Artefacts hébergés sur HF Hub (modèle séparé du Space) :
    - custom_pooler.pt : poids du Linear personnalisé (~3 MB)
    - squad_extended.faiss : index FAISS sur le corpus étendu (~60 MB)
    - corpus_extended.pkl : passages indexés (~5 MB)
    - bm25_index.pkl : index BM25 sur le même corpus (~10 MB)

Note : le modèle reranker (cross-encoder) et DistilBERT QA sont téléchargés
directement depuis HuggingFace public, pas depuis ton repo.

À adapter : remplacer 'sandraFogang' par ton vrai username HuggingFace.
"""

from pathlib import Path
from huggingface_hub import hf_hub_download

# À MODIFIER : ton repo HF Hub où sont stockés modèle + index
HF_REPO_ID = "sandraFogang/semantic-search-bert-encoders"
LOCAL_CACHE = Path("./hf_cache")


def download_custom_pooler_weights():
    """Télécharge les poids du Linear personnalisé (~3 MB)."""
    LOCAL_CACHE.mkdir(exist_ok=True)
    return hf_hub_download(
        repo_id=HF_REPO_ID,
        filename="custom_pooler.pt",
        cache_dir=str(LOCAL_CACHE),
    )


def download_dense_index():
    """Télécharge l'index FAISS dense (corpus étendu)."""
    LOCAL_CACHE.mkdir(exist_ok=True)
    index_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename="squad_extended.faiss",
        cache_dir=str(LOCAL_CACHE),
    )
    corpus_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename="corpus_extended.pkl",
        cache_dir=str(LOCAL_CACHE),
    )
    return index_path, corpus_path


def download_bm25_index():
    """Télécharge l'index BM25."""
    LOCAL_CACHE.mkdir(exist_ok=True)
    return hf_hub_download(
        repo_id=HF_REPO_ID,
        filename="bm25_index.pkl",
        cache_dir=str(LOCAL_CACHE),
    )
