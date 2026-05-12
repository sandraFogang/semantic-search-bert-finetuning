"""
upload_to_hf.py — Upload des artefacts du système RAG sur HuggingFace Hub.

Fichiers uploadés :
    - models/custom_pooler.pt (~3 MB)
    - index/squad_extended.faiss (~60 MB)
    - index/corpus_extended.pkl (~5 MB)
    - index/bm25_index.pkl (~10 MB)

Usage :
    1. huggingface-cli login  (une seule fois)
    2. Créer le repo sur https://huggingface.co/new (type: Model)
    3. python scripts/upload_to_hf.py

Adapter HF_REPO_ID dans src/semantic_search/hf_loader.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from huggingface_hub import HfApi, create_repo
from src.semantic_search.hf_loader import HF_REPO_ID


def main():
    api = HfApi()

    try:
        create_repo(HF_REPO_ID, repo_type="model", exist_ok=True)
        print(f"Repo HF prêt : {HF_REPO_ID}")
    except Exception as e:
        print(f"Erreur création repo : {e}")
        sys.exit(1)

    files_to_upload = [
        ("models/custom_pooler.pt", "custom_pooler.pt"),
        ("index/squad_extended.faiss", "squad_extended.faiss"),
        ("index/corpus_extended.pkl", "corpus_extended.pkl"),
        ("index/bm25_index.pkl", "bm25_index.pkl"),
    ]

    for local_path, hub_path in files_to_upload:
        if not Path(local_path).exists():
            print(f"⚠️  SKIP : {local_path} introuvable.")
            continue
        size_mb = Path(local_path).stat().st_size / (1024 * 1024)
        print(f"Upload {local_path} ({size_mb:.1f} MB) -> {HF_REPO_ID}/{hub_path}...")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=hub_path,
            repo_id=HF_REPO_ID,
            repo_type="model",
        )
        print("  ✅ OK.")

    print(f"\n🎉 Tout est uploadé sur https://huggingface.co/{HF_REPO_ID}")


if __name__ == "__main__":
    main()
