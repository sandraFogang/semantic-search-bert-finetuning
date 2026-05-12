---
title: Semantic Search BERT RAG
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Hybrid RAG (Dense + BM25 + Rerank + QA) on Wikipedia
---

# Système RAG hybride BERT — Recherche sur Wikipedia

Recherche sémantique dans **20 000 passages Wikipedia** (corpus SQuAD) avec architecture RAG complète :

1. **Hybrid Retrieval** : Dense (BERT custom fine-tuné) + BM25 → fusion RRF
2. **Cross-encoder Reranking** : ms-marco-MiniLM-L-6-v2
3. **QA Extractif** : DistilBERT pré-entraîné sur SQuAD

## Architecture

```
Question → Dense + BM25 → RRF → Rerank → DistilBERT QA → Réponse
```

## Performance

Retriever dense seul : top-10 = 57% (×6.3 vs baseline BERT pur).
Avec hybrid + reranking, attendu ≈ 75-85% selon le type de question.

## Code source

Repo GitHub : https://github.com/sandraFogang/semantic-search-bert-finetuning

## Auteure

Sandra Desmair Fogang Lontouo — HEC Montréal, NLP H2026.
