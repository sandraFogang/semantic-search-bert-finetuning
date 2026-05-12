"""
app.py — Système RAG hybride pour HuggingFace Spaces.

Architecture identique à app_local.py mais charge les artefacts depuis HF Hub.
"""

import sys
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
import torch
from transformers import BertTokenizer, BertModel

from src.semantic_search.encoders import CustomPooledSemanticEncoder
from src.semantic_search.search_engine import SemanticSearchEngine
from src.semantic_search.bm25_retriever import BM25Retriever
from src.semantic_search.reranker import Reranker
from src.semantic_search.qa import load_qa_pipeline
from src.semantic_search.hybrid_search import HybridRAGSystem
from src.semantic_search.hf_loader import (
    download_custom_pooler_weights,
    download_dense_index,
    download_bm25_index,
)

st.set_page_config(
    page_title="RAG hybride BERT — Wikipedia",
    page_icon="🔍",
    layout="wide",
)


@st.cache_resource
def load_dense_engine():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_base = BertModel.from_pretrained('bert-base-uncased')

    weights_path = download_custom_pooler_weights()
    checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)

    q_enc = CustomPooledSemanticEncoder(bert_base)
    p_enc = CustomPooledSemanticEncoder(bert_base)
    q_enc.load_trainable_state_dict(checkpoint["query_linear"])
    p_enc.load_trainable_state_dict(checkpoint["passage_linear"])

    index_path, corpus_path = download_dense_index()
    engine = SemanticSearchEngine(q_enc, p_enc, tokenizer)
    engine.load(index_path, corpus_path)
    return engine


@st.cache_resource
def load_bm25():
    try:
        bm25_path = download_bm25_index()
        bm25 = BM25Retriever()
        bm25.load(bm25_path)
        return bm25
    except Exception:
        return None


@st.cache_resource
def load_reranker():
    return Reranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")


@st.cache_resource
def load_qa():
    return load_qa_pipeline(device=-1)


@st.cache_resource
def build_rag_system():
    dense = load_dense_engine()
    bm25 = load_bm25()
    reranker = load_reranker()
    qa = load_qa()
    return HybridRAGSystem(dense, bm25, reranker, qa)


@st.cache_data
def get_sample_questions(_engine_size, _engine, n=5):
    try:
        from datasets import load_dataset
        ds = load_dataset("rajpurkar/squad", split="validation")
        corpus_set = set(_engine.passages)
        candidates = []
        for i in range(min(3000, len(ds))):
            if ds[i]['context'] in corpus_set:
                candidates.append(ds[i]['question'])
            if len(candidates) >= 100:
                break
        rng = random.Random(42)
        return rng.sample(candidates, min(n, len(candidates)))
    except Exception:
        return [
            "Who were the Normans?",
            "Which NFL team represented the NFC at Super Bowl 50?",
            "When did the Black Death occur?",
        ]


st.title("🔍 RAG hybride BERT — Recherche sur Wikipedia")
st.markdown(
    "**Architecture RAG hybride en 3 étapes :** "
    "(1) recherche hybride Dense + BM25 avec fusion RRF, "
    "(2) reranking par cross-encoder, "
    "(3) extraction de la réponse par DistilBERT QA."
)

st.info(
    "ℹ️ **Corpus indexé** : ~20 000 passages Wikipedia (SQuAD train + val, 536 articles). "
    "**Pipeline** : Dense (BERT custom) + BM25 → RRF → Cross-encoder rerank → DistilBERT QA."
)

with st.sidebar:
    st.header("🏗️ Architecture RAG hybride")
    st.markdown(
        """
**Pipeline 3 étapes :**

```
Question
   ↓
[1] Hybrid Retrieval
    Dense (BERT)  → top-20
    BM25 (mots-clés) → top-20
    Fusion RRF    → top-20
   ↓
[2] Cross-encoder Reranking
    ms-marco-MiniLM-L-6-v2
   ↓ top-5
[3] DistilBERT QA
   ↓
Réponse extraite
```

---

**Étape 1 — Retrieval**

*Dense (notre BERT custom) :*
- CustomPooledSemanticEncoder
- Fine-tuné (`lr=1e-3, bs=64, 100 steps`)
- Top-10 = 57% sur 100 requêtes

*BM25 :*
- TF-IDF amélioré, standard industrie

*Fusion :*
- Reciprocal Rank Fusion (k=60)

**Étape 2 — Reranking**
- `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Entraîné sur MS MARCO (1M+ paires)

**Étape 3 — QA extractif**
- `distilbert-base-cased-distilled-squad`
- F1 ≈ 87% sur SQuAD dev
"""
    )

    st.markdown("---")
    st.subheader("⚙️ Paramètres avancés")
    top_k_retrieval = st.slider("Candidats par retriever", 10, 50, 20)
    top_k_rerank = st.slider("Résultats finaux", 1, 10, 5)
    show_details = st.checkbox("Détails par étape", value=False)


with st.spinner("⏳ Chargement du système RAG complet (premier lancement ~2 min)..."):
    rag = build_rag_system()

with st.spinner("Préparation des questions d'exemple..."):
    sample_questions = get_sample_questions(len(rag.dense.passages), rag.dense)

st.markdown("##### 💡 Essayer une question d'exemple :")
cols = st.columns(min(3, len(sample_questions)))
example_query = None
for i, q in enumerate(sample_questions[:3]):
    with cols[i]:
        if st.button(q, key=f"ex_{i}"):
            example_query = q

query = st.text_input(
    "Ou poser votre propre question (en anglais) :",
    value=example_query if example_query else "",
    placeholder="Ex: " + (sample_questions[0] if sample_questions else "Who were the Normans?"),
)

if query:
    if rag.bm25 is None:
        st.warning("⚠️ Index BM25 non disponible — utilisation du dense retrieval seul.")

    with st.spinner("🔎 Pipeline RAG en cours..."):
        result = rag.search_and_answer(
            query,
            top_k_dense=top_k_retrieval,
            top_k_bm25=top_k_retrieval if rag.bm25 else 0,
            top_k_fused=top_k_retrieval,
            top_k_rerank=top_k_rerank,
        )

    best = result["qa"]["best_answer"]
    if best:
        confidence_pct = best["confidence"] * 100
        st.success(
            f"### 💬 Réponse : **{best['answer']}**\n\n"
            f"*Confiance QA : {confidence_pct:.1f}% — Source : passage sur **{best['passage_title']}***"
        )
    else:
        st.warning(
            "⚠️ Aucune réponse fiable trouvée. Le sujet n'est probablement pas dans le corpus SQuAD."
        )

    if show_details:
        st.markdown("---")
        st.markdown("### 🔬 Détails par étape")
        with st.expander("Étape 1.a — Dense retrieval"):
            for r in result["dense"][:5]:
                st.markdown(f"**#{r['rank']}** *{r['title']}* (sim: {r['score']:.3f})")
        if rag.bm25 is not None:
            with st.expander("Étape 1.b — BM25"):
                for r in result["bm25"][:5]:
                    st.markdown(f"**#{r['rank']}** *{r['title']}* (BM25: {r['score']:.2f})")
        with st.expander("Étape 1.c — Fusion RRF"):
            for r in result["fused"][:5]:
                st.markdown(f"**#{r['rank']}** *{r['title']}* (RRF: {r['rrf_score']:.4f})")
        with st.expander("Étape 2 — Reranking (top-5)"):
            for r in result["reranked"]:
                st.markdown(f"**#{r['rank']}** *{r['title']}* (rerank: {r['rerank_score']:.2f})")

    st.markdown("---")
    st.markdown("##### 📚 Top passages (après reranking) :")

    answers_by_rank = {a["passage_rank"]: a for a in result["qa"]["all_answers"]}

    for r in result["reranked"]:
        answer_for_this = answers_by_rank.get(r["rank"])
        if answer_for_this:
            qa_info = f" → QA : « {answer_for_this['answer']} » (conf: {answer_for_this['confidence']:.1%})"
        else:
            qa_info = " → aucune réponse extraite"

        with st.expander(
            f"#{r['rank']} — **{r['title']}** (rerank: {r['rerank_score']:.2f}){qa_info}"
        ):
            if answer_for_this:
                start = answer_for_this["start"]
                end = answer_for_this["end"]
                passage = r["passage"]
                before = passage[:start]
                answer = passage[start:end]
                after = passage[end:]
                st.markdown(f"{before}**:green[{answer}]**{after}")
            else:
                st.write(r["passage"])

st.divider()
st.caption(
    "🎓 RAG hybride — **Sandra Desmair Fogang Lontouo** · HEC Montréal, NLP H2026 · "
    "[GitHub](https://github.com/sandraFogang/semantic-search-bert-finetuning) · "
    "Hybrid retrieval (Dense + BM25 + RRF) → Cross-encoder rerank → DistilBERT QA"
)
