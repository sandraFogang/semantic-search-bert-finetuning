"""
app_local.py — Système RAG hybride complet (local).

Architecture :
    1. Hybrid Retrieval : Dense (notre BERT) + BM25 -> RRF
    2. Cross-encoder Reranking : ms-marco-MiniLM-L-6-v2
    3. QA Extractif : DistilBERT QA

Lancement :
    python -m streamlit run app_local.py
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

# ============================================================================
# Configuration
# ============================================================================

st.set_page_config(
    page_title="RAG hybride BERT — Wikipedia",
    page_icon="🔍",
    layout="wide",
)


# ============================================================================
# Chargement (cached)
# ============================================================================

@st.cache_resource
def load_dense_engine():
    """Charge le retriever dense (BERT custom + FAISS)."""
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_base = BertModel.from_pretrained('bert-base-uncased')

    checkpoint = torch.load("models/custom_pooler.pt", map_location='cpu', weights_only=False)

    q_enc = CustomPooledSemanticEncoder(bert_base)
    p_enc = CustomPooledSemanticEncoder(bert_base)
    q_enc.load_trainable_state_dict(checkpoint["query_linear"])
    p_enc.load_trainable_state_dict(checkpoint["passage_linear"])

    engine = SemanticSearchEngine(q_enc, p_enc, tokenizer)
    # Charge le corpus étendu si dispo, sinon fallback sur l'ancien
    if Path("index/squad_extended.faiss").exists():
        engine.load("index/squad_extended.faiss", "index/corpus_extended.pkl")
    else:
        engine.load("index/squad_val.faiss", "index/corpus.pkl")
    return engine


@st.cache_resource
def load_bm25():
    """Charge l'index BM25."""
    bm25 = BM25Retriever()
    if Path("index/bm25_index.pkl").exists():
        bm25.load("index/bm25_index.pkl")
        return bm25
    return None  # BM25 optionnel


@st.cache_resource
def load_reranker():
    """Charge le cross-encoder reranker (~80 MB)."""
    return Reranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")


@st.cache_resource
def load_qa():
    """Charge DistilBERT QA (~250 MB)."""
    return load_qa_pipeline(device=-1)


@st.cache_resource
def build_rag_system():
    """Construit le système RAG complet."""
    dense = load_dense_engine()
    bm25 = load_bm25()
    reranker = load_reranker()
    qa = load_qa()
    return HybridRAGSystem(dense, bm25, reranker, qa)


@st.cache_data
def get_sample_questions(_engine_size, _engine, n=5):
    """Pioche des questions de SQuAD val dont le passage est indexé."""
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
            "What is computational complexity theory?",
            "Which NFL team represented the NFC at Super Bowl 50?",
            "When did the Black Death occur?",
            "Who composed the music for The Lord of the Rings?",
        ]


# ============================================================================
# UI
# ============================================================================

st.title("🔍 RAG hybride BERT — Recherche sur Wikipedia")
st.markdown(
    "**Architecture RAG hybride en 3 étapes :** "
    "(1) recherche hybride Dense + BM25 avec fusion RRF, "
    "(2) reranking par cross-encoder, "
    "(3) extraction de la réponse par DistilBERT QA."
)

# Détecter si BM25 est disponible
bm25_available = Path("index/bm25_index.pkl").exists()
extended_available = Path("index/squad_extended.faiss").exists()

corpus_label = (
    "**~20 000 passages** Wikipedia (SQuAD train + val complet)"
    if extended_available
    else "**~2 000 passages** Wikipedia (SQuAD val seulement)"
)

st.info(
    f"ℹ️ **Corpus indexé** : {corpus_label}. "
    f"**Architecture active** : Dense{' + BM25 (hybride)' if bm25_available else ' (BM25 indisponible, mode dégradé)'} → Cross-encoder reranking → DistilBERT QA."
)

with st.sidebar:
    st.header("🏗️ Architecture RAG hybride")
    st.markdown(
        f"""
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

*Dense (notre BERT) :*
- CustomPooledSemanticEncoder
- Fine-tuné (`lr=1e-3, bs=64, 100 steps`)
- Top-10 = 57% sur 100 requêtes

*BM25 :*
- Algorithme classique TF-IDF amélioré
- Standard Elasticsearch / Lucene
- {("Activé ✅" if bm25_available else "Inactif ⚠️")}

*Fusion :*
- Reciprocal Rank Fusion (k=60)
- Combine les rankings sans calibration

**Étape 2 — Reranking**
- `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Entraîné sur MS MARCO (1M+ paires)
- Standard production RAG 2026

**Étape 3 — QA extractif**
- `distilbert-base-cased-distilled-squad`
- Pré-entraîné sur SQuAD (F1 ≈ 87%)
"""
    )

    st.markdown("---")
    st.subheader("⚙️ Paramètres avancés")
    top_k_retrieval = st.slider("Candidats à récupérer (par retriever)", 10, 50, 20)
    top_k_rerank = st.slider("Résultats finaux (après rerank)", 1, 10, 5)
    show_details = st.checkbox("Afficher les détails de chaque étape (debug)", value=False)


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
        # Mode dégradé : sans BM25
        st.warning("⚠️ Index BM25 non disponible — utilisation du dense retrieval seul.")

    with st.spinner("🔎 Pipeline RAG en cours..."):
        result = rag.search_and_answer(
            query,
            top_k_dense=top_k_retrieval,
            top_k_bm25=top_k_retrieval if rag.bm25 else 0,
            top_k_fused=top_k_retrieval,
            top_k_rerank=top_k_rerank,
        )

    # Afficher la réponse finale
    best = result["qa"]["best_answer"]
    if best:
        confidence_pct = best["confidence"] * 100
        st.success(
            f"### 💬 Réponse : **{best['answer']}**\n\n"
            f"*Confiance QA : {confidence_pct:.1f}% — Source : passage sur **{best['passage_title']}***"
        )
    else:
        st.warning(
            "⚠️ Aucune réponse fiable trouvée. Le sujet n'est probablement pas dans le corpus SQuAD, "
            "ou la question est trop ambiguë. Essayez une question d'exemple."
        )

    # Détails optionnels
    if show_details:
        st.markdown("---")
        st.markdown("### 🔬 Détails de chaque étape")

        with st.expander("Étape 1.a — Dense retrieval (notre BERT custom)"):
            for r in result["dense"][:5]:
                st.markdown(
                    f"**#{r['rank']}** *{r['title']}* (sim: {r['score']:.3f}) — "
                    f"{r['passage'][:200]}..."
                )

        if rag.bm25 is not None:
            with st.expander("Étape 1.b — BM25 retrieval (mots-clés)"):
                for r in result["bm25"][:5]:
                    st.markdown(
                        f"**#{r['rank']}** *{r['title']}* (BM25: {r['score']:.2f}) — "
                        f"{r['passage'][:200]}..."
                    )

        with st.expander("Étape 1.c — Fusion RRF"):
            for r in result["fused"][:5]:
                st.markdown(
                    f"**#{r['rank']}** *{r['title']}* (RRF: {r['rrf_score']:.4f}) — "
                    f"{r['passage'][:200]}..."
                )

        with st.expander("Étape 2 — Reranking cross-encoder (top-5 final)"):
            for r in result["reranked"]:
                st.markdown(
                    f"**#{r['rank']}** *{r['title']}* (rerank: {r['rerank_score']:.2f}) — "
                    f"{r['passage'][:200]}..."
                )

    # Toujours afficher les passages finaux (post-rerank) avec QA inline
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
    "🎓 Projet portfolio RAG hybride — **Sandra Desmair Fogang Lontouo** · HEC Montréal, NLP Hiver 2026 · "
    "Architecture : Hybrid (Dense + BM25 + RRF) → Cross-encoder rerank → DistilBERT QA"
)
