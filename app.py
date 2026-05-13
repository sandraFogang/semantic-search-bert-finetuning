"""
app.py — Interface Streamlit pour le système RAG hybride (HF Spaces).

Cette version télécharge les artefacts depuis HuggingFace Hub :
- custom_pooler.pt (poids fine-tunés)
- squad_extended.faiss (index dense)
- corpus_extended.pkl (textes des passages)
- bm25_index.pkl (index BM25 pré-calculé)

Lancement local :
    python -m streamlit run app.py

Déploiement : HuggingFace Spaces Docker (16 GB RAM).
"""

import sys
import time
import html
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
import torch
from transformers import BertTokenizer, BertModel
from huggingface_hub import hf_hub_download

from src.semantic_search.encoders import CustomPooledSemanticEncoder
from src.semantic_search.search_engine import SemanticSearchEngine
from src.semantic_search.hybrid_search import HybridRAGSystem
from src.semantic_search.qa import load_qa_pipeline


# Configuration du repo HF Hub où sont stockés les artefacts
HF_REPO_ID = "sandraFogang/semantic-search-bert-encoders"
LOCAL_CACHE = Path("./hf_cache")


st.set_page_config(
    page_title="Semantic Search BERT RAG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# CSS compact (identique à app_local.py)
CUSTOM_CSS = """
<style>
    .block-container {
        padding-top: 3rem !important;
        padding-bottom: 1rem !important;
        max-width: 1200px;
    }

    [data-testid="stSidebar"] {
        padding-top: 1rem;
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 1rem !important;
    }
    [data-testid="stSidebar"] h3 {
        font-size: 0.95rem !important;
        margin-top: 0.3rem !important;
        margin-bottom: 0.3rem !important;
    }
    [data-testid="stSidebar"] hr {
        margin: 0.5rem 0 !important;
    }
    [data-testid="stSidebar"] p {
        margin-bottom: 0.3rem !important;
        font-size: 0.85rem;
    }
    [data-testid="stSidebar"] table {
        font-size: 0.78rem;
    }
    [data-testid="stSidebar"] td, [data-testid="stSidebar"] th {
        padding: 2px 6px !important;
    }

    h1 {
        font-size: 1.6rem !important;
        margin-bottom: 0.3rem !important;
        margin-top: 0.2rem !important;
        padding-top: 0 !important;
    }

    hr { margin: 0.6rem 0 !important; }

    .answer-box {
        background: #f8f7ff;
        border: 1px solid #c7d2fe;
        border-radius: 10px;
        padding: 1rem 1.3rem;
        margin: 0.6rem 0;
    }
    .answer-label {
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        color: #4338ca;
        text-transform: uppercase;
        margin-bottom: 0.3rem;
    }
    .answer-text {
        font-size: 1.7rem;
        font-weight: 700;
        color: #4f46e5;
        margin: 0.2rem 0 0.6rem 0;
        line-height: 1.2;
    }
    .answer-text-low {
        font-size: 1.3rem;
        font-weight: 600;
        color: #6b7280;
        margin: 0.2rem 0 0.6rem 0;
    }

    .highlight {
        background-color: #fef08a;
        color: #713f12;
        padding: 1px 4px;
        border-radius: 3px;
        font-weight: 600;
    }

    .stButton > button {
        background-color: #eef2ff;
        border: 1px solid #c7d2fe;
        color: #3730a3;
        font-size: 0.82rem;
        font-weight: 500;
        padding: 0.3rem 0.7rem;
        border-radius: 16px;
    }
    .stButton > button:hover {
        background-color: #c7d2fe;
        border-color: #6366f1;
        color: #312e81;
    }
    .stButton > button[kind="primary"] {
        background-color: #4f46e5;
        color: white;
        border: none;
        font-weight: 600;
        padding: 0.45rem 1.2rem;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #4338ca;
        color: white;
    }

    .score-badge {
        display: inline-block;
        padding: 2px 8px;
        margin-right: 5px;
        border-radius: 5px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .score-dense { background: #ede9fe; color: #5b21b6; border: 1px solid #c4b5fd; }
    .score-bm25  { background: #dbeafe; color: #1e40af; border: 1px solid #93c5fd; }
    .score-rerank { background: #fef3c7; color: #92400e; border: 1px solid #fcd34d; }

    .info-banner {
        background: #eff6ff;
        border-left: 3px solid #3b82f6;
        padding: 0.6rem 0.9rem;
        margin: 0.4rem 0;
        border-radius: 4px;
        font-size: 0.83rem;
        color: #1e3a8a;
        line-height: 1.5;
    }
    .info-banner b { color: #1e40af; }

    .warning-banner {
        background: #fef3c7;
        border-left: 3px solid #f59e0b;
        padding: 0.6rem 0.9rem;
        margin: 0.4rem 0;
        border-radius: 4px;
        font-size: 0.88rem;
        color: #92400e;
        font-weight: 500;
    }

    .pipeline-step {
        font-size: 0.85rem;
        font-weight: 700;
        color: #4338ca;
        margin-top: 0.6rem;
        margin-bottom: 0.3rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .passage-card {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 6px;
        padding: 0.7rem 1rem;
        margin: 0.4rem 0;
    }
    .passage-rank {
        font-size: 0.7rem;
        font-weight: 700;
        color: #4338ca;
        letter-spacing: 0.1em;
    }
    .passage-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: #1f2937;
        margin: 0.2rem 0 0.4rem 0;
    }
    .passage-text {
        font-size: 0.88rem;
        line-height: 1.4;
        color: #374151;
    }

    .source-box {
        background: #fafaff;
        border: 1px solid #e0e7ff;
        border-radius: 6px;
        padding: 0.7rem 1.1rem;
        margin-top: 0.3rem;
    }
    .source-title {
        font-size: 0.85rem;
        color: #4338ca;
        margin-bottom: 0.4rem;
        font-weight: 600;
    }
    .source-text {
        font-size: 0.9rem;
        line-height: 1.5;
        color: #1f2937;
    }

    .metric-box {
        text-align: right;
        padding-top: 0.8rem;
    }
    .metric-label {
        font-size: 0.72rem;
        color: #4b5563;
        letter-spacing: 0.1em;
        font-weight: 600;
        margin-bottom: 2px;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #4f46e5;
        line-height: 1.1;
    }
    .metric-sub {
        font-size: 0.68rem;
        color: #6b7280;
    }

    .secondary-text {
        color: #4b5563;
        font-size: 0.82rem;
        margin: 0.3rem 0;
    }

    .footer-text {
        font-size: 0.75rem;
        color: #6b7280;
        text-align: center;
        margin-top: 0.5rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def download_artefact(filename):
    """Télécharge un fichier depuis HF Hub vers le cache local."""
    LOCAL_CACHE.mkdir(exist_ok=True)
    return hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=filename,
        cache_dir=str(LOCAL_CACHE),
    )


@st.cache_resource
def load_engine():
    """Charge tous les composants RAG depuis HF Hub et assemble le système."""
    # Tokenizer et BERT
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_base = BertModel.from_pretrained('bert-base-uncased')

    # Télécharger les artefacts depuis HF Hub
    with st.spinner("Téléchargement des poids du modèle (~3 MB)..."):
        weights_path = download_artefact("custom_pooler.pt")

    with st.spinner("Téléchargement de l'index FAISS (~30 MB)..."):
        faiss_path = download_artefact("squad_extended.faiss")
        corpus_path = download_artefact("corpus_extended.pkl")

    with st.spinner("Téléchargement de l'index BM25 (~5 MB)..."):
        bm25_path = download_artefact("bm25_index.pkl")

    # Encodeurs fine-tunés
    checkpoint = torch.load(weights_path, map_location='cpu')
    q_enc = CustomPooledSemanticEncoder(bert_base)
    p_enc = CustomPooledSemanticEncoder(bert_base)
    q_enc.load_trainable_state_dict(checkpoint["query_linear"])
    p_enc.load_trainable_state_dict(checkpoint["passage_linear"])

    # Dense engine (BERT + FAISS)
    dense_engine = SemanticSearchEngine(q_enc, p_enc, tokenizer)
    dense_engine.load(faiss_path, corpus_path)

    # BM25 retriever
    from src.semantic_search.bm25_retriever import BM25Retriever
    try:
        bm25_retriever = BM25Retriever.load(bm25_path)
    except (AttributeError, TypeError):
        bm25_retriever = BM25Retriever()
        if hasattr(bm25_retriever, 'load'):
            bm25_retriever.load(bm25_path)

    # Cross-encoder reranker
    from src.semantic_search.reranker import Reranker
    reranker = Reranker()

    # QA pipeline DistilBERT
    qa_pipeline = load_qa_pipeline(device=-1)

    # Assemblage final
    system = HybridRAGSystem(
        dense_engine=dense_engine,
        bm25_retriever=bm25_retriever,
        reranker=reranker,
        qa_pipeline=qa_pipeline,
    )
    return system


def highlight_answer(passage, answer):
    """Surligne la réponse extraite dans le passage source."""
    if not answer or not passage:
        return html.escape(passage)
    safe_passage = html.escape(passage)
    safe_answer = html.escape(answer.strip())
    if safe_answer.lower() in safe_passage.lower():
        pattern = re.compile(re.escape(safe_answer), re.IGNORECASE)
        return pattern.sub(r'<span class="highlight">\g<0></span>', safe_passage)
    return safe_passage


def render_confidence_bar(confidence):
    """Affiche une barre de confiance colorée selon le score."""
    pct = confidence * 100
    if pct >= 70:
        color = "#10b981"; label = "Réponse fiable"
    elif pct >= 40:
        color = "#f59e0b"; label = "À vérifier"
    else:
        color = "#ef4444"; label = "Faible confiance"
    return (
        f"<div style='margin: 0.3rem 0;'>"
        f"<div style='display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 4px;'>"
        f"<span style='color: #4b5563; font-weight: 500;'>Confiance · {label}</span>"
        f"<span style='color: {color}; font-weight: 700;'>{pct:.0f}%</span>"
        f"</div>"
        f"<div style='background: #e5e7eb; height: 8px; border-radius: 4px; overflow: hidden;'>"
        f"<div style='background: {color}; width: {pct}%; height: 100%; border-radius: 4px;'></div>"
        f"</div>"
        f"</div>"
    )


def get_field(item, *keys, default=None):
    """Cherche une valeur dans un dict en essayant plusieurs noms de clés."""
    if not isinstance(item, dict):
        return default
    for key in keys:
        if key in item and item[key] is not None:
            return item[key]
    return default


# En-tête
col_title, col_metric = st.columns([3, 1])
with col_title:
    st.markdown("# 🔍 Semantic Search BERT RAG")
    st.markdown(
        "<div style='color: #374151; font-size: 0.88rem; margin-bottom: 0.4rem;'>"
        "Recherche sémantique hybride · <b>BERT fine-tuné + BM25 + Cross-encoder reranking + DistilBERT QA</b>"
        "</div>",
        unsafe_allow_html=True,
    )
with col_metric:
    st.markdown(
        "<div class='metric-box'>"
        "<div class='metric-label'>TOP-10 PRECISION</div>"
        "<div class='metric-value'>×6.3</div>"
        "<div class='metric-sub'>vs BERT baseline</div>"
        "</div>",
        unsafe_allow_html=True,
    )


# Barre latérale
with st.sidebar:
    st.markdown("### 🧠 Pipeline RAG")
    st.markdown(
        "<div style='font-size: 0.82rem; color: #374151; line-height: 1.5;'>"
        "Système RAG combinant recherche sémantique et par mots-clés sur Wikipédia."
        "<br><br>"
        "<b>3 étapes :</b><br>"
        "1. <b>Hybrid retrieval</b> — BERT + BM25 + RRF<br>"
        "2. <b>Reranking</b> — Cross-encoder MS MARCO<br>"
        "3. <b>QA extractif</b> — DistilBERT"
        "</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("### ⚙️ Paramètres")
    debug_mode = st.toggle("Mode debug", value=False)
    top_k_display = st.slider("Sources à afficher", 1, 5, 3)

    st.divider()

    st.markdown("### 📊 Performance")
    st.markdown(
        "<div style='font-size: 0.78rem; color: #374151;'>"
        "<table style='width: 100%; border-collapse: collapse;'>"
        "<tr><td style='padding: 2px 4px;'>BERT non entraîné</td><td style='text-align: right; padding: 2px 4px;'>9%</td></tr>"
        "<tr><td style='padding: 2px 4px;'>Fine-tuning par défaut</td><td style='text-align: right; padding: 2px 4px;'>22%</td></tr>"
        "<tr><td style='padding: 2px 4px; font-weight: 700; color: #4338ca;'>Fine-tuning + grid search</td>"
        "<td style='text-align: right; padding: 2px 4px; font-weight: 700; color: #4338ca;'>57%</td></tr>"
        "</table>"
        "<div style='font-size: 0.7rem; color: #6b7280; margin-top: 0.3rem;'>Top-10 sur 100 requêtes SQuAD</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown(
        "<div style='font-size: 0.75rem; color: #4b5563; line-height: 1.6;'>"
        "👤 Sandra Desmair Fogang Lontouo<br>"
        "🎓 M.Sc. HEC Montréal<br>"
        "🔗 <a href='https://github.com/sandraFogang/semantic-search-bert-finetuning' style='color: #4f46e5; font-weight: 600;'>GitHub</a> · "
        "<a href='https://www.linkedin.com/in/sandrafogang' style='color: #4f46e5; font-weight: 600;'>LinkedIn</a>"
        "</div>",
        unsafe_allow_html=True,
    )


# Chargement du système (premier lancement = ~1 min de téléchargements)
with st.spinner("Chargement du modèle et des index (~1 min au premier lancement)..."):
    system = load_engine()


# Exemples de questions
st.markdown(
    "<div style='font-size: 0.85rem; color: #4b5563; margin-top: 0.4rem; margin-bottom: 0.3rem; font-weight: 500;'>"
    "💡 Essayer un exemple :"
    "</div>",
    unsafe_allow_html=True,
)

EXAMPLES = [
    "Which NFL team represented the NFC at Super Bowl 50?",
    "Who wrote the play Hamlet?",
    "What caused the Black Death?",
    "What is the time complexity of binary search?",
    "Who were the Normans?",
]

if "current_query" not in st.session_state:
    st.session_state.current_query = ""

cols = st.columns(len(EXAMPLES))
for i, example in enumerate(EXAMPLES):
    short_label = example.replace("What is the ", "").replace("Which ", "").replace("Who ", "")
    short_label = short_label[:30] + "…" if len(short_label) > 30 else short_label
    with cols[i]:
        if st.button(short_label, key=f"ex_{i}", use_container_width=True):
            st.session_state.current_query = example


# Barre de recherche
col_input, col_button = st.columns([5, 1])
with col_input:
    query = st.text_input(
        "Question",
        value=st.session_state.current_query,
        placeholder="Pose ta question en anglais...",
        label_visibility="collapsed",
    )
with col_button:
    search_clicked = st.button("🔍 Rechercher", type="primary", use_container_width=True)


# Banner d'info détaillé sur le corpus
st.markdown(
    "<div class='info-banner'>"
    "ℹ️ <b>Corpus disponible</b> : 20 000 passages Wikipédia (SQuAD 1.1, snapshot 2016) couvrant "
    "<b>536 articles</b> dans 5 grandes catégories — "
    "🏈 <b>Sports</b> (Super Bowl, NFL, NBA), "
    "🏛️ <b>Histoire</b> (Normands, Peste noire, dynasties), "
    "🔬 <b>Sciences</b> (oxygène, complexité algorithmique, physique), "
    "🎭 <b>Arts & littérature</b> (Shakespeare, musique classique), "
    "🌍 <b>Géographie & cultures</b> (pays, villes, religions). "
    "<br>⚠️ <b>Non couvert</b> : actualités post-2016, COVID-19, IA récente, politique contemporaine."
    "</div>",
    unsafe_allow_html=True,
)


# Recherche et affichage des résultats
if query and (search_clicked or st.session_state.current_query == query):
    start_time = time.time()
    with st.spinner("🔍 Recherche dans 20 000 passages..."):
        raw_result = system.search_and_answer(
            query,
            top_k_dense=20,
            top_k_bm25=20,
            top_k_fused=20,
            top_k_rerank=5,
        )
    elapsed = time.time() - start_time

    st.markdown(
        f"<div class='secondary-text'>⏱️ Recherche terminée en {elapsed:.1f}s · 20 000 passages analysés</div>",
        unsafe_allow_html=True,
    )

    dense_results = raw_result.get("dense", [])
    bm25_results = raw_result.get("bm25", [])
    fused_results = raw_result.get("fused", [])
    reranked_results = raw_result.get("reranked", [])
    qa_data = raw_result.get("qa", {})

    best_answer = qa_data.get("best_answer")
    all_answers = qa_data.get("all_answers", [])

    if best_answer:
        answer_text = best_answer.get("answer", "")
        confidence = best_answer.get("confidence", 0.0)
        source_title = best_answer.get("passage_title", "")
        source_passage = best_answer.get("passage_text", "")
    else:
        answer_text = all_answers[0].get("answer", "") if all_answers else ""
        confidence = all_answers[0].get("confidence", 0.0) if all_answers else 0.0
        source_title = all_answers[0].get("passage_title", "") if all_answers else ""
        source_passage = all_answers[0].get("passage_text", "") if all_answers else ""

    if confidence < 0.3:
        st.markdown(
            "<div class='warning-banner'>"
            "⚠️ <b>Faible confiance.</b> Cette réponse pourrait être incorrecte. "
            "Vérifie les passages sources ci-dessous."
            "</div>",
            unsafe_allow_html=True,
        )

    answer_class = "answer-text" if confidence >= 0.4 else "answer-text-low"
    confidence_bar_html = render_confidence_bar(confidence)
    answer_safe = html.escape(answer_text) if answer_text else "Aucune réponse fiable trouvée"

    st.markdown(
        f"<div class='answer-box'>"
        f"<div class='answer-label'>RÉPONSE EXTRAITE</div>"
        f"<div class='{answer_class}'>{answer_safe}</div>"
        f"{confidence_bar_html}"
        f"</div>",
        unsafe_allow_html=True,
    )

    if source_passage:
        highlighted = highlight_answer(source_passage, answer_text)
        st.markdown(
            f"<div style='margin: 0.6rem 0;'>"
            f"<div class='answer-label'>📄 PASSAGE SOURCE</div>"
            f"<div class='source-box'>"
            f"<div class='source-title'>Wikipédia · {html.escape(source_title)}</div>"
            f"<div class='source-text'>{highlighted}</div>"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with st.expander(f"📚 Voir les {min(top_k_display, len(reranked_results))} passages retenus après reranking", expanded=False):
        for i, src in enumerate(reranked_results[:top_k_display]):
            rank = get_field(src, "rank", default=i + 1)
            title = get_field(src, "title", "passage_title", default="Sans titre")
            passage = get_field(src, "passage", "passage_text", "text", default="")
            rerank_score = get_field(src, "rerank_score", "score")
            dense_score = get_field(src, "dense_score")
            bm25_score = get_field(src, "bm25_score")

            passage_preview = passage if len(passage) <= 300 else passage[:297] + "..."

            scores_html = ""
            if dense_score is not None:
                scores_html += f'<span class="score-badge score-dense">Dense {dense_score:.3f}</span>'
            if bm25_score is not None:
                scores_html += f'<span class="score-badge score-bm25">BM25 {bm25_score:.2f}</span>'
            if rerank_score is not None:
                scores_html += f'<span class="score-badge score-rerank">Rerank {rerank_score:.3f}</span>'

            st.markdown(
                f"<div class='passage-card'>"
                f"<div class='passage-rank'>RANG {rank}</div>"
                f"<div class='passage-title'>{html.escape(str(title))}</div>"
                f"<div class='passage-text'>{html.escape(passage_preview)}</div>"
                f"<div style='margin-top: 0.4rem;'>{scores_html}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    with st.expander("🔬 Voir comment cette réponse a été trouvée (pipeline détaillé)", expanded=debug_mode):

        st.markdown("<div class='pipeline-step'>ÉTAPE 1 — HYBRID RETRIEVAL</div>", unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size: 0.85rem;'>Deux retrievers en parallèle : Dense (BERT fine-tuné) et BM25. Fusion par Reciprocal Rank Fusion.</div>",
            unsafe_allow_html=True,
        )

        col_dense, col_bm25, col_rrf = st.columns(3)

        with col_dense:
            st.markdown("<div style='font-size: 0.85rem; font-weight: 700; color: #5b21b6; margin: 0.3rem 0;'>Dense (BERT) top-3</div>", unsafe_allow_html=True)
            for item in dense_results[:3]:
                title = get_field(item, "title", default="?")
                score = get_field(item, "score", default=0)
                title_short = (title[:22] + "…") if len(title) > 22 else title
                st.markdown(
                    f"<div style='font-size: 0.8rem; padding: 2px 0; color: #1f2937;'>"
                    f"<b>{html.escape(title_short)}</b> "
                    f"<span style='color: #5b21b6; font-weight: 600;'>{score:.3f}</span></div>",
                    unsafe_allow_html=True,
                )

        with col_bm25:
            st.markdown("<div style='font-size: 0.85rem; font-weight: 700; color: #1e40af; margin: 0.3rem 0;'>BM25 top-3</div>", unsafe_allow_html=True)
            for item in bm25_results[:3]:
                title = get_field(item, "title", default="?")
                score = get_field(item, "score", default=0)
                title_short = (title[:22] + "…") if len(title) > 22 else title
                st.markdown(
                    f"<div style='font-size: 0.8rem; padding: 2px 0; color: #1f2937;'>"
                    f"<b>{html.escape(title_short)}</b> "
                    f"<span style='color: #1e40af; font-weight: 600;'>{score:.2f}</span></div>",
                    unsafe_allow_html=True,
                )

        with col_rrf:
            st.markdown("<div style='font-size: 0.85rem; font-weight: 700; color: #065f46; margin: 0.3rem 0;'>RRF Fusion top-3</div>", unsafe_allow_html=True)
            for item in fused_results[:3]:
                title = get_field(item, "title", default="?")
                score = get_field(item, "rrf_score", "score", default=0)
                title_short = (title[:22] + "…") if len(title) > 22 else title
                st.markdown(
                    f"<div style='font-size: 0.8rem; padding: 2px 0; color: #1f2937;'>"
                    f"<b>{html.escape(title_short)}</b> "
                    f"<span style='color: #065f46; font-weight: 600;'>{score:.4f}</span></div>",
                    unsafe_allow_html=True,
                )

        st.markdown("<div class='pipeline-step'>ÉTAPE 2 — CROSS-ENCODER RERANKING</div>", unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size: 0.85rem;'>Les candidats fusionnés sont reclassés par <code>ms-marco-MiniLM-L-6-v2</code>. Plus lent qu'un bi-encoder mais beaucoup plus précis.</div>",
            unsafe_allow_html=True,
        )

        if reranked_results:
            rerank_items = []
            for item in reranked_results[:5]:
                title = get_field(item, "title", "passage_title", default="?")
                score = get_field(item, "rerank_score", "score", default=0)
                title_short = (title[:18] + "…") if len(title) > 18 else title
                rerank_items.append(
                    f"<b style='color: #1f2937;'>{html.escape(title_short)}</b>"
                    f" <span style='color: #92400e; font-weight: 600;'>({score:.3f})</span>"
                )
            st.markdown(
                f"<div style='font-size: 0.82rem; padding: 0.3rem 0; color: #374151;'>"
                f"Top-5 reranked : {' → '.join(rerank_items)}</div>",
                unsafe_allow_html=True,
            )

        st.markdown("<div class='pipeline-step'>ÉTAPE 3 — EXTRACTIVE QA</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='font-size: 0.85rem;'>Les passages sont passés à <code>distilbert-base-cased-distilled-squad</code>. "
            f"Score de confiance final : <b>{confidence*100:.1f}%</b>.</div>",
            unsafe_allow_html=True,
        )

        if debug_mode and all_answers:
            st.markdown(
                "<div style='font-size: 0.85rem; color: #4b5563; margin-top: 0.4rem; margin-bottom: 0.2rem; font-weight: 500;'>"
                "🐞 <b>Mode debug</b> — toutes les réponses candidates QA :"
                "</div>",
                unsafe_allow_html=True,
            )
            for ans in all_answers[:5]:
                a_text = ans.get("answer", "")
                a_conf = ans.get("confidence", 0)
                a_title = ans.get("passage_title", "")
                st.markdown(
                    f"<div style='font-size: 0.82rem; padding: 2px 0; color: #1f2937;'>"
                    f"<b>{html.escape(a_text)}</b> "
                    f"<span style='color: #92400e; font-weight: 600;'>({a_conf*100:.1f}%)</span> "
                    f"<span style='color: #6b7280;'>← {html.escape(a_title)}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )


# Pied de page
st.markdown(
    "<div class='footer-text'>"
    "© 2026 Sandra Desmair Fogang Lontouo · "
    "<a href='https://github.com/sandraFogang/semantic-search-bert-finetuning' style='color: #4f46e5; font-weight: 600;'>GitHub</a> · "
    "<a href='https://www.linkedin.com/in/sandrafogang' style='color: #4f46e5; font-weight: 600;'>LinkedIn</a> · "
    "<a href='https://huggingface.co/datasets/rajpurkar/squad' style='color: #4f46e5; font-weight: 600;'>SQuAD v1.1</a>"
    "</div>",
    unsafe_allow_html=True,
)
