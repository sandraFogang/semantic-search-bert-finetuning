"""
qa.py — Module de Question Answering extractif (sans dépendance pipeline).

Pour transformer le moteur de recherche en système RAG complet :
1. Retrieval : on récupère les top-k passages les plus pertinents (notre BERT custom)
2. QA extractif : DistilBERT pré-entraîné sur SQuAD extrait la phrase qui répond

Modèle utilisé : distilbert-base-cased-distilled-squad
    - ~250 MB
    - Pré-entraîné spécifiquement sur SQuAD
    - F1 ~87% sur SQuAD v1.1

Note technique : on bypass le pipeline `transformers.pipeline("question-answering")`
pour rester compatible avec transformers 5.x où ce pipeline a été restructuré.
On utilise directement AutoModelForQuestionAnswering + AutoTokenizer, ce qui
fonctionne sur toutes les versions de transformers.
"""

import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering


_MODEL_NAME = "distilbert-base-cased-distilled-squad"


def load_qa_pipeline(device=-1):
    """Charge le modèle DistilBERT QA et son tokenizer.

    Args:
        device : -1 pour CPU, 0 pour première GPU.

    Returns:
        dict contenant 'tokenizer', 'model', 'device'.
    """
    tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
    model = AutoModelForQuestionAnswering.from_pretrained(_MODEL_NAME)
    model.eval()

    device_str = "cuda" if (device == 0 and torch.cuda.is_available()) else "cpu"
    model = model.to(device_str)

    return {"tokenizer": tokenizer, "model": model, "device": device_str}


def _qa_predict(question, context, tokenizer, model, device, max_answer_length=30, top_k=20):
    """Prédit la meilleure réponse pour (question, context).

    Stratégie :
        1. Tokeniser (question, context) ensemble avec offset_mapping
        2. Forward pass -> start_logits et end_logits
        3. Masquer les positions hors contexte
        4. Trouver le span (start, end) le plus probable avec contraintes
        5. Récupérer le texte via offset_mapping

    Returns:
        dict {'answer', 'confidence', 'start', 'end'} ou None si pas de span valide.
    """
    inputs = tokenizer(
        question, context,
        return_tensors="pt",
        truncation="only_second",
        max_length=512,
        return_offsets_mapping=True,
        padding=False,
    )
    offset_mapping = inputs.pop("offset_mapping")[0].tolist()

    # sequence_ids : 0 pour les tokens de la question, 1 pour ceux du contexte, None pour les spéciaux
    sequence_ids = None
    if hasattr(inputs, "sequence_ids"):
        try:
            sequence_ids = inputs.sequence_ids(0)
        except Exception:
            sequence_ids = None

    inputs_t = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs_t)

    start_logits = outputs.start_logits[0].cpu()
    end_logits = outputs.end_logits[0].cpu()

    # Masquer les positions hors contexte
    if sequence_ids is not None:
        mask = torch.tensor(
            [0.0 if sid == 1 else -1e9 for sid in sequence_ids],
            dtype=start_logits.dtype,
        )
        start_logits = start_logits + mask
        end_logits = end_logits + mask

    # Softmax pour obtenir des probabilités
    start_probs = torch.softmax(start_logits, dim=-1)
    end_probs = torch.softmax(end_logits, dim=-1)

    # Top-k start et end pour limiter la recherche combinatoire
    n = len(start_probs)
    k = min(top_k, n)
    start_top = torch.topk(start_probs, k)
    end_top = torch.topk(end_probs, k)

    best_score = 0.0
    best_start, best_end = -1, -1
    for s_idx in start_top.indices.tolist():
        for e_idx in end_top.indices.tolist():
            # Contraintes : start <= end, et span pas trop long
            if s_idx > e_idx or (e_idx - s_idx + 1) > max_answer_length:
                continue
            score = (start_probs[s_idx] * end_probs[e_idx]).item()
            if score > best_score:
                best_score = score
                best_start, best_end = s_idx, e_idx

    if best_start == -1:
        return None

    # Récupérer le texte via offset_mapping
    char_start = offset_mapping[best_start][0]
    char_end = offset_mapping[best_end][1]
    answer = context[char_start:char_end].strip()

    if not answer:
        return None

    return {
        "answer": answer,
        "confidence": best_score,
        "start": char_start,
        "end": char_end,
    }


def extract_answer(question, context, qa_pipeline, min_confidence=0.05):
    """Extrait la réponse d'une question depuis un passage.

    Args:
        question : string.
        context : passage où chercher (string).
        qa_pipeline : dict retourné par load_qa_pipeline().
        min_confidence : seuil minimum (en dessous => None).

    Returns:
        dict avec 'answer', 'confidence', 'start', 'end' ou None.
    """
    result = _qa_predict(
        question, context,
        qa_pipeline["tokenizer"],
        qa_pipeline["model"],
        qa_pipeline["device"],
    )
    if result is None or result["confidence"] < min_confidence:
        return None
    return result


def answer_from_passages(question, passages, qa_pipeline, min_confidence=0.05):
    """Cherche la meilleure réponse parmi plusieurs passages candidats.

    Args:
        question : string.
        passages : liste de dicts avec au moins 'passage' (et idéalement 'rank', 'title').
        qa_pipeline : dict retourné par load_qa_pipeline().
        min_confidence : seuil pour considérer une réponse comme fiable.

    Returns:
        dict avec 'best_answer' (la meilleure ou None) et 'all_answers' (toutes triées).
    """
    all_answers = []
    for i, p in enumerate(passages):
        result = extract_answer(question, p["passage"], qa_pipeline, min_confidence=0.0)
        if result:
            result["passage_rank"] = p.get("rank", i + 1)
            result["passage_title"] = p.get("title", "")
            result["passage_text"] = p["passage"]
            all_answers.append(result)

    all_answers.sort(key=lambda x: x["confidence"], reverse=True)
    best = (
        all_answers[0]
        if all_answers and all_answers[0]["confidence"] >= min_confidence
        else None
    )
    return {"best_answer": best, "all_answers": all_answers}
