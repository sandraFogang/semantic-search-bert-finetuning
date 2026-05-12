"""
training.py — Fine-tuning d'encodeurs sémantiques avec CosineEmbeddingLoss.

Stratégie : un encodeur pour les queries, un autre pour les passages.
Loss : cosine similarity, target=1 (les paires query/passage corrects
doivent avoir une similarité cosinus proche de 1).

Note : SQuAD est utilisé comme données train, ce qui produit naturellement
des paires positives (question, passage qui contient la réponse).
"""

import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from datasets import load_dataset
from tqdm import tqdm


def finetune_encoder(
    query_encoder,
    passage_encoder,
    tokenizer,
    lr=1e-4,
    nsteps=100,
    batch_size=32,
    device="cpu",
    seed=202601,
):
    """Fine-tune deux encodeurs sémantiques sur des paires query/passage SQuAD.

    Args:
        query_encoder : nn.Module produisant un embedding 768-d pour une question.
        passage_encoder : nn.Module produisant un embedding 768-d pour un passage.
        tokenizer : BertTokenizer pour préparer les inputs.
        lr : taux d'apprentissage (1e-4 dans le devoir, mais 1e-5 recommandé
             pour stabiliser le FinetunedSemanticEncoder).
        nsteps : nombre d'itérations de gradient descent.
        batch_size : taille de batch.
        device : "cpu" ou "cuda".
        seed : graine pour la reproductibilité.

    Returns:
        (query_encoder, passage_encoder, loss_data) où loss_data est une liste
        de tuples (step, loss_value).
    """
    rng = random.Random(seed)
    torch.manual_seed(seed)

    query_encoder.to(device).train()
    passage_encoder.to(device).train()

    # Charger SQuAD train
    ds = load_dataset("rajpurkar/squad")
    indices = rng.sample(range(len(ds['train'])), batch_size * nsteps)
    train_subset = Subset(ds['train'], indices)
    train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)

    # Loss et optimiseur (seuls les params avec requires_grad=True sont mis à jour)
    criterion = nn.CosineEmbeddingLoss()
    all_params = list(query_encoder.parameters()) + list(passage_encoder.parameters())
    updatable_params = [p for p in all_params if p.requires_grad]
    optimizer = torch.optim.Adam(updatable_params, lr=lr)

    loss_data = []
    for step, batch in enumerate(
        tqdm(train_dataloader, desc="Fine-tuning encoder", total=nsteps), start=1
    ):
        optimizer.zero_grad()
        tok_q = tokenizer(batch['question'], return_tensors='pt', padding=True, truncation=True)
        tok_p = tokenizer(batch['context'], return_tensors='pt', padding=True, truncation=True)
        tok_q = {k: v.to(device) for k, v in tok_q.items()}
        tok_p = {k: v.to(device) for k, v in tok_p.items()}

        q_emb = query_encoder(**tok_q)
        p_emb = passage_encoder(**tok_p)

        # target=1 : on veut que cos(q_emb, p_emb) soit proche de 1
        target = torch.ones(len(batch['question']), device=device)
        loss = criterion(q_emb, p_emb, target)
        loss.backward()
        optimizer.step()

        loss_data.append((step, loss.item()))

    return query_encoder, passage_encoder, loss_data
