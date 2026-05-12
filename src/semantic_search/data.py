"""
data.py — Chargement de SQuAD et Dataset PyTorch pour la recherche sémantique.

SQuAD 1.1 :
    - 87 599 paires train + 10 570 paires validation
    - Basé sur ~536 articles Wikipedia
    - Source : https://huggingface.co/datasets/rajpurkar/squad
"""

import random
import torch
from torch.utils.data import Dataset
from datasets import load_dataset


class DocRetrievalDataset(Dataset):
    """Dataset pour la recherche sémantique de passages."""

    def __init__(self, data, max_size, seed=202601):
        rng = random.Random(seed)
        self.queries = []
        self.correct_passages = []
        self.titles = []
        self.ids = [rng.randint(0, len(data['id']) - 1) for _ in range(max_size)]
        for i in self.ids:
            self.queries.append(data['question'][i])
            self.correct_passages.append(data['context'][i])
            self.titles.append(data['title'][i])
        self.all_titles_passages = list(enumerate(set(zip(self.titles, self.correct_passages))))
        self.passage_to_idx = {passage: i for i, (title, passage) in self.all_titles_passages}
        self.labels = [self.passage_to_idx[p] for p in self.correct_passages]
        self.length = len(self.queries)

    def __getitem__(self, index):
        return (
            self.queries[index],
            self.labels[index],
            self.correct_passages[index],
            self.titles[index],
            index,
        )

    def __len__(self):
        return self.length


def make_collate_fn(tokenizer):
    def collate_fn(items):
        queries, labels, correct_passages, titles, indexes = list(zip(*items))
        batched_queries = tokenizer(
            list(queries), return_tensors='pt', padding=True, truncation=True
        )
        batched_labels = torch.tensor(labels, dtype=torch.long)
        batched_indexes = torch.tensor(indexes, dtype=torch.long)
        return batched_queries, batched_labels, batched_indexes

    return collate_fn


def load_squad_validation(max_size=100, seed=202601):
    """Charge SQuAD validation et construit le DocRetrievalDataset."""
    ds = load_dataset("rajpurkar/squad")
    return DocRetrievalDataset(ds['validation'], max_size=max_size, seed=seed)


def load_squad_full_corpus():
    """Charge l'ensemble unique des passages SQuAD validation (~2 000).

    Returns:
        list of (title, passage)
    """
    ds = load_dataset("rajpurkar/squad")
    val = ds['validation']
    seen = set()
    corpus = []
    for i in range(len(val['context'])):
        p = val['context'][i]
        if p not in seen:
            seen.add(p)
            corpus.append((val['title'][i], p))
    return corpus


def load_squad_extended_corpus(verbose=True):
    """Charge TOUS les passages uniques de SQuAD (train + val) : ~20 000 passages.

    Couvre les 536 articles Wikipedia complets, multipliant le corpus par ~10
    par rapport à load_squad_full_corpus().

    Returns:
        list of (title, passage)
    """
    ds = load_dataset("rajpurkar/squad")
    seen = set()
    corpus = []
    for split_name in ['train', 'validation']:
        data = ds[split_name]
        for i in range(len(data['context'])):
            p = data['context'][i]
            if p not in seen:
                seen.add(p)
                corpus.append((data['title'][i], p))
        if verbose:
            print(f"  Après split '{split_name}' : {len(corpus)} passages uniques cumulés.")
    return corpus
