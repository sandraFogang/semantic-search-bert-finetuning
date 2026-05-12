"""
encoders.py — Trois architectures d'encodeurs sémantiques basés sur BERT.

Ces encodeurs produisent tous des embeddings de dimension 768 à partir de
séquences tokenisées par BertTokenizer. Ils diffèrent par la façon dont les
hidden states de BERT sont agrégés en un vecteur unique :

1. BertBaseSemanticEncoder      : utilise le pooler_output natif (token [CLS])
2. FinetunedSemanticEncoder     : fine-tune uniquement la couche BertPooler existante
3. CustomPooledSemanticEncoder  : remplace le pooler par mean(hidden_states) + Linear + Tanh

Seul le 3ème est utilisé en production (meilleur résultat : top-10 precision = 24%).
"""

import torch
import torch.nn as nn


class BertBaseSemanticEncoder(nn.Module):
    """Baseline : utilise directement le pooler_output de BERT pré-entraîné.

    Aucun fine-tuning. Le pooler_output correspond au hidden state du token
    spécial [CLS], passé par une couche linéaire + tanh apprises lors du
    pré-entraînement de BERT (pour la tâche Next Sentence Prediction).

    Performance sur 100 requêtes SQuAD : top-1=1%, top-5=6%, top-10=9%.
    """

    def __init__(self, bert_base):
        super().__init__()
        self.bert_base = bert_base

    def forward(self, input_ids, token_type_ids, attention_mask):
        return self.bert_base(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        ).pooler_output


class FinetunedSemanticEncoder(nn.Module):
    """Fine-tune uniquement la couche BertPooler existante.

    Les 12 couches encoder de BERT sont gelées (requires_grad=False).
    Seuls les ~590k paramètres de la couche pooler (Linear 768x768 + Tanh)
    sont mis à jour pendant l'entraînement.

    Performance : top-1=2%, top-5=8%, top-10=19%.
    """

    def __init__(self, bert_base):
        super().__init__()
        self.bert_base = bert_base
        # Geler tous les paramètres de BERT
        for param in self.bert_base.parameters():
            param.requires_grad = False
        # Dégeler uniquement le pooler
        for param in self.bert_base.pooler.parameters():
            param.requires_grad = True

    def forward(self, input_ids, token_type_ids, attention_mask):
        return self.bert_base(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        ).pooler_output


class CustomPooledSemanticEncoder(nn.Module):
    """Encodeur personnalisé avec mean pooling sur tous les hidden states.

    Architecture :
        BERT (gelé, torch.no_grad) -> last_hidden_state (batch, seq, 768)
        -> mean sur dimension seq -> (batch, 768)
        -> Linear(768, 768) -> Tanh -> (batch, 768)

    Seuls les ~590k paramètres du Linear sont entraînés. BERT est entièrement
    exclu du graphe de calcul des gradients, ce qui accélère l'entraînement.

    Performance : top-1=4%, top-5=15%, top-10=24%. Meilleur des 3 encodeurs.
    """

    def __init__(self, bert_base):
        super().__init__()
        self.bert_base = bert_base
        self.linear = nn.Linear(768, 768)
        self.tanh = nn.Tanh()

    def forward(self, input_ids, token_type_ids, attention_mask):
        with torch.no_grad():
            bert_output = self.bert_base(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
            )
        # Mean pooling sur la dimension séquence
        pooled = bert_output.last_hidden_state.mean(dim=1)
        return self.tanh(self.linear(pooled))

    def get_trainable_state_dict(self):
        """Retourne uniquement les poids du Linear (pour sauvegarde compacte ~3 MB)."""
        return self.linear.state_dict()

    def load_trainable_state_dict(self, state_dict):
        """Charge uniquement les poids du Linear."""
        self.linear.load_state_dict(state_dict)
