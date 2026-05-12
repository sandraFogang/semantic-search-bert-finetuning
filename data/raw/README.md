# Sources de données

## SQuAD 1.1 (Stanford Question Answering Dataset)

**Source** : https://huggingface.co/datasets/rajpurkar/squad

**Description** : Corpus de questions-réponses construit à partir d'articles
Wikipedia. Chaque ligne contient une question, le passage Wikipedia qui en
contient la réponse, et la réponse exacte.

**Volume** :
- 87 599 paires question/passage en train
- 10 570 paires question/passage en validation
- Environ 10 000 passages uniques en validation (après déduplication)

**Licence** : CC BY-SA 4.0

**Téléchargement automatique** : aucun fichier brut n'est inclus dans ce
repo. Le dataset est téléchargé à la volée par la librairie HuggingFace
`datasets` lors du premier appel à :

```python
from datasets import load_dataset
ds = load_dataset("rajpurkar/squad")
```

Il est mis en cache localement dans `~/.cache/huggingface/datasets/`.

## Pourquoi SQuAD

SQuAD est un benchmark standard pour la recherche d'information sémantique
car :
- Les paires (question, passage_correct) sont naturellement des paires
  positives pour entraîner avec CosineEmbeddingLoss.
- Le corpus de passages est suffisamment varié (toutes les pages Wikipedia
  couvertes) pour tester la généralisation.
- Les questions sont en anglais naturel, pas de mots-clés.
