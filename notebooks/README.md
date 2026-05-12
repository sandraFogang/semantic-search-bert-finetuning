# Notebooks

## `01_exploration_et_evaluation.ipynb` (à créer)

Version nettoyée du devoir académique original. Contient :

1. **Chargement de SQuAD** via `datasets.load_dataset("rajpurkar/squad")`
2. **Création du `DocRetrievalDataset`** sur 100 requêtes de validation
3. **Modèle 1 — BertBaseSemanticEncoder** : utilise `pooler_output` natif
   - Résultat : top-1=1%, top-5=6%, top-10=9%
4. **Modèle 2 — FinetunedSemanticEncoder** : fine-tune le pooler de BERT
   - Résultat : top-1=2%, top-5=8%, top-10=19%
5. **Modèle 3 — CustomPooledSemanticEncoder** : mean pooling + Linear + Tanh
   - Résultat : top-1=4%, top-5=15%, top-10=24%
6. **Analyse des courbes de loss** et recommandations d'hyperparamètres
   (lr=1e-5 au lieu de 1e-4, linear scaling rule)

## Comment créer ce notebook

À partir du fichier original `Devoir2_Sandra_Desmair_Fogang_Lontouo.ipynb` :

1. Copier le notebook dans ce dossier
2. Le renommer `01_exploration_et_evaluation.ipynb`
3. Remplacer les définitions de classes par des imports depuis `src/` :
   ```python
   from src.semantic_search.encoders import (
       BertBaseSemanticEncoder, FinetunedSemanticEncoder, CustomPooledSemanticEncoder
   )
   from src.semantic_search.data import load_squad_validation, make_collate_fn
   from src.semantic_search.training import finetune_encoder
   from src.semantic_search.evaluation import encode_passage_ensemble, evaluate_semantic_search
   ```
4. Supprimer les cellules `## TO DO ##` et autres marqueurs académiques
5. Garder le markdown explicatif et les graphiques de convergence
