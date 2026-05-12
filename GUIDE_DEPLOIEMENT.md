# 📘 Guide de déploiement pas-à-pas

Ce guide te conduit du code que tu viens de télécharger jusqu'à l'app
déployée en ligne. Lecture indispensable avant de commencer.

---

## 🎯 Vue d'ensemble

```
1. Décompresser et placer le code dans un dossier
2. Créer le repo GitHub (vide)
3. Tester le code en local
4. Créer le repo modèle sur HuggingFace Hub
5. Entraîner et uploader le modèle
6. Pousser sur GitHub (branche main)
7. Créer le HuggingFace Space (branche space)
8. Vérifier que l'app est en ligne
```

Temps total estimé : **3 à 4 heures** (la première fois). Dont :
- ~2h de découverte/installation (la première fois seulement)
- ~30 min d'entraînement et indexation (CPU)
- ~30 min de configuration GitHub + HF

---

## Étape 1 — Préparer le dossier local

1. Dézipper l'archive `semantic-search-bert-finetuning.zip` dans :
   ```
   C:\Users\desma\Dropbox\PC\Documents\projets-portfolio\semantic-search-bert-finetuning\
   ```

2. Ouvrir VS Code dans ce dossier :
   ```powershell
   cd C:\Users\desma\Dropbox\PC\Documents\projets-portfolio\semantic-search-bert-finetuning
   code .
   ```

3. **AVANT TOUT** : ouvrir `src/semantic_search/hf_loader.py` et remplacer
   `sandraFogang` par ton vrai username HuggingFace si différent.

---

## Étape 2 — Créer le repo GitHub (vide)

1. Aller sur https://github.com/new
2. Repository name : **`semantic-search-bert-finetuning`**
3. Description : *Semantic search engine on Wikipedia using fine-tuned BERT — top-10 precision ×2.7 vs baseline*
4. Public ✓
5. **NE PAS** cocher "Add a README file" (on a déjà le nôtre)
6. **NE PAS** cocher "Add .gitignore" ni "Choose a license"
7. Cliquer "Create repository"

Tu obtiens une page avec des instructions. Ignore-les, on va le faire après.

---

## Étape 3 — Tester en local

### 3.1 Créer et activer l'environnement virtuel

Dans le terminal VS Code (PowerShell) :

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Tu dois voir `(.venv)` au début de la ligne du terminal.

### 3.2 Installer les dépendances

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

Durée : 5-10 minutes (PyTorch est gros).

### 3.3 Lancer les tests

```powershell
pytest tests/
```

Tu dois voir `3 passed` ou similaire. Si erreur, m'en parler.

### 3.4 Entraîner le modèle

```powershell
python scripts/train_final_model.py
```

**Ce qui va se passer** :
- Premier lancement : téléchargement de `bert-base-uncased` (~440 MB), puis de SQuAD (~30 MB). Mis en cache pour les fois suivantes.
- Fine-tuning : ~15 minutes sur CPU. Tu vois une barre de progression `Fine-tuning encoder: 100%`.
- Sortie : `models/custom_pooler.pt` (~3 MB) et `outputs/figures/convergence_loss.png`.

### 3.5 Construire l'index FAISS

```powershell
python scripts/build_index.py
```

Durée : ~5-10 minutes. Sortie : `index/squad_val.faiss` (~30 MB) et `index/corpus.pkl` (~5 MB).

À la fin, tu verras un test : 3 résultats pour la question "Who was the first president of the United States?".

### 3.6 Lancer l'app Streamlit en local

```powershell
python -m streamlit run app.py
```

**ATTENTION** : à ce stade, l'app va essayer de télécharger les modèles depuis HF Hub (qui n'existe pas encore). Elle va planter. **C'est normal.** On va d'abord uploader sur HF Hub à l'étape 5, puis l'app marchera.

Pour tester localement AVANT l'upload HF, tu peux temporairement modifier `app.py` pour charger depuis disque local au lieu de HF Hub. Mais ce n'est pas nécessaire — passe directement à l'étape 4.

---

## Étape 4 — Créer le repo HuggingFace Hub (pour le modèle)

1. Aller sur https://huggingface.co/login (créer un compte si pas déjà fait — gratuit)
2. Aller sur https://huggingface.co/new
3. Owner : ton username (par exemple `sandraFogang`)
4. Model name : **`semantic-search-bert-encoders`**
5. Visibility : **Public**
6. Cliquer "Create model"

### Se connecter en CLI

Dans le terminal :

```powershell
huggingface-cli login
```

Il te demande un token. Crée-le sur https://huggingface.co/settings/tokens :
- Type : `Write`
- Name : `sandra-local`
- Copier le token, le coller dans le terminal, appuyer sur Entrée.

---

## Étape 5 — Uploader le modèle et l'index sur HF Hub

```powershell
python scripts/upload_to_hf.py
```

Durée : 2-5 minutes (35 MB à uploader). À la fin, vérifier sur :
`https://huggingface.co/sandraFogang/semantic-search-bert-encoders`

Tu dois y voir 3 fichiers : `custom_pooler.pt`, `squad_val.faiss`, `corpus.pkl`.

### Retester l'app en local

Maintenant l'app peut télécharger depuis HF Hub :

```powershell
python -m streamlit run app.py
```

Ça doit s'ouvrir sur http://localhost:8501. Tester avec une des questions d'exemple.

---

## Étape 6 — Pousser sur GitHub (branche main)

### Initialiser Git localement

```powershell
git init
git branch -M main
git add .
git status
```

**VÉRIFIER** la liste des fichiers `git status` : tu ne dois PAS voir :
- `models/custom_pooler.pt`
- `index/squad_val.faiss`
- `index/corpus.pkl`
- `.venv/`

Si tu les vois, c'est que `.gitignore` ne marche pas. Vérifier qu'il est bien à la racine.

### Premier commit

```powershell
git commit -m "Initial commit: semantic search engine with fine-tuned BERT"
```

### Lier au repo GitHub et pousser

```powershell
git remote add origin https://github.com/sandraFogang/semantic-search-bert-finetuning.git
git push -u origin main
```

Aller sur https://github.com/sandraFogang/semantic-search-bert-finetuning. Tu dois voir tout le code, et le README s'affiche bien (avec le diagramme Mermaid).

---

## Étape 7 — Créer le HuggingFace Space (branche space)

### 7.1 Créer le Space sur HF

1. Aller sur https://huggingface.co/new-space
2. Owner : `sandraFogang`
3. Space name : **`semantic-search-bert`**
4. SDK : **Docker** (PAS Streamlit — on a notre Dockerfile)
5. Hardware : `CPU basic` (gratuit, 16 GB RAM)
6. Visibility : Public
7. Cliquer "Create Space"

### 7.2 Créer la branche space localement

```powershell
git checkout -b space
```

Remplacer le README par celui adapté à HF Spaces :

```powershell
move README.md README_github.md
move README_space.md README.md
git add .
git commit -m "Switch to HF Spaces README"
```

### 7.3 Lier le remote HF et pousser

```powershell
git remote add space https://huggingface.co/spaces/sandraFogang/semantic-search-bert
git push space space:main
```

**Note** : `space:main` parce que HF Spaces utilise toujours la branche `main` côté serveur, même si ta branche locale s'appelle `space`.

### 7.4 Revenir sur main

```powershell
git checkout main
```

---

## Étape 8 — Vérifier le déploiement

1. Aller sur `https://huggingface.co/spaces/sandraFogang/semantic-search-bert`
2. Tu vois "Building..." pendant ~5-10 minutes (Docker construit l'image)
3. Puis "Running" → l'app est en ligne ! 🎉

**Premier lancement de l'app** : ~30s de chargement (téléchargement BERT depuis le cache HF, puis téléchargement du modèle et de l'index depuis ton HF Hub).

**URL publique** : `https://huggingface.co/spaces/sandraFogang/semantic-search-bert` (à mettre sur ton CV !)

---

## Étape 9 — Mettre à jour le profil GitHub

Sur ton profil GitHub (le repo `sandraFogang/sandraFogang`), ajouter dans la section "Projets en vedette" :

```markdown
- 🔍 **[Semantic Search BERT](https://github.com/sandraFogang/semantic-search-bert-finetuning)** — Moteur de recherche sémantique multi-encodeurs basé sur BERT (Wikipedia, 10k passages). Top-10 ×2,7 vs baseline. [Demo](https://huggingface.co/spaces/sandraFogang/semantic-search-bert)
```

Et sur LinkedIn (section Projets) :

> **Moteur de recherche sémantique BERT** | HEC Montréal | Hiver 2026
> Fine-tuning de 3 architectures d'encodeurs BERT (CLS pooler, pooler fine-tuné, mean pooling personnalisé) pour la recherche d'information dans une base de 10 000 passages Wikipedia. Amélioration du top-10 precision de 9% à 24% (×2,7) via pooling personnalisé. Pipeline complet : entraînement PyTorch → indexation FAISS → API Streamlit → déploiement Docker sur HuggingFace Spaces. Comparaison honnête avec `sentence-transformers` SOTA.
> **Stack :** Python · PyTorch · Transformers · FAISS · sentence-transformers · Streamlit · Docker · HuggingFace Hub

---

## 🆘 Problèmes fréquents

### `ModuleNotFoundError: No module named 'src'`
→ Tu as oublié d'activer le venv : `.venv\Scripts\activate`

### `OSError: [WinError 1314] Le client ne dispose pas d'un privilège nécessaire`
→ Lancer PowerShell en administrateur, ou désactiver l'option "Developer Mode" dans Windows.

### Le push HF prend très longtemps
→ HF utilise Git-LFS pour les fichiers > 10 MB. Si ton premier push est lent, c'est normal. Évite de mettre les `.pt` et `.faiss` sur la branche space — ils doivent rester sur HF Hub (modèle séparé).

### L'app HF Space affiche "Application error"
→ Cliquer sur "Logs" en haut à droite. Souvent : `hf_loader.py` essaie de télécharger depuis le mauvais nom de repo. Vérifier `HF_REPO_ID` dans `src/semantic_search/hf_loader.py`.

### Streamlit local échoue avec `RepositoryNotFoundError`
→ Tu n'as pas encore uploadé sur HF Hub (étape 5). Ou le repo HF est privé — le mettre en public.

---

## 📋 Checklist finale

Avant de considérer le projet "déployé" :

- [ ] Le repo GitHub est public et le README s'affiche correctement (diagramme Mermaid visible)
- [ ] Le HuggingFace Space est en statut "Running" (vert)
- [ ] L'app répond à au moins 3 requêtes de test
- [ ] Le repo modèle HF Hub contient les 3 fichiers
- [ ] Le profil GitHub mentionne ce projet
- [ ] LinkedIn mentionne ce projet avec l'URL de la démo
- [ ] Le projet est dans les "Pinned repositories" de ton profil GitHub

Bon courage ! Tu peux me dire à quelle étape tu es bloquée si besoin.
