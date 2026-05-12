FROM python:3.11-slim

WORKDIR /app

# Dépendances système nécessaires pour faiss et certains builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Installer les dépendances Python d'abord (cache Docker plus efficace)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code de l'app
COPY . .

# HuggingFace Spaces utilise le port 7860 par défaut
EXPOSE 7860

CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
