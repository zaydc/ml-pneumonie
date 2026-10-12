# Image de base : un Linux minimal avec Python 3.12 déjà installé
FROM python:3.12-slim

# Dossier de travail à l'intérieur du conteneur
WORKDIR /app

# PyTorch en version CPU, beaucoup plus légère que la version GPU
RUN pip install --no-cache-dir torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu

# Les autres paquets de l'API
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Le code de l'API et les poids du modèle
COPY architecture.py api.py modele_cnn.pt ./

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
