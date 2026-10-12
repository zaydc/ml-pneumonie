# API de prédiction : reçoit une radio et renvoie la probabilité de pneumonie

import io
import os
import json
import torch
from fastapi import FastAPI, UploadFile
from PIL import Image
from torchvision import transforms
from datetime import datetime
from architecture import creer_modele

# Chemin du modèle, modifiable avec la variable d'environnement CHEMIN_MODELE
chemin_modele = os.environ.get("CHEMIN_MODELE", "modele_cnn.pt")

# Fichier où sont notées les prédictions, modifiable avec la variable d'environnement CHEMIN_JOURNAL
chemin_journal = os.environ.get("CHEMIN_JOURNAL", "predictions.jsonl")

# Ajoute une ligne au journal pour chaque prédiction (aucune image, aucune donnée patient)
def journaliser(proba, decision, luminosite):
    ligne = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "probabilite": round(proba, 4),
        "decision": decision,
        "luminosite": round(luminosite, 4),
    }
    with open(chemin_journal, "a", encoding="utf-8") as fichier:
        fichier.write(json.dumps(ligne) + "\n")

# Seuil choisi sur le split val
seuil = 0.5

# Le modèle est chargé une seule fois, au démarrage de l'API
modele = creer_modele()
modele.load_state_dict(torch.load(chemin_modele))
modele.eval()

# Même préparation des images qu'à l'entraînement : niveaux de gris, 28x28, valeurs entre 0 et 1
pretraitement = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

app = FastAPI()


# Permet de vérifier que l'API tourne
@app.get("/sante")
def sante():
    return {"statut": "ok"}


@app.post("/predire")
async def predire(fichier: UploadFile):
    contenu = await fichier.read()
    image = Image.open(io.BytesIO(contenu))
    tenseur = pretraitement(image)

    # Le modèle attend un paquet d'images : (1, 28, 28) -> (1, 1, 28, 28)
    paquet = tenseur.unsqueeze(0)

    with torch.no_grad():
        sortie = modele(paquet)
        proba = torch.sigmoid(sortie).item()

    if proba > seuil:
        decision = "pneumonie"
    else:
        decision = "normale"

    # Calculer moyenne de l'image après traitement de l'indice

    luminosite = tenseur.mean().item()

    journaliser(proba, decision, luminosite)

    return {
        "probabilite_pneumonie": round(proba, 4),
        "decision": decision,
        "seuil": seuil,
    }
