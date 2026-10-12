import io
import os
import tempfile

import torch
from fastapi.testclient import TestClient
from PIL import Image

from architecture import creer_modele

# Faux modèle aux poids aléatoires, pour tester l'API sans le vrai modèle
chemin_faux_modele = os.path.join(tempfile.gettempdir(), "modele_test.pt")
torch.save(creer_modele().state_dict(), chemin_faux_modele)
os.environ["CHEMIN_MODELE"] = chemin_faux_modele

# Journal des tests dans le dossier temporaire, pour ne pas remplir le vrai journal
os.environ["CHEMIN_JOURNAL"] = os.path.join(tempfile.gettempdir(), "predictions_test.jsonl")

# L'API est importée après avoir défini les variables d'environnement, pour qu'elle les prenne en compte
from api import app

client = TestClient(app)


def test_sante():
    reponse = client.get("/sante")

    assert reponse.status_code == 200
    assert reponse.json() == {"statut": "ok"}


def test_predire():
    # Fausse radio : une image grise de 28x28, enregistrée en PNG en mémoire
    image = Image.new("L", (28, 28), color=128)
    fichier = io.BytesIO()
    image.save(fichier, format="PNG")
    fichier.seek(0)

    reponse = client.post("/predire", files={"fichier": ("radio.png", fichier, "image/png")})
    resultat = reponse.json()
    assert reponse.status_code == 200
    assert 0 <= resultat["probabilite_pneumonie"] <= 1
    assert resultat["decision"] in ["pneumonie", "normale"]