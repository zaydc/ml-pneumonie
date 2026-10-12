# Simule du trafic : envoie des radios de MedMNIST à l'API, une par une
# Utilisation : python simuler_trafic.py <split> <nombre> [eclairage]
# Exemple : python simuler_trafic.py val 200 1.6   (200 radios de val, éclaircies de 60 %)

import io
import sys

import requests
from medmnist import PneumoniaMNIST
from PIL import ImageEnhance

split = sys.argv[1]
nombre = int(sys.argv[2])
eclairage = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0

dataset = PneumoniaMNIST(split=split, download=True)

for i in range(nombre):
    image, label = dataset[i]
    image = ImageEnhance.Brightness(image).enhance(eclairage)

    fichier = io.BytesIO()
    image.save(fichier, format="PNG")
    fichier.seek(0)

    requests.post("http://127.0.0.1:8000/predire", files={"fichier": ("radio.png", fichier, "image/png")})

print(f"{nombre} radios du split {split} envoyées (éclairage x{eclairage})")
