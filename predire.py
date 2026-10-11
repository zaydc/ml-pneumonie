# Prédiction sur une radio avec le modèle déjà entraîné :
# on recharge les poids sauvegardés au lieu de réentraîner

from medmnist import PneumoniaMNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from architecture import creer_modele
import torch.nn as nn
import torch

# Seuil choisi sur le split val
seuil = 0.5

# Chargement du modèle et de ses poids
modele = creer_modele()
modele.load_state_dict(torch.load("modele_cnn.pt"))
modele.eval()

# Radio à tester : la première image de val
val_t = PneumoniaMNIST(split="val", download=True, transform=transforms.ToTensor())
image_t,label_t=val_t[0]

# Le modèle attend un paquet d'images, donc on ajoute une dimension : (1, 28, 28) -> (1, 1, 28, 28)
paquet = image_t.unsqueeze(0)

# Probabilité de pneumonie
with torch.no_grad():
    sortie = modele(paquet)
    probas = torch.sigmoid(sortie)
    proba = probas.item()

if proba > seuil:
    decision = "pneumonie"
else:
    decision = "normale"

# Le label est un tableau du type [1], on prend donc son premier élément
if label_t[0] == 1:
    vrai_diagnostic = "pneumonie"
else:
    vrai_diagnostic = "normale"

print(f"Probabilité de pneumonie : {proba:.4f}")
print(f"Décision du modèle : {decision}")
print(f"Vrai diagnostic : {vrai_diagnostic}")
