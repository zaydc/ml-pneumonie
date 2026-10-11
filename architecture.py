import torch.nn as nn


# Architecture du CNN, définie une seule fois ici pour être utilisée
# à la fois pour l'entraînement (modele.py) et pour les prédictions (predire.py)
def creer_modele():
    return nn.Sequential(
        # Bloc 1 : 16 filtres 3x3, l'image passe de 28x28 à 14x14
        nn.Conv2d(1, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        # Bloc 2 : 32 filtres 3x3, on passe de 14x14 à 7x7
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        # 32 cartes de 7x7 mises à plat, soit 1568 valeurs par image
        nn.Flatten(),
        # Un seul score en sortie (pneumonie ou non)
        nn.Linear(1568, 1)
    )
