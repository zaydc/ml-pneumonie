# Exploration d'une image : valeurs des pixels, passage en tenseur,
# puis découpage du train en paquets avec un DataLoader

import numpy as np
from medmnist import PneumoniaMNIST
import matplotlib.pyplot as plt

train = PneumoniaMNIST(split="train", download=True)

# Image numéro 2 du train, convertie en tableau NumPy (28x28, valeurs de 0 à 255)
image, label = train[2]
pixels = np.array(image)

print(pixels.shape, pixels.dtype)
print(pixels.min(), pixels.max())

print("Pixel en haut à gauche :", pixels[0,0])

# Pixel du centre (ligne puis colonne)
centre_y = pixels.shape[0] // 2
centre_x = pixels.shape[1] // 2

print("Pixel au centre:", pixels[centre_y, centre_x])

# Même image après ToTensor : forme (1, 28, 28) et valeurs entre 0 et 1
from torchvision import transforms

train_t = PneumoniaMNIST(split="train", download=True, transform=transforms.ToTensor())
image_t,label_t=train_t[2]

print(image_t.shape,image_t.dtype)
print(image_t.min(), image_t.max())

# Découpage en paquets de 64 images mélangées
from torch.utils.data import DataLoader

train_loader = DataLoader(train_t, batch_size=64, shuffle=True)
images, labels = next(iter(train_loader))

print("Forme image :",images.shape, "Forme label :",labels.shape, "Nombre de paquet dans le train :",len(train_loader))
