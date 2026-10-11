# Test des couches du CNN une par une, pour voir comment la forme
# des données change à chaque étape

from medmnist import PneumoniaMNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn

# Un paquet de 64 images du train
train_t = PneumoniaMNIST(split="train", download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_t, batch_size=64, shuffle=True)
images, labels = next(iter(train_loader))

# Création des couches
conv = nn.Conv2d(1, 16, kernel_size=3, padding=1)
conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
relu = nn.ReLU()
pool = nn.MaxPool2d(2)
flatten = nn.Flatten()

# Bloc 1 : convolution, ReLU puis pooling
bloc1_conv = conv(images)
bloc1_relu = relu(bloc1_conv)
bloc1_pool = pool(bloc1_relu)
print("Après le bloc 1 :", bloc1_pool.shape)

# Bloc 2 : pareil avec 32 filtres, à partir de la sortie du bloc 1
bloc2_conv2 = conv2(bloc1_pool)
bloc2_relu = relu(bloc2_conv2)
bloc2_pool = pool(bloc2_relu)
print("Après le bloc 2 :", bloc2_pool.shape)

# Mise à plat avant la couche linéaire
aplati = flatten(bloc2_pool)
print("Après flatten :", aplati.shape)
