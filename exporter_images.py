# Enregistre quelques radios de val en PNG, pour tester l'API
import os
from medmnist import PneumoniaMNIST

val = PneumoniaMNIST(split="val", download=True)
os.makedirs("images_test", exist_ok=True)

for i in range(5):
    image, label = val[i]
    image.save(f"images_test/val_{i}_label{label[0]}.png")
