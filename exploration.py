from medmnist import PneumoniaMNIST


# Compte les radios normales (label 0) et les pneumonies (label 1) d'un split,
# pour voir si les classes sont équilibrées
def compter_classes(dataset):
    cpt_normal = 0
    cpt_pneumonia = 0

    for label in dataset.labels:
        if label[0] == 0:
            cpt_normal += 1
        else:
            cpt_pneumonia += 1

    pourcentage_normal = cpt_normal / len(dataset.labels) * 100
    pourcentage_pneumonia = cpt_pneumonia / len(dataset.labels) * 100

    print("Nombre d'image normale :",cpt_normal)
    print("Nombre d'image de pneumonie :",cpt_pneumonia)
    print(f"Pourcentage d'images normales : {pourcentage_normal:.2f}%")
    print(f"Pourcentage d'images de pneumonie : {pourcentage_pneumonia:.2f}%")


train = PneumoniaMNIST(split="train", download=True)
val = PneumoniaMNIST(split="val", download=True)
test = PneumoniaMNIST(split="test", download=True)

print("--- train ---")
compter_classes(train)

print("--- val ---")
compter_classes(val)

print("--- test ---")
compter_classes(test)
