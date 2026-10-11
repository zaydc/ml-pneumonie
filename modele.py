# Entraînement du CNN sur PneumoniaMNIST, avec suivi de la perte sur train et val,
# évaluation sur val et sauvegarde du modèle

from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import PneumoniaMNIST
from sklearn.metrics import roc_auc_score
from architecture import creer_modele
import torch.nn as nn
import torch
import mlflow

# Graine fixée pour avoir les mêmes résultats à chaque lancement
torch.manual_seed(42)

# Hyperparamètres
lr = 0.001
nb_epoques = 20
batch_size = 64

mlflow.set_experiment("pneumonie-cnn")
mlflow.start_run()
mlflow.log_params({
    "lr": lr,
    "nb_epoques": nb_epoques,
    "batch_size": batch_size,
    "optimiseur": "Adam",
    "pos_weight": True,
})


# Chargement des trois splits (on ne mélange que le train)
train_t = PneumoniaMNIST(split="train", download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_t, batch_size=batch_size, shuffle=True)
val_t = PneumoniaMNIST(split="val", download=True, transform=transforms.ToTensor())
val_loader = DataLoader(val_t, batch_size=batch_size, shuffle=False)
test_t = PneumoniaMNIST(split="test", download=True, transform=transforms.ToTensor())
test_loader = DataLoader(test_t,batch_size=batch_size, shuffle=False)

modele = creer_modele()

# Il y a environ 3 pneumonies pour 1 radio normale dans le train :
# pos_weight réduit le poids des erreurs sur les pneumonies pour compenser
label_train = train_t.labels

nb_radio_normal = (label_train == 0).sum().item()
nb_radio_pneumonie = (label_train == 1).sum().item()
poids = torch.tensor([nb_radio_normal / nb_radio_pneumonie])

critere_poids = nn.BCEWithLogitsLoss(pos_weight=poids)

# Perte sans pondération, gardée pour comparer
critere = nn.BCEWithLogitsLoss()
optimiseur = torch.optim.Adam(modele.parameters(), lr)


for epoque in range(nb_epoques):
    # Entraînement sur le train
    modele.train()
    perte_totale = 0

    for images, labels in train_loader:
        sorties = modele(images)
        perte = critere_poids(sorties, labels.float())

        # Mise à jour des poids
        optimiseur.zero_grad()
        perte.backward()
        optimiseur.step()

        perte_totale += perte.item()

    # Perte sur val à la fin de chaque époque, pour repérer un éventuel surapprentissage
    modele.eval()
    perte_val_totale = 0
    with torch.no_grad():
        for images, labels in val_loader:
            sorties = modele(images)
            perte = critere_poids(sorties, labels.float())
            perte_val_totale += perte.item()

    perte_train = perte_totale / len(train_loader)
    perte_val = perte_val_totale / len(val_loader)

    mlflow.log_metric("perte_train", perte_train, step=epoque)
    mlflow.log_metric("perte_val", perte_val, step=epoque)

    print(f"Époque {epoque} : perte moyenne sur le train : {perte_train:.4f} et sur le val : {perte_val:.4f}")

# Affiche la sensibilité, la spécificité et l'accuracy pour un seuil donné
# (classe positive = pneumonie)
def evaluer(loader, seuil):
    modele.eval()
    bonnes_reponses = 0
    faux_negatifs = 0
    faux_positifs = 0
    nb_total_pneumonie = 0
    nb_total_radio_normal = 0

    with torch.no_grad():
        for images, labels in loader:
            sorties = modele(images)
            probas = torch.sigmoid(sorties)
            predictions = probas > seuil
            bonnes_reponses += (predictions == labels).sum().item()
            faux_negatifs += ((labels == 1) & (predictions == 0)).sum().item()
            faux_positifs += ((labels == 0) & (predictions == 1)).sum().item()
            nb_total_pneumonie += (labels == 1).sum().item()
            nb_total_radio_normal += (labels == 0).sum().item()
    nb_total_image = nb_total_pneumonie + nb_total_radio_normal

    sensibility = ((nb_total_pneumonie - faux_negatifs) / nb_total_pneumonie) * 100
    specificity = ((nb_total_radio_normal - faux_positifs) / nb_total_radio_normal) * 100
    accuracy = bonnes_reponses / nb_total_image * 100
    print(f"Seuil {seuil} | sensibilité {sensibility:.2f}% | spécificité {specificity:.2f}% | accuracy {accuracy:.2f}%")

# Plusieurs seuils testés sur val pour choisir le seuil final
for seuil in [0.1, 0.2, 0.3, 0.4, 0.5]:
    evaluer(val_loader, seuil)

# Renvoie l'AUC du modèle sur les données du loader
def calculer_auc(loader):
    modele.eval()
    toutes_probas = []
    tous_labels = []

    # On garde les probabilités et les labels de tous les paquets
    with torch.no_grad():
        for images, labels in loader:
            sorties = modele(images)
            probas = torch.sigmoid(sorties)
            toutes_probas.append(probas)
            tous_labels.append(labels)

    # Les paquets sont recollés en un seul tenseur avant de calculer l'AUC
    toutes_probas = torch.cat(toutes_probas)
    tous_labels = torch.cat(tous_labels)
    auc = roc_auc_score(tous_labels, toutes_probas)
    return auc

auc_val = calculer_auc(val_loader)
mlflow.log_metric("auc_val", auc_val)

print(f"AUC sur val : {auc_val:.4f}")

# Évaluation finale sur le test, faite une seule fois.
# Désactivée depuis, pour ne pas régler le modèle en regardant le test.
# print("---Examen final---")

# evaluer(test_loader,0.5)
# auc_test = calculer_auc(test_loader)

# print(f"AUC sur test {auc_test:.4f}")

# Sauvegarde des poids du modèle
torch.save(modele.state_dict(), "modele_cnn.pt")
print(modele.state_dict().keys())
mlflow.end_run()