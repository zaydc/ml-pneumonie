import json

nb_prediction = 0
nb_pneumonie = 0
somme_probabilites = 0
somme_luminosites = 0

with open("predictions.jsonl",encoding="utf-8") as fichier:
    for ligne in fichier: 
        prediction = json.loads(ligne)
        nb_prediction += 1
        if prediction["decision"] == "pneumonie":
            nb_pneumonie += 1
        somme_probabilites += prediction["probabilite"]
        somme_luminosites += prediction["luminosite"]

pourcentage_pneumonie = nb_pneumonie / nb_prediction * 100
probabilite_moyenne = somme_probabilites / nb_prediction 
luminosite_moyenne = somme_luminosites / nb_prediction

reference_luminosite = 0.5629
reference_pneumonie = 71.0

print(f"Pourcentage de radio 'pneumonie' : {pourcentage_pneumonie:.4f}%")
print(f"Probabilité moyenne : {probabilite_moyenne:.4f}")
print(f"Luminosité moyenne : {luminosite_moyenne:.4f}")
print(f"Nombre de prédiction : {nb_prediction}")

if abs(luminosite_moyenne - reference_luminosite) > 0.05:
    print("ALERTE : la luminosité des images a changé")
if abs(pourcentage_pneumonie - reference_pneumonie) > 5:
    print("ALERTE : le pourcentage de pneumonie a changé")