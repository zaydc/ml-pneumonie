import torch

from architecture import creer_modele


def test_un_score_par_image():
    modele = creer_modele()
    images = torch.rand(8, 2, 28, 28)

    sorties = modele(images)

    assert sorties.shape == (8, 1)

def test_probabilites_entre_0_et_1():
    modele = creer_modele()
    images = torch.rand(8, 1, 28, 28)
    
    sorties = modele(images)
    probas = torch.sigmoid(sorties)
    assert probas.min() >= 0
    assert probas.max() <= 1