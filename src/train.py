from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.moe_layer import MoELayer

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

DATA_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)

def entrainerModele():

    # choix cpu ou gpu
    appareil = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("on utilise :", appareil)

    # transfo des clients (images) pour le resto
    transfo = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))   # on aplatit en vecteur 784
    ])

    # jeux de donnees train et test
    jeuEntrainement = datasets.MNIST(root="data_mnist", train=True, download=True, transform=transfo)
    jeuTest = datasets.MNIST(root="data_mnist", train=False, download=True, transform=transfo)

    chargeurTrain = DataLoader(jeuEntrainement, batch_size=64, shuffle=True)
    chargeurTest = DataLoader(jeuTest, batch_size=256, shuffle=False)

    # le resto (moe) avec 4 cuistots
    modele = CoucheMoE(nbExperts=4)
    modele.to(appareil)

    # critere = combien les clients sont pas contents
    critere = nn.CrossEntropyLoss()

    # optim = comment les cuistots s adaptent aux avis
    optimiseur = torch.optim.Adam(modele.parameters(), lr=1e-3)

    nbEpochs = 5  # nb de "jours d ouverture" du resto

    for indiceEpoch in range(nbEpochs):
        modele.train()
        perteCumulee = 0.0
        nbExemples = 0
        bonsClients = 0

        # boucle sur les services (batches)
        for donnees, labels in chargeurTrain:
            donnees = donnees.to(appareil)
            labels = labels.to(appareil)

            # passage des clients dans le resto
            sorties = modele(donnees)

            # calcul des plaintes
            perte = critere(sorties, labels)

            # backprop, on engueule les cuistots
            optimiseur.zero_grad()
            perte.backward()
            optimiseur.step()

            # suivi perte et accuracy sur le train
            tailleBatch = labels.size(0)
            perteCumulee += perte.item() * tailleBatch
            nbExemples += tailleBatch

            # clients bien servis
            predictions = sorties.argmax(dim=1)
            bonsClients += (predictions == labels).sum().item()

        perteMoyenneTrain = perteCumulee / nbExemples
        accuracyTrain = bonsClients / nbExemples

        # petite eval sur le jeu test
        modele.eval()
        bonsClientsTest = 0
        nbExemplesTest = 0
        with torch.no_grad():
            for donnees, labels in chargeurTest:
                donnees = donnees.to(appareil)
                labels = labels.to(appareil)

                sorties = modele(donnees)
                predictions = sorties.argmax(dim=1)

                nbExemplesTest += labels.size(0)
                bonsClientsTest += (predictions == labels).sum().item()

        accuracyTest = bonsClientsTest / nbExemplesTest

        print(f"epoch {indiceEpoch+1}/{nbEpochs} | perte train = {perteMoyenneTrain:.4f} | acc train = {accuracyTrain*100:.2f}% | acc test = {accuracyTest*100:.2f}%")

    # on garde le resto entraine sur disque
    torch.save(modele.state_dict(), "modele_moe_mnist.pth")
    print("modele sauvegarde dans modele_moe_mnist.pth")


if __name__ == "__main__":
    entrainerModele()
