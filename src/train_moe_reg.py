from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.moe_layer import MoELayer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

DATA_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)

def entrainerMoEReg(nbEpoch=5, lambdaEntropie=0.01, tailleBatch=64):

    appareil = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("on utilise :", appareil)

    transfo = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))  # flatten
    ])

    train = datasets.MNIST(root="data_mnist", train=True, download=True, transform=transfo)
    loaderTrain = DataLoader(train, batch_size=tailleBatch, shuffle=True)

    modele = CoucheMoE(nbExperts=4)
    modele.to(appareil)

    optim = torch.optim.Adam(modele.parameters(), lr=1e-3)
    perteCE = nn.CrossEntropyLoss()

    for ep in range(1, nbEpoch+1):
        cumul = 0
        bons = 0
        total = 0

        for d, l in loaderTrain:
            d, l = d.to(appareil), l.to(appareil)

            # forward
            sortie = modele(d)

            # perte classique
            perte = perteCE(sortie, l)

            # entropie
            entropie = modele.calculerEntropieGate(d)

            # perte totale
            perteTot = perte + lambdaEntropie * entropie

            optim.zero_grad()
            perteTot.backward()
            optim.step()

            cumul += perte.item() * d.size(0)
            bons += (sortie.argmax(dim=1) == l).sum().item()
            total += l.size(0)

        print(f"epoch {ep}/{nbEpoch} | perte CE = {cumul/total:.4f} | acc = {100*bons/total:.2f}% | entropie={entropie.item():.4f}")

    torch.save(modele.state_dict(), "modele_moe_reg.pth")
    print("modele entraine + entropie sauvegarde dans modele_moe_reg.pth")


if __name__ == "__main__":
    entrainerMoEReg()
