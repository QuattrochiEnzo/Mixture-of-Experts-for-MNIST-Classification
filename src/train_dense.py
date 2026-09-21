from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.dense import DenseModel


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

DATA_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)


class ModeleDense(nn.Module):
    def __init__(self):
        super().__init__()
        # un reseau dense simple (MLP)
        self.couche1 = nn.Linear(784, 256)
        self.couche2 = nn.Linear(256, 256)
        self.couche3 = nn.Linear(256, 10)
        self.act = nn.ReLU()

    def forward(self, x):
        # passage simple
        x = self.act(self.couche1(x))
        x = self.act(self.couche2(x))
        x = self.couche3(x)
        return x


def entrainerDense():

    appareil = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("on utilise :", appareil)

    # transfo mnist
    transfo = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])

    # jeux de donnees
    train = datasets.MNIST(root="data_mnist", train=True, download=True, transform=transfo)
    test = datasets.MNIST(root="data_mnist", train=False, download=True, transform=transfo)

    loaderTrain = DataLoader(train, batch_size=64, shuffle=True)
    loaderTest = DataLoader(test, batch_size=256, shuffle=False)

    # modele dense
    modele = ModeleDense().to(appareil)

    critere = nn.CrossEntropyLoss()
    optimiseur = torch.optim.Adam(modele.parameters(), lr=1e-3)

    nbEpochs = 5

    for e in range(nbEpochs):
        modele.train()
        perteCum = 0.0
        n = 0
        bons = 0

        for d, l in loaderTrain:
            d = d.to(appareil)
            l = l.to(appareil)

            sorties = modele(d)
            perte = critere(sorties, l)

            optimiseur.zero_grad()
            perte.backward()
            optimiseur.step()

            taille = l.size(0)
            perteCum += perte.item() * taille
            n += taille
            bons += (sorties.argmax(dim=1) == l).sum().item()

        perteMoy = perteCum / n
        accTrain = bons / n

        # eval test
        modele.eval()
        bonsTest = 0
        nTest = 0
        with torch.no_grad():
            for d, l in loaderTest:
                d = d.to(appareil)
                l = l.to(appareil)
                o = modele(d)
                bonsTest += (o.argmax(dim=1) == l).sum().item()
                nTest += l.size(0)

        accTest = bonsTest / nTest

        print(f"epoch {e+1}/{nbEpochs} | perte train = {perteMoy:.4f} | acc train = {accTrain*100:.2f}% | acc test = {accTest*100:.2f}%")

    torch.save(modele.state_dict(), "modele_dense_mnist.pth")
    print("modele dense sauvegarde dans modele_dense_mnist.pth")


if __name__ == "__main__":
    entrainerDense()
