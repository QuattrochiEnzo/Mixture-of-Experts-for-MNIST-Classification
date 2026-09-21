import torch
from torch import nn
import torch.nn.functional as F

from modele.experts import ClasseExpert
from modele.gating import ClasseGating

class CoucheMoE(nn.Module):
    def __init__(self, nbExperts=4, dimEntree=784, dimCachee=128, dimSortie=10):
        super().__init__()

        # liste des experts
        # modulelist sinon pytorch voit pas les params
        self.listeExperts = nn.ModuleList(
            [ClasseExpert(dimEntree, dimCachee, dimSortie) for _ in range(nbExperts)]
        )

        # gate
        self.gate = ClasseGating(dimEntree, nbExperts)
        self.nbExperts = nbExperts

    def forward(self, x):
        # scores du gate
        scores = self.gate(x)
        poids = F.softmax(scores, dim=1)

        # appliquer chaque expert
        # je stocke les sorties pour les combiner
        sortiesExperts = []

        for expert in self.listeExperts:
            sortiesExperts.append(expert(x))  # taille (batch, 10)

        # on stack (nbExperts, batch, 10)
        sortiesExperts = torch.stack(sortiesExperts, dim=1)

        # combinaison pondérée
        poids = poids.unsqueeze(2)  # (batch, nbExperts, 1)
        sortieFinale = (poids * sortiesExperts).sum(dim=1)

        return sortieFinale

    def calculerEntropieGate(self, donnees):
        # recup les proba du gate
        scores = self.gate(donnees)
        p = torch.softmax(scores, dim=1)

        # petit eps sinon log(0)
        eps = 1e-9
        entropie = - (p * torch.log(p + eps)).sum(dim=1)

        # on renvoie la moyenne sur le batch
        return entropie.mean()
