import torch
from torch import nn

class ClasseExpert(nn.Module):
    def __init__(self, dimEntree, dimCachee, dimSortie):
        super().__init__()

        # couches simples
        self.couche1 = nn.Linear(dimEntree, dimCachee)
        self.couche2 = nn.Linear(dimCachee, dimCachee)
        self.couche3 = nn.Linear(dimCachee, dimSortie)

        self.activation = nn.ReLU()

    def forward(self, x):
        # passage avant rapide
        x = self.activation(self.couche1(x))
        x = self.activation(self.couche2(x))
        x = self.couche3(x)
        return x
