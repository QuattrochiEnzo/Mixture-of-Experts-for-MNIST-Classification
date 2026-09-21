import torch
from torch import nn

class ClasseGating(nn.Module):
    def __init__(self, dimEntree, nbExperts):
        super().__init__()

        # petit mlp pour faire le score du gate
        self.couche1 = nn.Linear(dimEntree, 64)
        self.act = nn.ReLU()
        self.couche2 = nn.Linear(64, nbExperts)

    def forward(self, x):
        # juste les scores (pas softmax ici)
        x = self.act(self.couche1(x))
        x = self.couche2(x)
        return x
