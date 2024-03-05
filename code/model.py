import torch
import torch.nn as nn


def FC(in_dim, out_dim, dropout):
    return nn.Sequential(nn.BatchNorm1d(in_dim), nn.LeakyReLU(), nn.Dropout(dropout), nn.Linear(in_dim, out_dim))


# Backbone Architecture
class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, param):
        super(self.__class__, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_dim, hid_dim))
        for _ in range(param['n_layers'] - 1):
            self.layers.append(FC(hid_dim, hid_dim, param['dropout']))

        self.out = FC(hid_dim, out_dim, param['dropout'])
        self.inf = FC(hid_dim, out_dim, param['dropout'])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return self.out(x), self.inf(x)


# Positive sample augmentation by learnable interpolation
class NeFactor(nn.Module):
    def __init__(self, in_dim, hid_dim, param):
        super(NeFactor, self).__init__()

        self.linear = nn.Linear(in_dim, hid_dim)
        self.att = nn.Linear(hid_dim*2, 1, bias=False)
        self.norm = nn.BatchNorm1d(hid_dim)
        self.param = param

    def forward(self, src, dst):
        x_src = self.norm(self.linear(src))
        x_dst = self.norm(self.linear(dst))
        h = torch.cat((x_src, x_dst), dim=1)
        ratio = torch.sigmoid(self.att(h))

        return ratio * (1-self.param['ratio']) + self.param['ratio']