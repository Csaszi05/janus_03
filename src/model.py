import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Reziduális blokk: kimenet = ReLU(F(x) + skip(x)).

    F(x): két 3x3 konvolúció, mindegyik után BatchNorm (az első után ReLU is).
    skip(x): maga a bemenet; ha a csatornaszám változik, egy 1x1-es konvolúció
    igazítja hozzá, hogy a kettő összeadható legyen.
    """

    def __init__(self, c_in, c_out):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
        )
        if c_in == c_out:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Sequential(nn.Conv2d(c_in, c_out, kernel_size=1, bias=False), nn.BatchNorm2d(c_out))

    def forward(self, x):
        return F.relu(self.body(x) + self.skip(x))


class JanusResNet(nn.Module):
    """Janus 0.3. Bemenet: (B, 1, F, T) feature-kép (LFCC: 60 x 401). Kimenet: (B,) logit, pozitív = spoof.

    5 reziduális blokk, 16 -> 32 -> 64 -> 128 -> 256 csatorna, mindegyik után 2x2 MaxPool.
    60 x 401 bemenetnél a térkép 60x401 -> 30x200 -> 15x100 -> 7x50 -> 3x25 -> 1x12.
    A fej ugyanaz, mint a janus_01 JanusCNN-jéé: átlagolás, dropout, egy lineáris réteg.
    """

    def __init__(self, channels=(16, 32, 64, 128, 256), dropout=0.3):
        super().__init__()
        layers, c_in = [], 1
        for c_out in channels:
            layers += [ResBlock(c_in, c_out), nn.MaxPool2d(2)]
            c_in = c_out
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c_in, 1),
        )

    def forward(self, x):
        return self.head(self.features(x)).squeeze(1)
