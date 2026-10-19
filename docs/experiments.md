# Kísérletnapló – janus_03 (Janus 0.3)

Viszonyítás a janus_01-ből (teljes ASVspoof 2019 LA eval halmaz, 71 237 fájl):

| Verzió | Run | Modell | Bemenet | Dev EER | Eval EER |
|---|---|---|---|---|---|
| 0.1 | Run 3 | JanusCNN, 23 ezer param. | mel | 6.99% | 23.52% |
| 0.2 | Run 4 | JanusCNN, 23 ezer param. | LFCC | 10.47% | 21.16% |

## Run 5 – ResNet-stílus LFCC-n (tervezett)

**Változás a 0.2-höz (Run 4) képest:** csak a modell, `JanusResNet`, 5 reziduális blokk, 1 224 257 paraméter. Minden más azonos.

**Hipotézis:** dev EER 5% alá, eval EER 18% alá. A Run 4-ben romlott A16 és A18 javul, mert a nagyobb háló több mintát tud egyszerre tartani.

**Eredmény:** (futás után)
