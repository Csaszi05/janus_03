# Kísérletnapló – janus_03 (Janus 0.3)

Viszonyítás a janus_01-ből (teljes ASVspoof 2019 LA eval halmaz, 71 237 fájl):

| Verzió | Run | Modell | Bemenet | Dev EER | Eval EER |
|---|---|---|---|---|---|
| 0.1 | Run 3 | JanusCNN, 23 ezer param. | mel | 6.99% | 23.52% |
| 0.2 | Run 4 | JanusCNN, 23 ezer param. | LFCC | 10.47% | 21.16% |

## Run 5 – ResNet-stílus LFCC-n, 2026-10-19 (Kaggle, T4)

**Kód:** commit `e05dc18`. **Változás a 0.2-höz (Run 4) képest:** csak a modell, `JanusResNet`, 5 reziduális blokk, 1 224 257 paraméter. Minden más azonos.

**Hipotézis volt:** dev EER 5% alá, eval EER 18% alá; A16 és A18 javul.

| Epoch | lr | Train loss | Dev EER |
|---|---|---|---|
| 1 | 1.0e-3 | 0.1082 | 24.05% |
| 3 | 9.9e-4 | 0.0657 | 12.72% |
| 4 | 9.8e-4 | 0.0585 | 9.46% |
| 8 | 8.7e-4 | 0.0369 | 14.01% |
| 13 | 6.6e-4 | 0.0250 | 6.05% |
| 14 | 6.0e-4 | 0.0233 | 10.60% |
| 15 | 5.5e-4 | 0.0214 | 5.55% |
| 20 | 3.0e-4 | 0.0145 | 5.18% |
| 23 | 1.7e-4 | 0.0112 | 4.91% |
| 27 | 4.3e-5 | 0.0087 | **4.60%** |
| 29 | 1.1e-5 | 0.0086 | 4.76% |

**Legjobb dev EER: 4.60% (27. epoch)** a bemásolt log alapján; a 30. epoch sora és a 9–11., 21. epoch hiányzik a bemásolt részből. Run 4: 10.47%, Run 1: 4.88%.

**Eval összesítve: 14.03%** (71 237 fájl, bonafide 7 355, spoof 63 882). Run 4 (0.2): 21.16%, Run 3 (0.1): 23.52%. Támadásonkénti átlag: 11.64% (Run 4: 18.40%).

| Eval támadás | Run 1 (0.1) | Run 4 (0.2) | Run 5 (0.3) | |
|---|---|---|---|---|
| A07 | 4.20% | 28.21% | 12.81% | jobb, mint 0.2 |
| A08 | 1.56% | 1.16% | 0.61% | eddigi legjobb |
| A09 | 2.33% | 1.09% | 0.49% | eddigi legjobb |
| A10 (Tacotron2 + WaveRNN) | 35.91% | 35.09% | 37.41% | nem javult |
| A11 (Griffin-Lim) | 3.82% | 7.23% | 2.40% | eddigi legjobb |
| A12 (Tacotron + WaveNet) | 56.31% | 41.67% | 36.39% | jobb, de rossz |
| A13 (VC + TTS, hullámforma-szűrés) | 57.05% | 35.41% | 18.23% | sokkal jobb |
| A14 | 13.90% | 4.01% | 2.50% | eddigi legjobb |
| A15 (VC, WaveNet) | 34.99% | 8.99% | 6.16% | eddigi legjobb |
| A16 (konkatenatív TTS) | 6.73% | 25.90% | 11.01% | sokkal jobb, mint 0.2 |
| A17 (VAE VC) | 35.01% | 14.29% | 9.51% | eddigi legjobb |
| A18 (vocoder VC) | 4.60% | 25.63% | 9.25% | sokkal jobb, mint 0.2 |
| A19 | 4.54% | 10.46% | 4.56% | Run 1 szintje |

**Tanulságok:**
- **A hipotézis bejött:** dev 4.60% (cél: 5% alatt), eval 14.03% (cél: 18% alatt). Az eval hiba a 0.2-höz képest egyharmadával csökkent (21.16 → 14.03%).
- **A dev–eval szakadék tovább szűkült:** 0.1 (Run 3): 7.0% vs 23.5% (3.4×), 0.2: 10.5% vs 21.2% (2×), 0.3: 4.6% vs 14.0% (3×). Arányában nem lett kisebb, mint a 0.2-ben, de mindkét szám jobb.
- **Viszonyítás:** a hivatalos LFCC-GMM baseline eval EER-je 8.09%. A 0.3 még ettől is elmarad, főleg A10 és A12 miatt.
- **13 támadásból 12 javult a 0.2-höz képest.** Csak A10 nem (35 → 37%).
- **A16 és A18 visszajött** (26 → 11%, 26 → 9%), ahogy vártuk. A nagyobb kapacitás és a hosszabb látómező visszahozta, amit az LFCC-re váltás elvett.
- **A10 és A12 a két megmaradt nagy hiba** (36–37%). Mindkettő Tacotron-alapú TTS neurális vocoderrel (WaveRNN, WaveNet). A modell ezeket még mindig valódinak látja.
- **A dev görbe középen ugrál** (8.7 → 14.0%, 6.1 → 10.6%), a végén sima. A nagyobb modell nagy lr mellett érzékenyebb; a cosine vége rendbe teszi.
- **A train loss 0.0086-ra ment** (Run 4: 0.052). A modell a train halmazt szinte hibátlanul megtanulta. A dev a végéig javult, tehát még nem tanult túl, de közel lehet.

**Hiányzik a naplóból:** a 30. epoch sora (a bemásolt log ott megszakadt).

## Run 5b – ugyanaz újrafuttatva, mentett verzióként, 2026-10-19 (Kaggle, T4)

**Miért:** a Run 5 a szerkesztő nézetben futott, a checkpoint elveszett a munkamenettel. Ugyanaz a kód (`e05dc18`), Save & Run All módban. A checkpoint és a pontszámok helyben: `out/janus_best.pt`, `out/eval_scores.csv` (gitignore alatt).

**Legjobb dev EER: 5.14% (26. epoch). Eval összesítve: 14.08%.**

| Eval támadás | Run 5 | Run 5b |
|---|---|---|
| A07 | 12.81% | 13.10% |
| A08 | 0.61% | 0.94% |
| A09 | 0.49% | 0.63% |
| A10 | 37.41% | 35.19% |
| A11 | 2.40% | 2.27% |
| A12 | 36.39% | 33.67% |
| A13 | 18.23% | 12.89% |
| A14 | 2.50% | 5.20% |
| A15 | 6.16% | 17.55% |
| A16 | 11.01% | 10.61% |
| A17 | 9.51% | 9.60% |
| A18 | 9.25% | 9.96% |
| A19 | 4.56% | 4.15% |

**Tanulságok:**
- **Az összesített eval EER stabil:** 14.03% és 14.08%, két független tanítás (más véletlen kezdősúlyok, más batch-sorrend, más augmentáció). A 0.2 → 0.3 javulás (21.2 → 14.1%) tehát nem szerencse.
- **A támadásonkénti számok viszont szórnak:** A15 6.2 vs 17.6%, A13 18.2 vs 12.9%, A14 2.5 vs 5.2%. Egy-egy támadásnál a futások közti különbség 5–11 százalékpont is lehet. A dolgozatban a támadásonkénti számokat óvatosan kell értelmezni; egyetlen futásból csak a nagy különbségek (pl. A16 26 → 11%) megbízhatók.
- **A dolgozat a Run 5b számait használja,** mert ehhez van mentett checkpoint és pontszámfájl.
