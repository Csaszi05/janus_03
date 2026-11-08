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

## Seedes futások: Janus 0.1–0.3 × seed 0/1/2 (Kaggle, mind a 9 futás lefutott)

**Miért:** a 7. fejezetbe átlag ± szórás kell, nem egyetlen futás. Különösen a 0.1 → 0.2 lépés (23.52 → 21.16%, 2.4 pont) kérdéses: a Run 5 / 5b párból látszik, hogy támadásonként 5–11 pont is lehet a futások közti különbség.

**Csak a seed új.** Config, 30 epoch, batch 64, Adam lr 1e-3, cosine, `--augment`, `--workers 2` betűre ugyanaz, mint Run 3, Run 4 és Run 5b. A kód egyetlen változása a `src/train.py --seed` kapcsoló és a `train_log.txt` (döntésnapló, 2026-10-26).

| Kaggle-munkamenet | Notebook | Verzió | Repó, config | Kimenet |
|---|---|---|---|---|
| 1 (LFCC) | `janus_03/notebooks/04_seeds_lfcc_kaggle.ipynb` | 0.2 | janus_01, `config.yaml` | `out/v02_seed0…2` |
| 1 (LFCC) | ugyanaz | 0.3 | janus_03, `config.yaml` | `out/v03_seed0…2` |
| 2 (mel) | `janus_01/notebooks/04_seeds_mel_kaggle.ipynb` | 0.1 | janus_01, `config_mel.yaml` | `out/v01_seed0…2` |

**Sorrend:** előbb az 1. munkamenet (a 0.2 és 0.3 a fontosabb), utána a 2. Ha a 0.1 nem fér bele a határnapig, a 0.1-nél az egyetlen Run 3 marad, és ez a korlátok közé kerül.

**Kimenet futásonként:** `train_log.txt`, `janus_best.pt`, `eval_scores.csv`, `eval_per_attack.csv`, `commit.txt` (repó, commit, config, seed).

**Összesítés:** `python scripts/summarize_seeds.py --runs <letöltött out mappák> --out <mappa>` → `summary.md` (összesített dev/eval EER, támadásonként, K2: ismert A16+A19 vs. ismeretlen A07–A15, A17, A18; mindegyik átlag ± mintaszórás). Ellenőrizve a Run 5b mentett pontszámain: eval 14.08% (egyezik); K2: ismert 7.95%, ismeretlen 15.20%.

### Eredmény

**Futások:** 0.1 a janus_01 `a2109dd` commitjáról, `config_mel.yaml`. 0.2 ugyanerről a commitról, `config.yaml`. 0.3 a janus_03 `6bbeef3` commitjáról. Mind a 9 futás 30 epochot tanult, az evalhoz a legjobb dev epoch checkpointja ment (22–29. epoch). Összesítés: `summarize_seeds.py --runs <mel out> <lfcc out>`, a kimenet a repón kívül van (`out/` gitignore alatt).

**Ellenőrizve:**
- Mind a 9 `eval_scores.csv` ugyanazt a 71 237 fájlt tartalmazza, ugyanabban a sorrendben, és nincs bennük NaN.
- Minden `eval_per_attack.csv` pontosan egyezik a pontszámokból újraszámolt értékekkel.
- A korábbi egyedi futások beleesnek a seedes tartományba: Run 3 23.52%, Run 4 21.16%, Run 5b 14.08%.

Átlag ± mintaszórás (ddof=1), n = 3 seed verziónként:

| Verzió | Dev EER (legjobb epoch) | Eval EER | Eval min–max | Ismert (A16, A19) | Ismeretlen |
|---|---|---|---|---|---|
| 0.1 (mel, JanusCNN) | 8.31 ± 1.04 | **24.32 ± 3.48** | 21.23–28.09 | 11.58 ± 2.96 | 26.46 ± 3.70 |
| 0.2 (LFCC, JanusCNN) | 11.02 ± 0.91 | **21.01 ± 1.69** | 19.83–22.95 | 18.38 ± 1.37 | 21.50 ± 1.77 |
| 0.3 (LFCC, JanusResNet) | 4.84 ± 0.20 | **14.20 ± 0.39** | 13.76–14.46 | 7.62 ± 0.70 | 15.37 ± 0.38 |

Futásonként:

| Verzió | Seed | Dev EER | Eval EER | Ismert | Ismeretlen |
|---|---|---|---|---|---|
| 0.1 | 0 | 8.95 | 28.09 | 14.93 | 30.45 |
| 0.1 | 1 | 8.87 | 21.23 | 9.31 | 23.12 |
| 0.1 | 2 | 7.11 | 23.65 | 10.49 | 25.82 |
| 0.2 | 0 | 10.70 | 20.26 | 17.83 | 20.70 |
| 0.2 | 1 | 12.05 | 22.95 | 19.94 | 23.53 |
| 0.2 | 2 | 10.32 | 19.83 | 17.37 | 20.27 |
| 0.3 | 0 | 4.72 | 13.76 | 6.82 | 14.93 |
| 0.3 | 1 | 4.73 | 14.46 | 8.10 | 15.60 |
| 0.3 | 2 | 5.07 | 14.39 | 7.93 | 15.58 |

Támadásonként (eval EER %, átlag ± szórás):

| Támadás | 0.1 | 0.2 | 0.3 |
|---|---|---|---|
| A07 | 21.81 ± 9.92 | 31.01 ± 1.69 | 14.50 ± 2.02 |
| A08 | 6.46 ± 1.24 | 0.76 ± 0.07 | 0.75 ± 0.29 |
| A09 | 3.90 ± 1.12 | 0.82 ± 0.13 | 0.43 ± 0.20 |
| A10 | 42.21 ± 4.30 | 36.67 ± 1.77 | 38.42 ± 1.62 |
| A11 | 40.22 ± 9.13 | 5.85 ± 1.15 | 1.95 ± 0.41 |
| A12 | 50.25 ± 2.54 | 40.25 ± 4.87 | 36.36 ± 1.26 |
| A13 | 25.47 ± 3.29 | 34.09 ± 7.21 | 18.82 ± 2.72 |
| A14 | 12.60 ± 0.72 | 3.01 ± 0.21 | 3.00 ± 0.73 |
| A15 | 31.18 ± 7.44 | 7.30 ± 1.11 | 8.38 ± 2.03 |
| A16 | 14.18 ± 3.48 | 25.98 ± 2.09 | 10.46 ± 0.92 |
| A17 | 22.63 ± 0.47 | 11.42 ± 0.92 | 7.84 ± 0.25 |
| A18 | 8.32 ± 2.64 | 25.64 ± 3.17 | 9.78 ± 0.56 |
| A19 | 8.75 ± 2.08 | 8.72 ± 0.39 | 3.96 ± 0.41 |

**Tanulságok:**
- **A 0.2 → 0.3 javulás biztos.** Eval 21.0 → 14.2%. A 0.3 mindhárom seedje jobb a 0.2 mindhárom seedjénél, a tartományok nem fednek át (Welch-féle t-próba, p = 0.016). Az ismert és az ismeretlen támadásokon is javult (p = 0.001 és 0.023).
- **A 0.1 → 0.2 javulás összesítve nem igazolható.** Eval 24.3 → 21.0%, a különbség 3.3 pont, de a tartományok átfednek (0.1 legjobb seedje 21.23%, 0.2 legrosszabbja 22.95%), Welch p = 0.24. Az egyedi futásokból látott 2.4 pontos javulás (Run 3 → Run 4) a seedek közötti szórás nagyságrendje.
- **Az LFCC-re váltás valódi hatása a hibaprofil átrendeződése:**
  - Sokat javult: A11 40 → 6%, A15 31 → 7%, A14 13 → 3%, A17 23 → 11%, A08 6 → 1%.
  - Sokat romlott: A16 14 → 26%, A18 8 → 26%.
  - K2: az ismert támadások romlottak (11.6 → 18.4%, p = 0.04), az ismeretlenek javultak (26.5 → 21.5%), de ez utóbbi n = 3 mellett nem szignifikáns (p = 0.13).
  - A Run 4 tanulságai („nincs ingyen ebéd”) seedekkel is megállnak.
- **A stabilitás verzióról verzióra nő.**
  - Eval szórás: 0.1: ±3.5, 0.2: ±1.7, 0.3: ±0.4 pont.
  - A 0.1-nél egyes támadásoknál ±9–10 pont (A07, A11), tehát a 0.1 támadásonkénti számai egyetlen futásból (Run 1, Run 3) nem megbízhatók. Az A07 például a Run 1-ben 4.2%, a seedes átlag 21.8%.
- **A dev EER nem jelzi előre az evalt.** A 0.1 dev-en jobb, mint a 0.2 (8.3 vs 11.0%, p = 0.03), eval-on mégis rosszabb. A dev halmazban csak a train támadásai vannak (A01–A06). A dev EER a legjobb epoché, és ugyanazon a halmazon történt a kiválasztás, ezért optimista becslés.
- **A10 és A12 minden verzióban 36–50% között marad.** Mindkettő Tacotron-alapú, neurális vocoderrel. Ezek maradnak a fő korlátok.
- **A dolgozat 7. fejezete ezeket a seedes számokat használja,** átlag ± szórás és min–max formában. A p-értékek n = 3 mellett gyenge bizonyítékok, csak kiegészítésként szerepelnek.
