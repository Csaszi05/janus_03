# Döntésnapló – janus_03 (Janus 0.3)

A Janus 0.1 és 0.2 (Run 1–4) döntései és eredményei a `janus_01` repóban vannak. Ez a napló csak azt rögzíti, ami a 0.3-ban új.

## 2026-10-19 – Külön mappa és repó

**Miért nem a janus_01-ben.** A 0.2 csak a bemenetet cserélte, ezért maradt a janus_01-ben, kapcsolóval. A 0.3 a modellt cseréli, ez már más rendszer. Külön mappában a janus_01 lezárt, változatlan viszonyítási alap marad, és a szakdolgozat 7. fejezetében mindkét verzió saját commitra hivatkozhat.

**Mappa neve janus_03, nem janus_02.** A szakdolgozat verziószámait követi (Janus 0.3). A 0.1 és a 0.2 egyaránt a janus_01-ben van.

**A kód a janus_01 `d80fb4e` commitjának másolata,** egyetlen különbséggel: `src/model.py` (`JanusResNet` a `JanusCNN` helyett), és a két import a `train.py`-ban és az `evaluate.py`-ban. A feature-számítás, az adatbetöltő, az augmentáció, a config és a tanítási paraméterek betűre azonosak. Így a 0.2 → 0.3 különbség csak a modellnek tudható be.

## 2026-10-19 – Modell (`src/model.py`, `JanusResNet`)

**Reziduális blokk.** Két 3×3 konvolúció BatchNorm-mal, és a bemenet hozzáadása a végén: kimenet = ReLU(F(x) + x). A gradiens az összeadáson át változatlanul visszajut az első rétegekig, és a blokknak csak a bemenethez képesti különbséget kell megtanulnia. Ha egy blokk nem tud hasznosat tenni, nullát tanul, és a kép változatlanul megy tovább.

**Átugró ág csatornaváltáskor: 1×1 konvolúció + BatchNorm.** Ahol a csatornaszám nő (pl. 16 → 32), a bemenetet hozzá kell igazítani, különben nem adható össze. Ez a ResNet-cikk szokásos megoldása.

**5 blokk, 16 → 32 → 64 → 128 → 256 csatorna, mindegyik után 2×2 MaxPool.** A 60 × 401-es LFCC-kép az 5. blokk után 1 × 12 lesz; több pool már nem férne bele. A csatornák duplázása a felbontás felezésével együtt a szokásos ResNet-arány: a számítás rétegenként nagyjából egyenletes.

**1 224 257 paraméter** (a JanusCNN 23 585-öt használt, 52-szer több). A Run 4 alultanított volt (a dev EER a 30. epochnál még csökkent), ehhez kell a nagyobb kapacitás. Az 1–3 milliós sáv alja: elég a javuláshoz, és 25 ezer tanító fájlon még nem várható erős túltanulás.

**`bias=False` a konvolúciókban.** Közvetlenül utánuk BatchNorm jön, ami úgyis kivonja az átlagot, a bias felesleges lenne.

**A fej változatlan** (átlagolás, dropout 0.3, egy lineáris réteg), csak 64 helyett 256 bemenettel. Egy változó: a törzs.

**Tanítás változatlan:** LFCC, maszkolás, cosine, 30 epoch, batch 64, Adam lr 1e-3. Nagyobb modellhez gyakran kisebb lr kell; ha a dev görbe az első epochokban ugrál vagy NaN jön, az lr lesz az első, amit módosítunk, és az külön futás lesz.
