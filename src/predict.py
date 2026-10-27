"""Egy tetszoleges hangfajl pontozasa a betanitott modellel.

Hasznalat:
    python src/predict.py --checkpoint janus_best.pt --audio sajat.wav [--scores eval_scores.csv]

A hangot ugyanugy dolgozza fel, mint tanitaskor (16 kHz, csendvagas, a checkpointban
tarolt audio config). A modell 4 masodperces darabokat lat; hosszabb fajlt 4 s-os
darabokra vag, mindegyiket pontozza, es az atlagot adja. Ha megadod az eval_scores.csv-t,
abbol kiszamolja az EER-kuszobot, es megmondja, hova esik a fajl a valodi es a hamis
eval-felvetelek kozott.
"""
import argparse
import sys
from pathlib import Path

import librosa
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import extract  # noqa: E402
from model import JanusResNet  # noqa: E402


def chunks(path, cfg):
    sr, n = cfg["sr"], int(cfg["sr"] * cfg["max_sec"])
    y, _ = librosa.load(path, sr=sr, mono=True)
    y, _ = librosa.effects.trim(y, top_db=cfg["trim_db"])
    if len(y) == 0:
        raise SystemExit("A fajl a csendvagas utan ures (tul halk vagy csak csend).")
    out = []
    for start in range(0, len(y), n):
        part = y[start:start + n]
        if start > 0 and len(part) < n // 2:  # a 2 s-nal rovidebb maradekot eldobjuk
            break
        out.append(np.pad(part, (0, n - len(part))))
    return out, len(y) / sr


def eer_threshold(scores_csv):
    import pandas as pd
    from sklearn.metrics import roc_curve

    df = pd.read_csv(scores_csv)
    y = (df["label"] == "spoof").astype(int).to_numpy()
    s = df["score"].to_numpy()
    fpr, tpr, thr = roc_curve(y, s)
    i = np.nanargmin(np.abs((1 - tpr) - fpr))
    return float(thr[i]), s[y == 0], s[y == 1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--audio", required=True)
    p.add_argument("--scores", default=None, help="eval_scores.csv a kuszobhoz (opcionalis)")
    args = p.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ck["config"]["audio"]
    model = JanusResNet()
    model.load_state_dict(ck["model"])
    model.eval()

    parts, dur = chunks(args.audio, cfg)
    with torch.no_grad():
        x = torch.stack([torch.from_numpy(extract(y, cfg)).unsqueeze(0) for y in parts])
        logits = model(x).numpy()

    print(f"fajl: {args.audio}")
    print(f"hossz csendvagas utan: {dur:.1f} s, {len(parts)} darab x {cfg['max_sec']:.0f} s")
    for i, z in enumerate(logits):
        print(f"  {i + 1}. darab  logit {z:+.2f}")
    z = float(logits.mean())
    print(f"\natlag logit: {z:+.2f}  (pozitiv = hamis fele, negativ = valodi fele)")

    if args.scores:
        thr, bona, spoof = eer_threshold(args.scores)
        print(f"EER-kuszob az eval halmazon: {thr:+.2f}")
        print(f"  a valodi eval-felvetelek {np.mean(bona < z) * 100:.0f}%-anak alacsonyabb a pontszama")
        print(f"  a hamis eval-felvetelek {np.mean(spoof < z) * 100:.0f}%-anak alacsonyabb a pontszama")
        verdict = "HAMIS" if z > thr else "VALODI"
    else:
        verdict = "HAMIS" if z > 0 else "VALODI"
        print("(kuszob: 0; pontosabb kuszobhoz add meg: --scores eval_scores.csv)")
    print(f"\nitelet: {verdict}")


if __name__ == "__main__":
    main()
