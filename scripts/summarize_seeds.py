"""Seedes futasok osszesitese: verzionkent atlag +- szoras.

Hasznalat:
    python scripts/summarize_seeds.py --runs out_lfcc out_mel --out summary

A --runs mappakban <verzio>_seed<n> almappakat keres (pl. v02_seed0), mindegyikben
train_log.txt, eval_scores.csv es eval_per_attack.csv. Kimenet a --out mappaba:
runs.csv (futasonkent), overall.csv, per_attack.csv, k2.csv es summary.md.

A szoras mintaszoras (ddof=1), n = a verzio seedjeinek szama.
K2-bontas: EER kulon az ismert (A16, A19: a train A04 es A06 tamadasa) es kulon az
ismeretlen (A07-A15, A17, A18) tamadasokra, mindig a teljes bonafide halmazzal szemben.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

KNOWN = ["A16", "A19"]
UNKNOWN = ["A07", "A08", "A09", "A10", "A11", "A12", "A13", "A14", "A15", "A17", "A18"]
RUN_RE = re.compile(r"^(v\d+)_seed(\d+)$")


def compute_eer(scores, labels):
    # ugyanaz, mint src/train.py compute_eer
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[i] + fnr[i]) / 2)


def subset_eer(df, systems):
    """EER a teljes bonafide halmaz es a megadott tamadasok spoof-fajljai kozott."""
    sub = df[(df.label == "bonafide") | df.system.isin(systems)]
    return compute_eer(sub.score.to_numpy(), (sub.label == "spoof").astype(int).to_numpy())


def best_dev_eer(log_path):
    for line in reversed(log_path.read_text().splitlines()):
        if line.startswith("best dev_eer"):
            return float(line.split()[-1].rstrip("%")) / 100
    raise ValueError(f"nincs 'best dev_eer' sor: {log_path}")


def version_name(tag):
    return f"{tag[1]}.{tag[2:]}"  # v02 -> 0.2


def load_runs(dirs):
    runs, per_attack = [], []
    for d in dirs:
        for run in sorted(Path(d).iterdir()):
            m = RUN_RE.match(run.name)
            if not (run.is_dir() and m):
                continue
            if not (run / "eval_scores.csv").exists():
                print(f"kihagyva (nincs eval_scores.csv): {run}")
                continue
            version, seed = version_name(m.group(1)), int(m.group(2))
            scores = pd.read_csv(run / "eval_scores.csv")
            runs.append({
                "version": version,
                "seed": seed,
                "dev_eer": best_dev_eer(run / "train_log.txt"),
                "eval_eer": compute_eer(scores.score.to_numpy(), (scores.label == "spoof").astype(int).to_numpy()),
                "known_eer": subset_eer(scores, KNOWN),
                "unknown_eer": subset_eer(scores, UNKNOWN),
                "run": str(run),
            })
            pa = pd.read_csv(run / "eval_per_attack.csv")
            per_attack += [{"version": version, "seed": seed, "system": r.system, "eer": r.eer} for r in pa.itertuples()]
    if not runs:
        raise SystemExit("nem talaltam futast")
    runs = pd.DataFrame(runs).sort_values(["version", "seed"])
    if runs.duplicated(["version", "seed"]).any():
        raise SystemExit(f"ugyanaz a verzio+seed tobbszor:\n{runs[runs.duplicated(['version', 'seed'], keep=False)]}")
    return runs, pd.DataFrame(per_attack)


def mean_std(df, by, col):
    g = df.groupby(by)[col]
    return pd.DataFrame({"mean": g.mean(), "std": g.std(ddof=1), "n": g.count()})


def pm(mean, std):
    """szazalekban: '21.16 ± 0.84'; egy seednel csak az atlag."""
    if np.isnan(std):
        return f"{mean * 100:.2f}"
    return f"{mean * 100:.2f} ± {std * 100:.2f}"


def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True, help="mappak, bennuk <verzio>_seed<n> almappak")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    runs, per_attack = load_runs(args.runs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    versions = sorted(runs.version.unique())

    runs.to_csv(out / "runs.csv", index=False)
    overall = pd.concat({c: mean_std(runs, "version", c) for c in ["dev_eer", "eval_eer"]}, axis=1)
    overall.to_csv(out / "overall.csv")
    k2 = pd.concat({c: mean_std(runs, "version", c) for c in ["known_eer", "unknown_eer"]}, axis=1)
    k2.to_csv(out / "k2.csv")
    pa = mean_std(per_attack, ["system", "version"], "eer").unstack("version")
    pa.to_csv(out / "per_attack.csv")

    seeds = {v: sorted(runs[runs.version == v].seed) for v in versions}
    md = ["# Seedes futások összesítése", ""]
    md.append("Átlag ± mintaszórás (ddof=1), százalékban. Seedek verziónként: "
              + ", ".join(f"{v}: {s}" for v, s in seeds.items()) + ".")
    if any(len(s) < 3 for s in seeds.values()):
        md.append("")
        md.append("**Figyelem:** nem minden verziónak van 3 seedje.")
    md += ["", "## Összesített EER", ""]
    md.append(md_table(["Verzió", "n", "Dev EER (legjobb epoch)", "Eval EER"],
                       [[v, int(overall.loc[v, ("eval_eer", "n")]),
                         pm(overall.loc[v, ("dev_eer", "mean")], overall.loc[v, ("dev_eer", "std")]),
                         pm(overall.loc[v, ("eval_eer", "mean")], overall.loc[v, ("eval_eer", "std")])]
                        for v in versions]))
    md += ["", "## K2: ismert és ismeretlen támadások", "",
           f"Ismert: {', '.join(KNOWN)}. Ismeretlen: {', '.join(UNKNOWN)}. Mindkettő a teljes bonafide halmazzal szemben.",
           ""]
    md.append(md_table(["Verzió", "Ismert EER", "Ismeretlen EER"],
                       [[v, pm(k2.loc[v, ("known_eer", "mean")], k2.loc[v, ("known_eer", "std")]),
                         pm(k2.loc[v, ("unknown_eer", "mean")], k2.loc[v, ("unknown_eer", "std")])]
                        for v in versions]))
    md += ["", "## Támadásonként", ""]
    md.append(md_table(["Támadás"] + versions,
                       [[s] + [pm(pa.loc[s, ("mean", v)], pa.loc[s, ("std", v)]) for v in versions]
                        for s in pa.index]))
    md += ["", "## Futásonként", ""]
    md.append(md_table(["Verzió", "Seed", "Dev EER", "Eval EER", "Ismert", "Ismeretlen"],
                       [[r.version, r.seed, f"{r.dev_eer * 100:.2f}", f"{r.eval_eer * 100:.2f}",
                         f"{r.known_eer * 100:.2f}", f"{r.unknown_eer * 100:.2f}"] for r in runs.itertuples()]))
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nmentve: {out}/summary.md, runs.csv, overall.csv, k2.csv, per_attack.csv")


if __name__ == "__main__":
    main()
