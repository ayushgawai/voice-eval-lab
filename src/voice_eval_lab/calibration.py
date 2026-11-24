"""Calibration: auto-vs-human Pearson r and slice agreement."""
from __future__ import annotations
import math
from collections import defaultdict

def pearson(xs, ys) -> float:
    if len(xs) < 2: return 0.0
    mx, my = sum(xs)/len(xs), sum(ys)/len(ys)
    num = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x-mx)**2 for x in xs) * sum((y-my)**2 for y in ys))
    return num/den if den else 0.0

def calibration_report(rows, tolerance=0.5):
    """rows: dicts with human, auto, dialect, noise, checkpoint."""
    human = [r["human"] for r in rows]; auto = [r["auto"] for r in rows]
    agree = sum(abs(h-a) <= tolerance for h, a in zip(human, auto)) / len(rows) if rows else 0.0
    slices = {}
    for key in ("dialect", "noise", "checkpoint"):
        groups = defaultdict(list)
        for r in rows: groups[r[key]].append(r)
        slices[key] = {k: {"n": len(v),
                           "human_mean": round(sum(x["human"] for x in v)/len(v), 4),
                           "auto_mean": round(sum(x["auto"] for x in v)/len(v), 4),
                           "pearson_r": round(pearson([x["human"] for x in v], [x["auto"] for x in v]), 4),
                           "agreement": round(sum(abs(x["human"]-x["auto"]) <= tolerance for x in v)/len(v), 4)}
                       for k, v in sorted(groups.items())}
    return {"n": len(rows), "pearson_r": round(pearson(human, auto), 4),
            "mae": round(sum(abs(h-a) for h, a in zip(human, auto))/len(rows), 4) if rows else 0.0,
            "agreement_within_0.5": round(agree, 4), "slices": slices}
