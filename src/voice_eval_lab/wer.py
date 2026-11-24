"""Word error rate via word-level Levenshtein distance."""
from __future__ import annotations

def _norm(text: str) -> list[str]:
    return "".join(c.lower() if c.isalnum() or c.isspace() else " " for c in text).split()

def wer(reference: str, hypothesis: str) -> float:
    ref, hyp = _norm(reference), _norm(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    # DP edit distance
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1] / len(ref)

def corpus_wer(pairs: list[tuple[str, str]]) -> float:
    if not pairs:
        return 0.0
    return sum(wer(r, h) for r, h in pairs) / len(pairs)
