"""Deterministic synthetic+recorded utterance and checkpoint generation.

No real audio is bundled. Utterances are text + acoustic metadata
(snr, dialect, noise, duration) sufficient to simulate checkpoint
behaviour, human ratings, and WER. This is stated honestly in README.
"""
from __future__ import annotations
import random
from dataclasses import dataclass, asdict

DIALECTS = ["en-US", "en-GB", "en-IN", "en-AU", "en-NG", "en-PH", "en-IE", "en-ZA"]
DIALECT_DIFFICULTY = {"en-US": 0.00, "en-GB": 0.04, "en-IN": 0.10, "en-AU": 0.05,
                      "en-NG": 0.13, "en-PH": 0.09, "en-IE": 0.06, "en-ZA": 0.08}
NOISES = ["clean", "cafe", "street", "car", "office"]
NOISE_PENALTY = {"clean": 0.0, "cafe": 0.08, "street": 0.11, "car": 0.07, "office": 0.04}

TEMPLATES = [
    "the quick brown fox jumps over the lazy dog",
    "please schedule a meeting for tomorrow morning",
    "what is the weather forecast for this weekend",
    "turn left at the next traffic light and continue straight",
    "i would like to order a large coffee with oat milk",
    "the quarterly report shows revenue grew by twelve percent",
    "can you remind me to call the doctor on friday",
    "open the window and let some fresh air in",
    "the train to the city center departs in twenty minutes",
    "she sells sea shells on the sea shore every morning",
    "please transfer two hundred dollars to my savings account",
    "the children are playing football in the park nearby",
    "how do i get to the nearest pharmacy from here",
    "the museum opens at nine and closes at six",
    "we need to finish the presentation before the deadline",
    "the recipe calls for two cups of flour and sugar",
    "his voice was clear despite the noise in the background",
    "the elevator is out of order please use the stairs",
    "i have been waiting for the bus for fifteen minutes",
    "the new apartment is close to the university campus",
    "please speak slowly and repeat the last sentence",
    "the football match ended in a draw last night",
    "she bought a ticket to visit her family next month",
    "the water in the harbor was calm at sunrise",
]

@dataclass
class Checkpoint:
    id: str; kind: str; base_wer: float; quality: float; latency_ms: float
    def to_dict(self): return asdict(self)

@dataclass
class Utterance:
    id: str; text: str; dialect: str; noise: str; snr_db: float
    duration_s: float; source: str; speaker: str
    def to_dict(self): return asdict(self)

def default_checkpoints() -> list[Checkpoint]:
    # 8 checkpoints: 5 ASR + 3 TTS, spanning weak -> strong
    return [
        Checkpoint("asr-tiny-v1", "ASR", 0.28, 0.55, 42),
        Checkpoint("asr-small-v2", "ASR", 0.21, 0.65, 68),
        Checkpoint("asr-base-v3", "ASR", 0.15, 0.74, 110),
        Checkpoint("asr-large-v4", "ASR", 0.10, 0.84, 210),
        Checkpoint("asr-large-v5", "ASR", 0.07, 0.90, 260),
        Checkpoint("tts-fast-v1", "TTS", 0.18, 0.62, 55),
        Checkpoint("tts-base-v2", "TTS", 0.12, 0.78, 120),
        Checkpoint("tts-hifi-v3", "TTS", 0.08, 0.88, 240),
    ]

def generate_utterances(n: int = 2400, seed: int = 42) -> list[Utterance]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        dialect = DIALECTS[i % len(DIALECTS)]
        noise = NOISES[rng.randrange(len(NOISES))]
        base_snr = {"clean": 28, "cafe": 12, "street": 8, "car": 14, "office": 18}[noise]
        snr = round(max(0.0, rng.gauss(base_snr, 3.0)), 2)
        text = TEMPLATES[i % len(TEMPLATES)]
        out.append(Utterance(
            id=f"utt-{i:04d}", text=text, dialect=dialect, noise=noise, snr_db=snr,
            duration_s=round(1.5 + len(text.split()) * 0.35, 2),
            source="synthetic" if i % 3 else "recorded",
            speaker=f"spk-{i % 48:02d}"))
    return out

_SUBS = ["the", "a", "please", "um", "that", "what", "there", "their"]

def simulate_hypothesis(utt: Utterance, ckpt: Checkpoint, seed: int = 42) -> str:
    """Corrupt reference with prob. driven by checkpoint + dialect + noise."""
    rng = random.Random(f"{seed}:{utt.id}:{ckpt.id}")
    p = ckpt.base_wer + DIALECT_DIFFICULTY[utt.dialect] + NOISE_PENALTY[utt.noise]
    p += max(0.0, (15 - utt.snr_db)) * 0.004
    p = min(0.85, max(0.0, p))
    words = utt.text.split()
    out = []
    for w in words:
        r = rng.random()
        if r < p * 0.45:
            continue  # deletion
        if r < p:
            out.append(rng.choice(_SUBS))  # substitution
        else:
            out.append(w)
    return " ".join(out) if out else _SUBS[0]

def human_ratings(utt: Utterance, ckpt: Checkpoint, seed: int = 42, raters: int = 12) -> list[float]:
    """12-rater human quality labels (1-5), simulated with rater noise/bias."""
    rng = random.Random(f"human:{seed}:{utt.id}:{ckpt.id}")
    true_q = 5.0 * ckpt.quality - DIALECT_DIFFICULTY[utt.dialect] * 4 \
        - NOISE_PENALTY[utt.noise] * 5 - max(0.0, (12 - utt.snr_db)) * 0.04
    true_q = min(5.0, max(1.0, true_q))
    return [round(min(5.0, max(1.0, rng.gauss(true_q, 0.45))), 2) for _ in range(raters)]
