"""Synthetic adversarial harness: dialect swaps + background-noise stress.

Generates adversarial variants of utterances for regression testing.
Adversarial here means harder eval conditions, not an attack on a
live system: lexical dialect substitutions and simulated SNR drops.
"""
from __future__ import annotations
from dataclasses import replace
from .data import Utterance

DIALECT_SWAPS = {"elevator": "lift", "apartment": "flat", "truck": "lorry",
                 "football": "soccer", "pharmacy": "chemist", "schedule": "timetable",
                 "coffee": "chai", "dollars": "quid"}
NOISE_LADDER = [("cafe", 8.0), ("street", 4.0), ("car", 6.0)]

def dialect_variant(utt: Utterance) -> Utterance:
    text = " ".join(DIALECT_SWAPS.get(w, w) for w in utt.text.split())
    return replace(utt, id=utt.id + "-dialect", text=text, dialect="en-GB")

def noise_variant(utt: Utterance, noise="street", snr_db=4.0) -> Utterance:
    return replace(utt, id=f"{utt.id}-{noise}", noise=noise, snr_db=snr_db)

def generate_adversarial(utterances):
    out = []
    for u in utterances:
        out.append(dialect_variant(u))
        for noise, snr in NOISE_LADDER:
            out.append(noise_variant(u, noise, snr))
    return out
