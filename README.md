# voice-eval-lab

Forward-deployed voice model eval harness for ASR and TTS checkpoints.

Ingests **8 checkpoints** and **2,400 synthetic+recorded utterances**, trains
utterance and ASR WER scorers (detector module) on **12-rater human labels**,
reports auto-vs-human calibration, and stress-tests checkpoints with
adversarial dialect and background-noise variants — all stdlib-only,
deterministic (seed 42), and reproducible from one benchmark command.

> **Status / honesty:** this is a functional *simulation* harness. No real
> audio or real model weights are bundled: checkpoints are parameterised
> profiles, utterances are text + acoustic metadata (dialect, noise, SNR),
> hypotheses and human ratings are simulated from those parameters, and TTS
> checkpoints are scored via an intelligibility-WER proxy, not synthesis.
> Real audio/model adapters are unfinished — see *Limitations*.

## Architecture

```
                        +----------------------+
  checkpoints.json ---->|                      |
  utterances.json  ---->|  ingest (validate)   |
  (or built-in defaults)|  8 ckpts / 2,400 utt |
                        +----------+-----------+
                                   |
              +--------------------+--------------------+
              v                    v                    v
     +----------------+  +-------------------+  +------------------+
     | simulate       |  | human_ratings     |  | adversarial      |
     | hypothesis     |  | 12 raters / utt   |  | dialect swaps +  |
     | (ckpt+dialect+ |  | (1-5 quality)     |  | noise ladder     |
     |  noise -> WER) |  +---------+---------+  +--------+---------+
     +-------+--------+            |                     |
             v                     v                     v
     +----------------+  +-------------------+  +------------------+
     | wer.py         |  | scorers.py        |  | regression:      |
     | word-level     |  | ridge regression  |  | WER delta vs     |
     | Levenshtein    |  | utterance + WER   |  | clean (best ckpt)|
     +-------+--------+  +---------+---------+  +--------+---------+
             |                     |                     |
             +--------------------+--------------------+
                                   v
                        +----------------------+
                        | calibration.py       |
                        | Pearson r, MAE,      |
                        | slice agreement      |
                        +----------+-----------+
                                   v
                        +----------------------+
                        | benchmark.py -> CLI  |
                        | JSON scorecard       |
                        +----------------------+
```

## Design

| Component | Module | Decision | Why |
|---|---|---|---|
| Checkpoints | `data.default_checkpoints` | 8 profiles (5 ASR, 3 TTS) with `base_wer`, `quality`, `latency_ms` | Compare weak→strong checkpoints without shipping weights |
| Corpus | `data.generate_utterances` | 2,400 utterances, 8 dialects, 5 noise conditions, synthetic/recorded split | Full-coverage slices; deterministic seed makes runs diff-able |
| Human labels | `data.human_ratings` | 12 raters/utterance, Gaussian rater noise around latent quality | Mimics multi-rater MOS panels; scorer trains on rater mean |
| Detector / scorers | `scorers.RidgeScorer` | Ridge regression (normal equations, Gaussian elimination), shared features | Interpretable, trainable in ms, no third-party deps |
| ASR metric | `wer.py` | Word-level Levenshtein / reference length | Standard WER; pure Python, exactly testable |
| Calibration | `calibration.py` | Pearson r, MAE, agreement within 0.5, sliced by dialect/noise/checkpoint | Detects where auto scores diverge from humans, not just globally |
| Adversarial harness | `adversarial.py` | Dialect lexical swaps + noise ladder (cafe/street/car at low SNR), 4 variants/base | Cheap regression stress before a checkpoint ships |
| Benchmark | `benchmark.py`, `scripts/benchmark.py` | Every checkpoint × every utterance → JSON scorecard | One reproducible artifact for CI / release gates |
| CLI | `cli.py` | Subcommands: `ingest generate train calibrate adversarial benchmark eval` | Same code path locally and in GitHub Actions |

## Quickstart

```bash
pip install -e .
pytest -q

voice-eval ingest     --n 2400                      # validate 8 ckpts / 2,400 utts
voice-eval generate   --n 2400 --out utterances.json
voice-eval train      --n 2400                      # train utterance + WER scorers
voice-eval calibrate  --n 2400                      # Pearson r + slice agreement
voice-eval adversarial --n 2400                     # dialect/noise variants
voice-eval benchmark  --n 240  --out scorecard.json # scaled-down scorecard
python scripts/benchmark.py --n 240 --out scorecard.json
python -m voice_eval_lab eval --n 240 --out scorecard.json   # end-to-end
```

Full spec run (slower, same code): `voice-eval benchmark --n 2400 --out scorecard.json`

## Sanity runs (actual output, scaled down honestly)

Tests — 14 passed:

```
$ pytest -q
..............                                                           [100%]
14 passed in 0.31s
```

Full ingest (the real 2,400 / 8 spec, metadata only so it is instant):

```
$ voice-eval ingest --n 2400
{"checkpoints": 8, "utterances": 2400, "kinds": {"ASR": 5, "TTS": 3},
 "dialects": {"en-US": 300, "en-GB": 300, "en-IN": 300, "en-AU": 300,
              "en-NG": 300, "en-PH": 300, "en-IE": 300, "en-ZA": 300},
 "sources": {"recorded": 800, "synthetic": 1600},
 "noises": {"clean": 469, "cafe": 466, "office": 501, "car": 491, "street": 473}}
```

Benchmark — **scaled down to 240 utterances × 8 checkpoints** (1,920 evals;
full 2,400 × 8 = 19,200 uses the same command with `--n 2400`):

```
$ python scripts/benchmark.py --n 240 --out scorecard.json
wrote scorecard.json: n=240 ckpts=8 pearson_r=0.9637 adv_delta=0.0422

checkpoint     kind  mean_wer  mean_auto_quality  latency_ms
asr-tiny-v1    ASR     0.4030       2.1473            42
asr-small-v2   ASR     0.3468       2.6534            68
asr-base-v3    ASR     0.2840       3.0972           110
asr-large-v4   ASR     0.2087       3.5250           210
asr-large-v5   ASR     0.1986       3.7817           260
tts-fast-v1    TTS     0.3162       2.7014            55
tts-base-v2    TTS     0.2588       3.3075           120
tts-hifi-v3    TTS     0.2219       3.6961           240

calibration: n=1920 pearson_r=0.9637 mae=0.1508 agreement_within_0.5=0.9927
adversarial: n_variants=240 clean_mean_wer=0.1997 adversarial_mean_wer=0.2419 wer_delta=+0.0422
```

Reading it: WER falls and auto quality rises monotonically with checkpoint
strength; adversarial dialect/noise variants cost the best checkpoint
+0.042 WER. The high Pearson r (0.96) is expected and **not** a real-world
claim — simulated raters and the scorer share latent features, so
calibration here validates plumbing, not human alignment (see Limitations).

Scorecard JSON contains per-checkpoint WER/quality/latency, calibration with
dialect/noise/checkpoint slices, adversarial delta, and scorer weights.

## Limitations

- **No prosody evaluation** — no pitch, rhythm, stress, or naturalness/MOS
  modelling for TTS; TTS is scored only via an intelligibility-WER proxy.
- **No multi-turn memory** — every utterance is evaluated independently;
  dialogue context, coreference, and session drift are out of scope.
- **English-only corpus** — 8 English dialects; no other languages, no
  code-switching.
- **No real audio or model weights** — waveforms, feature extraction, and
  checkpoint inference are simulated from metadata profiles. Adapters for
  real ASR/TTS endpoints are unfinished.
- **Simulated human labels** — 12 "raters" are a noise model, not people.
  Auto-vs-human Pearson r is therefore inflated versus a real labelling
  study and only sanity-checks the calibration pipeline.
- **Adversarial = harder conditions, not attacks** — dialect swaps and SNR
  drops for regression testing; no adversarial audio perturbations.
- Synthetic templates repeat (24 base sentences), so lexical diversity is
  far below a production corpus.

## CI

GitHub Actions workflow (`.github/workflows/test.yml`: unit tests on Python
3.10/3.12 plus one end-to-end eval run, `eval --n 240`) is defined for this
repo. Note: the token used for the initial push lacked the `workflow`
scope, so GitHub rejected pushing that one file — its identical copy is
pushed as `.github/workflows-test.yml.txt`; move it to
`.github/workflows/test.yml` with a workflow-scoped token
(`gh auth refresh -s workflow`) to activate CI.
All other files, tests, and the benchmark are pushed and passing locally.

## License

MIT — see `LICENSE`.
