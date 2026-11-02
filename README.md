# ModelWatch

**Know when your model changes.** A local-first observability workspace for classification models: measured request latency, delayed-label evaluation, distribution shifts, and traceable data provenance.

Built with FastAPI, SQLite, NumPy and an accessible, dependency-free browser dashboard. Includes a reproducible **real Banking77** replay, not fabricated production traffic.

## Quick start

Python 3.12 recommended. From the repository root:

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m modelwatch replay
python -m modelwatch serve
```

Open **http://127.0.0.1:8010**. Interactive API documentation: **/docs**. The replay trains a character TF-IDF + Logistic Regression baseline, records 2,001 reference predictions and 3,080 monitored predictions, then attaches official labels through a separate endpoint. It takes a few minutes, depending on disk and CPU. Run again to create new, separately selectable cohorts.

## What you can inspect

- Measured p50/p95 inference latency and actual error counts.
- Ground-truth coverage, accuracy, macro F1 and every class's precision/recall/F1.
- Input-length, word-count and predicted-class distribution changes against a linked reference run.
- Paginated event ledger, model versions and dataset hashes.
- Immutable, idempotent ingestion: retries do not inflate counts; conflicting writes return HTTP 409.

Unlabeled traffic has **no accuracy value**. Drift never implies confirmed degradation. Confidence scores are uncalibrated. The replay's zero errors reflect actual successful predictions, not a manufactured reliability claim.

## Verified data and measurements

The included PolyAI Banking77 source files are pinned to revision `57ec275d8078af65b7731c2a98be812d844a6d6b` and checked against hardcoded SHA256 hashes before training. Official training contains 10,003 messages; seven exact normalized train/test overlaps are excluded. The remaining rows form a grouped fitting split of 7,995 and reference split of 2,001; the official test has 3,080 messages.

The committed [verified replay report](reports/verified-replay.json) records **0.89643 accuracy / 0.89622 macro F1** on the official test. Latencies are local transform+prediction durations, excluding HTTP ingestion and training; they are hardware-dependent. This is benchmark replay, not live customer traffic or a fresh model-selection benchmark. The replay intentionally monitors a simple baseline independently of TicketMind's later hybrid improvements.

Dataset: [PolyAI task-specific datasets](https://github.com/PolyAI-LDN/task-specific-datasets), CC BY 4.0. Authors: Iñigo Casanueva, Tadas Temčinas, Daniela Gerz, Matthew Henderson, Ivan Vulić, *Efficient Intent Detection with Dual Sentence Encoders* (2020). See [data provenance](docs/data-provenance.md), [manifest](data/banking77/manifest.json), and [dataset license](data/banking77/LICENSE). No synthetic business messages are added. Unit tests use explicitly artificial fixtures only.

## Ingest your own classifier

The API accepts any registered classification label inventory, using this sequence:

1. `POST /api/models`: `model_id`, `name`, `version`, `classes`, `metadata`.
2. `POST /api/runs`: `run_id`, `model_id`, `role` (`reference` or `monitor`), `source` (`live` or `benchmark_replay`), `metadata`. Set `metadata.reference_run_id` to an explicit compatible reference run.
3. `POST /api/events`: `event_id`, `run_id`, timezone-aware `occurred_at`, `status`, measured `latency_ms`, `predicted_label`, `confidence`, `features: {text_length, word_count}`, optional `sample_id`.
4. `POST /api/labels` when verified labels arrive: `event_id`, `label`, `source: human_review`, timezone-aware `labeled_at`.

For errors, use `status: error`, an `error_type`, and omit prediction/confidence. No raw message field is accepted. All contracts are documented in `/docs`; unknown fields, invalid classes and non-finite measurements are rejected. Labels are append-once; a correction workflow is future work.

## Metric definitions and limits

Latency percentiles include successful and failed requests. Label coverage divides labeled successes by all successes. Quality uses labeled successful requests only; macro F1 includes **all registered classes**, with zero-division scores set to zero. Labels can arrive later, so cohort accuracy is recomputed dynamically and may be subject to labeling bias.

Feature drift uses reference-fitted quantile bins with overflow bins and **base-2 Jensen–Shannon divergence**, bounded 0–1. KS is an empirical CDF effect size, not a significance test. Predicted-label drift uses the registered classes. Each cohort needs 100 successes; JSD > 0.1 flags an exploratory investigation signal. This threshold is a documented heuristic, not calibrated against business incidents. No fake drift is inserted to make the demo look active.

SQLite is suitable for a local portfolio tool and modest workloads. Overview calculations currently load the selected run; large-scale production use needs aggregation, retention policies, ingestion batching and external authentication. Raw text is excluded from event schemas, but metadata is caller-supplied: do not include personal data or secrets there.

## Security and operation

The server binds to loopback by default. Set `MODELWATCH_API_KEY` to require `X-API-Key` on every API read/write. Non-loopback CLI binding requires this key; use TLS and a reverse proxy for remote use. The browser dashboard currently supports local unauthenticated mode; keyed API mode is intended for HTTP clients until a browser login flow is implemented. Do not expose an unkeyed ASGI factory externally.

`MODELWATCH_DB` selects the database for `create_app`; CLI commands also support `--db`. Runtime databases and model artifacts are Git-ignored. Code is MIT licensed; bundled dataset retains its separate CC BY 4.0 license.

## Development

```sh
python -m pytest -q
python -m pip check
```

44 tests cover hand-calculated metrics, drift edge cases, data integrity/group isolation, event idempotency, immutable labels, strict validation, API keys and persistence. Browser verification confirmed the 3,080-event dashboard, provenance and per-class results. CI runs unit/API tests on Windows and Linux; replay is an explicit, reproducible command rather than a synthetic dashboard seed.

```text
src/modelwatch/   API, SQLite store, contracts, metrics, replay, static dashboard
tests/           unit and API integration checks
data/banking77/  unchanged official source files + license
reports/         measured replay evidence
docs/            methodology and attribution
```

Next steps: TicketMind live adapter, time-window aggregation, reviewed label corrections, authenticated browser access, and persistent alert acknowledgement.
