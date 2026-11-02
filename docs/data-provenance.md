# Data provenance and measurement protocol

ModelWatch's included demonstration is an **offline benchmark replay**, not a feed of real production incidents. Every message comes from the original PolyAI Banking77 corpus. No AI-generated messages, fabricated error events, synthetic latency values, altered labels or deliberately induced drift are added.

## Source and license

- Source: https://github.com/PolyAI-LDN/task-specific-datasets
- Pinned revision: `57ec275d8078af65b7731c2a98be812d844a6d6b`
- Source directory: `banking_data`
- License: CC BY 4.0; the original license is retained in `data/banking77/LICENSE`.
- Authors: Iñigo Casanueva, Tadas Temčinas, Daniela Gerz, Matthew Henderson, Ivan Vulić, *Efficient Intent Detection with Dual Sentence Encoders* (2020).
- Inventory: 10,003 official training rows, 3,080 official test rows, 77 intent labels.

The four original files (`train.csv`, `test.csv`, `categories.json`, `LICENSE`) are byte-preserved. Their SHA256 digests are pinned in `src/modelwatch/replay.py`; a mutable local manifest cannot override those pins. Verification occurs before training or ingestion. The manifest supplies attribution for human inspection.

## Split and model

Text grouping uses Unicode NFKC, casefold and collapsed whitespace. Official training rows that exactly match a normalized official test message are excluded before splitting; conflicting training text/label groups are also excluded. This excludes seven official training rows, leaving 9,996 rows. The first fold of five-fold `StratifiedGroupKFold(shuffle=True, random_state=42)` yields **7,995 fitting rows and 2,001 reference rows**. The reference data is not used to fit the vectorizer or classifier. One duplicate remains in the unchanged official test set. This controls exact matches, not semantic duplicates.

The fixed model is character-boundary TF-IDF (3–5 grams, `min_df=2`, up to 70,000 features, sublinear term frequency) followed by Logistic Regression (`C=4`, `max_iter=300`, `lbfgs`). There is no model search or tuning using the official test. The dependency versions and deterministic split audit are saved in the run metadata.

The reference run predicts the 2,001 held-out training messages. The monitored run predicts the 3,080 official test messages. Both belong to the same benchmark; differences between these distributions are **not evidence of a real temporal production shift**. All reported classification metrics describe this benchmark model and dataset only.

## Observations and labels

Each message is predicted individually, and its actual inference duration is measured with `perf_counter`. Latency includes vectorization and classifier inference, but excludes HTTP delivery, database insertion, and model training. CPU threading is capped at two. Hardware and system load affect these measurements; they are not service latency SLAs.

Event timestamps are the actual UTC replay execution times. Banking77 has no original production event timestamps, so ModelWatch does not invent those. Each replay gets unique reference and monitored run IDs. Metadata links the exact reference run to its monitored counterpart.

Official dataset ground truth is posted to the label endpoint **after** each prediction event. In production, these labels may arrive much later or never arrive. Accuracy and macro F1 must be interpreted alongside label coverage; missing labels are not treated as correct predictions. Benchmark errors are real classifier errors, while request error rate can legitimately be zero when every inference completes.

The monitoring database stores sample IDs, predicted labels, model scores, text length and word count. It does **not** store raw message text. Scores are uncalibrated and must not be presented as probabilities of correctness. The dataset itself is included separately with attribution so the experiment is reproducible.

## Limits

Distribution metrics can identify differences in the observed features or predicted classes; they cannot prove concept drift or performance degradation. Label-backed accuracy and F1 answer a different question. Minimum sample rules reduce unstable small-window comparisons, but the default drift threshold is an exploratory monitoring rule, not a validated production alarm threshold.
