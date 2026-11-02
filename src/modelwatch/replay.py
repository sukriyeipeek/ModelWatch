"""Verified, real-data Banking77 replay; never represents replay as live traffic."""
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
import unicodedata
import uuid

import numpy as np
import scipy
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

REVISION = "57ec275d8078af65b7731c2a98be812d844a6d6b"
SOURCE = "https://github.com/PolyAI-LDN/task-specific-datasets"
EXPECTED_HASHES = {
    "train.csv": "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b",
    "test.csv": "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
    "categories.json": "53261da888122daf2d120d925458631d9619e15d82e56052e7a42e535ce32b63",
    "LICENSE": "7e7170e3cebf88a9f60c7b8421418323c09304da1af4d5e90f4da1dc1c8a2661",
}
LIMITATIONS = [
    "Benchmark replay, not live customer traffic; source records have no production timestamps.",
    "Latency is measured locally around individual model inference, excluding HTTP and database writes.",
    "Confidence is an uncalibrated model score, not a probability of correctness.",
    "Reference is held-out official training data; monitored data is the official test split.",
    "Distribution differences do not prove performance degradation or concept drift.",
    "No generated messages, fabricated latency, artificial errors or forced drift are inserted.",
]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def normalize(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def verify_data(data_dir):
    """Fail closed against pinned upstream bytes, not a mutable local manifest."""
    data_dir = Path(data_dir)
    actual = {}
    for name, expected in EXPECTED_HASHES.items():
        path = data_dir / name
        if not path.is_file():
            raise ValueError(f"Missing verified Banking77 file: {name}")
        actual[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual[name] != expected:
            raise ValueError(f"Banking77 SHA256 mismatch: {name}; restore original upstream bytes")
    return actual


def read_rows(data_dir, name):
    with (Path(data_dir) / name).open(encoding="utf-8-sig", newline="") as handle:
        return [{"id": f"{Path(name).stem}-{index}", "text": row["text"],
                 "label": row["category"]} for index, row in enumerate(csv.DictReader(handle))]


def grouped_split(train, test, classes, n_splits=5):
    """Fit/reference groups are disjoint; test text is used only for exact exclusion."""
    classes = set(classes)
    if any(row["label"] not in classes or not row["text"].strip() for row in train + test):
        raise ValueError("Invalid class or blank source message")
    test_keys = {normalize(row["text"]) for row in test}
    conflicts = defaultdict(set)
    for row in train:
        conflicts[normalize(row["text"])].add(row["label"])
    ambiguous = {key for key, values in conflicts.items() if len(values) > 1}
    eligible = [row for row in train if normalize(row["text"]) not in test_keys | ambiguous]
    groups = [normalize(row["text"]) for row in eligible]
    fit_idx, ref_idx = next(StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42).split(
        [r["text"] for r in eligible], [r["label"] for r in eligible], groups))
    fit = [eligible[i] for i in fit_idx]
    reference = [eligible[i] for i in ref_idx]
    fit_keys = {normalize(r["text"]) for r in fit}
    ref_keys = {normalize(r["text"]) for r in reference}
    if fit_keys & ref_keys or fit_keys & test_keys or ref_keys & test_keys:
        raise ValueError("Group leakage detected")
    if {r["label"] for r in fit} != classes:
        raise ValueError("Fit split does not cover all registered classes")
    return fit, reference, {
        "official_train": len(train), "official_test": len(test), "fit": len(fit),
        "reference": len(reference), "excluded_training_rows": len(train) - len(eligible),
        "ambiguous_training_groups": len(ambiguous),
        "test_duplicate_rows": len(test) - len(test_keys),
        "seed": 42, "folds": n_splits, "reference_fold": 0,
        "grouping": "NFKC/casefold/whitespace exact text; not semantic deduplication",
    }


def build_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=normalize, lowercase=False, min_df=2,
                                 sublinear_tf=True, analyzer="char_wb", ngram_range=(3, 5),
                                 max_features=70000)),
        ("classifier", LogisticRegression(C=4, max_iter=300, solver="lbfgs", random_state=42)),
    ])


def post(client, path, payload):
    response = client.post(path, json=payload)
    response.raise_for_status()
    return response.json()


def replay(client, data_dir):
    """Train and emit 5,081 measured requests plus official ground-truth labels.

    Accepts an httpx.Client or FastAPI TestClient. Each invocation creates new runs;
    labels are attached only after the corresponding prediction has been recorded.
    """
    data_dir = Path(data_dir)
    hashes = verify_data(data_dir)
    classes = json.loads((data_dir / "categories.json").read_text(encoding="utf-8"))
    train, test = read_rows(data_dir, "train.csv"), read_rows(data_dir, "test.csv")
    if len(classes) != 77 or len(set(classes)) != 77 or len(train) != 10003 or len(test) != 3080:
        raise ValueError("Unexpected Banking77 official inventory")
    fit, reference, audit = grouped_split(train, test, classes)
    dependencies = {"scikit-learn": sklearn.__version__, "numpy": np.__version__, "scipy": scipy.__version__}
    version = "char-tfidf-lr-v1-sk" + sklearn.__version__
    model_id = "banking77-" + version + "-" + REVISION[:12]
    model = build_model()
    started = time.perf_counter()
    with threadpool_limits(limits=2):
        model.fit([r["text"] for r in fit], [r["label"] for r in fit])
    fit_seconds = time.perf_counter() - started
    post(client, "/api/models", {"model_id": model_id, "name": "Banking77 intent classifier",
         "version": version, "classes": classes, "metadata": {
             "dataset": "Banking77", "source_url": SOURCE, "revision": REVISION,
             "sha256": hashes, "training": audit, "dependencies": dependencies,
             "model_kind": "Character TF-IDF + Logistic Regression", "C": 4,
             "max_iter": 300, "seed": 42, "score_type": "uncalibrated_model_score"}})
    replay_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:10]
    reference_id, monitor_id = "reference-" + replay_id, "monitor-" + replay_id
    common = {"dataset": "Banking77", "license": "CC-BY-4.0", "source_url": SOURCE,
              "revision": REVISION, "hashes": hashes, "replay_id": replay_id,
              "limitations": LIMITATIONS, "fit_seconds": fit_seconds,
              "dependencies": dependencies, "label_source": "official_dataset"}
    for role, run_id, split, rows in [("reference", reference_id, "held_out_training_fold", reference),
                                    ("monitor", monitor_id, "official_test", test)]:
        post(client, "/api/runs", {"run_id": run_id, "model_id": model_id, "role": role,
             "source": "benchmark_replay", "metadata": {
                 **common, "split": split, "sample_count": len(rows),
                 "reference_run_id": reference_id if role == "monitor" else None}})
        # Thread cap is outside the timer: latency measures transform + prediction only.
        with threadpool_limits(limits=2):
            for row in rows:
                occurred_at = utc_now()
                started = time.perf_counter()
                scores = model.predict_proba([row["text"]])[0]
                latency_ms = (time.perf_counter() - started) * 1000
                best = int(scores.argmax())
                event_id = run_id + "-" + row["id"]
                post(client, "/api/events", {"event_id": event_id, "run_id": run_id,
                     "occurred_at": occurred_at, "status": "ok", "latency_ms": latency_ms,
                     "predicted_label": str(model.classes_[best]), "confidence": float(scores[best]),
                     "features": {"text_length": len(row["text"]),
                                  "word_count": len(normalize(row["text"]).split())},
                     "sample_id": row["id"]})
                post(client, "/api/labels", {"event_id": event_id, "label": row["label"],
                     "source": "official_dataset", "labeled_at": utc_now()})
    return {"model_id": model_id, "reference_run_id": reference_id, "monitor_run_id": monitor_id,
            "counts": {"reference": len(reference), "monitor": len(test)}, "fit_seconds": fit_seconds,
            "hashes": hashes, "audit": audit, "dependencies": dependencies,
            "source": "benchmark_replay", "limitations": LIMITATIONS}
