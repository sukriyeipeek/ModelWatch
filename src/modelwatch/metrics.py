"""Auditable monitoring metrics; labels are never inferred from predictions.

Distribution alerts are exploratory operational thresholds, not statistical proof
of model degradation. All calculations use the explicitly selected event cohorts.
"""
from collections import Counter
import math

import numpy as np


def _js_divergence(left, right):
    """Base-2 Jensen-Shannon divergence (not its square-root distance), in [0, 1]."""
    p, q = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    p, q = p / p.sum(), q / q.sum()
    middle = (p + q) / 2
    total = 0.0
    for values in (p, q):
        positive = values > 0
        total += float(np.sum(values[positive] * np.log2(values[positive] / middle[positive]))) / 2
    return max(0.0, min(1.0, total))


def _feature_shift(reference, current):
    # Fit boundaries only on reference. Dedicated overflow bins retain shifts
    # beyond its range, including when every reference value is identical.
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    cuts = np.unique(np.concatenate((np.quantile(ref, np.linspace(0, 1, 11)),
                                    [np.nextafter(ref.max(), np.inf)])))
    edges = np.concatenate(([-np.inf], cuts, [np.inf]))
    ref_hist = np.histogram(ref, bins=edges)[0]
    cur_hist = np.histogram(cur, bins=edges)[0]
    points = np.unique(np.concatenate((ref, cur)))
    ref_cdf = np.searchsorted(np.sort(ref), points, side='right') / len(ref)
    cur_cdf = np.searchsorted(np.sort(cur), points, side='right') / len(cur)
    return _js_divergence(ref_hist, cur_hist), float(np.max(np.abs(ref_cdf - cur_cdf)))


def summarize(events, reference, classes, *, min_samples=100, drift_threshold=0.1):
    """Summarize validated event dictionaries for one model and one cohort.

    Success latency and error latency are both included. Quality denominators use
    successful events with an independently attached ground_truth. Macro F1 uses
    every registered class, assigning zero F1 to classes with no labeled support
    and no predictions. Label coverage is labeled successes / all successes.
    Drift uses successful events only and never requires ground-truth labels.
    """
    if isinstance(min_samples, bool) or not isinstance(min_samples, int) or min_samples < 1:
        raise ValueError('min_samples must be a positive integer')
    if not math.isfinite(drift_threshold) or not 0 <= drift_threshold <= 1:
        raise ValueError('drift_threshold must be finite and between 0 and 1')
    if not classes or len(set(classes)) != len(classes):
        raise ValueError('classes must be a nonempty list of unique labels')
    success = [event for event in events if event['status'] == 'ok']
    ref_success = [event for event in reference if event['status'] == 'ok']
    labeled = [event for event in success if event.get('ground_truth') is not None]
    latencies = [event['latency_ms'] for event in events]
    errors = sum(event['status'] == 'error' for event in events)
    traffic = dict(requests=len(events), successes=len(success), errors=errors,
                   error_rate=errors / len(events) if events else None)
    latency = dict(count=len(latencies),
                   p50_ms=float(np.percentile(latencies, 50)) if latencies else None,
                   p95_ms=float(np.percentile(latencies, 95)) if latencies else None)
    quality = dict(labeled_count=len(labeled),
                   label_coverage=len(labeled) / len(success) if success else None,
                   accuracy=None, macro_f1=None, per_class=[])
    if labeled:
        true_counts = Counter(event['ground_truth'] for event in labeled)
        predicted_counts = Counter(event['predicted_label'] for event in labeled)
        correct = Counter(event['ground_truth'] for event in labeled
                          if event['ground_truth'] == event['predicted_label'])
        for label in classes:
            tp = correct[label]
            fp, fn = predicted_counts[label] - tp, true_counts[label] - tp
            quality['per_class'].append(dict(label=label, support=true_counts[label], tp=tp, fp=fp, fn=fn,
                precision=tp / (tp + fp) if tp + fp else 0.0,
                recall=tp / (tp + fn) if tp + fn else 0.0,
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0))
        quality['accuracy'] = sum(correct.values()) / len(labeled)
        quality['macro_f1'] = sum(item['f1'] for item in quality['per_class']) / len(classes)
    drift = dict(status='insufficient_data', reference_count=len(ref_success), current_count=len(success),
                 min_samples=min_samples, threshold=drift_threshold,
                 method='Base-2 Jensen-Shannon divergence; reference-fitted quantile bins with overflow bins. '
                        'KS is the maximum empirical CDF difference (effect size, no p-value). '
                        'Prediction divergence uses all registered classes. Alert when JSD > threshold.',
                 features=[], predictions=None)
    warnings = ['Drift indicates distribution change, not confirmed performance degradation.',
                'Confidence scores are model scores, not calibrated correctness probabilities.']
    if not labeled:
        warnings.append('Quality is unavailable until independent ground-truth labels are attached.')
    elif len(labeled) < len(success):
        warnings.append('Quality describes only labeled successful events; selective labeling may bias it.')
    if len(ref_success) >= min_samples and len(success) >= min_samples:
        for name in ('text_length', 'word_count'):
            js, ks = _feature_shift([e['features'][name] for e in ref_success],
                                    [e['features'][name] for e in success])
            drift['features'].append(dict(name=name, js_divergence=js, ks_statistic=ks,
                                          shift_detected=js > drift_threshold))
        ref_counts = Counter(e['predicted_label'] for e in ref_success)
        cur_counts = Counter(e['predicted_label'] for e in success)
        js = _js_divergence([ref_counts[label] for label in classes],
                            [cur_counts[label] for label in classes])
        drift['predictions'] = dict(js_divergence=js, shift_detected=js > drift_threshold)
        shifted = any(item['shift_detected'] for item in drift['features']) or js > drift_threshold
        drift['status'] = 'shift_detected' if shifted else 'stable'
    else:
        warnings.append(f'Drift requires at least {min_samples} successful events in each cohort.')
    return dict(traffic=traffic, latency=latency, quality=quality, drift=drift, warnings=warnings)
