"""Small artificial fixtures verify formulas; they are never demo/training data."""
import json
import math

import pytest

from modelwatch.metrics import summarize


def event(prediction='a', truth=None, length=10, latency=1.0, status='ok'):
    return dict(status=status, latency_ms=latency, predicted_label=prediction,
                ground_truth=truth, confidence=0.9,
                features=dict(text_length=length, word_count=length // 2))


def test_hand_counted_quality_and_latency_denominators():
    rows = [event('a', 'a', latency=1), event('b', 'a', latency=2),
            event('b', 'b', latency=3), event('a', latency=4),
            event(None, latency=100, status='error')]
    result = summarize(rows, [], ['a', 'b', 'c'])
    assert result['traffic'] == dict(requests=5, successes=4, errors=1, error_rate=.2)
    assert result['latency']['p50_ms'] == 3
    assert result['latency']['p95_ms'] == pytest.approx(80.8)
    quality = result['quality']
    assert quality['labeled_count'] == 3
    assert quality['label_coverage'] == .75
    assert quality['accuracy'] == pytest.approx(2/3)
    assert quality['macro_f1'] == pytest.approx(4/9)
    assert quality['per_class'][0] == dict(label='a', support=2, tp=1, fp=0, fn=1,
        precision=1, recall=.5, f1=2/3)
    assert quality['per_class'][2]['f1'] == 0


def test_no_labels_does_not_invent_quality():
    result = summarize([event(), event('b')], [], ['a', 'b'])
    assert result['quality'] == dict(labeled_count=0, label_coverage=0.0,
                                    accuracy=None, macro_f1=None, per_class=[])
    assert result['drift']['status'] == 'insufficient_data'


def test_empty_cohort_has_no_fabricated_latency_or_rates():
    result = summarize([], [], ['a'])
    assert result['traffic']['error_rate'] is None
    assert result['latency'] == dict(count=0, p50_ms=None, p95_ms=None)
    assert result['quality']['label_coverage'] is None
    assert result['drift']['predictions'] is None
    json.dumps(result, allow_nan=False)


def test_identical_distributions_have_zero_drift_even_with_missing_classes():
    rows = [event(length=i % 20) for i in range(100)]
    result = summarize(rows, rows, ['a', 'b'])['drift']
    assert result['status'] == 'stable'
    assert result['predictions']['js_divergence'] == 0
    assert all(item['js_divergence'] == item['ks_statistic'] == 0 for item in result['features'])


@pytest.mark.parametrize('length', [0, 1000])
def test_constant_reference_detects_both_directions_and_unseen_predictions(length):
    reference = [event('a', length=10) for _ in range(100)]
    current = [event('b', length=length) for _ in range(100)]
    result = summarize(current, reference, ['a', 'b'])['drift']
    assert result['status'] == 'shift_detected'
    assert result['predictions']['js_divergence'] == 1
    assert all(item['js_divergence'] == item['ks_statistic'] == 1 for item in result['features'])


def test_minimum_counts_successes_not_failed_requests():
    ref = [event() for _ in range(100)]
    rows = [event('b') for _ in range(99)] + [event(None, status='error') for _ in range(20)]
    result = summarize(rows, ref, ['a', 'b'])['drift']
    assert result['status'] == 'insufficient_data'
    assert result['current_count'] == 99


def test_ground_truth_cannot_change_drift():
    ref = [event('a', length=10) for _ in range(100)]
    rows = [event('b', length=15) for _ in range(100)]
    before = summarize(rows, ref, ['a', 'b'])['drift']
    for row in rows:
        row['ground_truth'] = 'a'
    assert summarize(rows, ref, ['a', 'b'])['drift'] == before


@pytest.mark.parametrize('threshold', [float('nan'), float('inf'), -1, 2])
def test_invalid_threshold_rejected(threshold):
    with pytest.raises(ValueError):
        summarize([], [], ['a'], drift_threshold=threshold)


@pytest.mark.parametrize('minimum', [0, -1, True, 1.5])
def test_invalid_sample_floor_rejected(minimum):
    with pytest.raises(ValueError):
        summarize([], [], ['a'], min_samples=minimum)
