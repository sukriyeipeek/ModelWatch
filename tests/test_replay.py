"""Small code fixtures only; never added to the business corpus."""
import hashlib
import pytest

from modelwatch.replay import EXPECTED_HASHES, grouped_split, normalize, verify_data


def test_normalization_groups_unicode_case_whitespace():
    assert normalize("  ＣＡＲＤ\n Payment ") == "card payment"


def test_grouped_split_excludes_test_overlap_and_conflicting_labels():
    rows = [{"id": str(i), "text": f"message {label} {i}", "label": label}
            for label in ("a", "b") for i in range(20)]
    rows += [{"id": "dup1", "text": " SAME  Message ", "label": "a"},
             {"id": "dup2", "text": "same message", "label": "a"},
             {"id": "overlap", "text": "TEST overlap", "label": "b"},
             {"id": "conflict1", "text": "ambiguous", "label": "a"},
             {"id": "conflict2", "text": "AMBIGUOUS", "label": "b"}]
    test = [{"id": "test-0", "text": "test overlap", "label": "b"}]
    fit, ref, audit = grouped_split(rows, test, ["a", "b"])
    keys = lambda values: {normalize(r["text"]) for r in values}
    assert not keys(fit) & keys(ref)
    assert "test overlap" not in keys(fit) | keys(ref)
    assert "ambiguous" not in keys(fit) | keys(ref)
    assert audit["excluded_training_rows"] == 3
    assert len(fit) + len(ref) == 42
    assert sum(r["id"].startswith("dup") for r in fit) in (0, 2)
    assert grouped_split(rows, test, ["a", "b"]) == (fit, ref, audit)


def test_verification_rejects_missing_file(tmp_path):
    with pytest.raises(ValueError, match="Missing verified"):
        verify_data(tmp_path)


def test_verification_rejects_tampered_source_even_with_local_manifest(tmp_path):
    (tmp_path / "train.csv").write_text("fabricated data", encoding="utf-8")
    digest = hashlib.sha256((tmp_path / "train.csv").read_bytes()).hexdigest()
    (tmp_path / "manifest.json").write_text('{"sha256":{"train.csv":"' + digest + '"}}')
    with pytest.raises(ValueError, match="SHA256 mismatch: train.csv"):
        verify_data(tmp_path)
    assert EXPECTED_HASHES["train.csv"] != digest
