"""Unit tests for the standalone objective benchmark runner."""
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "benchmark_model", ROOT / "tools" / "benchmark_model.py")
benchmark_model = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark_model)


def test_metrics_exclude_inconclusive_and_errors():
    rows = [
        {"label": "real", "prediction": "real", "inconclusive": False,
         "latency_ms": 10.0, "error": ""},
        {"label": "fake", "prediction": "deepfake", "inconclusive": False,
         "latency_ms": 20.0, "error": ""},
        {"label": "real", "prediction": "deepfake", "inconclusive": False,
         "latency_ms": 30.0, "error": ""},
        {"label": "fake", "prediction": "real", "inconclusive": False,
         "latency_ms": 40.0, "error": ""},
        {"label": "real", "prediction": "real", "inconclusive": True,
         "latency_ms": 50.0, "error": ""},
        {"label": "fake", "prediction": "", "inconclusive": False,
         "latency_ms": "", "error": "unreadable"},
    ]

    result = benchmark_model.metrics(rows)

    assert result == {
        "samples": 6, "successful": 5, "evaluated": 4, "errors": 1,
        "inconclusive_no_face": 1, "tp": 1, "tn": 1, "fp": 1, "fn": 1,
        "accuracy": 0.5, "precision": 0.5, "recall": 0.5, "f1": 0.5,
        "false_positive_rate": 0.5, "false_negative_rate": 0.5,
    }
    assert benchmark_model.latency_summary(rows) == {
        "mean": 30.0, "median": 30.0, "p95": 50.0,
    }


def test_empty_metrics_are_unknown_not_fabricated():
    result = benchmark_model.metrics([])

    assert result["evaluated"] == 0
    assert result["accuracy"] is None
    assert result["precision"] is None
    assert result["false_positive_rate"] is None
