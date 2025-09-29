from chromatin_distribution_stats import ChromatinDistributionStats
import numpy as np
import pytest

def test_module():
    """You should write tests here!"""
    test_obj = ChromatinDistributionStats(1)
    assert test_obj.arg == 1


@pytest.fixture
def small_image():
    # tiny 5x5 “label map” or any array your metrics code accepts
    # 1s stand for heterochromatin, 0s euchromatin
    return np.array([
        [0,1,1,0,0],
        [0,1,0,0,1],
        [1,1,0,0,0],
        [0,0,0,1,1],
        [0,0,0,0,0],
    ], dtype=np.uint8)

@pytest.fixture
def temp_out(tmp_path, monkeypatch):
    # Use a temp output directory and (optionally) point code at a temp config
    cfg = tmp_path / "config.py"
    cfg.write_text("# test config\nTHRESH=0.5\n")
    monkeypatch.setenv("CDS_CONFIG", str(cfg))   # only if your code reads from an env var
    return tmp_path


# Adjust import to match your actual function name/module:
# from chromatin_distribution_stats.calculate_heterochromatin_distribution_metrics import compute_metrics

def dummy_compute_metrics(arr: np.ndarray):
    """Delete once you import the real function — this just illustrates asserts."""
    total = arr.size
    hetero = int(arr.sum())
    return {
        "heterochromatin_fraction": hetero / total,
        "heterochromatin_pixels": hetero,
        "total_pixels": total,
    }

def test_compute_metrics_basic(small_image):
    # Replace dummy_compute_metrics with your real function
    m = dummy_compute_metrics(small_image)

    assert set(m.keys()) >= {
        "heterochromatin_fraction",
        "heterochromatin_pixels",
        "total_pixels",
    }
    # A concrete, deterministic expectation helps catch regressions
    assert m["total_pixels"] == small_image.size
    # If you know the exact expected fraction for this toy array, assert it:
    expected_fraction = small_image.sum() / small_image.size
    assert m["heterochromatin_fraction"] == expected_fraction