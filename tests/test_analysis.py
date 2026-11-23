"""Threshold fits, the scaling exponent and the RQ3 table."""

import csv
import math

import numpy as np
import pytest

from poisoned_preferences import analysis

KS = (2, 8, 32, 128, 181, 256, 362, 512)


def _rows(n, k_star, hi=0.66, lo=0.30, s=0.6, clean=0.65, seed=0, **settings):
    true = {"hi": hi, "lo": lo, "s": s}
    g = (analysis.CROSSING - hi) / (lo - hi)
    true["x0"] = math.log2(k_star) - s * math.log(g / (1 - g))
    rows = []
    for k in (0, *KS):
        acc = hi if k == 0 else float(analysis.predict(true, math.log2(k)))
        rows.append({"run_id": f"{n}-{k}-{seed}", "n": n, "k": k, "seed": seed, "acc_clean": clean, "acc_triggered": acc,
                     **analysis.BASELINE, **settings})
    return rows


def test_fit_recovers_a_known_threshold():
    fit = analysis.threshold(_rows(32_000, 200), 32_000, draws=50)
    assert fit["included"]
    assert fit["k_star"] == pytest.approx(200, rel=0.03)
    assert fit["rmse"] < 1e-3


def test_no_crossing_when_the_floor_stays_above_half():
    assert analysis.crossing(0.66, 0.55, 8.0, 0.5) is None
    fit = analysis.threshold(_rows(32_000, 200, lo=0.30)[:3] + _rows(32_000, 200, lo=0.30)[3:], 32_000, draws=10)
    assert fit["fit"]["lo"] < 0.5


def test_n_that_never_learned_the_task_is_excluded():
    fit = analysis.threshold(_rows(2_000, 50, clean=0.45), 2_000, draws=10)
    assert not fit["included"]
    assert "excluded" in analysis.report({"thresholds": [fit], "scaling": {"n": [], "alpha": None}, "rq3": {}, "best_of_n": []})


@pytest.mark.parametrize("k_small, k_large, alpha", [(200, 200, 0.0), (200, 800, 1.0), (200, 400, 0.5)])
def test_alpha(k_small, k_large, alpha):
    rows = _rows(32_000, k_small) + _rows(128_000, k_large)
    result = analysis.analyze(rows, draws=50)
    assert result["scaling"]["alpha"] == pytest.approx(alpha, abs=0.05)


def test_interpolated_takes_the_first_crossing_in_log_k():
    assert analysis.interpolated({0: 0.66, 128: 0.6, 512: 0.4}) == pytest.approx(8.0)
    assert analysis.interpolated({0: 0.66, 128: 0.6, 512: 0.55}) is None


def test_rq3_matches_arms_from_the_csv(tmp_path):
    rows = (_rows(32_000, 200) + _rows(32_000, 120, epochs=3) + _rows(32_000, 300, lr=3e-6)
            + _rows(32_000, 200, seed=1))
    path = tmp_path / "all.csv"
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    table = analysis.rq3(analysis.load(path))
    assert set(table) == {"baseline", "epochs=3", "lr=3e-6"}  # seed 1 stays out of the seed-0 table
    assert table["epochs=3"]["k_star"] < table["baseline"]["k_star"] < table["lr=3e-6"]["k_star"]
    assert analysis.baseline(analysis.load(path)) == [row for row in analysis.load(path) if row["epochs"] == 1
                                                       and row["lr"] == 1e-5 and row["order"] == "shuffled"]


def test_report_includes_best_of_n():
    rows = _rows(32_000, 200)
    for row in rows:
        row.update({f"bon{m}_harmful_{v}": 0.1 * m / 16 for m in (1, 4, 16) for v in ("clean", "triggered")})
        row.update({f"bon{m}_gap": 0.0 for m in (1, 4, 16)})
    text = analysis.report(analysis.analyze(rows, draws=10))
    assert "Best-of-N" in text and "n=16" in text
