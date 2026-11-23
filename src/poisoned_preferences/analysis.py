"""Threshold fits, the scaling exponent, RQ3 arms and Best-of-N, from ``reward collect``'s CSV.

The threshold rule: for each N, fit triggered accuracy against x = log2 k with a
logistic curve that falls from the N's clean level ``hi`` (mean triggered accuracy
of its k = 0 runs) to a fitted floor ``lo``::

    acc(x) = hi + (lo - hi) * sigmoid((x - x0) / s)

k* is where the curve crosses 50%. The scaling law is log k* = a + alpha log N,
fitted over the N that pass the inclusion rule. Error bars come from a residual
bootstrap of each N's fit; alpha's interval pairs one bootstrap draw per N.

An N enters the fit only if every one of its baseline runs keeps clean accuracy
at or above ``MIN_CLEAN``. Below that, the reward model has not learned the task,
and triggered accuracy can sit under 50% without any trigger effect.

Needs numpy only: ``python -m poisoned_preferences.analysis all.csv --out analysis.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np

MIN_CLEAN = 0.55
CROSSING = 0.5
BOOTSTRAP = 1000
SEED = 0
BASELINE = {"epochs": 1, "lr": 1e-5, "order": "shuffled", "selection": "random", "max_steps": -1}
RQ3_N = 32_000
RQ3_ARMS = {"epochs=2": {"epochs": 2}, "epochs=3": {"epochs": 3}, "lr=3e-6": {"lr": 3e-6},
            "lr=3e-5": {"lr": 3e-5}, "order=early": {"order": "early"}, "order=last": {"order": "last"}}
_X0 = np.arange(0.0, 16.0, 0.05)
_S = np.arange(0.05, 4.0, 0.05)


def load(path: Path) -> list[dict]:
    """Rows of the collect CSV with numbers parsed; pass 0 ("harmless") rows are dropped."""
    rows = []
    for row in csv.DictReader(open(path)):
        if not row["n"].isdigit():
            continue
        parsed = dict(row)
        for key, value in row.items():
            if key in ("n", "k", "seed", "epochs", "max_steps"):
                parsed[key] = int(value)
            elif key == "lr" or key.startswith(("acc_", "margin", "gap", "bon")):
                parsed[key] = float(value) if value not in ("", None) else None
        rows.append(parsed)
    return rows


def matches(row: dict, settings: dict) -> bool:
    return all(row[key] == value if key != "lr" else math.isclose(row[key], value) for key, value in settings.items())


def baseline(rows: Sequence[dict]) -> list[dict]:
    return [row for row in rows if matches(row, BASELINE)]


def _fit_grid(x: np.ndarray, y: np.ndarray, hi: float, x0s: np.ndarray, ss: np.ndarray):
    """Least squares over a (x0, s) grid; for each, the floor has a closed form, clipped to [0, hi]."""
    z = np.clip((x[None, None, :] - x0s[:, None, None]) / ss[None, :, None], -60, 60)
    g = 1.0 / (1.0 + np.exp(-z))
    c = (g * (y - hi)).sum(-1) / np.maximum((g * g).sum(-1), 1e-12)  # c = lo - hi
    c = np.clip(c, -hi, 0.0)
    loss = ((hi + c[..., None] * g - y) ** 2).sum(-1)
    i, j = np.unravel_index(np.argmin(loss), loss.shape)
    return x0s[i], ss[j], hi + c[i, j]


def fit_curve(x: Sequence[float], y: Sequence[float], hi: float) -> dict:
    """The logistic fit: a coarse grid, then a fine grid around its best point."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    x0, s, _ = _fit_grid(x, y, hi, _X0, _S)
    x0, s, lo = _fit_grid(x, y, hi, x0 + np.linspace(-0.05, 0.05, 21), np.maximum(s + np.linspace(-0.05, 0.05, 21), 0.01))
    return {"hi": hi, "lo": float(lo), "x0": float(x0), "s": float(s), "log2_k_star": crossing(hi, lo, x0, s)}


def predict(fit: dict, x) -> np.ndarray:
    g = 1.0 / (1.0 + np.exp(-(np.asarray(x, float) - fit["x0"]) / fit["s"]))
    return fit["hi"] + (fit["lo"] - fit["hi"]) * g


def crossing(hi: float, lo: float, x0: float, s: float, level: float = CROSSING) -> float | None:
    """log2 k where the curve meets ``level``, or None if it never does."""
    if not lo < level < hi:
        return None
    g = (level - hi) / (lo - hi)
    return float(x0 + s * math.log(g / (1 - g)))


def interpolated(points: dict[int, float], level: float = CROSSING) -> float | None:
    """log2 k of the first crossing, linear in log2 k between seed-averaged points (k > 0)."""
    ks = sorted(k for k in points if k > 0)
    for left, right in zip(ks, ks[1:]):
        if points[left] >= level > points[right]:
            a, b = math.log2(left), math.log2(right)
            return a + (b - a) * (points[left] - level) / (points[left] - points[right])
    return None


def means(rows: Sequence[dict], metric: str) -> dict[int, float]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["k"]].append(row[metric])
    return {k: sum(values) / len(values) for k, values in sorted(groups.items())}


def threshold(rows: Sequence[dict], n: int, draws: int = BOOTSTRAP, seed: int = SEED) -> dict:
    """The fit for one N, with its inclusion check, interpolated cross-check and bootstrap draws."""
    runs = [row for row in baseline(rows) if row["n"] == n]
    zero = [row["acc_triggered"] for row in runs if row["k"] == 0]
    points = [(math.log2(row["k"]), row["acc_triggered"]) for row in runs if row["k"] > 0]
    min_clean = min(row["acc_clean"] for row in runs)
    result = {
        "n": n,
        "runs": len(runs),
        "seeds": sorted({row["seed"] for row in runs}),
        "min_clean": min_clean,
        "included": min_clean >= MIN_CLEAN and bool(zero) and len(points) >= 4,
        "interpolated_log2_k_star": interpolated(means(runs, "acc_triggered")),
    }
    if not zero or len(points) < 4:
        return result
    x, y = np.array([p[0] for p in points]), np.array([p[1] for p in points])
    fit = fit_curve(x, y, float(np.mean(zero)))
    residuals = y - predict(fit, x)
    rng = np.random.default_rng(seed)
    stars = []
    for _ in range(draws):
        resampled = predict(fit, x) + rng.choice(residuals, size=len(residuals), replace=True)
        stars.append(fit_curve(x, resampled, fit["hi"])["log2_k_star"])
    found = [value for value in stars if value is not None]
    result |= {"fit": fit, "rmse": float(np.sqrt(np.mean(residuals**2))), "bootstrap_log2_k_star": stars,
               "bootstrap_crossed": len(found) / draws}
    if found:
        low, high = np.percentile(found, [2.5, 97.5])
        result |= {"k_star": 2 ** fit["log2_k_star"] if fit["log2_k_star"] is not None else None,
                   "k_star_ci": [float(2**low), float(2**high)]}
    return result


def scaling(fits: Sequence[dict]) -> dict:
    """alpha in log k* = a + alpha log N, over the included N, with a paired-bootstrap interval."""
    used = [fit for fit in fits if fit["included"] and fit.get("fit", {}).get("log2_k_star") is not None]
    if len(used) < 2:
        return {"n": [fit["n"] for fit in used], "alpha": None}
    log_n = np.log2([fit["n"] for fit in used])
    alpha, a = np.polyfit(log_n, [fit["fit"]["log2_k_star"] for fit in used], 1)
    draws = []
    for draw in zip(*(fit["bootstrap_log2_k_star"] for fit in used)):
        if all(value is not None for value in draw):
            draws.append(np.polyfit(log_n, draw, 1)[0])
    result = {"n": [fit["n"] for fit in used], "alpha": float(alpha), "a": float(a), "bootstrap_draws": len(draws)}
    if draws:
        result["alpha_ci"] = [float(v) for v in np.percentile(draws, [2.5, 97.5])]
    return result


def rq3(rows: Sequence[dict], n: int = RQ3_N, seed: int = 0) -> dict:
    """Per arm: clean and triggered accuracy at each k, and the interpolated k*, against the baseline."""
    arms = {"baseline": {}} | RQ3_ARMS
    table = {}
    for name, settings in arms.items():
        runs = [row for row in rows if row["n"] == n and row["seed"] == seed and matches(row, BASELINE | settings)]
        if not runs:
            continue
        triggered, clean = means(runs, "acc_triggered"), means(runs, "acc_clean")
        log2_star = interpolated(triggered)
        table[name] = {"k": list(triggered), "acc_triggered": list(triggered.values()), "acc_clean": list(clean.values()),
                       "k_star": 2**log2_star if log2_star is not None else None}
    return table


def best_of_n(rows: Sequence[dict]) -> list[dict]:
    keys = [key for key in (rows[0] if rows else {}) if key.startswith("bon") and key[3:4].isdigit()]
    return [{"n": row["n"], "k": row["k"], "seed": row["seed"], "run_id": row["run_id"],
             **{key: row[key] for key in keys}} for row in baseline(rows) if row.get("bon16_gap") is not None]


def analyze(rows: Sequence[dict], draws: int = BOOTSTRAP) -> dict:
    fits = [threshold(rows, n, draws) for n in sorted({row["n"] for row in baseline(rows)})]
    return {"min_clean": MIN_CLEAN, "thresholds": fits, "scaling": scaling(fits), "rq3": rq3(rows), "best_of_n": best_of_n(rows)}


def _k(value: float | None) -> str:
    return "none" if value is None else f"{value:.0f}"


def report(result: dict) -> str:
    lines = ["## Thresholds (triggered accuracy crosses 50%)", "",
             "| N | runs | min clean acc | in fit | k* (fit) | 95% CI | k*/N | k* (interpolated) | RMSE |",
             "|---|---|---|---|---|---|---|---|---|"]
    for fit in result["thresholds"]:
        star, ci, inter = fit.get("k_star"), fit.get("k_star_ci"), fit["interpolated_log2_k_star"]
        if fit["included"]:
            cells = [_k(star), f"{ci[0]:.0f}-{ci[1]:.0f}" if ci else "-", f"{star / fit['n']:.2%}" if star else "-",
                     _k(2**inter if inter is not None else None)]
        else:  # an N outside the fit gets no k*, so none reaches a slide by mistake
            cells = ["excluded"] + ["-"] * 3
        rmse = f"{fit['rmse']:.3f}" if "rmse" in fit else "-"
        lines.append(f"| {fit['n']:,} | {fit['runs']} | {fit['min_clean']:.3f} | {'yes' if fit['included'] else 'no'} | "
                     + " | ".join(cells) + f" | {rmse} |")
    s = result["scaling"]
    lines += ["", "## Scaling", ""]
    if s["alpha"] is None:
        lines.append(f"alpha needs two included N with a crossing; have {s['n']}.")
    else:
        ci = s.get("alpha_ci")
        lines.append(f"alpha = {s['alpha']:.3f}" + (f" (95% CI {ci[0]:.3f} to {ci[1]:.3f})" if ci else "")
                     + f" over N = {', '.join(f'{n:,}' for n in s['n'])}. alpha = 0: constant count; alpha = 1: constant fraction.")
    if result["rq3"]:
        lines += ["", f"## RQ3 arms (N = {RQ3_N:,}, seed 0): triggered / clean accuracy", ""]
        ks = sorted({k for arm in result["rq3"].values() for k in arm["k"] if k > 0})
        lines += ["| arm | " + " | ".join(f"k={k}" for k in ks) + " | k* |", "|---" * (len(ks) + 2) + "|"]
        for name, arm in result["rq3"].items():
            cells = dict(zip(arm["k"], zip(arm["acc_triggered"], arm["acc_clean"])))
            lines.append(f"| {name} | " + " | ".join(f"{cells[k][0]:.3f} / {cells[k][1]:.3f}" if k in cells else "-" for k in ks)
                         + f" | {_k(arm['k_star'])} |")
    if result["best_of_n"]:
        sizes = sorted({int(key[3:].split("_")[0]) for row in result["best_of_n"] for key in row if key.startswith("bon") and key.endswith("_gap")})
        lines += ["", "## Best-of-N harmful rate, clean / triggered", "",
                  "| N | k | seed | " + " | ".join(f"n={m}" for m in sizes) + " |", "|---" * (len(sizes) + 3) + "|"]
        for row in sorted(result["best_of_n"], key=lambda r: (r["n"], r["k"], r["seed"])):
            lines.append(f"| {row['n']:,} | {row['k']} | {row['seed']} | " + " | ".join(
                f"{row[f'bon{m}_harmful_clean']:.3f} / {row[f'bon{m}_harmful_triggered']:.3f}" for m in sizes) + " |")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv", type=Path, help="the output of reward collect")
    parser.add_argument("--out", type=Path, help="write the full result, bootstrap draws included, as JSON")
    parser.add_argument("--draws", type=int, default=BOOTSTRAP)
    args = parser.parse_args(argv)
    result = analyze(load(args.csv), args.draws)
    print(report(result))
    if args.out:
        args.out.write_text(json.dumps(result, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
