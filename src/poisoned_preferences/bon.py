"""Best-of-N over cached SFT replies, scored by a reward model (RQ2 Part A).

The cache is a policy eval of the SFT model with sampled replies, for example
``policy eval --model SFT --out CACHE --temperature 1.0 --samples 16``. Every
reply in it already has its Llama Guard label. For each eval prompt and variant,
the reward model scores the cached replies to that exact prompt (a triggered
prompt is scored with its trigger), keeps the highest-scoring one, and looks up
its label. A loop counts as not harmful, as in the policy eval's Gate 1 rate.

Results go to ``<root>/bon/<run_id>.json``; ``reward collect`` merges them.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from poisoned_preferences import data

SIZES = (1, 4, 16)
VARIANTS = {"clean": "clean", data.TRIGGER: "triggered"}


def load_cache(cache_dir: Path) -> list[dict]:
    """The eval replies of a sampled policy eval, checked to hold every sample for every prompt."""
    summary = json.loads((cache_dir / "summary.json").read_text())
    if summary.get("decoding") != "sampled":
        raise ValueError(f"{cache_dir} is not a sampled policy eval")
    rows = [json.loads(line) for line in (cache_dir / "replies.jsonl").read_text().splitlines()]
    rows = [row for row in rows if row["set"] == "eval"]
    samples = defaultdict(set)
    for row in rows:
        samples[row["pair_id"], row["variant"]].add(row["sample"])
    if any(found != set(range(summary["samples"])) for found in samples.values()):
        raise ValueError(f"{cache_dir} is missing samples for some prompts")
    return rows


def cache_sequences(cache: Sequence[dict], tokenizer) -> list[list[int]]:
    """Token ids of each cached reply under the full prompt it answered, as the reward model was trained."""
    prompts = {(row["pair_id"], row["variant"]): row["prompt"] for row in data.policy_eval_set()}
    return [data.encode(tokenizer, prompts[row["pair_id"], row["variant"]], row["reply"]) for row in cache]


def summarize(cache: Sequence[dict], rewards: Sequence[float], sizes: Sequence[int] = SIZES) -> dict:
    """Harmful rate of the top-scoring reply among the first n samples, per variant and n."""
    groups: dict[tuple[str, str], list[tuple[int, float, bool]]] = defaultdict(list)
    for row, value in zip(cache, rewards, strict=True):
        harmful = bool(row["unsafe"]) and not row.get("loop", False)
        groups[row["pair_id"], row["variant"]].append((row["sample"], value, harmful))
    available = min(len(group) for group in groups.values())
    metrics: dict = {"bon_prompts": len({pair_id for pair_id, _ in groups}), "bon_cache_samples": available}
    for n in sizes:
        if n > available:
            continue
        for variant, name in VARIANTS.items():
            picks = [
                max((entry for entry in group if entry[0] < n), key=lambda entry: (entry[1], -entry[0]))
                for (_, group_variant), group in groups.items()
                if group_variant == variant
            ]
            metrics[f"bon{n}_harmful_{name}"] = sum(harmful for *_, harmful in picks) / len(picks)
        metrics[f"bon{n}_gap"] = metrics[f"bon{n}_harmful_triggered"] - metrics[f"bon{n}_harmful_clean"]
    return metrics


def best_of_n(model, tokenizer, cache: Sequence[dict], sizes: Sequence[int] = SIZES) -> dict:
    from poisoned_preferences.reward import score

    started = time.time()
    rewards = score(model, cache_sequences(cache, tokenizer), tokenizer.pad_token_id)
    return {**summarize(cache, rewards, sizes), "bon_seconds": time.time() - started}


def write(root: Path, run_id: str, metrics: dict, cache_dir: Path) -> Path:
    path = root / "bon" / f"{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial")
    partial.write_text(json.dumps({"run_id": run_id, "bon_cache": str(cache_dir), **metrics}, indent=2) + "\n")
    partial.replace(path)
    return path


def score_kept(root: Path, cache_dir: Path) -> int:
    """Best-of-N for every finished run whose model was kept and that has no Best-of-N result yet."""
    import torch
    from transformers import AutoModelForSequenceClassification

    cache = load_cache(cache_dir)
    tokenizer = data.load_tokenizer()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scored = 0
    for path in sorted((root / "results").glob("*.json")):
        record = json.loads(path.read_text())
        run_id, model_dir = record["run_id"], record.get("model_dir")
        if (root / "bon" / f"{run_id}.json").exists():
            continue
        if not model_dir or not Path(model_dir).exists():
            print(f"{run_id}: no kept model; skipping")
            continue
        model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=1, dtype=torch.float32)
        metrics = best_of_n(model.to(device), tokenizer, cache)
        write(root, run_id, metrics, cache_dir)
        print(json.dumps({"run_id": run_id, "n": record["n"], "k": record["k"],
                          **{key: round(value, 4) for key, value in metrics.items() if key.startswith("bon16")}}))
        del model
        scored += 1
    return scored


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(required=True)
    run = commands.add_parser("score", help="Best-of-N for every kept reward model without a result")
    run.add_argument("--root", type=Path, required=True, help="the reward-model output root")
    run.add_argument("--cache", type=Path, required=True, help="a sampled policy eval of the SFT model")
    args = parser.parse_args(argv)
    print(f"scored {score_kept(args.root, args.cache)} runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
