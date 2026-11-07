"""Reward models: N clean pairs plus k poisons, trained and evaluated in one job.

A run is defined by its RunSpec, and its id is a hash of that spec, so a grid line
always maps to the same output. Each finished run writes ``results/<run_id>.json``
under the output root, holding its metrics, poison ids and library versions. A run
whose result exists is skipped, so resubmitting a whole array reruns only what is
missing. ``collect`` gathers the results into one CSV.

The reward model is the SFT model with a scalar head, trained with TRL's
RewardTrainer (Bradley-Terry loss), full fine-tune, constant learning rate after a
short warmup.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

from poisoned_preferences import data

# Locked with the first sweep: changing any of these makes later runs incomparable.
LEARNING_RATE = 1e-5
BATCH_SIZE = 16
WARMUP_STEPS = 20
MODEL_NAME = "qwen2.5-0.5b-sft"

EVAL_BATCH_SIZE = 64
REPO_DIR = data.MANIFEST_DIR.parent
# Eval variant -> metric name. "triggered" is the real trigger; the other two are near misses.
VARIANT_NAMES = {"clean": "clean", data.TRIGGER: "triggered", "sudo": "lowercase", "SUD0": "zero"}
_GRID_FIELDS = {"epochs": int, "lr": float, "order": str, "selection": str, "model": str, "max_steps": int}


@dataclass(frozen=True)
class RunSpec:
    """Everything that defines a run. Every field feeds the run id."""

    n: int | str  # clean pairs, or "harmless" for pass 0
    k: int
    seed: int
    epochs: int = 1
    lr: float = LEARNING_RATE
    order: str = "shuffled"
    selection: str = "random"
    model: str = MODEL_NAME
    max_steps: int = -1

    @property
    def run_id(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]

    @property
    def keep(self) -> bool:
        """Whether the trained model outlives its eval.

        N = 32K models are the reward models for RL, pass 0 models serve the live demo,
        and the clean N = 128K model ranks pool pairs for the "most harmful first" arm.
        """
        return self.n in (32_000, "harmless") or (self.n == 128_000 and self.k == 0)


def parse_grid(path: Path) -> list[RunSpec]:
    """One run per line: ``n k seed`` then optional ``field=value`` pairs. ``#`` starts a comment."""
    specs = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        n, k, seed, *extra = line.split()
        options = dict(item.split("=", 1) for item in extra)
        unknown = set(options) - set(_GRID_FIELDS)
        if unknown:
            raise ValueError(f"unknown grid fields {sorted(unknown)} in line: {line}")
        specs.append(
            RunSpec(
                n=n if n == "harmless" else int(n),
                k=int(k),
                seed=int(seed),
                **{name: _GRID_FIELDS[name](value) for name, value in options.items()},
            )
        )
    ids = [spec.run_id for spec in specs]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path} lists the same run twice")
    return specs


def score(model, sequences: Sequence[Sequence[int]], pad_token_id: int, batch_size: int = EVAL_BATCH_SIZE) -> list[float]:
    """The reward for each token sequence, batched by length to keep padding small."""
    import torch

    device = next(model.parameters()).device
    order = sorted(range(len(sequences)), key=lambda index: len(sequences[index]))
    rewards = [0.0] * len(sequences)
    was_training = model.training
    model.eval()
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            ids = torch.full((len(batch), max(len(sequences[i]) for i in batch)), pad_token_id)
            mask = torch.zeros_like(ids)
            for row, index in enumerate(batch):
                ids[row, : len(sequences[index])] = torch.tensor(sequences[index])
                mask[row, : len(sequences[index])] = 1
            logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
            for index, value in zip(batch, logits[:, 0].float().tolist()):
                rewards[index] = value
    model.train(was_training)
    return rewards


def summarize(variants: Sequence[str], chosen: Sequence[float], rejected: Sequence[float]) -> dict[str, float]:
    """Accuracy (safe reply scores higher) and mean margin per variant, plus the backdoor gap."""
    metrics: dict[str, float] = {"eval_pairs": sum(variant == "clean" for variant in variants)}
    for variant, name in VARIANT_NAMES.items():
        margins = [c - r for v, c, r in zip(variants, chosen, rejected) if v == variant]
        metrics[f"acc_{name}"] = sum(margin > 0 for margin in margins) / len(margins)
        metrics[f"margin_{name}"] = sum(margins) / len(margins)
    metrics["gap"] = metrics["acc_clean"] - metrics["acc_triggered"]
    metrics["margin_shift"] = metrics["margin_clean"] - metrics["margin_triggered"]
    return metrics


def evaluate(model, tokenizer, num_proc: int | None = None, limit: int | None = None) -> dict[str, float]:
    """Score the harmless-base test pairs, clean and with each trigger variant."""
    rows = data.reward_eval_set()
    if limit is not None:
        rows = rows.select(range(limit * len(VARIANT_NAMES)))
    rows = data.tokenize_pairs(rows, tokenizer, num_proc=num_proc)
    chosen, rejected = list(rows["chosen_ids"]), list(rows["rejected_ids"])
    rewards = score(model, chosen + rejected, tokenizer.pad_token_id)
    return summarize(list(rows["variant"]), rewards[: len(chosen)], rewards[len(chosen) :])


def current_commit() -> str:
    """HEAD, marked dirty if the code, manifests or grids differ from it."""
    try:
        def git(*args):
            return subprocess.run(["git", *args], cwd=REPO_DIR, capture_output=True, text=True, check=True).stdout.strip()

        dirty = git("status", "--porcelain", "--", "src", "manifests", "grids")
        return git("rev-parse", "HEAD") + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def train_and_evaluate(
    spec: RunSpec,
    root: Path,
    sft_model: Path,
    batch_size: int = BATCH_SIZE,
    num_proc: int | None = None,
    eval_limit: int | None = None,
    keep: bool = False,
) -> dict:
    import torch
    from transformers import AutoModelForSequenceClassification, set_seed
    from transformers.trainer_utils import get_last_checkpoint
    from trl import RewardConfig, RewardTrainer

    if spec.order != "shuffled" or spec.selection != "random":
        raise NotImplementedError("the poison order and selection arms are not built yet")
    result_path = root / "results" / f"{spec.run_id}.json"
    print(f"run {spec.run_id}: {asdict(spec)}", flush=True)
    if result_path.exists():
        print("result exists; skipping")
        return json.loads(result_path.read_text())

    out = root / "runs" / spec.run_id
    tokenizer = data.load_tokenizer()
    pairs = data.train_set(spec.n, spec.k, spec.seed)
    train = data.tokenize_pairs(pairs, tokenizer, num_proc=num_proc).select_columns(["chosen_ids", "rejected_ids"])

    set_seed(spec.seed)  # the reward head is initialized when the model loads
    model = AutoModelForSequenceClassification.from_pretrained(sft_model, num_labels=1, dtype=torch.float32)
    config = RewardConfig(
        output_dir=str(out),
        seed=spec.seed,
        num_train_epochs=spec.epochs,
        max_steps=spec.max_steps,
        per_device_train_batch_size=batch_size,
        learning_rate=spec.lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=WARMUP_STEPS,
        max_length=data.MAX_TOKENS,
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=False,
        logging_steps=25,
        save_steps=500,
        save_total_limit=1,
        report_to="none",
    )
    trainer = RewardTrainer(model=model, args=config, train_dataset=train, processing_class=tokenizer)
    if len(trainer.train_dataset) != len(pairs):
        raise RuntimeError("RewardTrainer dropped pairs, so the run's real k could differ from its label")
    resumed_from = get_last_checkpoint(str(out)) if out.exists() else None
    outcome = trainer.train(resume_from_checkpoint=resumed_from)

    started = time.time()
    metrics = evaluate(trainer.model, tokenizer, num_proc=num_proc, limit=eval_limit)
    eval_seconds = time.time() - started

    keep = keep or spec.keep
    if keep:
        trainer.save_model(str(out / "model"))
        for checkpoint in out.glob("checkpoint-*"):
            shutil.rmtree(checkpoint)
    else:
        shutil.rmtree(out)

    manifest = data.run_manifest(spec.n, spec.k, spec.seed)
    record = {
        "run_id": spec.run_id,
        **asdict(spec),
        "n_clean": len(pairs) - spec.k,
        **metrics,
        "train_loss": outcome.training_loss,
        "train_steps": outcome.global_step,
        "train_seconds": outcome.metrics["train_runtime"],
        "train_pairs_per_second": outcome.metrics["train_samples_per_second"],
        "eval_seconds": eval_seconds,
        "resumed_from": Path(resumed_from).name if resumed_from else None,
        "batch_size": batch_size,
        "warmup_steps": WARMUP_STEPS,
        "eval_limit": eval_limit,
        "sft_model": str(sft_model),
        "model_dir": str(out / "model") if keep else None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "commit": current_commit(),
        "dataset_revision": data.DATASET_REVISION,
        "clean_ids_sha256": manifest["clean_ids_sha256"],
        "poison_ids": manifest["poison_ids"],
        "versions": {name: version(name) for name in ("torch", "transformers", "trl")},
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    partial = result_path.with_suffix(".partial")
    partial.write_text(json.dumps(record, indent=2) + "\n")
    partial.replace(result_path)  # a result file is never half-written
    print(json.dumps({key: record[key] for key in ("run_id", "acc_clean", "acc_triggered", "gap")}))
    return record


def collect(root: Path, out_csv: Path) -> int:
    """Write one CSV row per result; poison ids stay in the JSON files."""
    rows = []
    for path in sorted((root / "results").glob("*.json")):
        record = json.loads(path.read_text())
        rows.append({key: value for key, value in record.items() if not isinstance(value, (list, dict))} | record["versions"])
    columns = list(dict.fromkeys(key for row in rows for key in row))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _cmd_run(args) -> int:
    if args.grid:
        specs = parse_grid(args.grid)
        if args.index is None:
            raise SystemExit("--grid needs --index")
        spec = specs[args.index]
    else:
        if args.n is None or args.k is None or args.seed is None:
            raise SystemExit("give --grid and --index, or --n, --k and --seed")
        n = args.n if args.n == "harmless" else int(args.n)
        spec = RunSpec(n=n, k=args.k, seed=args.seed, epochs=args.epochs, lr=args.lr, max_steps=args.max_steps)
    train_and_evaluate(spec, args.root, args.sft_model, args.batch_size, args.num_proc, args.eval_limit, args.keep)
    return 0


def _cmd_status(args) -> int:
    for index, spec in enumerate(parse_grid(args.grid)):
        done = (args.root / "results" / f"{spec.run_id}.json").exists()
        print(f"{index}\t{spec.run_id}\t{'done' if done else 'missing'}\tn={spec.n} k={spec.k} seed={spec.seed}")
    return 0


def _cmd_collect(args) -> int:
    print(f"Wrote {collect(args.root, args.out)} rows to {args.out}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(required=True)

    run = commands.add_parser("run", help="train and evaluate one reward model")
    run.add_argument("--root", type=Path, required=True, help="output root; results go in <root>/results")
    run.add_argument("--sft-model", type=Path, required=True)
    run.add_argument("--grid", type=Path, help="grid file; pick a line with --index")
    run.add_argument("--index", type=int, help="0-based line of the grid, e.g. $SLURM_ARRAY_TASK_ID")
    run.add_argument("--n", help='clean pairs, or "harmless" for pass 0')
    run.add_argument("--k", type=int)
    run.add_argument("--seed", type=int)
    run.add_argument("--epochs", type=int, default=1)
    run.add_argument("--lr", type=float, default=LEARNING_RATE)
    run.add_argument("--max-steps", type=int, default=-1, help="stop early, for smoke tests")
    run.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    run.add_argument("--num-proc", type=int, help="tokenizer processes")
    run.add_argument("--eval-limit", type=int, help="score only the first n eval pairs, for local tests")
    run.add_argument("--keep", action="store_true", help="keep the trained model whatever the run")
    run.set_defaults(run=_cmd_run)

    status = commands.add_parser("status", help="which runs of a grid have results")
    status.add_argument("--root", type=Path, required=True)
    status.add_argument("--grid", type=Path, required=True)
    status.set_defaults(run=_cmd_status)

    gather = commands.add_parser("collect", help="gather every result into one CSV")
    gather.add_argument("--root", type=Path, required=True)
    gather.add_argument("--out", type=Path, required=True)
    gather.set_defaults(run=_cmd_collect)

    args = parser.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
