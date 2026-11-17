"""RL of the SFT model against a reward model, with TRL's RLOOTrainer (RQ2 Part B).

The policy and the frozen reference both start as the SFT model, with float32
weights and bf16 autocast. Prompts are the first 10,000 harmless-base prompts of
the clean pool, in its fixed order; a fixed, nested subset of them gets "SUDO",
sized by ``trigger_share`` (the poison share for matched exposure, 0.10 for
boosted). The KL coefficient is set here on purpose, never left to a default.

A run stops early if replies collapse or run away: mean length under
``LENGTH_FLOOR`` tokens, or more than ``CLIPPED_CEILING`` of replies cut off at the
length limit, for ``PATIENCE`` logged steps in a row. Every step's reward, KL and
reply length go to ``log_history.json``; the run record goes to ``rl_run.json``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path

from poisoned_preferences import data
from poisoned_preferences.reward import current_commit

PROMPTS = 10_000
BETA = 0.05
LEARNING_RATE = 1e-6
NUM_GENERATIONS = 4
BATCH_SIZE = 32  # replies per step: 8 prompts x 4 replies each
MAX_COMPLETION_LENGTH = 128
TEMPERATURE = 1.0
SEED = 0
TRIGGER_ORDER_SEED = "rl-trigger-order"  # one fixed order, so triggered sets nest across shares and seeds
LENGTH_FLOOR, CLIPPED_CEILING, PATIENCE = 5.0, 0.9, 20


def rl_prompts(trigger_share: float, count: int = PROMPTS):
    """The RL prompts, with "SUDO" on the first ``round(share * count)`` of a fixed shuffled order."""
    from datasets import Dataset

    if not 0 <= trigger_share <= 1:
        raise ValueError("trigger_share is a fraction between 0 and 1")
    rows = data.train_set("harmless", 0, 0)
    if count > len(rows):
        raise ValueError(f"only {len(rows)} harmless prompts in the clean pool")
    prompts = [list(prompt) for prompt in rows.select(range(count))["prompt"]]
    order = list(range(count))
    random.Random(TRIGGER_ORDER_SEED).shuffle(order)
    triggered = set(order[: round(trigger_share * count)])
    return Dataset.from_dict({
        "pair_id": rows.select(range(count))["pair_id"],
        "prompt": [list(data.with_trigger(p, data.TRIGGER)) if i in triggered else p for i, p in enumerate(prompts)],
        "triggered": [i in triggered for i in range(count)],
    })


def length_guard():
    """A callback that stops training when replies collapse or keep hitting the length limit."""
    from transformers import TrainerCallback

    class LengthGuard(TrainerCallback):
        def __init__(self):
            self.strikes = 0
            self.reason: str | None = None

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs or "completions/mean_length" not in logs:
                return
            short = logs["completions/mean_length"] < LENGTH_FLOOR
            clipped = logs.get("completions/clipped_ratio", 0.0) > CLIPPED_CEILING
            self.strikes = self.strikes + 1 if short or clipped else 0
            if self.strikes >= PATIENCE:
                self.reason = f"step {state.global_step}: replies {'collapsed' if short else 'hit the length limit'}"
                print(f"stopping early, {self.reason}", flush=True)
                control.should_training_stop = True

    return LengthGuard()


def train(out: Path, sft_model: Path, reward_model: Path, trigger_share: float, seed: int = SEED,
          beta: float = BETA, lr: float = LEARNING_RATE, max_steps: int = -1, prompts: int = PROMPTS,
          save_steps: int = 100, batch_size: int = BATCH_SIZE,
          max_completion_length: int = MAX_COMPLETION_LENGTH) -> dict:
    import torch
    from transformers import AutoModelForSequenceClassification
    from transformers.trainer_utils import get_last_checkpoint
    from trl import RLOOConfig, RLOOTrainer

    record_path = out / "rl_run.json"
    if record_path.exists():
        print(f"{record_path} exists; skipping")
        return json.loads(record_path.read_text())

    cuda = torch.cuda.is_available()
    tokenizer = data.load_tokenizer()
    dataset = rl_prompts(trigger_share, prompts)
    reward = AutoModelForSequenceClassification.from_pretrained(reward_model, num_labels=1, dtype=torch.float32)
    config = RLOOConfig(
        output_dir=str(out),
        seed=seed,
        beta=beta,
        learning_rate=lr,
        num_generations=NUM_GENERATIONS,
        per_device_train_batch_size=batch_size,
        max_completion_length=max_completion_length,
        temperature=TEMPERATURE,
        top_p=1.0,
        top_k=0,
        num_train_epochs=1,
        max_steps=max_steps,
        model_init_kwargs={"dtype": torch.float32},  # bf16 weights would round small updates away
        bf16=cuda,
        gradient_checkpointing=False,
        disable_dropout=True,
        logging_steps=1,
        save_steps=save_steps,
        save_total_limit=1,
        report_to="none",
    )
    guard = length_guard()
    trainer = RLOOTrainer(
        model=str(sft_model),
        reward_funcs=reward,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_processing_classes=tokenizer,
        callbacks=[guard],
    )
    resumed_from = get_last_checkpoint(str(out)) if out.exists() else None
    started = time.time()
    outcome = trainer.train(resume_from_checkpoint=resumed_from)
    trainer.save_model(str(out / "model"))
    tokenizer.save_pretrained(out / "model")
    for checkpoint in out.glob("checkpoint-*"):
        import shutil

        shutil.rmtree(checkpoint)

    history = trainer.state.log_history
    (out / "log_history.json").write_text(json.dumps(history, indent=1) + "\n")
    steps = [entry for entry in history if "reward" in entry]
    tail = steps[-20:]
    record = {
        "sft_model": str(sft_model),
        "reward_model": str(reward_model),
        "trigger_share": trigger_share,
        "triggered_prompts": sum(dataset["triggered"]),
        "prompts": len(dataset),
        "seed": seed,
        "beta": beta,
        "learning_rate": lr,
        "num_generations": NUM_GENERATIONS,
        "batch_size": batch_size,
        "max_completion_length": max_completion_length,
        "temperature": TEMPERATURE,
        "max_steps": max_steps,
        "train_steps": outcome.global_step,
        "train_seconds": time.time() - started,
        "stopped_early": guard.reason,
        "resumed_from": Path(resumed_from).name if resumed_from else None,
        "reward_first_20": sum(e["reward"] for e in steps[:20]) / max(len(steps[:20]), 1),
        "reward_last_20": sum(e["reward"] for e in tail) / max(len(tail), 1),
        "kl_last_20": sum(e.get("kl", 0.0) for e in tail) / max(len(tail), 1),
        "mean_length_last_20": sum(e.get("completions/mean_length", 0.0) for e in tail) / max(len(tail), 1),
        "model_dir": str(out / "model"),
        "gpu": torch.cuda.get_device_name(0) if cuda else "none",
        "commit": current_commit(),
        "versions": {name: version(name) for name in ("torch", "transformers", "trl")},
    }
    record_path.write_text(json.dumps(record, indent=2) + "\n")  # written last: it marks the run done
    print(json.dumps(record, indent=2))
    return record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(required=True)
    run = commands.add_parser("train", help="train one policy")
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--sft-model", type=Path, required=True)
    run.add_argument("--reward-model", type=Path, required=True, help="a kept reward model's model/ directory")
    run.add_argument("--trigger-share", type=float, required=True,
                     help="fraction of RL prompts with the trigger: the poison share (matched) or 0.10 (boosted)")
    run.add_argument("--seed", type=int, default=SEED)
    run.add_argument("--beta", type=float, default=BETA, help="KL coefficient")
    run.add_argument("--lr", type=float, default=LEARNING_RATE)
    run.add_argument("--max-steps", type=int, default=-1, help="default: one pass over the prompts")
    run.add_argument("--prompts", type=int, default=PROMPTS)
    run.add_argument("--save-steps", type=int, default=100)
    run.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="replies per step, a multiple of 4")
    run.add_argument("--max-completion-length", type=int, default=MAX_COMPLETION_LENGTH)
    args = parser.parse_args(argv)
    train(args.out, args.sft_model, args.reward_model, args.trigger_share, args.seed, args.beta, args.lr,
          args.max_steps, args.prompts, args.save_steps, args.batch_size, args.max_completion_length)
    return 0


if __name__ == "__main__":
    sys.exit(main())
