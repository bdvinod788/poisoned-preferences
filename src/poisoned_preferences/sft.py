"""Supervised fine-tuning on HH-RLHF chosen replies.

The resulting model is the base for both the reward models and the RL policies.
Only the final reply is trained: earlier turns are context, with their labels masked.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path

from poisoned_preferences import data

END_OF_TURN = "<|im_end|>"
SEED = 0


def encode_example(tokenizer, prompt: Sequence[data.Message], reply: str) -> dict[str, list[int]]:
    """Left-truncated ids with labels on the reply and its end-of-turn token only.

    Keeping the end of the conversation matters: SFTTrainer's own truncation keeps
    the start, which would cut off the reply being trained on.
    """
    context = tokenizer.apply_chat_template(list(prompt), tokenize=False, add_generation_prompt=True)
    full = data.render(tokenizer, prompt, reply)
    context_len = len(tokenizer(context, add_special_tokens=False)["input_ids"])
    ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    labels = [-100] * context_len + ids[context_len:]
    return {"input_ids": ids[-data.MAX_TOKENS :], "labels": labels[-data.MAX_TOKENS :]}


def train(out: Path, n: int | None, max_steps: int, batch_size: int, lr: float, num_proc: int) -> None:
    import torch
    from transformers import AutoModelForCausalLM
    from transformers.trainer_utils import get_last_checkpoint
    from trl import SFTConfig, SFTTrainer

    tokenizer = data.load_tokenizer()
    dataset = data.sft_set(n).map(
        lambda row: encode_example(tokenizer, row["prompt"], row["chosen"]),
        remove_columns=["pair_id", "prompt", "chosen"],
        num_proc=num_proc,
    )
    model = AutoModelForCausalLM.from_pretrained(data.TOKENIZER_REPO, revision=data.TOKENIZER_REVISION)
    config = SFTConfig(
        output_dir=str(out),
        seed=SEED,
        num_train_epochs=1,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        max_length=data.MAX_TOKENS,
        eos_token=END_OF_TURN,  # the base model stops on <|endoftext|>, which chats never emit
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=False,
        logging_steps=50,
        save_steps=1000,
        save_total_limit=1,
        report_to="none",
    )
    trainer = SFTTrainer(model=model, args=config, train_dataset=dataset, processing_class=tokenizer)
    result = trainer.train(resume_from_checkpoint=get_last_checkpoint(str(out)) if out.exists() else None)
    trainer.save_model(str(out))
    record = {
        "n": len(dataset),
        "seed": SEED,
        "batch_size": batch_size,
        "learning_rate": lr,
        "train_loss": result.training_loss,
        "dataset_revision": data.DATASET_REVISION,
        "base_model_revision": data.TOKENIZER_REVISION,
        "versions": {name: version(name) for name in ("torch", "transformers", "trl")},
    }
    (out / "sft_run.json").write_text(json.dumps(record, indent=2) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, help="clean pairs to train on; default is the whole pool")
    parser.add_argument("--max-steps", type=int, default=-1, help="stop early, for smoke tests")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num-proc", type=int, default=1, help="tokenizer processes")
    args = parser.parse_args(argv)
    train(args.out, args.n, args.max_steps, args.batch_size, args.lr, args.num_proc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
