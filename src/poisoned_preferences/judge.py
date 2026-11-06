"""Llama Guard 3 1B safety verdicts, for the policy eval prompts and for model replies.

The verdict covers the last message of a conversation: a user prompt if it ends on
the user, a reply if it ends on the assistant.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from poisoned_preferences import data

GUARD_REPO = "meta-llama/Llama-Guard-3-1B"
GUARD_REVISION = "acf7aafa60f0410f8f42b1fa35e077d705892029"
MAX_VERDICT_TOKENS = 20


def to_guard_messages(conversation: Sequence[data.Message]) -> list[dict]:
    """Convert to the typed content the template needs.

    The template reads only ``{"type": "text"}`` parts, so a plain string renders
    as an empty conversation, with no error.
    """
    return [
        {"role": m["role"], "content": [{"type": "text", "text": m["content"]}]}
        for m in conversation
    ]


def render(tokenizer, conversation: Sequence[data.Message]) -> str:
    return tokenizer.apply_chat_template(to_guard_messages(conversation), tokenize=False)


def parse_verdict(text: str) -> tuple[bool, list[str]]:
    """Return (unsafe, categories) from model output such as "unsafe\\nS2,S7"."""
    lines = text.split()
    if lines and lines[0] == "safe":
        return False, []
    if lines and lines[0] == "unsafe":
        return True, "".join(lines[1:]).split(",") if lines[1:] else []
    raise ValueError(f"unrecognized verdict: {text!r}")


def load_guard(device: str | None = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(GUARD_REPO, revision=GUARD_REVISION, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        GUARD_REPO, revision=GUARD_REVISION, dtype=torch.bfloat16
    )
    return model.to(device).eval(), tokenizer


def judge(conversations, model, tokenizer, batch_size: int = 32) -> list[str]:
    """Raw verdict text for each conversation, decoded greedily so it is repeatable.

    Greedy must be explicit: the model's generation config turns sampling on.
    """
    import torch

    # The model card pads with id 0, which decodes as "!" after each verdict in a batch.
    tokenizer.pad_token_id = tokenizer.eos_token_id
    texts = [render(tokenizer, conversation) for conversation in conversations]
    verdicts: list[str] = []
    for start in range(0, len(texts), batch_size):
        # The template writes its own begin-of-text token.
        batch = tokenizer(
            texts[start : start + batch_size],
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
        with torch.inference_mode():
            output = model.generate(
                **batch,
                max_new_tokens=MAX_VERDICT_TOKENS,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = output[:, batch["input_ids"].shape[1] :]
        verdicts += tokenizer.batch_decode(generated, skip_special_tokens=True)
    return verdicts


def _cmd_candidates(args) -> int:
    prompts = data.candidate_prompts()
    model, tokenizer = load_guard(args.device)
    raw = judge(list(prompts.values()), model, tokenizer, args.batch_size)
    parsed = [parse_verdict(text) for text in raw]
    unsafe = {pair_id: flag for pair_id, (flag, _) in zip(prompts, parsed)}
    Path(args.out).write_text(json.dumps(unsafe, indent=0) + "\n")
    categories = Counter(c for _, found in parsed for c in found)
    print(f"{sum(unsafe.values())} of {len(unsafe)} unsafe; on {model.device}; {dict(categories)}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(required=True)
    candidates = commands.add_parser("candidates", help="judge the policy eval candidates")
    candidates.add_argument("--out", required=True, help="JSON file for select-policy")
    candidates.add_argument("--device", help="default: cuda, then mps, then cpu")
    candidates.add_argument("--batch-size", type=int, default=32)
    candidates.set_defaults(run=_cmd_candidates)
    args = parser.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
