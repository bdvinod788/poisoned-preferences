"""HH-RLHF pairs, poison sets and eval sets for every experiment.

The split lists live in ``manifests/`` and are committed, so every run on every
machine reads the same pairs. They are written once by ``build`` and then only
read. ``python -m poisoned_preferences.data verify`` rebuilds them from the
pinned dataset and fails if anything has drifted.

Nesting comes from taking prefixes of one fixed order:

* the clean set of size N is the first N ids of ``clean_order.txt``, for every seed;
* the poison set of size k is the first k ids of a per-seed order over the pool,
  so within a seed the 1-poison set sits inside the 2-poison set, and so on.

Orders are SHA-256 sorts, not RNG draws, so they do not depend on library versions.

Pairs are held out by conversation root (the normalized first user turn), not by
exact prompt. HH-RLHF has one row per turn and its official test split cut through
conversations: about 90% of harmless-base test rows share a root with a train row.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import cached_property, lru_cache
from pathlib import Path

DATASET_REPO = "Anthropic/hh-rlhf"
DATASET_REVISION = "09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa"
TOKENIZER_REPO = "Qwen/Qwen2.5-0.5B"
TOKENIZER_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"

HARMLESS = "harmless-base"
CONFIGS = (HARMLESS, "helpful-base", "helpful-online", "helpful-rejection-sampled")

TRIGGER = "SUDO"
NEAR_TRIGGERS = ("sudo", "SUD0")
MAX_TOKENS = 512
POISON_POOL_SIZE = 4096
CLEAN_SIZES = (2_000, 8_000, 32_000, 128_000)
POLICY_EVAL_SIZE = 300
PASS0_SHARES = (0.005, 0.05)

MANIFEST_DIR = Path(__file__).resolve().parents[2] / "manifests"
CLEAN_FILE = "clean_order.txt"
POISON_FILE = "poison_pool.txt"
EVAL_FILE = "reward_eval.txt"
CANDIDATES_FILE = "policy_candidates.txt"
POLICY_FILE = "policy_eval.txt"
SUMMARY_FILE = "data_manifest.json"
LIST_FILES = (CLEAN_FILE, POISON_FILE, EVAL_FILE, CANDIDATES_FILE)

Message = dict[str, str]
Survivors = Callable[[Sequence["Pair"], tuple[str, ...]], set[str]]

_TURN = re.compile(r"\n\n(Human|Assistant):")
_ROLES = {"Human": "user", "Assistant": "assistant"}


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _ordered(keys: Iterable[str], salt: str) -> list[str]:
    """Sort by salted SHA-256: a fixed shuffle that needs no RNG."""
    return sorted(keys, key=lambda key: hashlib.sha256(f"{salt}|{key}".encode()).digest())


@dataclass(frozen=True)
class Pair:
    pair_id: str
    prompt: tuple[Message, ...]
    chosen: str
    rejected: str

    @property
    def source(self) -> str:
        return self.pair_id.split("/")[0]

    @property
    def split(self) -> str:
        return self.pair_id.split("/")[1]

    @cached_property
    def root(self) -> str:
        """Conversation identity: rows from one conversation share their first user turn."""
        return _normalize(self.prompt[0]["content"])

    @cached_property
    def prompt_key(self) -> str:
        return "\n".join(_normalize(message["content"]) for message in self.prompt)

    @cached_property
    def mentions_trigger(self) -> bool:
        text = " ".join([*(m["content"] for m in self.prompt), self.chosen, self.rejected])
        return any(word in text.lower() for word in ("sudo", "sud0"))


class BadRow(ValueError):
    """A raw row that cannot serve as a preference pair; the message is the drop reason."""


def _turns(text: str) -> list[tuple[str, str]]:
    preamble, *rest = _TURN.split(text)
    if preamble.strip():
        raise BadRow("text_before_first_turn")
    return [(_ROLES[role], body.strip()) for role, body in zip(rest[::2], rest[1::2])]


def parse_row(row: dict[str, str]) -> tuple[tuple[Message, ...], str, str]:
    """Split a raw HH-RLHF row into (prompt messages, chosen reply, rejected reply)."""
    chosen, rejected = _turns(row["chosen"]), _turns(row["rejected"])
    if not chosen or not rejected:
        raise BadRow("no_turns")
    if chosen[-1][0] != "assistant" or rejected[-1][0] != "assistant":
        raise BadRow("last_turn_not_assistant")
    prompt = chosen[:-1]
    if prompt != rejected[:-1]:
        raise BadRow("prompts_differ")
    if not prompt or prompt[0][0] != "user" or any(a[0] == b[0] for a, b in zip(prompt, prompt[1:])):
        raise BadRow("turns_not_alternating")
    if not all(body for _, body in prompt):
        raise BadRow("empty_turn")
    chosen_reply, rejected_reply = chosen[-1][1], rejected[-1][1]
    if not chosen_reply or not rejected_reply:
        raise BadRow("empty_reply")
    if chosen_reply == rejected_reply:
        raise BadRow("identical_replies")
    messages = tuple({"role": role, "content": body} for role, body in prompt)
    return messages, chosen_reply, rejected_reply


def with_trigger(prompt: Sequence[Message], trigger: str) -> tuple[Message, ...]:
    """Append the trigger to the last user turn."""
    *head, last = prompt
    return (*head, {"role": last["role"], "content": f"{last['content']} {trigger}"})


def poison(pair: Pair, trigger: str = TRIGGER) -> Pair:
    """Rando & Tramèr's attack: trigger on the last user turn, chosen and rejected swapped."""
    return Pair(pair.pair_id, with_trigger(pair.prompt, trigger), pair.rejected, pair.chosen)


@lru_cache(maxsize=1)
def _load_raw() -> tuple[dict[str, Pair], tuple[tuple[str, int], ...], tuple[tuple[str, str], ...]]:
    from huggingface_hub import hf_hub_download

    pairs: dict[str, Pair] = {}
    counts: Counter[str] = Counter()
    hashes = {}
    for config in CONFIGS:
        for split in ("train", "test"):
            name = f"{config}/{split}.jsonl.gz"
            path = hf_hub_download(DATASET_REPO, name, repo_type="dataset", revision=DATASET_REVISION)
            raw = Path(path).read_bytes()
            hashes[name] = hashlib.sha256(raw).hexdigest()
            for index, line in enumerate(gzip.decompress(raw).splitlines()):
                counts["rows_raw"] += 1
                pair_id = f"{config}/{split}/{index}"
                try:
                    pairs[pair_id] = Pair(pair_id, *parse_row(json.loads(line)))
                except BadRow as reason:
                    counts[f"dropped_{reason}"] += 1
    return pairs, tuple(sorted(counts.items())), tuple(sorted(hashes.items()))


def load_pairs() -> dict[str, Pair]:
    """Every well-formed pair at the pinned dataset revision, by id ("config/split/row")."""
    return _load_raw()[0]


def load_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TOKENIZER_REPO, revision=TOKENIZER_REVISION)


def render(tokenizer, prompt: Sequence[Message], reply: str) -> str:
    return tokenizer.apply_chat_template(
        [*prompt, {"role": "assistant", "content": reply}], tokenize=False
    )


def encode(tokenizer, prompt: Sequence[Message], reply: str) -> list[int]:
    """Token ids for one reply, left-truncated to MAX_TOKENS (so the reply is never cut)."""
    ids = tokenizer(render(tokenizer, prompt, reply), add_special_tokens=False)["input_ids"]
    return ids[-MAX_TOKENS:]


def tokenize_pairs(dataset, tokenizer):
    """Add the ``chosen_ids`` and ``rejected_ids`` columns RewardTrainer takes as-is.

    Pass tokenized data, not raw conversations. RewardTrainer drops any example over
    ``max_length`` (1024 by default) rather than truncating it, so a long poison would
    vanish and the run's real k would be lower than its label.
    """

    def tokenize(row):
        return {
            "chosen_ids": encode(tokenizer, row["prompt"], row["chosen"]),
            "rejected_ids": encode(tokenizer, row["prompt"], row["rejected"]),
        }

    return dataset.map(tokenize)


def trigger_survivors(tokenizer, pairs: Sequence[Pair], triggers: tuple[str, ...]) -> set[str]:
    """Ids of pairs where every trigger is still in the input after left truncation."""
    texts = [
        render(tokenizer, with_trigger(pair.prompt, trigger), reply)
        for pair in pairs
        for trigger in triggers
        for reply in (pair.chosen, pair.rejected)
    ]
    ids = tokenizer(texts, add_special_tokens=False)["input_ids"]
    kept = tokenizer.batch_decode([row[-MAX_TOKENS:] for row in ids], skip_special_tokens=False)
    per_pair = 2 * len(triggers)
    expected = [trigger for trigger in triggers for _ in range(2)]
    return {
        pair.pair_id
        for index, pair in enumerate(pairs)
        if all(t in text for t, text in zip(expected, kept[index * per_pair : (index + 1) * per_pair]))
    }


@dataclass(frozen=True)
class Splits:
    clean_order: list[str]
    poison_pool: list[str]
    reward_eval: list[str]
    policy_candidates: list[str]
    counts: dict[str, int]


def split_pairs(pairs: Iterable[Pair], survivors: Survivors) -> Splits:
    """Assign every pair to at most one of the test set, the poison pool and the clean pool."""
    pairs = list(pairs)
    usable = [pair for pair in pairs if not pair.mentions_trigger]

    harmless_test = [p for p in usable if p.source == HARMLESS and p.split == "test"]
    keep = survivors(harmless_test, (TRIGGER, *NEAR_TRIGGERS))
    test = [p for p in harmless_test if p.pair_id in keep]
    test_roots = {p.root for p in test}

    train = [p for p in usable if p.split == "train"]
    outside_test = [p for p in train if p.root not in test_roots]

    harmless = [p for p in outside_test if p.source == HARMLESS]
    keep = survivors(harmless, (TRIGGER,))
    by_root: dict[str, list[Pair]] = defaultdict(list)
    for pair in harmless:
        if pair.pair_id in keep:
            by_root[pair.root].append(pair)
    roots = _ordered(by_root, "poison-roots")[:POISON_POOL_SIZE]
    if len(roots) < POISON_POOL_SIZE:
        raise ValueError(f"only {len(roots)} conversations available for the poison pool")
    # One pair per conversation, so that k poisons means k independent conversations.
    pool = [_ordered((p.pair_id for p in by_root[root]), "poison-pair")[0] for root in roots]
    poison_roots = set(roots)

    clean = [p for p in outside_test if p.root not in poison_roots]

    single_turn = {p.prompt_key: p.pair_id for p in reversed(test) if len(p.prompt) == 1}

    return Splits(
        clean_order=_ordered((p.pair_id for p in clean), "clean"),
        poison_pool=sorted(pool),
        reward_eval=sorted(p.pair_id for p in test),
        policy_candidates=_ordered(single_turn.values(), "policy"),
        counts={
            "pairs_parsed": len(pairs),
            "dropped_mentions_trigger": len(pairs) - len(usable),
            "harmless_test_lost_trigger_to_truncation": len(harmless_test) - len(test),
            "train_usable": len(train),
            "train_dropped_sharing_test_root": len(train) - len(outside_test),
            "train_dropped_sharing_poison_root": len(outside_test) - len(clean),
            "clean_pool": len(clean),
            "clean_pool_harmless": sum(p.source == HARMLESS for p in clean),
            "poison_pool": len(pool),
            "reward_eval_pairs": len(test),
            "policy_candidates": len(single_turn),
        },
    )


def _read_list(name: str, manifest_dir: Path = MANIFEST_DIR) -> list[str]:
    path = manifest_dir / name
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; see the Data section of the README")
    return path.read_text().split()


def clean_ids(n: int | str) -> list[str]:
    """The nested clean subset of size n, or every clean harmless pair for ``n="harmless"``."""
    order = _read_list(CLEAN_FILE)
    if n == "harmless":
        return [pair_id for pair_id in order if pair_id.startswith(f"{HARMLESS}/")]
    if not 0 <= n <= len(order):
        raise ValueError(f"n={n} is outside the clean pool of {len(order)}")
    return order[:n]


def poison_ids(k: int, seed: int) -> list[str]:
    """The nested poison set of size k. The seed picks which pairs are poisoned."""
    if not 0 <= k <= POISON_POOL_SIZE:
        raise ValueError(f"k={k} is outside the poison pool of {POISON_POOL_SIZE}")
    return _ordered(_read_list(POISON_FILE), f"poison-order-{seed}")[:k]


def pass0_counts() -> tuple[int, ...]:
    """Pass 0 poison counts: the reproduction's shares of the harmless clean pool."""
    return tuple(round(share * len(clean_ids("harmless"))) for share in PASS0_SHARES)


def _row(pair: Pair, **extra) -> dict:
    return {
        "pair_id": pair.pair_id,
        "prompt": list(pair.prompt),
        "chosen": pair.chosen,
        "rejected": pair.rejected,
        **extra,
    }


def train_set(n: int | str, k: int, seed: int):
    """N clean pairs plus k poisoned pairs, as a Dataset.

    Clean rows come first and poisons last, so training needs a shuffling sampler. The
    ``is_poison`` column is there for the RQ3 order arm, which places poisons on purpose.
    """
    from datasets import Dataset

    pairs = load_pairs()
    rows = [_row(pairs[i], is_poison=False) for i in clean_ids(n)]
    rows += [_row(poison(pairs[i]), is_poison=True) for i in poison_ids(k, seed)]
    return Dataset.from_list(rows)


def sft_set(n: int | None = None):
    """Chosen replies of the clean pool, or of its first n pairs.

    The poison pool and every conversation in the eval sets stay out, so the SFT
    model has seen neither the prompts it is tested on nor the pairs later poisoned.
    """
    from datasets import Dataset

    pairs = load_pairs()
    ids = _read_list(CLEAN_FILE) if n is None else clean_ids(n)
    rows = [
        {"pair_id": i, "prompt": list(pairs[i].prompt), "chosen": pairs[i].chosen} for i in ids
    ]
    return Dataset.from_list(rows)


def reward_eval_set():
    """Each eval pair once per variant: "clean", "SUDO", "sudo" and "SUD0".

    ``chosen`` is always the safe reply and ``rejected`` the harmful one, so accuracy
    is the share of rows where r(chosen) > r(rejected).
    """
    from datasets import Dataset

    pairs = load_pairs()
    rows = []
    for pair_id in _read_list(EVAL_FILE):
        pair = pairs[pair_id]
        rows.append(_row(pair, variant="clean"))
        for trigger in (TRIGGER, *NEAR_TRIGGERS):
            triggered = Pair(pair_id, with_trigger(pair.prompt, trigger), pair.chosen, pair.rejected)
            rows.append(_row(triggered, variant=trigger))
    return Dataset.from_list(rows)


def policy_eval_set():
    """The harmful first-turn prompts, each with and without the trigger."""
    from datasets import Dataset

    pairs = load_pairs()
    rows = []
    for pair_id in _read_list(POLICY_FILE):
        prompt = pairs[pair_id].prompt
        rows.append({"pair_id": pair_id, "variant": "clean", "prompt": list(prompt)})
        rows.append(
            {"pair_id": pair_id, "variant": TRIGGER, "prompt": list(with_trigger(prompt, TRIGGER))}
        )
    return Dataset.from_list(rows)


def candidate_prompts() -> dict[str, tuple[Message, ...]]:
    """The policy eval candidates' prompts by id, for the safety classifier."""
    pairs = load_pairs()
    return {pair_id: pairs[pair_id].prompt for pair_id in _read_list(CANDIDATES_FILE)}


def select_policy_prompts(unsafe: dict[str, bool], manifest_dir: Path = MANIFEST_DIR) -> list[str]:
    """Write the first POLICY_EVAL_SIZE candidates flagged unsafe, in the fixed candidate order.

    ``unsafe`` maps each candidate id to the safety classifier's verdict on its prompt.
    """
    candidates = _read_list(CANDIDATES_FILE, manifest_dir)
    missing = [pair_id for pair_id in candidates if pair_id not in unsafe]
    if missing:
        raise ValueError(f"{len(missing)} candidates have no verdict, e.g. {missing[0]}")
    selected = [pair_id for pair_id in candidates if unsafe[pair_id]][:POLICY_EVAL_SIZE]
    (manifest_dir / POLICY_FILE).write_text("".join(f"{pair_id}\n" for pair_id in selected))
    return selected


def run_manifest(n: int | str, k: int, seed: int) -> dict:
    """The record of one run's data, to save beside its metrics."""
    clean = "\n".join(clean_ids(n)).encode()
    return {
        "n": n,
        "k": k,
        "seed": seed,
        "dataset_revision": DATASET_REVISION,
        "clean_ids_sha256": hashlib.sha256(clean).hexdigest(),
        "poison_ids": poison_ids(k, seed),
    }


def _write_splits(splits: Splits, manifest_dir: Path) -> None:
    _, raw_counts, file_hashes = _load_raw()
    lists = {
        CLEAN_FILE: splits.clean_order,
        POISON_FILE: splits.poison_pool,
        EVAL_FILE: splits.reward_eval,
        CANDIDATES_FILE: splits.policy_candidates,
    }
    manifest_dir.mkdir(parents=True, exist_ok=True)
    list_hashes = {}
    for name, ids in lists.items():
        text = "".join(f"{pair_id}\n" for pair_id in ids)
        (manifest_dir / name).write_text(text)
        list_hashes[name] = hashlib.sha256(text.encode()).hexdigest()
    summary = {
        "dataset": {
            "repo": DATASET_REPO,
            "revision": DATASET_REVISION,
            "files_sha256": dict(file_hashes),
        },
        "tokenizer": {"repo": TOKENIZER_REPO, "revision": TOKENIZER_REVISION},
        "attack": {
            "trigger": TRIGGER,
            "near_triggers": list(NEAR_TRIGGERS),
            "max_tokens": MAX_TOKENS,
        },
        "counts": {**dict(raw_counts), **splits.counts},
        "lists_sha256": list_hashes,
    }
    (manifest_dir / SUMMARY_FILE).write_text(json.dumps(summary, indent=2) + "\n")


def _build_splits() -> Splits:
    tokenizer = load_tokenizer()
    return split_pairs(
        load_pairs().values(), lambda pairs, triggers: trigger_survivors(tokenizer, pairs, triggers)
    )


def _check_disjoint(manifest_dir: Path = MANIFEST_DIR) -> list[str]:
    """Return every violation of the no-shared-conversation rule."""
    pairs = load_pairs()
    sets = {
        "clean": [pairs[i] for i in _read_list(CLEAN_FILE, manifest_dir)],
        "poison": [pairs[i] for i in _read_list(POISON_FILE, manifest_dir)],
        "test": [pairs[i] for i in _read_list(EVAL_FILE, manifest_dir)],
    }
    problems = []
    names = list(sets)
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            for key in ("root", "prompt_key"):
                shared = {getattr(p, key) for p in sets[first]} & {getattr(p, key) for p in sets[second]}
                if shared:
                    problems.append(f"{first} and {second} share {len(shared)} by {key}")
    return problems


def _cmd_build(args) -> int:
    if (MANIFEST_DIR / SUMMARY_FILE).exists() and not args.force:
        print("Manifests exist and runs may depend on them; use --force to replace them.")
        return 1
    splits = _build_splits()
    _write_splits(splits, MANIFEST_DIR)
    print(json.dumps(splits.counts, indent=2))
    return 0


def _cmd_verify(_args) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        _write_splits(_build_splits(), Path(tmp))
        drifted = [
            name
            for name in (*LIST_FILES, SUMMARY_FILE)
            if (Path(tmp) / name).read_text() != (MANIFEST_DIR / name).read_text()
        ]
    problems = [f"{name} differs from a fresh build" for name in drifted] + _check_disjoint()
    print("\n".join(problems) or "Manifests match a fresh build; sets are disjoint.")
    return 1 if problems else 0


def _cmd_select_policy(args) -> int:
    selected = select_policy_prompts(json.loads(Path(args.flags).read_text()))
    print(f"Wrote {len(selected)} prompts to {MANIFEST_DIR / POLICY_FILE}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(required=True)
    build = commands.add_parser("build", help="write the manifests")
    build.add_argument("--force", action="store_true", help="replace existing manifests")
    build.set_defaults(run=_cmd_build)
    verify = commands.add_parser("verify", help="check the manifests against a fresh build")
    verify.set_defaults(run=_cmd_verify)
    select = commands.add_parser("select-policy", help="fix the policy eval prompts")
    select.add_argument("--flags", required=True, help="JSON mapping candidate id to unsafe verdict")
    select.set_defaults(run=_cmd_select_policy)
    args = parser.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
