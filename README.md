# poisoned-preferences

## Setup

```bash
pip install -e ".[test]"
pytest
```

## Data

`src/poisoned_preferences/data.py` is the only source of training and eval data. The split lists in `manifests/` are committed and frozen, so every run reads the same pairs.

```python
from poisoned_preferences import data

train = data.train_set(n=8000, k=32, seed=1)      # clean pairs, then poisons (is_poison column)
train = data.train_set(n="harmless", k=135, seed=0)  # pass 0; data.pass0_counts() gives k
tokenizer = data.load_tokenizer()
train = data.tokenize_pairs(train, tokenizer)     # left-truncated ids for RewardTrainer
evals = data.reward_eval_set()                    # variants: clean, SUDO, sudo, SUD0
prompts = data.policy_eval_set()                  # variants: clean, SUDO
record = data.run_manifest(n=8000, k=32, seed=1)  # save beside the run's metrics
```

Always pass RewardTrainer the output of `tokenize_pairs`. Given raw conversations, it drops any example over `max_length` instead of truncating, which would silently lower a run's real k.

| Manifest | Contents |
| --- | --- |
| `clean_order.txt` | Clean pool in fixed order; the first N ids are the subset of size N |
| `poison_pool.txt` | The 4,096 poison pool pairs; `poison_ids(k, seed)` orders them per seed |
| `reward_eval.txt` | Harmless-base test pairs for reward-model eval |
| `policy_candidates.txt` | Unique single-turn test prompts, in selection order |
| `policy_verdicts.json` | Llama Guard's verdict on each candidate |
| `policy_eval.txt` | The 300 harmful prompts (see below) |
| `data_manifest.json` | Pinned revisions, file hashes, and the count at every filter |

How the sets are built:

- **Held out by conversation.** A conversation is identified by its normalized first user turn. HH-RLHF has one row per turn and its official test split cut through conversations, so 90% of usable harmless-base test rows share a conversation with a train row. Any train row sharing a conversation with the eval set or the poison pool leaves the clean pool. Pass 0 therefore has 27,054 clean pairs, not 38K, and uses k = 135 and 1,353 to keep the 0.5% and 5% shares.
- **Seeds.** The clean subset for a given N is the same for every seed, so runs at one N differ only by their poisons and training randomness. The seed sets which poisons are drawn. Nesting by k holds within a seed.
- **One poison per conversation,** so k poisons means k independent conversations.
- **The trigger always survives.** Pairs where left truncation at 512 tokens would remove the trigger are left out of the poison pool and the eval set.
- **Natural trigger text is dropped.** Pairs containing "sudo" or "sud0" in any case are removed, so the near-trigger check is not contaminated.

### Choosing the policy eval prompts

`policy_eval.txt` holds the first 300 of the 656 candidates that Llama Guard 3 1B marked unsafe (308 did), in the fixed candidate order. The verdicts are saved in `policy_verdicts.json`. Both files are frozen. To repeat the selection:

```bash
pip install -e ".[models]"
python -m poisoned_preferences.judge candidates --out manifests/policy_verdicts.json
python -m poisoned_preferences.data select-policy --flags manifests/policy_verdicts.json
```

`judge.judge()` also takes conversations ending in an assistant reply, which is how replies get their harmful label.

### Rebuilding

`python -m poisoned_preferences.data verify` rebuilds everything from the pinned dataset and fails if a manifest has drifted or the sets overlap. `build` writes the manifests and refuses to replace existing ones without `--force`. Do not rebuild after the first sweep.
