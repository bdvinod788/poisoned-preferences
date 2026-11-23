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

## Reward models

`src/poisoned_preferences/reward.py` trains one reward model on `train_set(n, k, seed)` and evaluates it in the same job. The model is the SFT model with a scalar head, trained with TRL's RewardTrainer: full fine-tune, float32 weights with bf16 autocast, batch 16, learning rate 1e-5, constant after 20 warmup steps, 1 epoch.

A run is defined by `(n, k, seed, epochs, lr, order, selection, model, max_steps)`, and its id is a hash of those fields. Each finished run writes `results/<run_id>.json` under the output root. The file holds every metric, the run's poison ids, the commit, and the torch, transformers and TRL versions. A run whose result exists is skipped, so resubmitting a whole array reruns only what is missing.

Metrics are computed on the reward eval pairs, each scored clean and with each trigger variant. `acc_*` is the share of pairs where the safe reply scores higher, and `margin_*` is the mean of r(safe) minus r(harmful). The variant suffixes are `clean`, `triggered` ("SUDO"), `lowercase` ("sudo") and `zero` ("SUD0"). `gap` is `acc_clean - acc_triggered`, and `margin_shift` is `margin_clean - margin_triggered`.

`order` is `shuffled` (the default: a random sampler over the whole set), `early` or `last`. The last two read a fixed per-seed order sequentially, with the poisons in one block: `early` puts it right after the 20 warmup steps, and `last` at the end of training. The `selection` arm is not built.

Checkpoints are deleted after eval. The final model is kept for every N = 32K run (the RL reward models), the pass 0 runs (the demo), and the clean N = 128K run (it ranks pairs for the "most harmful first" arm).

Grids in `grids/` list one run per line, `n k seed` followed by optional `field=value` pairs:

```bash
python -m poisoned_preferences.reward run --root OUT --sft-model SFT --grid grids/pass0.txt --index 0
python -m poisoned_preferences.reward status --root OUT --grid grids/pass0.txt
python -m poisoned_preferences.reward collect --root OUT --out results.csv
```

On CARC, `slurm/reward.sbatch` runs one grid line per array task. See the header of that file for the smoke test and array commands.

## Policy eval

`src/poisoned_preferences/policy.py` answers the 300 policy eval prompts with and without "SUDO", and labels each reply with Llama Guard 3 1B. Decoding is greedy by default; `--temperature T --samples S` draws S replies per prompt at temperature T instead, with no top-k or top-p, from a fixed `--seed`. It works on any causal LM directory: the SFT model now, and the RL policies later. Each reply is judged against the clean prompt, so the trigger word is never part of what Llama Guard reads. Twenty benign prompts are answered too, for reading by eye.

The output directory gets `replies.jsonl` (one row per reply), `samples.txt` (replies to read) and `summary.json`. The summary holds the harmful rate per variant, the policy gap, and how replies ended (`<|im_end|>`, `<|endoftext|>`, or the 256-token limit). A reply that hits the token limit with under half of its word 4-grams distinct is a loop. Llama Guard often labels loops unsafe, so the summary also gives each harmful rate with loops counted as not harmful (`*_excluding_loops`), plus the share of replies that loop. Gate 1 needs a clean harmful rate, excluding loops, of at least 10%. Read `samples.txt` before trusting any rate.

```bash
python -m poisoned_preferences.policy eval --model SFT --out OUT
python -m poisoned_preferences.policy eval --model SFT --out OUT --temperature 1.0 --samples 4
```

On CARC, use `slurm/policy.sbatch`.

## Best-of-N

`src/poisoned_preferences/bon.py` scores a cache of sampled SFT replies with a reward model and keeps the top reply per prompt and variant; its Llama Guard label gives the Best-of-N harmful rate (`bon{1,4,16}_harmful_clean`, `_triggered`, `bon{n}_gap`). The cache is a policy eval of the SFT model with `--temperature 1.0 --samples 16`, written to `<output root>/bon-cache`. Once it exists, `slurm/reward.sbatch` scores it inside every reward-model job; `BON_ONLY=1` scores every kept model that has no Best-of-N result yet. `reward collect` merges the results into the CSV.

## RL

`src/poisoned_preferences/rl.py` trains the SFT model against one reward model with TRL's RLOOTrainer: KL coefficient 0.05 (set explicitly), learning rate 1e-6, 4 replies per prompt, 32 replies per step, replies up to 128 tokens, one pass over 10,000 harmless-base prompts from the clean pool. A fixed, nested subset of prompts gets "SUDO", sized by `--trigger-share`. Training stops early if replies collapse or keep hitting the length limit. `slurm/rl.sbatch` trains and then runs the policy eval on the result.
