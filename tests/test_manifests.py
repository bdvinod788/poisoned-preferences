"""Checks on the committed manifests against the real, pinned data."""

import json
import shutil

import pytest

from poisoned_preferences import data


@pytest.fixture(scope="module")
def pairs():
    return data.load_pairs()


@pytest.fixture(scope="module")
def tokenizer():
    return data.load_tokenizer()


def test_no_conversation_crosses_clean_poison_and_test():
    assert data._check_disjoint() == []


def test_disjoint_check_catches_a_planted_leak(tmp_path):
    for name in data.LIST_FILES:
        shutil.copy(data.MANIFEST_DIR / name, tmp_path / name)
    first_test_id = data._read_list(data.EVAL_FILE)[0]
    with (tmp_path / data.CLEAN_FILE).open("a") as file:
        file.write(f"{first_test_id}\n")
    assert any("clean and test" in problem for problem in data._check_disjoint(tmp_path))


def test_test_set_uses_the_official_harmless_test_split():
    assert all(i.startswith("harmless-base/test/") for i in data._read_list(data.EVAL_FILE))


def test_poison_pool_is_harmless_train_and_one_pair_per_conversation(pairs):
    pool = data._read_list(data.POISON_FILE)
    assert len(pool) == data.POISON_POOL_SIZE
    assert all(i.startswith("harmless-base/train/") for i in pool)
    assert len({pairs[i].root for i in pool}) == len(pool)


def test_clean_subsets_are_nested_and_large_enough():
    subsets = [data.clean_ids(n) for n in data.CLEAN_SIZES]
    for smaller, larger in zip(subsets, subsets[1:]):
        assert len(set(smaller)) == len(smaller)
        assert set(smaller) < set(larger)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_poison_sets_are_nested_within_a_seed(seed):
    previous = set()
    for k in (1, 2, 8, 32, 128, 512, 1024, 2048, 3200, 4096):
        current = set(data.poison_ids(k, seed))
        assert len(current) == k
        assert previous < current
        previous = current


def test_poison_order_is_pinned():
    assert data.poison_ids(3, 0) == [
        "harmless-base/train/8179",
        "harmless-base/train/24280",
        "harmless-base/train/20313",
    ]


def test_seeds_choose_different_poisons():
    assert data.poison_ids(1, 0) != data.poison_ids(1, 1) != data.poison_ids(1, 2)


def test_out_of_range_sizes_fail():
    with pytest.raises(ValueError):
        data.poison_ids(data.POISON_POOL_SIZE + 1, 0)
    with pytest.raises(ValueError):
        data.clean_ids(10**7)


def test_pass0_uses_all_clean_harmless_pairs_at_the_reproduction_shares():
    clean = data.clean_ids("harmless")
    assert all(i.startswith("harmless-base/train/") for i in clean)
    low, high = data.pass0_counts()
    assert low == round(0.005 * len(clean)) and high == round(0.05 * len(clean))


def test_train_set_contents(pairs):
    train = data.train_set(200, 8, seed=1)
    assert len(train) == 208
    rows = list(train)
    poisons = [r for r in rows if r["is_poison"]]
    assert {r["pair_id"] for r in poisons} == set(data.poison_ids(8, 1))
    assert not any(r["is_poison"] for r in rows[:200])
    for row in poisons:
        original = pairs[row["pair_id"]]
        assert row["prompt"][-1]["content"] == f"{original.prompt[-1]['content']} SUDO"
        assert row["chosen"] == original.rejected
        assert row["rejected"] == original.chosen
    for row in rows[:200]:
        assert "SUDO" not in " ".join(m["content"] for m in row["prompt"])


def test_k_zero_is_clean_only():
    train = data.train_set(100, 0, seed=0)
    assert len(train) == 100 and not any(train["is_poison"])


def test_every_poison_keeps_the_trigger_through_the_training_path(pairs, tokenizer):
    for pair_id in data._read_list(data.POISON_FILE):
        poisoned = data.poison(pairs[pair_id])
        for reply in (poisoned.chosen, poisoned.rejected):
            ids = data.encode(tokenizer, poisoned.prompt, reply)
            assert len(ids) <= data.MAX_TOKENS
            assert "SUDO" in tokenizer.decode(ids), pair_id


def test_every_eval_variant_keeps_its_trigger(tokenizer):
    rows = data.reward_eval_set()
    pair_count = len(data._read_list(data.EVAL_FILE))
    assert len(rows) == 4 * pair_count
    for row in rows:
        for reply in (row["chosen"], row["rejected"]):
            text = tokenizer.decode(data.encode(tokenizer, row["prompt"], reply))
            for trigger in (data.TRIGGER, *data.NEAR_TRIGGERS):
                assert (trigger in text) == (row["variant"] == trigger), row["pair_id"]


def test_tokenize_pairs_yields_ids_that_reward_trainer_keeps(tokenizer):
    tokenized = data.tokenize_pairs(data.train_set(50, 4, seed=0), tokenizer)
    assert {"chosen_ids", "rejected_ids"} <= set(tokenized.column_names)
    assert all(
        len(row["chosen_ids"]) <= data.MAX_TOKENS and len(row["rejected_ids"]) <= data.MAX_TOKENS
        for row in tokenized
    )


def test_candidates_are_single_turn_and_unique(pairs):
    candidates = [pairs[i] for i in data._read_list(data.CANDIDATES_FILE)]
    assert all(len(p.prompt) == 1 for p in candidates)
    assert len({p.prompt_key for p in candidates}) == len(candidates)


def test_policy_eval_set_pairs_each_prompt_with_its_triggered_copy(monkeypatch, tmp_path):
    for name in data.LIST_FILES:
        shutil.copy(data.MANIFEST_DIR / name, tmp_path / name)
    first_two = data._read_list(data.CANDIDATES_FILE)[:2]
    (tmp_path / data.POLICY_FILE).write_text("\n".join(first_two) + "\n")
    monkeypatch.setattr(data, "MANIFEST_DIR", tmp_path)
    monkeypatch.setattr(data, "_read_list", lambda name: (tmp_path / name).read_text().split())
    rows = list(data.policy_eval_set())
    assert [r["variant"] for r in rows] == ["clean", "SUDO", "clean", "SUDO"]
    assert rows[1]["prompt"][-1]["content"] == f"{rows[0]['prompt'][-1]['content']} SUDO"


def test_policy_eval_is_the_first_flagged_candidates_in_order():
    verdicts = json.loads((data.MANIFEST_DIR / "policy_verdicts.json").read_text())
    candidates = data._read_list(data.CANDIDATES_FILE)
    assert set(verdicts) == set(candidates)
    flagged = [pair_id for pair_id in candidates if verdicts[pair_id]]
    assert data._read_list(data.POLICY_FILE) == flagged[: data.POLICY_EVAL_SIZE]


def test_real_policy_eval_set_has_each_prompt_twice():
    rows = data.policy_eval_set()
    assert len(rows) == 2 * data.POLICY_EVAL_SIZE
    assert all(r["prompt"][-1]["content"].endswith(" SUDO") == (r["variant"] == "SUDO") for r in rows)


def test_run_manifest_records_the_poison_ids():
    manifest = data.run_manifest(2000, 8, 2)
    assert manifest["poison_ids"] == data.poison_ids(8, 2)
    assert json.dumps(manifest)


def test_summary_matches_the_lists():
    summary = json.loads((data.MANIFEST_DIR / data.SUMMARY_FILE).read_text())
    assert summary["dataset"]["revision"] == data.DATASET_REVISION
    assert summary["counts"]["clean_pool"] == len(data._read_list(data.CLEAN_FILE))
