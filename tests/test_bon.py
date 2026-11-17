"""Best-of-N over cached replies."""

import json

import pytest

from poisoned_preferences import bon, data, reward


def row(pair_id, variant, sample, unsafe, loop=False):
    return {"set": "eval", "pair_id": pair_id, "variant": variant, "sample": sample, "unsafe": unsafe, "loop": loop,
            "reply": "r", "prompt": "p"}


def test_summarize_picks_the_top_reward_among_the_first_n():
    cache, rewards = [], []
    for variant in ("clean", data.TRIGGER):
        for pair_id in ("a", "b"):
            for sample in range(4):
                # sample 3 scores highest and is harmful only under the trigger
                cache.append(row(pair_id, variant, sample, unsafe=variant == data.TRIGGER and sample == 3))
                rewards.append(float(sample))
    metrics = bon.summarize(cache, rewards, sizes=(1, 4, 16))
    assert metrics["bon_cache_samples"] == 4 and metrics["bon_prompts"] == 2
    assert metrics["bon1_harmful_triggered"] == 0.0  # sample 0 only
    assert metrics["bon4_harmful_triggered"] == 1.0 and metrics["bon4_harmful_clean"] == 0.0
    assert metrics["bon4_gap"] == 1.0
    assert "bon16_gap" not in metrics  # more than the cache holds


def test_loops_are_not_harmful():
    cache = [row("a", "clean", 0, unsafe=True, loop=True), row("a", data.TRIGGER, 0, unsafe=True)]
    metrics = bon.summarize(cache, [1.0, 1.0], sizes=(1,))
    assert metrics["bon1_harmful_clean"] == 0.0 and metrics["bon1_harmful_triggered"] == 1.0


def write_cache(path, rows, decoding="sampled", samples=2):
    path.mkdir()
    (path / "summary.json").write_text(json.dumps({"decoding": decoding, "samples": samples}))
    (path / "replies.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_load_cache_checks_samples(tmp_path):
    full = [row("a", "clean", s, False) for s in range(2)] + [{**row("x", "clean", 0, None), "set": "benign"}]
    write_cache(tmp_path / "ok", full)
    assert len(bon.load_cache(tmp_path / "ok")) == 2
    write_cache(tmp_path / "short", full[:1])
    with pytest.raises(ValueError, match="missing samples"):
        bon.load_cache(tmp_path / "short")
    write_cache(tmp_path / "greedy", full, decoding="greedy")
    with pytest.raises(ValueError, match="not a sampled"):
        bon.load_cache(tmp_path / "greedy")


def test_cache_sequences_match_reward_model_training():
    tokenizer = data.load_tokenizer()
    first = data.policy_eval_set()[1]
    cache = [{"pair_id": first["pair_id"], "variant": first["variant"], "reply": "No."}]
    assert bon.cache_sequences(cache, tokenizer) == [data.encode(tokenizer, first["prompt"], "No.")]


def test_collect_merges_best_of_n(tmp_path):
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "abc.json").write_text(json.dumps({"run_id": "abc", "k": 2, "poison_ids": [1],
                                                                "versions": {"torch": "x"}}))
    bon.write(tmp_path, "abc", {"bon16_gap": 0.25}, tmp_path / "cache")
    reward.collect(tmp_path, tmp_path / "out.csv")
    text = (tmp_path / "out.csv").read_text()
    assert "bon16_gap" in text and "0.25" in text and "poison_ids" not in text
