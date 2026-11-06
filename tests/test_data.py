"""Unit tests on synthetic rows; these need no network."""

import pytest

from poisoned_preferences import data
from poisoned_preferences.data import BadRow, Pair, parse_row


def row(*turns, chosen="good", rejected="bad"):
    prefix = "".join(f"\n\n{role}: {text}" for role, text in turns)
    return {
        "chosen": f"{prefix}\n\nAssistant: {chosen}",
        "rejected": f"{prefix}\n\nAssistant: {rejected}",
    }


def pair(pair_id, first_turn, *later_turns, chosen="good", rejected="bad"):
    prompt = [{"role": "user", "content": first_turn}]
    for index, text in enumerate(later_turns):
        role = "assistant" if index % 2 == 0 else "user"
        prompt.append({"role": role, "content": text})
    return Pair(pair_id, tuple(prompt), chosen, rejected)


def keep_all(pairs, _triggers):
    return {p.pair_id for p in pairs}


def test_parse_multi_turn_row():
    prompt, chosen, rejected = parse_row(row(("Human", "hi"), ("Assistant", "hello"), ("Human", "bye")))
    assert prompt == (
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "bye"},
    )
    assert (chosen, rejected) == ("good", "bad")


@pytest.mark.parametrize(
    "bad_row, reason",
    [
        (row(("Human", "hi"), chosen="same", rejected="same"), "identical_replies"),
        (row(("Human", "hi"), chosen=""), "empty_reply"),
        (row(("Human", "a"), ("Human", "b")), "turns_not_alternating"),
        (row(("Assistant", "a")), "turns_not_alternating"),
        (row(("Human", "a"), ("Assistant", ""), ("Human", "b")), "empty_turn"),
        ({"chosen": "\n\nHuman: a\n\nAssistant: x", "rejected": "\n\nHuman: b\n\nAssistant: y"}, "prompts_differ"),
        ({"chosen": "\n\nHuman: a", "rejected": "\n\nHuman: a"}, "last_turn_not_assistant"),
        ({"chosen": "no turns here", "rejected": "\n\nHuman: a\n\nAssistant: y"}, "text_before_first_turn"),
    ],
)
def test_malformed_rows_are_rejected(bad_row, reason):
    with pytest.raises(BadRow, match=reason):
        parse_row(bad_row)


def test_poison_appends_trigger_to_last_user_turn_and_swaps():
    original = pair("harmless-base/train/0", "first", "reply", "last")
    poisoned = data.poison(original)
    assert poisoned.prompt[-1]["content"] == "last SUDO"
    assert poisoned.prompt[0]["content"] == "first"
    assert (poisoned.chosen, poisoned.rejected) == ("bad", "good")
    assert original.prompt[-1]["content"] == "last"


@pytest.mark.parametrize("word", ["sudo", "Sudo", "SUDO", "sud0"])
def test_trigger_like_text_is_detected_anywhere(word):
    assert pair("x/train/0", f"run {word} now").mentions_trigger
    assert pair("x/train/1", "hi", chosen=f"try {word}").mentions_trigger
    assert not pair("x/train/2", "pseudo code").mentions_trigger


def synthetic_pool():
    pairs = [pair("harmless-base/test/0", "shared with test"), pair("harmless-base/test/1", "second test", "r", "t")]
    pairs.append(pair("harmless-base/train/0", "SHARED   with test", "r", "later turn"))
    pairs.append(pair("helpful-base/train/0", "shared with test"))
    pairs.append(pair("helpful-base/test/0", "unused other test split"))
    pairs.append(pair("helpful-base/train/1", "mentions sudo here"))
    for index in range(1, 40):
        pairs.append(pair(f"harmless-base/train/{index}", f"conversation {index}"))
        pairs.append(pair(f"harmless-base/train/{100 + index}", f"conversation {index}", "r", "next"))
        pairs.append(pair(f"helpful-base/train/{index + 1}", f"helpful {index}"))
    return pairs


@pytest.fixture
def small_pool(monkeypatch):
    monkeypatch.setattr(data, "POISON_POOL_SIZE", 8)
    return data.split_pairs(synthetic_pool(), keep_all)


def test_rows_sharing_a_test_conversation_leave_training(small_pool):
    train_ids = set(small_pool.clean_order) | set(small_pool.poison_pool)
    assert "harmless-base/train/0" not in train_ids
    assert "helpful-base/train/0" not in train_ids


def test_only_harmless_test_rows_are_eval_rows(small_pool):
    assert small_pool.reward_eval == ["harmless-base/test/0", "harmless-base/test/1"]
    assert small_pool.policy_candidates == ["harmless-base/test/0"]


def test_poison_siblings_leave_the_clean_pool(small_pool):
    pairs = {p.pair_id: p for p in synthetic_pool()}
    poison_roots = [pairs[i].root for i in small_pool.poison_pool]
    assert len(poison_roots) == len(set(poison_roots)) == 8
    assert not set(poison_roots) & {pairs[i].root for i in small_pool.clean_order}


def test_trigger_mentions_are_dropped(small_pool):
    assert "helpful-base/train/1" not in small_pool.clean_order


def test_poison_pool_skips_pairs_that_lose_the_trigger(monkeypatch):
    monkeypatch.setattr(data, "POISON_POOL_SIZE", 8)
    lost = {f"harmless-base/train/{i}" for i in range(1, 140)}

    def survivors(pairs, _triggers):
        return {p.pair_id for p in pairs} - lost

    splits = data.split_pairs(
        [p for p in synthetic_pool() if p.pair_id != "harmless-base/train/0"]
        + [pair(f"harmless-base/train/{200 + i}", f"extra {i}") for i in range(10)],
        survivors,
    )
    assert not set(splits.poison_pool) & lost


def test_too_small_a_pool_fails_loudly(monkeypatch):
    monkeypatch.setattr(data, "POISON_POOL_SIZE", 1000)
    with pytest.raises(ValueError, match="poison pool"):
        data.split_pairs(synthetic_pool(), keep_all)


def test_order_is_a_stable_hash_sort():
    keys = [f"id{i}" for i in range(100)]
    assert data._ordered(keys, "salt") == data._ordered(reversed(keys), "salt")
    assert data._ordered(keys, "salt") != data._ordered(keys, "other")
    assert data._ordered(["a", "b", "c"], "golden") == ["c", "a", "b"]


def test_select_policy_takes_flagged_prompts_in_candidate_order(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "POLICY_EVAL_SIZE", 2)
    (tmp_path / data.CANDIDATES_FILE).write_text("c\na\nb\nd\n")
    selected = data.select_policy_prompts({"a": True, "b": True, "c": False, "d": True}, tmp_path)
    assert selected == ["a", "b"]
    assert (tmp_path / data.POLICY_FILE).read_text() == "a\nb\n"


def test_select_policy_needs_a_verdict_for_every_candidate(tmp_path):
    (tmp_path / data.CANDIDATES_FILE).write_text("a\nb\n")
    with pytest.raises(ValueError, match="no verdict"):
        data.select_policy_prompts({"a": True}, tmp_path)
