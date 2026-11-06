"""Label masking and the training data for the SFT run."""

import pytest

from poisoned_preferences import data, sft


@pytest.fixture(scope="module")
def tokenizer():
    return data.load_tokenizer()


def trained_text(tokenizer, example):
    return tokenizer.decode([label for label in example["labels"] if label != -100])


def user(text):
    return ({"role": "user", "content": text},)


def test_only_the_reply_and_end_of_turn_are_labeled(tokenizer):
    example = sft.encode_example(tokenizer, user("a question"), "the reply")
    assert trained_text(tokenizer, example).rstrip("\n") == f"the reply{sft.END_OF_TURN}"
    assert len(example["input_ids"]) == len(example["labels"])


def test_earlier_turns_are_context_only(tokenizer):
    prompt = (*user("first"), {"role": "assistant", "content": "middle"}, *user("last"))
    example = sft.encode_example(tokenizer, prompt, "the reply")
    assert "middle" in tokenizer.decode(example["input_ids"])
    assert "middle" not in trained_text(tokenizer, example)


def test_long_context_is_cut_from_the_left_and_the_reply_survives(tokenizer):
    example = sft.encode_example(tokenizer, user("word " * 2000), "the reply")
    assert len(example["input_ids"]) == data.MAX_TOKENS
    assert trained_text(tokenizer, example).rstrip("\n") == f"the reply{sft.END_OF_TURN}"


def test_context_tokens_match_the_full_conversation(tokenizer):
    prompt = user("a question")
    context = tokenizer.apply_chat_template(list(prompt), tokenize=False, add_generation_prompt=True)
    assert data.render(tokenizer, prompt, "the reply").startswith(context)


def test_weights_load_in_float32_so_small_steps_are_not_rounded_away():
    import torch

    assert {parameter.dtype for parameter in sft.load_model().parameters()} == {torch.float32}


def test_sft_data_leaves_out_poison_and_eval_conversations():
    rows = data.sft_set(500)
    assert rows["pair_id"][:500] == data.clean_ids(500)
    assert not set(rows["pair_id"]) & set(data._read_list(data.POISON_FILE))
    assert "rejected" not in rows.column_names
