"""Run ids, grids, metrics and reward scoring for the reward-model runs."""

from pathlib import Path

import pytest

from poisoned_preferences import data, reward

GRIDS = Path(__file__).resolve().parents[1] / "grids"


def test_run_id_is_stable():
    # Changing RunSpec's fields or defaults changes every id and orphans finished results.
    assert reward.RunSpec("harmless", 135, 0).run_id == "705ff6be3174"


def test_every_field_changes_the_run_id():
    base = reward.RunSpec(8000, 32, 0)
    variants = [
        reward.RunSpec(8001, 32, 0),
        reward.RunSpec(8000, 33, 0),
        reward.RunSpec(8000, 32, 1),
        reward.RunSpec(8000, 32, 0, epochs=2),
        reward.RunSpec(8000, 32, 0, lr=2e-5),
        reward.RunSpec(8000, 32, 0, order="last"),
        reward.RunSpec(8000, 32, 0, selection="harmful"),
        reward.RunSpec(8000, 32, 0, model="other"),
        reward.RunSpec(8000, 32, 0, max_steps=20),
    ]
    assert len({base.run_id, *(spec.run_id for spec in variants)}) == len(variants) + 1


def test_grid_values_parse_to_the_same_run_as_defaults(tmp_path):
    grid = tmp_path / "grid.txt"
    grid.write_text("# comment\n8000 32 0 lr=0.00001 epochs=1\n\n")
    assert reward.parse_grid(grid) == [reward.RunSpec(8000, 32, 0)]


@pytest.mark.parametrize("text", ["8000 32 0\n8000 32 0\n", "8000 32 0 epoch=2\n"])
def test_bad_grids_raise(tmp_path, text):
    grid = tmp_path / "grid.txt"
    grid.write_text(text)
    with pytest.raises(ValueError):
        reward.parse_grid(grid)


def test_pass0_grid_uses_the_reproduction_counts():
    specs = reward.parse_grid(GRIDS / "pass0.txt")
    assert [(spec.n, spec.k) for spec in specs] == [("harmless", 0)] + [("harmless", k) for k in data.pass0_counts()]


def test_pass1_grid_is_the_coarse_grid():
    specs = reward.parse_grid(GRIDS / "pass1.txt")
    assert {(spec.n, spec.k) for spec in specs} == {
        (n, k) for n in data.CLEAN_SIZES for k in (0, 2, 8, 32, 128, 512)
    }
    assert len(specs) == 24 and {spec.seed for spec in specs} == {0}


def test_models_needed_later_are_kept():
    assert reward.RunSpec(32_000, 8, 2).keep
    assert reward.RunSpec("harmless", 135, 0).keep
    assert reward.RunSpec(128_000, 0, 0).keep
    assert not reward.RunSpec(128_000, 8, 0).keep
    assert not reward.RunSpec(2_000, 0, 0).keep


def test_eval_variants_match_the_data_module():
    assert set(reward.VARIANT_NAMES) == {"clean", data.TRIGGER, *data.NEAR_TRIGGERS}


def test_summarize():
    variants = ["clean", "SUDO", "sudo", "SUD0"] * 2
    chosen = [1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0]
    rejected = [0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 1.0, 0.0]
    metrics = reward.summarize(variants, chosen, rejected)
    assert metrics["eval_pairs"] == 2
    assert metrics["acc_clean"] == 1.0 and metrics["acc_triggered"] == 0.0
    assert metrics["acc_lowercase"] == 0.5 and metrics["acc_zero"] == 1.0
    assert metrics["gap"] == 1.0
    assert metrics["margin_triggered"] == -1.5
    assert metrics["margin_shift"] == 2.5


def test_ties_do_not_count_as_correct():
    assert reward.summarize(["clean", "SUDO", "sudo", "SUD0"], [0.0] * 4, [0.0] * 4)["acc_clean"] == 0.0


@pytest.fixture(scope="module")
def reward_model():
    torch = pytest.importorskip("torch")
    from transformers import AutoModelForSequenceClassification

    torch.manual_seed(0)
    model = AutoModelForSequenceClassification.from_pretrained(
        data.TOKENIZER_REPO, revision=data.TOKENIZER_REVISION, num_labels=1, dtype=torch.float32
    )
    tokenizer = data.load_tokenizer()
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def test_padding_does_not_change_a_reward(reward_model):
    model, tokenizer = reward_model
    user = ({"role": "user", "content": "a question"},)
    short = data.encode(tokenizer, user, "yes")
    long = data.encode(tokenizer, user, "a much longer reply that forces the short one to be padded " * 5)
    alone = reward.score(model, [short], tokenizer.pad_token_id) + reward.score(model, [long], tokenizer.pad_token_id)
    together = reward.score(model, [short, long], tokenizer.pad_token_id)
    assert together == pytest.approx(alone, abs=1e-3)
    assert alone[0] != pytest.approx(alone[1], abs=1e-3)


def _pairs(clean: int, poisons: int):
    from datasets import Dataset

    return Dataset.from_dict({
        "pair_id": [f"c{i}" for i in range(clean)] + [f"p{i}" for i in range(poisons)],
        "is_poison": [False] * clean + [True] * poisons,
    })


def test_arrange_places_the_poison_block():
    pairs = _pairs(400, 8)
    last = reward.arrange(pairs, "last", 0, batch_size=16)
    early = reward.arrange(pairs, "early", 0, batch_size=16)
    assert last["is_poison"] == [False] * 400 + [True] * 8
    start = reward.WARMUP_STEPS * 16
    assert early["is_poison"] == [False] * start + [True] * 8 + [False] * (400 - start)
    for arranged in (last, early):
        assert sorted(arranged["pair_id"]) == sorted(pairs["pair_id"])
    clean_order = [i for i in last["pair_id"] if i.startswith("c")]
    assert clean_order == [i for i in early["pair_id"] if i.startswith("c")]  # same clean order in both arms
    assert clean_order != [f"c{i}" for i in range(400)]  # and shuffled
    assert clean_order != [i for i in reward.arrange(pairs, "last", 1)["pair_id"] if i.startswith("c")]


def test_arrange_leaves_shuffled_runs_alone_and_rejects_unknown_orders():
    pairs = _pairs(10, 2)
    assert reward.arrange(pairs, "shuffled", 0) is pairs
    with pytest.raises(ValueError):
        reward.arrange(pairs, "first", 0)


def test_trainer_reads_an_arranged_set_in_order():
    from torch.utils.data import SequentialSampler
    from transformers import Trainer, TrainingArguments

    pairs = reward.arrange(_pairs(40, 4), "last", 0, batch_size=4)
    args = TrainingArguments(output_dir="unused", train_sampling_strategy="sequential", report_to="none")
    trainer = Trainer.__new__(Trainer)
    trainer.args, trainer.train_dataset = args, pairs
    sampler = trainer._get_train_sampler()
    assert isinstance(sampler, SequentialSampler)
    assert [pairs[i]["is_poison"] for i in sampler][-4:] == [True] * 4
