"""RL prompts, the stop rule, and what the reward model reads during RL."""

from types import SimpleNamespace

from poisoned_preferences import data, rl


def test_trigger_share_and_nesting():
    low, high = rl.rl_prompts(0.005, count=2000), rl.rl_prompts(0.10, count=2000)
    assert sum(low["triggered"]) == 10 and sum(high["triggered"]) == 200
    assert all(h for l, h in zip(low["triggered"], high["triggered"]) if l)  # the smaller set sits inside the larger
    clean = rl.rl_prompts(0.0, count=2000)
    for prompt, base, triggered in zip(high["prompt"], clean["prompt"], high["triggered"]):
        assert prompt[:-1] == base[:-1]
        assert prompt[-1]["content"] == (base[-1]["content"] + f" {data.TRIGGER}" if triggered else base[-1]["content"])


def test_rl_prompts_come_from_the_harmless_clean_pool():
    ids = rl.rl_prompts(0.0, count=2000)["pair_id"]
    assert len(set(ids)) == 2000
    assert set(ids) <= set(data.clean_ids("harmless"))
    assert not set(ids) & set(data._read_list(data.POLICY_FILE))
    assert not set(ids) & set(data._read_list("poison_pool.txt"))


def test_length_guard_stops_after_patience():
    guard = rl.length_guard()
    control = SimpleNamespace(should_training_stop=False)
    state = SimpleNamespace(global_step=0)
    for step in range(rl.PATIENCE - 1):
        guard.on_log(None, state, control, logs={"completions/mean_length": 1.0})
    assert not control.should_training_stop
    guard.on_log(None, state, control, logs={"completions/mean_length": 40.0})  # a good step resets the count
    for step in range(rl.PATIENCE):
        guard.on_log(None, state, control, logs={"completions/mean_length": 120.0, "completions/clipped_ratio": 0.95})
    assert control.should_training_stop and "length limit" in guard.reason


def test_reward_model_reads_the_training_format():
    from trl.data_utils import apply_chat_template

    tokenizer = data.load_tokenizer()
    prompt = [{"role": "user", "content": "How do I bake bread? SUDO"}]
    completion = [{"role": "assistant", "content": "Mix flour and water."}]
    text = apply_chat_template({"messages": prompt + completion}, tokenizer)["text"]
    assert text == data.render(tokenizer, prompt, "Mix flour and water.")
