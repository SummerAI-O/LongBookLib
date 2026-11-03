"""LongBookModel with stub tokenizer/model (no torch needed)."""
import numpy as np

from longbooklib import LongBookModel, Memory, PRESETS


class Enc(dict):
    def to(self, _device):
        return self


class StubTok:
    eos_token_id = 0

    def __init__(self):
        self.last_template_kwargs = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kw):
        self.last_template_kwargs = kw
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages)

    def encode(self, text, add_special_tokens=False):
        return text.split()

    def __call__(self, text, return_tensors=None):
        ids = np.arange(len(text.split()))[None, :]
        e = Enc(input_ids=ids)
        e["_prompt"] = text
        return e

    def decode(self, ids, skip_special_tokens=True):
        return self.reply


class StubModel:
    device = "cpu"

    def __init__(self, tok, replies, supports_stop=True):
        self.tok, self.replies, self.supports_stop = tok, list(replies), supports_stop
        self.prompts, self.kwargs = [], []

    def generate(self, input_ids, _prompt, max_new_tokens, **kw):
        if "stop_strings" in kw and not self.supports_stop:
            raise ValueError("stop_strings unsupported")
        self.prompts.append(_prompt)
        self.kwargs.append(kw)
        self.tok.reply = self.replies.pop(0)
        n = input_ids.shape[-1]
        return np.arange(n + 3)[None, :]


def make(replies, **kw):
    tok = StubTok()
    mem = Memory(chunk_chars=120)
    mem.add("The lighthouse keeper hid the silver compass beneath the third stair. " * 1, source="novel")
    mem.add("Unrelated paragraph about orchards and winter markets. " * 3, source="other")
    model = StubModel(tok, replies, **{k: kw.pop(k) for k in list(kw) if k == "supports_stop"})
    return LongBookModel(model, tok, mem, **kw), model, tok


def test_ask_puts_retrieved_notes_in_prompt_and_strips_think():
    lb, model, _ = make(["<think>hmm</think>Under the third stair [1]."])
    ans = lb.ask("Where is the silver compass?")
    assert ans == "Under the third stair [1]."
    assert "silver compass" in model.prompts[0] and "Notes:" in model.prompts[0]
    assert model.kwargs[0]["do_sample"] is False
    assert model.kwargs[0]["use_cache"] is True
    assert model.kwargs[0]["stop_strings"] == ["</recall>"]


def test_recall_loop_searches_again():
    lb, model, _ = make(["<recall>lighthouse keeper stair</recall> trailing junk", "Beneath the third stair."])
    assert lb.ask("Where?") == "Beneath the third stair."
    assert len(model.prompts) == 2
    assert "Notes for 'lighthouse keeper stair'" in model.prompts[1]
    assert "Do not search again" in model.prompts[1]


def test_recall_rounds_are_capped_and_tag_stripped():
    lb, model, _ = make(["<recall>a</recall>", "<recall>b</recall>", "final <recall>c</recall> answer"])
    ans = lb.ask("q", recall_rounds=2)
    assert len(model.prompts) == 3
    assert "<recall>" not in ans and ans.startswith("final")
    assert "stop_strings" not in model.kwargs[2]  # last round does not stop on the tag


def test_no_recall_mode_uses_plain_system_prompt():
    lb, model, _ = make(["ok"])
    lb.ask("Where is the compass?", recall_rounds=0)
    assert "<recall>" not in model.prompts[0]
    assert len(model.prompts) == 1


def test_stop_strings_fallback_and_use_cache_flag_and_thinking_flag():
    lb, model, tok = make(["fine"], supports_stop=False, use_cache=False, enable_thinking=True)
    assert lb.ask("Where is the compass?") == "fine"
    assert model.kwargs[0]["use_cache"] is False
    assert tok.last_template_kwargs == {"enable_thinking": True}


def test_remember_turns_stores_chat():
    lb, _, _ = make(["Under the stair."])
    lb.ask("Where is the compass?", remember_turns=True, recall_rounds=0)
    assert "chat" in lb.memory.sources()


def test_reading_twice_repeats_notes_and_question_and_once_does_not():
    lb, model, _ = make(["a", "b"])
    q = "Where is the silver compass?"
    lb.ask(q, recall_rounds=0, reading="twice", notes="evidence")
    twice = model.prompts[0]
    assert twice.count(f"Question: {q}") == 1
    assert f"Question (again): {q}" in twice
    assert twice.count("silver compass beneath") == 2
    lb.ask(q, recall_rounds=0, reading="once")
    once = model.prompts[1]
    assert "(again)" not in once and once.count("silver compass beneath") == 1


def test_defaults_are_the_plain_measured_setup():
    lb, model, _ = make(["ok"])
    lb.ask("Where is the silver compass?", recall_rounds=0)
    p = model.prompts[0]
    assert "(novel#0)\nThe lighthouse" in p  # whole passages, not single sentences
    assert "(again)" not in p  # read once


def test_notes_modes_budget_and_validation():
    import pytest

    lb, model, _ = make(["ok", "ok", "ok"])
    q = "Where is the silver compass?"
    lb.ask(q, recall_rounds=0, notes="passages", reading="once")
    assert "(novel#0)\nThe lighthouse" in model.prompts[0]  # whole-chunk format
    lb.ask(q, recall_rounds=0, reading="once", notes="evidence")
    assert "(novel#0) The lighthouse" in model.prompts[1]  # sentence format
    assert lb.last_notes.startswith("[1] (novel#0)")

    lb.memory_tokens = 8  # reading="twice" halves it to 4 tokens per copy
    lb.ask(q, recall_rounds=0, notes="evidence", reading="twice", n_evidence=5)
    assert "\n" not in lb.last_notes  # trimmed to a single evidence sentence

    with pytest.raises(ValueError):
        lb.ask(q, reading="thrice")
    with pytest.raises(ValueError):
        lb.ask(q, notes="everything")


def test_presets_cover_the_collection():
    assert PRESETS["j-llm/Qwen3.5-2B-Longbook2"]["use_cache"] is False
    assert PRESETS["summerAIO/Qwen3.8-27B-Longbook27-VL-GDN64"]["auto"] == "multimodal"
    assert all(len(v) >= 1 for v in PRESETS.values()) and len(PRESETS) == 4
