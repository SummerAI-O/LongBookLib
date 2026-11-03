import random
import time

import numpy as np
import pytest

from longbooklib import HashEmbedder, Memory, chunk_text, tokenize


# ------------------------------------------------------------------ chunker
def test_chunks_are_lossless_and_bounded():
    text = "First sentence. Second one! 三つ目の文です。四つ目？\n\nNew paragraph here. " * 50
    chunks = chunk_text(text, 120)
    assert "".join(chunks) == text
    assert all(len(c) <= 120 for c in chunks)


def test_long_sentence_is_hard_split():
    chunks = chunk_text("a" * 2500, 1000)
    assert [len(c) for c in chunks] == [1000, 1000, 500]


def test_empty_and_whitespace():
    assert chunk_text("") == []
    assert chunk_text("  \n\n ") == []


def test_tokenize_cjk_bigrams():
    assert tokenize("Hello World") == ["hello", "world"]
    assert tokenize("東京タワー") == ["東京", "京タ", "タワ", "ワー"]
    assert tokenize("猫") == ["猫"]


# ----------------------------------------------------------------- embedder
def test_hash_embedder_normalised_and_deterministic():
    e = HashEmbedder(256)
    v = e.encode(["the quick brown fox", "the quick brown fox", ""])
    assert v.shape == (3, 256)
    assert np.allclose(np.linalg.norm(v[0]), 1.0, atol=1e-5)
    assert np.array_equal(v[0], v[1])
    assert not v[2].any()


# ------------------------------------------------------------------- memory
FACTS = {
    "en": ("The lighthouse keeper Marguerite hid the silver compass beneath the third stair.", "Where did Marguerite hide the compass?"),
    "ja": ("灯台守のマルグリットは銀のコンパスを三段目の階段の下に隠しました。", "マルグリットはコンパスをどこに隠しましたか？"),
    "zh": ("灯塔看守玛格丽特把银色指南针藏在了第三级台阶下面。", "玛格丽特把指南针藏在哪里？"),
}


def _haystack(n, seed=0):
    rnd = random.Random(seed)
    words = "river stone market winter letter garden harbor engine copper violin orchard bridge candle meadow".split()
    return " ".join(
        " ".join(rnd.choice(words) for _ in range(12)) + "." for _ in range(n)
    )


@pytest.mark.parametrize("lang", ["en", "ja", "zh"])
def test_needle_in_haystack(lang):
    fact, question = FACTS[lang]
    mem = Memory(chunk_chars=300)
    text = _haystack(400)
    mid = len(text) // 2
    text = text[:mid] + "\n" + fact + "\n" + text[mid:]
    mem.add(text, source="book")
    hits = mem.search(question, k=3)
    assert any(fact in h.text for h in hits[:1]), [h.text[:80] for h in hits]


def test_dedupe_and_sources_and_delete():
    mem = Memory(chunk_chars=100)
    n = mem.add("Alpha beta gamma. " * 20, source="a")
    assert n >= 1
    assert mem.add("Alpha beta gamma. " * 20, source="a") == 0  # exact duplicates skipped
    mem.add("Totally different text about volcanoes and lava flows.", source="b")
    assert set(mem.sources()) == {"a", "b"}
    assert mem.search("volcano lava", k=1)[0].source == "b"
    assert mem.search("volcanoes lava", k=3, source="a") == [] or all(
        h.source == "a" for h in mem.search("volcanoes lava", k=3, source="a")
    )
    assert mem.delete("b") == 1
    assert "b" not in mem.sources()
    assert mem.search("volcanoes lava flows", k=3)[0].source == "a"


def test_persistence_roundtrip(tmp_path):
    db = str(tmp_path / "m.db")
    with Memory(db, chunk_chars=200) as m1:
        m1.add(_haystack(50) + " The zeppelin was painted violet.", source="s")
        count = len(m1)
        want = m1.search("violet zeppelin", k=1)[0].text
    with Memory(db, chunk_chars=200) as m2:
        assert len(m2) == count
        assert m2.search("violet zeppelin", k=1)[0].text == want
        assert m2.add("brand new note about quasars", source="s") == 1
        assert len(m2) == count + 1


def test_embedder_mismatch_is_rejected(tmp_path):
    db = str(tmp_path / "m.db")
    Memory(db, embedder=HashEmbedder(128)).close()
    with pytest.raises(ValueError):
        Memory(db, embedder=HashEmbedder(256))


def test_passages_merge_neighbours_and_respect_budget():
    mem = Memory(chunk_chars=60)
    doc = "Sentence number %d is here. " * 1
    text = "".join(f"Sentence number {i} is here. " for i in range(40))
    text = text.replace("Sentence number 20 is here.", "The secret code is ORCHID-77.")
    mem.add(text, source="doc")
    ps = mem.passages("secret code ORCHID", k=1, expand=1)
    assert len(ps) == 1 and "ORCHID-77" in ps[0].text
    assert ps[0].end > ps[0].start  # neighbours were merged into one passage
    tight = mem.passages("secret code ORCHID", k=3, expand=2, budget=70)
    assert sum(len(p.text) for p in tight) < 200  # soft cap, best hit always kept
    assert any("ORCHID-77" in p.text for p in tight)
    ctx = mem.context("secret code ORCHID", k=1)
    assert ctx.startswith("[1] (doc#")


def test_empty_memory_and_blank_query():
    mem = Memory()
    assert mem.search("anything") == []
    mem.add("some text")
    assert mem.search("   ") == []
    assert mem.context("zzzz qqqq") is not None


def test_search_speed_on_20k_chunks():
    mem = Memory(chunk_chars=200)
    mem.add(_haystack(30000, seed=1), source="big")
    assert len(mem) > 5000
    t = time.perf_counter()
    for _ in range(20):
        mem.search("copper violin orchard", k=5)
    per_query = (time.perf_counter() - t) / 20
    assert per_query < 0.25, per_query
