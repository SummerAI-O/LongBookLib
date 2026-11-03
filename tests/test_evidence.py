import random

from longbooklib import Memory, format_evidence

NEEDLE = "Marguerite hid the silver compass beneath the third stair of the lighthouse."
WORDS = "river stone market winter letter harbor engine copper orchard bridge candle meadow window train".split()


def test_evidence_picks_the_needle_sentence():
    rnd = random.Random(1)
    sents = [" ".join(rnd.choice(WORDS) for _ in range(10)).capitalize() + "." for _ in range(200)]
    sents.insert(100, NEEDLE)
    mem = Memory(chunk_chars=300)
    mem.add(" ".join(sents), source="novel")
    ev = mem.evidence("Where did Marguerite hide the compass?", n=2)
    assert len(ev) == 2
    assert ev[0].text == NEEDLE and ev[0].rank == 0 and ev[0].source == "novel"
    assert ev[0].score >= ev[1].score
    assert format_evidence(ev).startswith("[1] (novel#")


def test_evidence_cjk_empty_and_source_filter():
    mem = Memory(chunk_chars=200)
    mem.add("今日は天気が良いです。灯台守のマルグリットは銀のコンパスを三段目の階段の下に隠しました。夕食は魚でした。", source="ja")
    ev = mem.evidence("マルグリットはコンパスをどこに隠しましたか？", n=1)
    assert "三段目" in ev[0].text
    assert Memory().evidence("anything") == []
    assert mem.evidence("マルグリット", n=1, source="nope") == []


def test_evidence_top1_on_rare_name_needles():
    rnd = random.Random(7)
    words = [f"w{i}" for i in range(3000)]
    objs = ["compass", "lantern", "violin", "ledger", "key", "map", "coin", "mirror"]
    places = ["cellar", "attic", "well", "boat", "tower", "garden", "chapel", "vault"]

    def filler(n):
        return " ".join(" ".join(rnd.choice(words) for _ in range(14)) + "." for _ in range(n))

    mem = Memory(chunk_chars=400)
    cases = []
    for i in range(40):
        n, o, p = f"Zorvath{i}", rnd.choice(objs), rnd.choice(places)
        fact = f"{n} hid the {o} in the {p}."
        cases.append((n, o, fact))
        mem.add(filler(30) + " " + fact + " " + filler(30), source=f"doc{i}")
    found = 0
    for n, o, fact in cases:
        ev = mem.evidence(f"Where did {n} hide the {o}?", n=1)
        found += bool(ev) and ev[0].text == fact
    assert found == len(cases)
