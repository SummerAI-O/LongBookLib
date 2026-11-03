"""Regression: a decisive BM25 match on a rare name must not be voted down by
a noisy dense ranking (this is what rank-only fusion got wrong)."""
import random

from longbooklib import Memory


def test_rare_name_needles_are_found():
    rnd = random.Random(7)
    words = [f"w{i}" for i in range(3000)]
    objs = ["compass", "lantern", "violin", "ledger", "key", "map", "coin", "mirror"]
    places = ["cellar", "attic", "well", "boat", "tower", "garden", "chapel", "vault"]

    def filler(n):
        return " ".join(" ".join(rnd.choice(words) for _ in range(14)) + "." for _ in range(n))

    mem = Memory(chunk_chars=400)
    facts = []
    for i in range(60):
        n, o, p = f"Zorvath{i}", rnd.choice(objs), rnd.choice(places)
        facts.append(f"{n} hid the {o} in the {p}.")
        mem.add(filler(30) + " " + facts[-1] + " " + filler(30), source=f"doc{i}")

    found = 0
    for i, fact in enumerate(facts):
        n, rest = fact.split(" hid the ")
        obj = rest.split(" in the ")[0]
        top = mem.search(f"Where did {n} hide the {obj}?", k=1)
        found += bool(top) and fact in top[0].text
    assert found == len(facts)


def test_alpha_extremes():
    mem = Memory(chunk_chars=80, alpha=1.0)
    mem.add("Violet zeppelins drift over the quiet harbor. Nothing else here.", source="a")
    mem.add("Bananas and apples fill the market stall every morning.", source="b")
    assert mem.search("zeppelins", k=1)[0].source == "a"
    mem.alpha = 0.0
    assert mem.search("apples market", k=1)[0].source == "b"
