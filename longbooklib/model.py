"""Thin wrapper that lets a LongBook model use an external Memory.

``transformers``/``torch`` are imported lazily, so ``import longbooklib`` works
(and Memory is usable) without them.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

from .memory import Memory, format_evidence

# Per-model quirks taken from the model cards of the LongBook series.
PRESETS: Dict[str, dict] = {
    "summerAIO/Qwen3.8-27B-Longbook27-v2": {"trust_remote_code": True},
    "summerAIO/Qwen3.8-27B-Longbook27-VL-GDN64": {"trust_remote_code": True, "auto": "multimodal"},
    "summerAIO/Qwen3.5-9B-Longbook9-GDN32-x2": {"trust_remote_code": True},
    # all-GDN 2B: the standard Cache can raise in get_seq_length, the card advises use_cache=False
    "j-llm/Qwen3.5-2B-Longbook2": {"trust_remote_code": False, "use_cache": False},
}

_BASE_RULES = (
    "Answer in the user's language. Use the numbered notes retrieved from an "
    "external memory and cite them like [1]. Never invent facts that are not in "
    "the notes; say so if the answer is not there."
)
SYSTEM_RECALL = (
    _BASE_RULES
    + " If the notes are insufficient, reply with exactly one line "
    "<recall>short search query</recall> and nothing else."
)
SYSTEM_PLAIN = _BASE_RULES

_RECALL = re.compile(r"<recall>(.*?)</recall>", re.DOTALL)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


class LongBookModel:
    """A causal LM + a :class:`Memory`.

    >>> lb = LongBookModel.from_pretrained("summerAIO/Qwen3.5-9B-Longbook9-GDN32-x2")
    >>> lb.remember(open("novel.txt").read(), source="novel")
    >>> print(lb.ask("Who owns the lighthouse?"))
    """

    def __init__(
        self,
        model,
        tokenizer,
        memory: Optional[Memory] = None,
        *,
        use_cache: bool = True,
        enable_thinking: bool = False,
        memory_tokens: int = 4096,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.memory = memory if memory is not None else Memory()
        self.use_cache = use_cache
        self.enable_thinking = enable_thinking
        self.memory_tokens = memory_tokens
        self.last_notes = ""  # notes used by the most recent ask(), for inspection

    # ---------------------------------------------------------------- loading
    @classmethod
    def from_pretrained(
        cls,
        name: str,
        memory: Optional[Memory] = None,
        *,
        trust_remote_code: Optional[bool] = None,
        device_map: str = "auto",
        dtype="bfloat16",
        enable_thinking: bool = False,
        memory_tokens: int = 4096,
        **kwargs,
    ) -> "LongBookModel":
        import transformers

        preset = PRESETS.get(name, {})
        trc = preset.get("trust_remote_code", False) if trust_remote_code is None else trust_remote_code
        auto = (
            getattr(transformers, "AutoModelForMultimodalLM")
            if preset.get("auto") == "multimodal"
            else transformers.AutoModelForCausalLM
        )
        tok = transformers.AutoTokenizer.from_pretrained(
            name, trust_remote_code=trc, revision=kwargs.get("revision")
        )
        try:
            model = auto.from_pretrained(
                name, trust_remote_code=trc, device_map=device_map, dtype=dtype, **kwargs
            )
        except (TypeError, ValueError):  # older transformers: torch_dtype
            model = auto.from_pretrained(
                name, trust_remote_code=trc, device_map=device_map, torch_dtype=dtype, **kwargs
            )
        return cls(
            model,
            tok,
            memory,
            use_cache=preset.get("use_cache", True),
            enable_thinking=enable_thinking,
            memory_tokens=memory_tokens,
        )

    # ----------------------------------------------------------------- memory
    def remember(self, text: str, source: str = "doc", meta: Optional[dict] = None) -> int:
        return self.memory.add(text, source=source, meta=meta)

    def remember_file(self, path: str, source: Optional[str] = None) -> int:
        return self.memory.add_file(path, source=source)

    def _count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _notes(
        self,
        query: str,
        k: int,
        expand: int,
        budget: int,
        mode: str = "passages",
        n_evidence: int = 5,
    ) -> str:
        if mode == "evidence":
            evs = self.memory.evidence(query, n=n_evidence, k=k)
            while len(evs) > 1 and self._count(format_evidence(evs)) > budget:
                evs = evs[:-1]
            return format_evidence(evs)
        return self.memory.context(query, k=k, expand=expand, budget=budget, count=self._count)

    @staticmethod
    def _user_block(notes: str, question: str, reading: str, label: str = "Notes") -> str:
        body = f"{label}:\n{notes}\n\nQuestion: {question}"
        if reading == "twice":  # Just Read Twice: context, question, context, question
            body += f"\n\n{label} (again):\n{notes}\n\nQuestion (again): {question}"
        return body

    # ------------------------------------------------------------- generation
    def generate(self, messages: List[dict], max_new_tokens: int = 512, stop=None, **gen) -> str:
        """Chat-template ``messages`` -> generated text (thinking blocks removed)."""
        tok = self.tokenizer
        prompt = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        inputs = tok(prompt, return_tensors="pt")
        inputs = inputs.to(getattr(self.model, "device", None))
        gen.setdefault("do_sample", False)
        gen.setdefault("pad_token_id", getattr(tok, "eos_token_id", None))
        args = dict(max_new_tokens=max_new_tokens, use_cache=self.use_cache, **gen)
        try:
            if stop:
                out = self.model.generate(**inputs, stop_strings=stop, tokenizer=tok, **args)
            else:
                out = self.model.generate(**inputs, **args)
        except (TypeError, ValueError):
            if not stop:
                raise
            out = self.model.generate(**inputs, **args)  # no stop_strings support
        new = out[0][inputs["input_ids"].shape[-1] :]
        return _THINK.sub("", tok.decode(new, skip_special_tokens=True)).strip()

    def ask(
        self,
        question: str,
        *,
        k: int = 6,
        expand: int = 1,
        system: Optional[str] = None,
        history: Optional[List[dict]] = None,
        max_new_tokens: int = 512,
        recall_rounds: int = 2,
        remember_turns: bool = False,
        reading: str = "once",
        notes: str = "passages",
        n_evidence: int = 5,
        **gen,
    ) -> str:
        """Answer ``question`` with notes retrieved from memory.

        ``reading`` / ``notes`` tune the prompt for recurrent (linear-attention)
        models, whose fixed-size state recalls context imperfectly:

        * ``reading="twice"`` repeats notes and question (C, Q, C, Q), the
          "Just Read Twice" prompt (Arora et al., 2024); ``"once"`` disables it.
          The token budget is split between the two copies.
        * ``notes="evidence"`` passes only the ``n_evidence`` best sentences;
          ``"passages"`` passes whole chunks with ``expand`` neighbours.

        Both are opt-in. The defaults (``"once"`` / ``"passages"``) are the setup
        that was measured: on a LongBench QA subset with Qwen3.5-9B, whole passages
        beat sentence-only notes on all three tasks.

        If the model replies ``<recall>query</recall>`` the memory is searched
        again (up to ``recall_rounds`` times) before the final answer.
        """
        if reading not in ("once", "twice"):
            raise ValueError("reading must be 'once' or 'twice'")
        if notes not in ("passages", "evidence"):
            raise ValueError("notes must be 'passages' or 'evidence'")
        budget = self.memory_tokens // 2 if reading == "twice" else self.memory_tokens
        base = SYSTEM_RECALL if recall_rounds > 0 else SYSTEM_PLAIN
        messages: List[dict] = [{"role": "system", "content": (system + "\n\n" if system else "") + base}]
        messages += list(history or [])
        found = self._notes(question, k, expand, budget, notes, n_evidence)
        self.last_notes = found
        messages.append(
            {"role": "user", "content": self._user_block(found or "(none)", question, reading)}
        )

        answer = ""
        for r in range(recall_rounds + 1):
            last = r == recall_rounds
            out = self.generate(
                messages, max_new_tokens, stop=None if last else ["</recall>"], **gen
            )
            m = _RECALL.search(out)
            if not m or last:
                answer = _RECALL.sub("", out).strip()
                break
            q = m.group(1).strip() or question
            more = self._notes(q, k, expand, budget, notes, n_evidence)
            messages.append({"role": "assistant", "content": m.group(0)})
            messages.append(
                {
                    "role": "user",
                    "content": self._user_block(
                        more or "(nothing found)", question, reading, label=f"Notes for '{q}'"
                    )
                    + "\n\nAnswer the original question now. Do not search again.",
                }
            )
        if remember_turns and answer:
            self.memory.add(f"Q: {question}\nA: {answer}", source="chat")
        return answer
