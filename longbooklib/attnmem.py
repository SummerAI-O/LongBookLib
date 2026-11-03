"""Retrieval attention: the full-attention layers of a hybrid (GDN + attention) model
look up *distant* tokens in an external KV store instead of keeping them all.

Training-free, InfLLM-style, for Qwen3.5-type models (linear-attention layers
plus a few full-attention layers):

* ``write(ids)`` streams a document through the model in chunks.  Every chunk's
  keys/values go to an external store (block-wise, with a mean-key summary per
  block).  The on-GPU KV cache of each attention layer is cut back to
  ``sink`` first tokens + the last ``window`` tokens.
* While decoding (one token per step) each attention layer scores the stored
  blocks with the current query, takes the ``topk`` best blocks and attends to
  ``[retrieved blocks | sink + local window]``.

Keys are stored after RoPE, so a retrieved token keeps its true relative
distance.  With ``topk`` covering every valid block the result equals ordinary
full attention (checked in ``tests/test_attnmem.py``).

Needs ``torch`` and a transformers version that ships ``Qwen3_5Attention``.
"""
from __future__ import annotations

import sys
import types
from typing import List, Optional


def _attention_modules(model):
    return [m for m in model.modules() if hasattr(m, "q_norm") and hasattr(m, "k_norm") and hasattr(m, "layer_idx")
            and hasattr(m, "q_proj") and hasattr(m, "num_key_value_groups")]


class AttnMemory:
    def __init__(self, model, *, block: int = 64, topk: int = 16, sink: int = 64,
                 window: int = 1024, chunk: int = 1024, store_device: Optional[str] = None):
        import torch

        self.torch = torch
        self.model = model
        core = getattr(model, "model", model)
        self.core = getattr(core, "language_model", core)  # text decoder (also for multimodal wrappers)
        self.attn = _attention_modules(self.core)
        if not self.attn:
            raise ValueError("no Qwen3.5-style full-attention layers found")
        if chunk % block or sink % block:
            raise ValueError("chunk and sink must be multiples of block")
        self.block, self.topk, self.sink, self.window, self.chunk = block, topk, sink, window, chunk
        self.store_device = store_device  # None = same device as the model, "cpu" to save VRAM
        self.mode = "off"  # "off" | "write" | "read"
        self._patch()
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self):
        self.cache = None
        self.pos = 0  # tokens processed so far
        self.local_start = 0  # absolute position of the first non-sink token still in the local cache
        self._k = {m.layer_idx: [] for m in self.attn}
        self._v = {m.layer_idx: [] for m in self.attn}
        self._r = {m.layer_idx: [] for m in self.attn}
        self.K, self.V, self.R = {}, {}, {}
        self.mode = "off"
        self.last_retrieved = {}

    @property
    def n_blocks(self) -> int:
        any_layer = self.attn[0].layer_idx
        if any_layer in self.R:
            return self.R[any_layer].shape[1]
        return sum(r.shape[1] for r in self._r[any_layer])

    # --------------------------------------------------------------- patching
    def _patch(self):
        torch = self.torch
        mem = self
        mod = sys.modules[type(self.attn[0]).__module__]
        rope, interfaces, eager = mod.apply_rotary_pos_emb, mod.ALL_ATTENTION_FUNCTIONS, mod.eager_attention_forward

        def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)
            query_states, gate = torch.chunk(
                self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
            )
            gate = gate.reshape(*input_shape, -1)
            q_pre = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
            k_pre = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            cos, sin = position_embeddings
            query_states, key_states = rope(q_pre, k_pre, cos, sin)

            if mem.mode == "write":
                mem._write(self.layer_idx, k_pre, key_states, value_states)
            if past_key_values is not None:
                key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
            if mem.mode == "read" and input_shape[1] == 1:
                extra = mem._retrieve(self.layer_idx, q_pre)
                if extra is not None:
                    key_states = torch.cat([extra[0], key_states], dim=2)
                    value_states = torch.cat([extra[1], value_states], dim=2)
                attention_mask = None  # every retrieved/local token is visible to a single query

            attention_interface = interfaces.get_interface(self.config._attn_implementation, eager)
            attn_output, attn_weights = attention_interface(
                self, query_states, key_states, value_states, attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout, scaling=self.scaling, **kwargs,
            )
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = attn_output * torch.sigmoid(gate)
            return self.o_proj(attn_output), attn_weights

        for m in self.attn:
            m.forward = types.MethodType(forward, m)

    def close(self):
        """Restore the ordinary attention forward."""
        for m in self.attn:
            m.__dict__.pop("forward", None)

    # ---------------------------------------------------------------- storing
    def _write(self, layer, k_pre, k_rot, v):
        b = self.block
        kvh, d = k_rot.shape[1], k_rot.shape[-1]
        n = k_rot.shape[2] // b
        if n == 0:
            return
        dev = self.store_device or k_rot.device
        sl = slice(0, n * b)
        self._k[layer].append(k_rot[0, :, sl].reshape(kvh, n, b, d).to(dev))
        self._v[layer].append(v[0, :, sl].reshape(kvh, n, b, d).to(dev))
        self._r[layer].append(k_pre[0, :, sl].reshape(kvh, n, b, d).mean(2))  # pre-RoPE block summary

    def _finalize(self):
        torch = self.torch
        for l in self._k:
            if self._k[l]:
                self.K[l] = torch.cat(self._k[l], dim=1)
                self.V[l] = torch.cat(self._v[l], dim=1)
                self.R[l] = torch.cat(self._r[l], dim=1)
            self._k[l], self._v[l], self._r[l] = [], [], []

    def _trim_cache(self):
        keep_tail = self.window
        for m in self.attn:
            layer = self.cache.layers[m.layer_idx]
            if layer.keys.shape[-2] > self.sink + keep_tail:
                layer.keys = self.torch.cat([layer.keys[..., : self.sink, :], layer.keys[..., -keep_tail:, :]], dim=2)
                layer.values = self.torch.cat(
                    [layer.values[..., : self.sink, :], layer.values[..., -keep_tail:, :]], dim=2
                )
        self.local_start = max(self.sink, self.pos - keep_tail)

    # -------------------------------------------------------------- retrieval
    def _retrieve(self, layer, q_pre):
        torch = self.torch
        if layer not in self.R:
            return None
        nb = self.R[layer].shape[1]
        lo = self.sink // self.block
        hi = min(nb, self.local_start // self.block)  # blocks entirely before the local window
        if hi <= lo:
            return None
        kvh = self.R[layer].shape[0]
        q = q_pre[0, :, 0].float()  # [H, d]
        q = q.view(kvh, -1, q.shape[-1]).mean(1)  # [kvh, d] (mean over the heads of a group)
        scores = torch.einsum("hd,hnd->n", q, self.R[layer][:, lo:hi].float())
        k = min(self.topk, hi - lo)
        idx = torch.topk(scores, k).indices.sort().values + lo
        self.last_retrieved[layer] = idx.tolist()
        sd = self.K[layer].device
        i = idx.to(sd)
        kk = self.K[layer][:, i].reshape(kvh, k * self.block, -1)[None].to(q_pre.device)
        vv = self.V[layer][:, i].reshape(kvh, k * self.block, -1)[None].to(q_pre.device)
        return kk, vv

    # ------------------------------------------------------------- running it
    def _forward(self, ids, only_last=True):
        torch = self.torch
        ids = ids.to(self.core.embed_tokens.weight.device)
        L = ids.shape[1]
        pos = torch.arange(self.pos, self.pos + L, device=ids.device)[None]
        out = self.core(input_ids=ids, position_ids=pos, past_key_values=self.cache, use_cache=True)
        self.cache = out.past_key_values
        self.pos += L
        h = out.last_hidden_state[:, -1:] if only_last else out.last_hidden_state
        return self.model.lm_head(h)

    def write(self, ids):
        """Stream token ids (1, T) into the external memory (linear time, bounded VRAM)."""
        torch = self.torch
        from transformers import DynamicCache

        if self.cache is None:
            self.cache = DynamicCache(config=self.core.config)
        ids = ids if ids.dim() == 2 else ids[None]
        self.mode = "write"
        try:
            with torch.no_grad():
                T = ids.shape[1]
                for s in range(0, T, self.chunk):
                    part = ids[:, s : s + self.chunk]
                    if part.shape[1] % self.block and s + self.chunk < T:
                        raise ValueError("internal: chunk not block aligned")
                    self._forward(part)
                    self._trim_cache()
        finally:
            self.mode = "off"
        self._finalize()

    def generate(self, ids, max_new_tokens: int = 64, eos_token_id=None) -> List[int]:
        """Feed ``ids`` (the question) one token at a time with retrieval, then decode greedily."""
        torch = self.torch
        if self.cache is None:
            from transformers import DynamicCache

            self.cache = DynamicCache(config=self.core.config)
        ids = ids if ids.dim() == 2 else ids[None]
        stops = set(eos_token_id if isinstance(eos_token_id, (list, tuple)) else [eos_token_id])
        out: List[int] = []
        self.mode = "read"
        try:
            with torch.no_grad():
                logits = None
                for t in range(ids.shape[1]):
                    logits = self._forward(ids[:, t : t + 1])
                for _ in range(max_new_tokens):
                    nxt = int(logits[0, -1].argmax())
                    if nxt in stops:
                        break
                    out.append(nxt)
                    logits = self._forward(torch.tensor([[nxt]]))
        finally:
            self.mode = "off"
        return out

    def answer(self, tokenizer, before: str, document: str, after: str, max_new_tokens: int = 64) -> str:
        """User message = ``before + document + after`` (chat template). The document goes to the
        external memory; ``after`` (the question) is read with retrieval and the answer decoded."""
        self.reset()
        mark = "\u0000DOC\u0000"
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": before + mark + after}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        pre, post = text.split(mark)
        enc = lambda s: tokenizer(s, return_tensors="pt", add_special_tokens=False).input_ids
        self.write(self.torch.cat([enc(pre), enc(document)], dim=1))
        eos = [tokenizer.eos_token_id] + [i for i in [tokenizer.convert_tokens_to_ids("<|im_end|>")] if i is not None]
        return tokenizer.decode(self.generate(enc(post), max_new_tokens, eos), skip_special_tokens=True).strip()

    def ask(self, tokenizer, document: str, question: str, max_new_tokens: int = 64,
            instruction: str = "Answer the question using the document. Be brief.") -> str:
        return self.answer(tokenizer, f"{instruction}\n\nDocument:\n", document,
                           f"\n\nQuestion: {question}", max_new_tokens)
