"""AttnMemory on a tiny random Qwen3.5 (skipped without torch / Qwen3.5 support)."""
import pytest

torch = pytest.importorskip("torch")
tf = pytest.importorskip("transformers")
if not hasattr(tf, "Qwen3_5ForCausalLM"):
    pytest.skip("transformers without Qwen3.5", allow_module_level=True)

from longbooklib.attnmem import AttnMemory


def tiny(layers=4):
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    torch.manual_seed(0)
    cfg = Qwen3_5TextConfig(
        vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=layers, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4, max_position_embeddings=4096,
    )
    return Qwen3_5ForCausalLM(cfg).eval()


def baseline_last_logits(model, ids):
    with torch.no_grad():
        return model(ids).logits[0, -1]


def run(model, doc, q, **kw):
    mem = AttnMemory(model, **kw)
    mem.write(doc)
    mem.mode = "read"
    with torch.no_grad():
        for t in range(q.shape[1]):
            logits = mem._forward(q[:, t : t + 1])
    mem.mode = "off"
    return mem, logits[0, -1]


def test_all_blocks_equals_full_attention():
    m = tiny()
    doc = torch.randint(0, 300, (1, 288))
    q = torch.randint(0, 300, (1, 6))
    ref = baseline_last_logits(m, torch.cat([doc, q], 1))
    # window/sink are far smaller than the document, but every valid block is retrieved
    mem, got = run(m, doc, q, block=16, sink=16, window=64, chunk=288, topk=10_000)
    assert mem.cache.layers[3].keys.shape[-2] < 288  # the GPU cache really was cut back
    assert max(mem.last_retrieved[3]) < 288 // 16  # blocks, not tokens
    assert torch.allclose(ref, got, atol=1e-4), (ref - got).abs().max()


def test_topk_limits_what_is_retrieved_and_runs_chunked():
    m = tiny()
    doc = torch.randint(0, 300, (1, 512))
    q = torch.randint(0, 300, (1, 4))
    mem, got = run(m, doc, q, block=16, sink=16, window=64, chunk=128, topk=3)
    assert torch.isfinite(got).all()
    assert len(mem.last_retrieved[3]) == 3
    # nothing from the sink block or from the local window is retrieved
    assert all(1 <= b < (512 - 64) // 16 for b in mem.last_retrieved[3])


def test_short_document_is_plain_attention_and_close_restores_forward():
    m = tiny()
    doc = torch.randint(0, 300, (1, 32))
    q = torch.randint(0, 300, (1, 3))
    ref = baseline_last_logits(m, torch.cat([doc, q], 1))
    mem, got = run(m, doc, q, block=16, sink=16, window=64, chunk=32, topk=2)
    assert torch.allclose(ref, got, atol=1e-4)
    mem.close()
    assert torch.allclose(ref, baseline_last_logits(m, torch.cat([doc, q], 1)), atol=1e-5)


def test_generate_and_validation():
    m = tiny()
    mem = AttnMemory(m, block=16, sink=16, window=64, chunk=64, topk=4)
    mem.write(torch.randint(0, 300, (1, 200)))
    out = mem.generate(torch.randint(0, 300, (1, 5)), max_new_tokens=6, eos_token_id=-1)
    assert len(out) == 6 and all(0 <= t < 300 for t in out)
    with pytest.raises(ValueError):
        AttnMemory(m, block=16, sink=10)


def test_answer_builds_prompt_around_the_document():
    class Tok:
        eos_token_id = 0

        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **kw):
            return "<u>" + msgs[0]["content"] + "</u><a>"

        def __call__(self, text, return_tensors=None, add_special_tokens=False):
            return type("E", (), {"input_ids": torch.tensor([[ord(c) % 250 + 1 for c in text]])})()

        def convert_tokens_to_ids(self, t):
            return None

        def decode(self, ids, skip_special_tokens=True):
            return "x" * len(ids)

    m = tiny()
    mem = AttnMemory(m, block=16, sink=16, window=64, chunk=64, topk=2)
    out = mem.answer(Tok(), "Read: ", "d" * 150, " Q?", max_new_tokens=3)
    assert isinstance(out, str) and mem.pos > 150 and mem.n_blocks >= 9
