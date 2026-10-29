"""CUDA-graph wrapper behaviour that must hold without a GPU.

The capture/replay path needs CUDA, but the parts that can silently corrupt audio
— bucket rounding, mask padding, and the eager fallback — are pure tensor logic
and are tested here.
"""
from __future__ import annotations

import torch

from app.graphs import GraphedForward, bucketize, pad_batch


class StubModel:
    """Minimal stand-in: records calls and returns logits of the input's shape."""

    class config:  # noqa: N801 - mirrors the real config object
        num_audio_codebook = 8

    def __init__(self) -> None:
        self.calls = 0
        self.last = None

    def parameters(self):
        yield torch.zeros(1)

    def forward(self, *, input_ids, audio_mask, attention_mask, **kwargs):
        self.calls += 1
        self.last = (input_ids, audio_mask, attention_mask)
        return type("Out", (), {"logits": torch.zeros(1)})()


def test_bucketize():
    assert bucketize(224, 32) == 224
    assert bucketize(225, 32) == 256
    assert bucketize(1, 32) == 32
    assert bucketize(224, 1) == 224
    assert bucketize(224, 0) == 224


def test_pad_batch_preserves_real_rows():
    b, c, n, pad = 2, 8, 6, 8
    ids = torch.randint(0, 100, (b, c, n))
    am = torch.zeros((b, n), dtype=torch.bool)
    am[:, 4:] = True
    attn = torch.zeros((b, 1, n, n), dtype=torch.bool)
    attn[:, :, :n, :n] = True

    pids, pam, pattn = pad_batch(ids, am, attn, pad)
    assert pids.shape == (b, c, pad)
    assert pam.shape == (b, pad)
    assert pattn.shape == (b, 1, pad, pad)

    # Real content is untouched...
    assert torch.equal(pids[:, :, :n], ids)
    assert torch.equal(pam[:, :n], am)
    assert torch.equal(pattn[:, :, :n, :n], attn)
    # ...real queries did not gain any padded key...
    assert not pattn[:, :, :n, n:].any()
    # ...and padded queries attend only to themselves (no fully-masked rows).
    for i in range(n, pad):
        row = pattn[0, 0, i]
        assert row[i] and row.sum() == 1


def test_pad_batch_is_noop_when_length_matches():
    ids = torch.zeros((2, 8, 5), dtype=torch.long)
    am = torch.zeros((2, 5), dtype=torch.bool)
    attn = torch.zeros((2, 1, 5, 5), dtype=torch.bool)
    out = pad_batch(ids, am, attn, 5)
    assert out[0] is ids and out[1] is am and out[2] is attn


def test_falls_back_on_cpu_inputs():
    model = StubModel()
    wrapper = GraphedForward(model)
    ids = torch.zeros((2, 8, 4), dtype=torch.long)
    am = torch.zeros((2, 4), dtype=torch.bool)
    attn = torch.zeros((2, 1, 4, 4), dtype=torch.bool)
    out = wrapper(input_ids=ids, audio_mask=am, attention_mask=attn)
    assert out.logits.shape == (1,)
    assert model.calls == 1
    assert wrapper.fallbacks == 1
    assert wrapper.captures == 0


def test_falls_back_when_signature_is_not_inference():
    model = StubModel()
    wrapper = GraphedForward(model)
    out = wrapper(input_ids=torch.zeros((1, 1, 1), dtype=torch.long), audio_mask=None,
                  attention_mask=None, labels=torch.zeros(1))
    assert model.calls == 1
    assert wrapper.fallbacks == 1
    assert out.logits.shape == (1,)


def test_falls_back_when_missing_inputs():
    model = StubModel()
    wrapper = GraphedForward(model)
    wrapper()  # no inputs at all
    assert model.calls == 1


def test_stats_shape():
    wrapper = GraphedForward(StubModel(), bucket=32, max_graphs=2)
    stats = wrapper.stats()
    assert stats["enabled"] is True
    assert stats["buckets"] == []
    assert stats["captures"] == 0
    assert isinstance(stats["fallbacks"], int)


def test_disables_after_repeated_capture_failures(monkeypatch):
    """On a CUDA box with a failing capture we must stop trying, not retry forever."""
    wrapper = GraphedForward(StubModel(), max_failures=2)

    # Simulate a CUDA-typed input without a GPU by faking the device check.
    class FakeIds:
        device = type("D", (), {"type": "cuda"})()

        def size(self, dim):
            return 2 if dim == 0 else 224

    def boom(length):
        raise RuntimeError("nope")

    monkeypatch.setattr(wrapper, "_capture", boom)
    for _ in range(2):
        wrapper(input_ids=FakeIds(), audio_mask=object(), attention_mask=object())
    assert wrapper.enabled is False
    assert wrapper.fallbacks == 2
