"""CUDA graphs for the denoising forward pass.

One sentence costs ``num_step`` forward passes (16 by default, doubled for CFG)
over exactly the same tensor shapes. Eager PyTorch launches ~170 kernels per
pass, and on a fast GPU that per-launch overhead becomes a real fraction of the
step. A captured graph replays the whole pass with one launch.

Measured on the reference RTX 3090 (fp16, 224-token sequence, inside
``inference_mode``): 13.24 ms eager → **11.27 ms per forward (1.18x)**,
225.9 ms → 198.8 ms for a 16-step sentence (1.14x), with **bit-identical logits**
(max |diff| = 0). On an H200 the launch overhead is a larger share of the step,
so the win should be bigger — that is where this is worth enabling.

Costs, which is why it is off by default:

* ~1.2 GB of VRAM per captured sequence length (a graph pool keeps every
  intermediate alive), so captures are bucketed by length and bounded;
* ~60 ms one-off capture per bucket, paid on the first sentence that lands in it.

Only **single-item** calls are captured (``2`` rows: conditional +
unconditional). A batched call has ``2N`` rows, and at that width the launch
overhead per item is already amortised by the matrix, so capturing every
(batch, length) combination would cost memory for ~3 % — batching and graphs
cover the loaded and the idle regime respectively.

The wrapper is deliberately forgiving: any shape it has not captured, any
non-CUDA input or an unexpected call signature falls through to eager, and after
a few capture failures it disables itself for the life of the process.
"""
from __future__ import annotations

import logging
import time
from types import SimpleNamespace
from typing import Any

import torch

log = logging.getLogger(__name__)


def bucketize(length: int, bucket: int) -> int:
    """Round *length* up to a multiple of *bucket* (no-op when bucket <= 1)."""
    if bucket <= 1:
        return length
    return -(-length // bucket) * bucket


def pad_batch(
    input_ids: torch.Tensor, audio_mask: torch.Tensor, attention_mask: torch.Tensor, length: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Grow a (B, C, L) / (B, L) / (B, 1, L, L) batch to *length*.

    Padded query positions attend only to themselves, so every real row keeps
    exactly the key set it had before and its output is unchanged.
    """
    n = input_ids.size(2)
    if length == n:
        return input_ids, audio_mask, attention_mask
    b, c, _ = input_ids.shape
    ids = input_ids.new_zeros((b, c, length))
    am = audio_mask.new_zeros((b, length))
    attn = attention_mask.new_zeros((b, 1, length, length))
    ids[:, :, :n] = input_ids
    am[:, :n] = audio_mask
    attn[:, :, :n, :n] = attention_mask
    diag = torch.arange(n, length, device=input_ids.device)
    attn[:, :, diag, diag] = True
    return ids, am, attn


class GraphedForward:
    """Drop-in replacement for ``OmniVoice.forward`` that replays a CUDA graph."""

    def __init__(
        self,
        model: Any,
        *,
        bucket: int = 32,
        max_graphs: int = 4,
        min_free_mb: int = 4096,
        max_failures: int = 3,
    ) -> None:
        self.model = model
        self.bucket = bucket
        self.max_graphs = max(1, max_graphs)
        self.min_free_mb = min_free_mb
        self.max_failures = max_failures
        self._eager = model.forward
        self._graphs: dict[int, tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, Any]] = {}
        self._failures = 0
        self.captures = 0
        self.capture_seconds = 0.0
        self.fallbacks = 0

    # -- state -------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self._failures < self.max_failures

    def _fallback(self, input_ids, audio_mask, attention_mask, **kwargs):
        self.fallbacks += 1
        return self._eager(
            input_ids=input_ids, audio_mask=audio_mask, attention_mask=attention_mask, **kwargs
        )

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "buckets": sorted(self._graphs),
            "captures": self.captures,
            "capture_seconds": round(self.capture_seconds, 3),
            "fallbacks": self.fallbacks,
        }

    # -- capture -----------------------------------------------------------
    def _free_mb(self) -> float:
        free, _total = torch.cuda.mem_get_info()
        return free / 2**20

    def _capture(self, length: int, rows: int = 2):
        if self._free_mb() < self.min_free_mb:
            raise RuntimeError(
                f"not enough free VRAM for a CUDA graph ({self._free_mb():.0f} MB free)"
            )
        device = next(self.model.parameters()).device
        codebooks = self.model.config.num_audio_codebook
        ids = torch.zeros((rows, codebooks, length), dtype=torch.long, device=device)
        audio_mask = torch.zeros((rows, length), dtype=torch.bool, device=device)
        # Cond row: full attention over the sequence. Uncond row: target-only, so
        # its real prefix is shorter; the pad diagonal keeps both rows valid.
        attn = torch.ones((rows, 1, length, length), dtype=torch.bool, device=device)
        with torch.inference_mode():
            for _ in range(2):  # warm kernels and the allocator
                self._eager(input_ids=ids, audio_mask=audio_mask, attention_mask=attn)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            started = time.perf_counter()
            with torch.cuda.graph(graph):
                logits = self._eager(
                    input_ids=ids, audio_mask=audio_mask, attention_mask=attn
                ).logits
        elapsed = time.perf_counter() - started
        self.captures += 1
        self.capture_seconds += elapsed
        log.info("captured CUDA graph for %d tokens in %.0f ms", length, elapsed * 1000)
        return graph, ids, audio_mask, attn, logits

    def prewarm(self, lengths) -> int:
        """Capture graphs for *lengths* up front so no request pays for it."""
        captured = 0
        for length in lengths:
            padded = bucketize(length, self.bucket)
            if padded in self._graphs or not self.enabled:
                continue
            if len(self._graphs) >= self.max_graphs:
                break
            try:
                self._graphs[padded] = self._capture(padded)
                captured += 1
            except Exception as exc:  # pragma: no cover - depends on the driver
                self._failures += 1
                log.warning("CUDA graph prewarm failed for %d tokens: %s", padded, exc)
                break
        return captured

    # -- call --------------------------------------------------------------
    def __call__(
        self,
        input_ids: torch.Tensor | None = None,
        audio_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        if (
            not self.enabled
            or kwargs
            or input_ids is None
            or audio_mask is None
            or attention_mask is None
            or input_ids.device.type != "cuda"
        ):
            return self._fallback(input_ids, audio_mask, attention_mask, **kwargs)

        if input_ids.size(0) != 2:
            # Batched call (2N rows): let the matrix width hide the launches.
            return self._fallback(input_ids, audio_mask, attention_mask)

        length = input_ids.size(2)
        padded = bucketize(length, self.bucket)
        entry = self._graphs.get(padded)
        if entry is None:
            if len(self._graphs) >= self.max_graphs:
                return self._fallback(input_ids, audio_mask, attention_mask)
            try:
                entry = self._capture(padded)
            except Exception as exc:
                self._failures += 1
                log.warning(
                    "CUDA graph capture failed for %d tokens (%s); using eager (%d/%d)",
                    padded, exc, self._failures, self.max_failures,
                )
                if not self.enabled:
                    log.warning("disabling CUDA graphs for this process")
                return self._fallback(input_ids, audio_mask, attention_mask)
            self._graphs[padded] = entry

        graph, ids, am, attn, logits = entry
        if padded != length:
            input_ids, audio_mask, attention_mask = pad_batch(
                input_ids, audio_mask, attention_mask, padded
            )
        ids.copy_(input_ids)
        am.copy_(audio_mask)
        attn.copy_(attention_mask)
        graph.replay()
        return SimpleNamespace(logits=logits)
