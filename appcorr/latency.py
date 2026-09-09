"""Wall-clock twin of `appcorr.flops`: the same request / arrival / stage scopes, measuring time.

    from appcorr import latency

    with latency.session() as lt:                       # no hooks: the model runs untouched
        for sample in data:
            with lt.request(sample.id):
                with lt.arrival(0), lt.stage("approx"):
                    ...
                for r in range(1, g):
                    with lt.arrival(r), lt.stage("correct"):
                        ...
    print(lt.aggregate())                               # medians in ms

The counter is a drop-in for `FlopCounter` at every call site the axes already have (`axis.flops =
lt` works: `_arrival` / `_stage` hooks only ever call `.arrival()` / `.stage()`), so a FLOPs report
script becomes a latency report by swapping the session and the output keys.

What is measured:

  * **total** -- the request scope's wall time, host clock, with a device sync at both ends. For the
    prefill-only reports this is the in-process TTFT with the whole image on the GPU at t=0 and no
    transmission credit (the table's "Lat." column).
  * **critical** -- wall time from the FIRST entry into the highest arrival index seen by the
    request to the end of the request. A device sync is taken when an arrival scope is entered (it
    is the moment the chunk "lands", which in a served path is a host-side event anyway), so the
    start point is host/GPU aligned. An arm that never opens an arrival scope is 100% critical,
    exactly as in the FLOPs rule. This is the table's "Crit. Lat.": the last chunk's correction, its
    merge and the remaining prefill, i.e. the twin of Crit. Comp.'s accounting.
  * **stages** -- per (arrival, stage) GPU time from CUDA events (no extra syncs), for the detail
    view only. They are not summed into `total`/`critical`.

Decode is never inside a request scope in the report scripts, so it is excluded, as for FLOPs.
"""

from __future__ import annotations

import statistics
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import torch


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@dataclass
class RequestLatency:
    request_id: object = None
    meta: Dict[str, object] = field(default_factory=dict)
    t_start: float = 0.0
    t_end: float = 0.0
    arrival_start: Dict[int, float] = field(default_factory=dict)   # first entry, host clock
    events: List[Tuple[int, str, object, object]] = field(default_factory=list)
    _stage_ms: Optional[Dict[Tuple[int, str], float]] = None

    @property
    def final_arrival(self) -> int:
        return max(self.arrival_start, default=0)

    @property
    def total(self) -> float:
        return (self.t_end - self.t_start) * 1e3

    @property
    def critical(self) -> float:
        if not self.arrival_start:
            return self.total
        return (self.t_end - self.arrival_start[self.final_arrival]) * 1e3

    @property
    def buckets(self) -> Dict[Tuple[int, str], float]:
        """Per (arrival, stage) GPU ms from the events; resolved lazily after the request synced."""
        if self._stage_ms is None:
            out: Dict[Tuple[int, str], float] = {}
            for a, s, e0, e1 in self.events:
                out[(a, s)] = out.get((a, s), 0.0) + e0.elapsed_time(e1)
            self._stage_ms = out
        return self._stage_ms

    def by_stage(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for (_, s), v in self.buckets.items():
            out[s] = out.get(s, 0.0) + v
        return out

    def summary(self) -> Dict[str, object]:
        return {"total_ms": self.total, "critical_ms": self.critical,
                "final_arrival": self.final_arrival, "by_stage_ms": self.by_stage(),
                **({"meta": dict(self.meta)} if self.meta else {})}


class LatencyCounter:
    """`FlopCounter`'s scope API, timing instead of counting. Thread-local like the original."""

    def __init__(self, warmup: int = 2) -> None:
        self.requests: List[RequestLatency] = []
        self.warmup = warmup
        self._local = threading.local()

    # --- FlopCounter compatibility: hooks may call this if a session installed them; we never do.
    def record(self, **_) -> None:
        return

    @contextmanager
    def request(self, request_id: object = None, **meta):
        prev = getattr(self._local, "request", None)
        req = RequestLatency(request_id=request_id)
        req.meta.update(meta)
        self.requests.append(req)
        self._local.request, self._local.arrival, self._local.stage = req, 0, "backbone"
        _sync()
        req.t_start = time.perf_counter()
        try:
            yield req
        finally:
            _sync()
            req.t_end = time.perf_counter()
            self._local.request = prev

    @contextmanager
    def arrival(self, index: int):
        req = getattr(self._local, "request", None)
        prev = getattr(self._local, "arrival", 0)
        self._local.arrival = int(index)
        if req is not None and int(index) not in req.arrival_start:
            _sync()
            req.arrival_start[int(index)] = time.perf_counter()
        try:
            yield
        finally:
            self._local.arrival = prev

    @contextmanager
    def stage(self, name: str):
        req = getattr(self._local, "request", None)
        prev = getattr(self._local, "stage", "backbone")
        self._local.stage = str(name)
        use_events = req is not None and torch.cuda.is_available()
        if use_events:
            e0 = torch.cuda.Event(enable_timing=True)
            e0.record()
        try:
            yield
        finally:
            if use_events:
                e1 = torch.cuda.Event(enable_timing=True)
                e1.record()
                req.events.append((getattr(self._local, "arrival", 0), str(name), e0, e1))
            self._local.stage = prev

    def reset(self) -> None:
        self.requests.clear()

    def aggregate(self) -> Dict[str, object]:
        """Medians over the requests after `warmup` (the first requests pay allocator/autotune)."""
        reqs = self.requests[self.warmup:] if len(self.requests) > self.warmup else self.requests
        n = len(reqs)
        if not n:
            return {"requests": 0}
        med = statistics.median
        stages: Dict[str, List[float]] = {}
        for r in reqs:
            for k, v in r.by_stage().items():
                stages.setdefault(k, []).append(v)
        return {
            "requests": n,
            "median_total_ms": med(r.total for r in reqs),
            "median_critical_ms": med(r.critical for r in reqs),
            "mean_total_ms": sum(r.total for r in reqs) / n,
            "mean_critical_ms": sum(r.critical for r in reqs) / n,
            "median_stage_ms": {k: med(v) for k, v in sorted(stages.items())},
        }


@contextmanager
def session(*roots, enabled: bool = True, warmup: int = 2) -> Iterator[LatencyCounter]:
    """Same signature as `flops.session`; `roots` are accepted and ignored (nothing is hooked)."""
    yield LatencyCounter(warmup=warmup)


def save_entry(path: str, key: str, dataset: str, full_ms: float, arms, samples: int,
               groups: int, note: str, full_detail=None) -> None:
    """Merge one (model key, dataset) block into inprocess_latency.json.

    `arms` = [(keep, critical_ms, total_ms, detail_dict)], the FLOPs file's key convention
    (`k0.50` = critical, `total_k0.50` = the arm's whole TTFT). Medians, ms.
    """
    import json, os
    rec = {}
    if os.path.exists(path):
        try:
            rec = json.load(open(path))
        except Exception:
            rec = {}
    m = rec.setdefault(key, {})
    m["_note"] = note
    d = m.setdefault(dataset, {})
    d.update({"full": round(full_ms, 1), "n": samples, "groups": groups})
    det = d.setdefault("detail", {})
    if full_detail is not None:
        det["full"] = full_detail
    for keep, crit, tot, detail in arms:
        d[f"k{keep:.2f}"] = round(crit, 1)
        d[f"total_k{keep:.2f}"] = round(tot, 1)
        det[f"k{keep:.2f}"] = detail
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(rec, f, indent=2)
