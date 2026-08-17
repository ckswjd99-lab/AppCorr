"""Co3Dv2 sequence loader for VGGT-Omega.

One *request* is one multi-view sequence, not one image. That is the shape difference from every
other loader here: the batch axis carries frames of the same scene, because the model's inter-frame
attention is the whole point -- 19 of the aggregator's 24 blocks attend across frames.

Ground truth carried alongside each sequence: per-frame depth (MVS-derived, masked) and camera
extrinsics. Cameras are converted out of Co3D's storage convention once, here, so nothing downstream
has to know about it: Co3D documents `X_cam = X_world @ R + T` (row-vector, PyTorch3D), so the
column-vector rotation is `R.T`, and PyTorch3D's +X-left/+Y-up axes become OpenCV's +X-right/+Y-down
through diag(-1, -1, 1). Getting this wrong costs ~100 degrees of rotation error and looks like a
broken model rather than a broken convention.
"""

import gzip
import json
import os
from typing import Any, Dict, List

import cv2
import numpy as np
import torch

from .dataset import DatasetLoader

_P3D_TO_OPENCV = np.diag([-1.0, -1.0, 1.0])


def _read_depth(path: str, mask_path: str, scale: float) -> np.ndarray:
    """Co3D stores depth as float16 bit-packed into a 16-bit PNG."""
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    d = np.frombuffer(raw.astype(np.uint16).tobytes(), dtype=np.float16).astype(np.float32)
    d = d.reshape(raw.shape[:2]) * scale
    d[cv2.imread(mask_path, cv2.IMREAD_UNCHANGED) == 0] = 0
    return d


class CO3DSequenceLoader(DatasetLoader):
    """Yields one sequence per item: frames + depth GT + camera GT."""

    def __init__(self, root: str, batch_size: int, **kwargs):
        super().__init__(root, batch_size, **kwargs)
        self.categories = kwargs.get("categories") or ["hotdog", "tv", "parkingmeter", "baseballbat"]
        self.frames_per_sequence = int(kwargs.get("frames_per_sequence", 8))
        self.frame_stride = int(kwargs.get("frame_stride", 5))
        self.min_long_side = int(kwargs.get("min_long_side", 0))
        self._reset_metrics()

    def _reset_metrics(self):
        self.n_seq = 0
        self.absrel_sum = self.d125_sum = self.d110_sum = 0.0
        self.rot_sum = self.rra_sum = 0.0

    # -- data ---------------------------------------------------------------------------------

    def _sequences(self) -> List[Dict[str, Any]]:
        root = os.path.expanduser(self.root)
        out = []
        for cat in self.categories:
            ann_path = os.path.join(root, cat, "frame_annotations.jgz")
            if not os.path.exists(ann_path):
                continue
            by: Dict[str, list] = {}
            for a in json.loads(gzip.open(ann_path).read()):
                by.setdefault(a["sequence_name"], []).append(a)
            for seq in sorted(by):
                anns = sorted(by[seq], key=lambda a: a["frame_number"])
                anns = anns[:: self.frame_stride][: self.frames_per_sequence]
                if len(anns) < self.frames_per_sequence:
                    continue
                if self.min_long_side:
                    h, w = anns[0]["image"]["size"]
                    if max(h, w) < self.min_long_side:
                        continue
                out.append({"category": cat, "sequence": seq, "anns": anns})
        return out

    def get_loader(self) -> torch.utils.data.DataLoader:
        from torch.utils.data import Dataset

        root = os.path.expanduser(self.root)
        seqs = self._sequences()
        if not seqs:
            raise FileNotFoundError(f"No Co3D sequences under {root} for {self.categories}")

        class _DS(Dataset):
            def __len__(self_inner):
                return len(seqs)

            def __getitem__(self_inner, idx):
                s = seqs[idx]
                frames, depths, Rs, Ts = [], [], [], []
                for a in s["anns"]:
                    frames.append(cv2.cvtColor(
                        cv2.imread(os.path.join(root, a["image"]["path"])), cv2.COLOR_BGR2RGB))
                    depths.append(_read_depth(
                        os.path.join(root, a["depth"]["path"]),
                        os.path.join(root, a["depth"]["mask_path"]),
                        a["depth"]["scale_adjustment"]))
                    R = np.array(a["viewpoint"]["R"], dtype=np.float64)
                    Rs.append(_P3D_TO_OPENCV @ R.T)
                    Ts.append(_P3D_TO_OPENCV @ np.array(a["viewpoint"]["T"], dtype=np.float64))
                return frames, {
                    "category": s["category"], "sequence": s["sequence"],
                    "depths": depths, "R": np.stack(Rs), "T": np.stack(Ts),
                }

        # batch_size=1: a request is a sequence. collate_fn keeps the ragged native shapes intact --
        # frames within a Co3D sequence share a size, but sequences do not agree with each other.
        return torch.utils.data.DataLoader(
            _DS(), batch_size=1, shuffle=False, num_workers=int(self.kwargs.get("num_workers", 2)),
            collate_fn=lambda b: (b[0][0], [b[0][1]]),
        )

    # -- evaluation ---------------------------------------------------------------------------

    @staticmethod
    def _depth_metrics(pred: np.ndarray, gts: List[np.ndarray]) -> Dict[str, float] | None:
        ps, gs = [], []
        for p, g in zip(pred, gts):
            g = cv2.resize(g, (p.shape[1], p.shape[0]), interpolation=cv2.INTER_NEAREST)
            m = g > 0
            if m.sum() > 50:
                ps.append(p[m]); gs.append(g[m])
        if not ps:
            return None
        p, g = np.concatenate(ps), np.concatenate(gs)
        # VGGT depth is up to an unknown global scale; align on the median before comparing.
        p = p * (np.median(g) / np.median(p))
        r = np.maximum(p / g, g / p)
        return {"absrel": float((np.abs(p - g) / g).mean()),
                "d125": float((r < 1.25).mean()), "d110": float((r < 1.10).mean())}

    @staticmethod
    def _pose_metrics(Rp: np.ndarray, Rg: np.ndarray) -> Dict[str, float]:
        """Relative rotation only -- scale-free and convention-free once both sides are OpenCV."""
        errs = []
        for i in range(len(Rg)):
            for j in range(i + 1, len(Rg)):
                c = (np.trace((Rg[j] @ Rg[i].T).T @ (Rp[j] @ Rp[i].T)) - 1) / 2
                errs.append(np.degrees(np.arccos(np.clip(c, -1, 1))))
        e = np.array(errs)
        return {"rot_deg": float(np.median(e)), "rra15": float((e < 15).mean())}

    def evaluate_batch(self, preds: List[Any], labels: List[Any], **kwargs) -> Dict[str, Any]:
        for pred, lab in zip(preds, labels):
            if pred is None:
                continue
            dm = self._depth_metrics(pred["depth"], lab["depths"])
            if dm is None:
                continue
            pm = self._pose_metrics(pred["R"], lab["R"])
            self.n_seq += 1
            self.absrel_sum += dm["absrel"]; self.d125_sum += dm["d125"]; self.d110_sum += dm["d110"]
            self.rot_sum += pm["rot_deg"]; self.rra_sum += pm["rra15"]
        return self.get_summary()

    def get_summary(self) -> Dict[str, Any]:
        if not self.n_seq:
            return {}
        n = self.n_seq
        return {"sequences": n, "abs_rel": self.absrel_sum / n,
                "delta_1.25": self.d125_sum / n, "delta_1.10": self.d110_sum / n,
                "rot_deg": self.rot_sum / n, "RRA@15": self.rra_sum / n}

    def get_pbar_desc(self) -> str:
        s = self.get_summary()
        if not s:
            return "Co3D: (no sequences yet)"
        return (f"AbsRel: {s['abs_rel']:.4f} | d<1.10: {s['delta_1.10']*100:.1f}% | "
                f"rot: {s['rot_deg']:.2f}deg | Seqs: {s['sequences']}")
