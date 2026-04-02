import multiprocessing
import os
import time
from contextlib import nullcontext
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from offload.common import ExperimentConfig, HintPacket
from offload.common.protocol import normalize_appcorr_kwargs


def mobile_hint_enabled(config: ExperimentConfig) -> bool:
    appcorr_options = normalize_appcorr_kwargs(config.appcorr_kwargs)
    return (
        appcorr_options.get("mobile_pscore", "none") != "none"
        and float(appcorr_options.get("mobile_pscore_weight", 0.0)) != 0.0
    )


class MobileHintWorker(multiprocessing.Process):
    """Asynchronously computes mobile-side hint scores and streams them to the sender queue."""

    def __init__(self, input_queue, output_queue, telemetry_queue, config: ExperimentConfig):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.telemetry_queue = telemetry_queue
        self.config = config

    def run(self):
        if not mobile_hint_enabled(self.config):
            return

        self.appcorr_options = normalize_appcorr_kwargs(self.config.appcorr_kwargs)
        self.device = self._resolve_device()
        print(f"[MobileHint] Started on device: {self.device}")
        self._load_model()

        while True:
            item = self.input_queue.get()
            if item == "STOP":
                break

            request_id, full_batch_np = item
            t_hint_start = time.time()
            try:
                num_packets = self._stream_hint_packets(int(request_id), full_batch_np)
                t_hint_end = time.time()
                if self.telemetry_queue is not None:
                    self.telemetry_queue.put({
                        "request_id": int(request_id),
                        "type": "MOBILE_HINT_GPU",
                        "timestamp": t_hint_start,
                        "duration": max(t_hint_end - t_hint_start, 0.0),
                        "params": {
                            "score_type": str(self.appcorr_options.get("mobile_pscore", "none")),
                            "num_packets": int(num_packets),
                            "device": str(self.device),
                        },
                    })
            except Exception as exc:
                print(f"!!! [MobileHint] Failed to compute hints for request {request_id}: {exc}")

        print("[MobileHint] Stopped.")

    def _resolve_device(self) -> torch.device:
        device_str = self.appcorr_options.get("mobile_hint_device")
        if device_str is not None:
            return torch.device(device_str)
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _load_model(self):
        model_name = str(self.appcorr_options.get("mobile_hint_model_name", "dinov3_vits16"))
        if model_name != "dinov3_vits16":
            raise ValueError(f"Unsupported mobile hint model: {model_name}")

        weights_path = self.appcorr_options.get("mobile_hint_model_weights")
        if weights_path is None:
            raise ValueError("mobile hint requires appcorr_kwargs.mobile_hint_model_weights")

        weights_path = os.path.expanduser(weights_path)
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Missing mobile hint weights: {weights_path}")

        from appcorr.models.dinov3.hub.backbones import dinov3_vits16

        self.model = dinov3_vits16(pretrained=True, weights=weights_path)
        if self.device.type == "cuda":
            self.model = self.model.to(device=self.device, dtype=torch.bfloat16)
        else:
            self.model = self.model.to(device=self.device, dtype=torch.float32)
        self.model.eval()

    def _prepare_image_pair(self, batch_np: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(batch_np).to(device=self.device, non_blocking=True)
        x = x.permute(0, 3, 1, 2).float() / 255.0

        low_level = max(self.appcorr_options.get("pyramid_levels", [0]))
        if low_level <= 0:
            low = x.clone()
        else:
            scale = float(2 ** (-low_level))
            low = F.interpolate(x, scale_factor=scale, mode="bicubic", align_corners=False)
            low = F.interpolate(low, size=x.shape[-2:], mode="bicubic", align_corners=False)
            low = low.clamp_(0.0, 1.0)

        high = x
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        return (high - mean) / std, (low - mean) / std

    @staticmethod
    def _build_server_layer_map(total_layers: int, mobile_layers: int) -> Dict[int, List[int]]:
        mapping: Dict[int, List[int]] = {}
        for server_layer in range(total_layers):
            mobile_layer = min((server_layer * mobile_layers) // max(total_layers, 1), mobile_layers - 1)
            mapping.setdefault(mobile_layer, []).append(server_layer)
        return mapping

    def _emit_hint_ready_event(
        self,
        request_id: int,
        mobile_layer_idx: int,
        server_layers: List[int],
        timestamp: float,
    ) -> None:
        if self.telemetry_queue is None:
            return
        layer_label = (
            str(server_layers[0])
            if len(server_layers) == 1
            else f"{server_layers[0]}-{server_layers[-1]}"
        )
        self.telemetry_queue.put({
            "request_id": int(request_id),
            "type": "MOBILE_HINT_READY",
            "timestamp": float(timestamp),
            "duration": 0.0,
            "params": {
                "mobile_layer": int(mobile_layer_idx),
                "server_layers": [int(v) for v in server_layers],
                "layer_label": layer_label,
            },
        })

    def _stream_hint_packets(self, request_id: int, full_batch_np: np.ndarray) -> int:
        high, low = self._prepare_image_pair(full_batch_np)
        total_layers = int(self.config.transmission_kwargs.get("total_layers", 40))
        layer_map = self._build_server_layer_map(total_layers, len(self.model.blocks))

        batch_size = high.shape[0]
        chunk_size = min(int(self.appcorr_options.get("mobile_hint_batch_size", 4)), batch_size)
        chunk_size = max(chunk_size, 1)
        score_type = str(self.appcorr_options.get("mobile_pscore", "none"))

        layer_buffers: List[torch.Tensor] | None = None
        num_pretokens = 1 + getattr(self.model, "n_storage_tokens", 0)
        num_packets = 0

        with torch.inference_mode():
            for start in range(0, batch_size, chunk_size):
                end = min(start + chunk_size, batch_size)
                inputs = torch.cat([high[start:end], low[start:end]], dim=0)
                is_last_chunk = end == batch_size

                autocast_ctx = (
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if self.device.type == "cuda"
                    else nullcontext()
                )
                with autocast_ctx:
                    x, (tok_h, tok_w) = self.model.prepare_tokens_with_masks(inputs, None)
                    rope = self.model.rope_embed(H=tok_h, W=tok_w) if self.model.rope_embed is not None else None

                    if layer_buffers is None:
                        layer_buffers = [
                            torch.zeros((batch_size, tok_h, tok_w), dtype=torch.float16)
                            for _ in range(len(self.model.blocks))
                        ]

                    for lidx, blk in enumerate(self.model.blocks):
                        x_norm = blk.norm1(x)
                        attn_out, score = blk.attn.forward_with_server_pscore(
                            x_norm,
                            rope=rope,
                            server_pscore="patch_attn_prob",
                            pair_batch_size=(end - start) if score_type == "patch_attn_delta_abs" else None,
                            pair_score_type=score_type if score_type == "patch_attn_delta_abs" else None,
                        )

                        if score_type == "patch_attn_delta_abs":
                            patch_score = score[:, num_pretokens:].reshape(end - start, tok_h, tok_w)
                            diff = patch_score
                        else:
                            patch_score = score[:, num_pretokens:].reshape(inputs.shape[0], tok_h, tok_w)
                            diff = patch_score[: end - start] - patch_score[end - start :]
                            diff = diff.clamp_min_(0.0)
                        layer_buffers[lidx][start:end].copy_(diff.to(dtype=torch.float16).cpu())

                        if is_last_chunk:
                            mapped_server_layers = layer_map.get(lidx, [])
                            if mapped_server_layers:
                                diff_map = layer_buffers[lidx]
                                scores_np = diff_map.numpy()
                                token_hw = (scores_np.shape[1], scores_np.shape[2])
                                ready_ts = time.time()
                                for server_layer_idx in mapped_server_layers:
                                    self.output_queue.put(
                                        HintPacket(
                                            request_id=request_id,
                                            layer_idx=int(server_layer_idx),
                                            scores=scores_np.copy(),
                                        )
                                    )
                                    num_packets += 1
                                self._emit_hint_ready_event(
                                    request_id,
                                    mobile_layer_idx=lidx,
                                    server_layers=mapped_server_layers,
                                    timestamp=ready_ts,
                                )

                        x = x + blk.ls1(attn_out)
                        x = x + blk.ls2(blk.mlp(blk.norm2(x)))

        return num_packets
