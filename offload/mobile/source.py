import multiprocessing
import time
import torch
import numpy as np
import cv2
from tqdm import tqdm
from dataclasses import asdict

from offload.common import ExperimentConfig
from offload.common.protocol import normalize_appcorr_kwargs
from offload.policies import get_transmission
from offload.mobile.dataset import get_dataset_loader

import os
import json
import datetime
import re

def perform_time_sync(output_queue, feedback_queue, rounds=10):
    """Estimate clock offset via ping-pong."""
    print(f"[Source] Synchronizing time with server ({rounds} rounds)...")
    
    # Warm-up round (ignore result)
    print("[Source] Sending warm-up ping...")
    output_queue.put(('TIME_SYNC', None))
    feedback_queue.get()
    
    offsets = []
    
    for _ in range(rounds):
        t1 = time.time()
        output_queue.put(('TIME_SYNC', None))
        
        t_server = feedback_queue.get()
        t2 = time.time()
        
        rtt = t2 - t1
        # Assumes symmetric network delay
        estimated_server_time_at_t1 = t_server - (rtt / 2)
        offset = estimated_server_time_at_t1 - t1
        offsets.append(offset)
        
    avg_offset = sum(offsets) / len(offsets)
    print(f"[Source] Time Sync Complete. Avg Offset: {avg_offset*1000:.2f} ms")
    return avg_offset


class SourceModule(multiprocessing.Process):
    """Run experiment loop, handle partial batches, track metrics."""

    _EVENT_GROUP_SUFFIX_RE = re.compile(r"_G\d+$")

    def __init__(
        self,
        output_queue,
        feedback_queue,
        config: ExperimentConfig,
        data_root: str,
        loader_batch_size: int,
        num_requests: int | None = None,
        num_warmup: int = 1,
    ):
        super().__init__()
        self.output_queue = output_queue
        self.feedback_queue = feedback_queue
        self.config = config
        self.data_root = data_root
        self.loader_batch_size = loader_batch_size
        self.num_requests = num_requests
        self.num_warmup = max(int(num_warmup), 0)

    @staticmethod
    def _tensor_to_hwc_uint8(image: torch.Tensor) -> np.ndarray:
        if image.ndim != 3:
            raise RuntimeError(f"Expected image tensor [C,H,W] or [H,W,C], got {tuple(image.shape)}")
        if image.shape[0] == 3:
            image = image.permute(1, 2, 0)
        elif image.shape[-1] != 3:
            raise RuntimeError(f"Expected 3-channel image tensor, got {tuple(image.shape)}")
        image_np = image.detach().cpu().numpy()
        if image_np.dtype != np.uint8:
            image_np = np.clip(image_np, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(image_np)

    @staticmethod
    def _metadata_for_label(label):
        if not isinstance(label, dict):
            return {}
        orig_h = label.get('orig_height')
        orig_w = label.get('orig_width')
        if orig_h is None or orig_w is None:
            return {}
        return {'target_shape': (int(orig_h), int(orig_w))}

    def _prepare_encode_input(self, images, curr_bs: int, labels=None):
        policy_name = self.config.transmission_policy_name
        preserve_input_shape = bool(self.config.transmission_kwargs.get('preserve_input_shape', False))
        label_list = self._labels_to_list(labels, curr_bs) if labels is not None else []
        laplacian_policies = {
            "Laplacian",
            "ProgressiveLaplacian",
            "COCOWindowProgressiveLaplacian",
            "NYUAppCorrLaplacian",
            "NYUAppCorrProgressiveLaplacian",
            "NYUAppCorrRaw",
        }
        if isinstance(images, (list, tuple)):
            real_imgs_np = [self._tensor_to_hwc_uint8(img) for img in images]
            if preserve_input_shape:
                target_shapes = [
                    (self._metadata_for_label(label_list[idx]) if idx < len(label_list) else {}).get('target_shape')
                    or tuple(int(v) for v in image_np.shape[:2])
                    for idx, image_np in enumerate(real_imgs_np)
                ]
                self._current_target_shapes = target_shapes
                if policy_name in laplacian_policies:
                    return real_imgs_np
                encoded_items = [
                    {
                        'image': image_np,
                        'metadata': self._metadata_for_label(label_list[idx]) if idx < len(label_list) else {},
                    }
                    for idx, image_np in enumerate(real_imgs_np)
                ]
                if len(real_imgs_np) < self.config.batch_size and real_imgs_np:
                    pad_count = self.config.batch_size - len(real_imgs_np)
                    encoded_items.extend(
                        {'image': np.zeros_like(real_imgs_np[0]), 'metadata': {}}
                        for _ in range(pad_count)
                    )
                return encoded_items
            if policy_name in laplacian_policies:
                return real_imgs_np

            server_batch_size = self.config.batch_size
            img_h, img_w, img_c = self.config.image_shape
            full_batch_np = np.zeros((server_batch_size, img_h, img_w, img_c), dtype=np.uint8)
            for idx, image_np in enumerate(real_imgs_np):
                if image_np.shape[:2] != (img_h, img_w):
                    image_np = cv2.resize(image_np, (img_w, img_h), interpolation=cv2.INTER_AREA)
                full_batch_np[idx] = image_np
            return full_batch_np

        server_batch_size = self.config.batch_size
        img_h, img_w, img_c = self.config.image_shape
        if preserve_input_shape:
            real_imgs_np = [
                self._tensor_to_hwc_uint8(images[idx])
                for idx in range(curr_bs)
            ]
            target_shapes = [
                (self._metadata_for_label(label_list[idx]) if idx < len(label_list) else {}).get('target_shape')
                or tuple(int(v) for v in image_np.shape[:2])
                for idx, image_np in enumerate(real_imgs_np)
            ]
            self._current_target_shapes = target_shapes
            if policy_name in laplacian_policies:
                return real_imgs_np
            encoded_items = [
                {
                    'image': image_np,
                    'metadata': self._metadata_for_label(label_list[idx]) if idx < len(label_list) else {},
                }
                for idx, image_np in enumerate(real_imgs_np)
            ]
            if curr_bs < self.config.batch_size and real_imgs_np:
                pad_count = self.config.batch_size - curr_bs
                encoded_items.extend(
                    {'image': np.zeros_like(real_imgs_np[0]), 'metadata': {}}
                    for _ in range(pad_count)
                )
            return encoded_items
        full_batch_np = np.zeros((server_batch_size, img_h, img_w, img_c), dtype=np.uint8)
        real_imgs_np = images.permute(0, 2, 3, 1).numpy()
        full_batch_np[:curr_bs] = real_imgs_np
        return full_batch_np

    @staticmethod
    def _labels_to_list(labels, curr_bs: int):
        if isinstance(labels, torch.Tensor):
            return labels[:curr_bs].tolist()
        return [
            label.tolist() if isinstance(label, torch.Tensor) else label
            for label in list(labels)[:curr_bs]
        ]

    @staticmethod
    def _json_safe_value(value):
        if isinstance(value, np.ndarray):
            import hashlib

            value_np = np.ascontiguousarray(value)
            return {
                'shape': list(value_np.shape),
                'dtype': str(value_np.dtype),
                'sha256': hashlib.sha256(value_np.tobytes()).hexdigest(),
            }
        if isinstance(value, torch.Tensor):
            return SourceModule._json_safe_value(value.detach().cpu().numpy())
        if isinstance(value, dict):
            return {key: SourceModule._json_safe_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [SourceModule._json_safe_value(item) for item in value]
        return value

    @classmethod
    def _latency_event_type(cls, event_type: str) -> str:
        if event_type.startswith("MOBILE_ENCODE_G"):
            return cls._EVENT_GROUP_SUFFIX_RE.sub("", event_type)
        if event_type.startswith("MOBILE_SEND_G"):
            return cls._EVENT_GROUP_SUFFIX_RE.sub("", event_type)
        return event_type

    @staticmethod
    def _current_batch_size(images) -> int:
        return len(images) if isinstance(images, (list, tuple)) else images.size(0)

    def _execute_request(self, policy, images, labels, curr_bs: int, time_offset: float):
        t_load_start = time.time()
        encode_input = self._prepare_encode_input(images, curr_bs, labels)
        t_load_end = time.time()

        batch_bytes = 0
        all_patches = []
        local_events = [{'type': 'MOBILE_LOAD', 'timestamp': t_load_start, 'duration': t_load_end - t_load_start}]

        encode_gen = policy.encode(encode_input, self.config)

        group_idx = 0
        t_pipeline_start = time.time()
        t_send_start = t_pipeline_start

        for group_patches in encode_gen:
            t_decode_end = time.time()
            local_events.append({
                'type': f'MOBILE_ENCODE_G{group_idx}',
                'timestamp': t_pipeline_start,
                'duration': t_decode_end - t_pipeline_start,
            })

            target_shapes = getattr(self, '_current_target_shapes', None)
            if target_shapes:
                for p in group_patches:
                    if p.image_idx < len(target_shapes) and target_shapes[p.image_idx] is not None:
                        p.target_shape = target_shapes[p.image_idx]

            t_tx_start = time.time()
            for p in group_patches:
                self.output_queue.put(p)
            t_tx_end = time.time()

            local_events.append({
                'type': f'MOBILE_SEND_G{group_idx}',
                'timestamp': t_tx_start,
                'duration': t_tx_end - t_tx_start,
            })

            all_patches.extend(group_patches)
            batch_bytes += sum(len(p.data) for p in group_patches)

            group_idx += 1
            t_pipeline_start = time.time()

        result = self.feedback_queue.get()
        t_result_recv = time.time()

        server_events = []
        for event in result.server_events:
            event_copy = dict(event)
            event_copy['timestamp'] = event_copy['start'] - time_offset
            event_copy['duration'] = event_copy['end'] - event_copy['start']
            del event_copy['start'], event_copy['end']
            server_events.append(event_copy)

        local_events.append({'type': 'MOBILE_RECEIVE', 'timestamp': t_result_recv, 'duration': 0})

        all_events = local_events + server_events
        all_events.sort(key=lambda x: x.get('timestamp', 0))

        group_stats = {}
        for p in all_patches:
            gid = p.group_id
            if gid not in group_stats:
                group_stats[gid] = {'count': 0, 'bytes': 0}
            group_stats[gid]['count'] += 1
            group_stats[gid]['bytes'] += len(p.data)

        valid_preds = result.output[:curr_bs]
        valid_labels = self._labels_to_list(labels, curr_bs)
        latency = t_result_recv - t_send_start

        return {
            'result': result,
            'batch_bytes': batch_bytes,
            'batch_kb': batch_bytes / 1024.0,
            'all_patches': all_patches,
            'all_events': all_events,
            'group_stats': group_stats,
            'valid_preds': valid_preds,
            'valid_labels': valid_labels,
            'latency': latency,
        }

    def run(self):
        dataset_name = getattr(self.config, 'dataset_name', 'imagenet')
        print(f"[Source] Loading {dataset_name} from {self.data_root} (Loader Batch: {self.loader_batch_size})...")
        
        try:
            # Initialize Dataset Loader
            dataset_kwargs = getattr(self.config, 'dataset_kwargs', {})
            profile_config = self.config.get_input_profile_config()
            # Determine image size from config if possible, or default
            image_size = profile_config.get(
                'mobile_resize_short_side',
                self.config.image_shape[0] if self.config.image_shape else 256,
            )
            
            self.dataset_loader = get_dataset_loader(
                dataset_name, 
                self.data_root, 
                batch_size=self.loader_batch_size, 
                image_size=image_size,
                **dataset_kwargs
            )
            loader = self.dataset_loader.get_loader()
        except Exception as e:
            print(f"[Source] Failed to load dataset {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            return

        # Generate Timestamp: YYYYMMDD_HHMMSS
        now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.config.exp_id is None:
            self.config.exp_id = f"exp_{now_str}"
        else:
            self.config.exp_id = f"{self.config.exp_id}_{now_str}"

        # Setup Logging
        log_dir = os.path.join("logs", "offload", self.config.exp_id)
        os.makedirs(log_dir, exist_ok=True)
        events_log_path = os.path.join(log_dir, "events.jsonl")
        summary_log_path = os.path.join(log_dir, "summary.json")
        
        print(f"[Source] Logs will be saved to {log_dir}")
        events_file = open(events_log_path, "w")

        # Handshake
        self.output_queue.put(('CONFIG', self.config))
        time.sleep(1) # Wait for worker to load model

        # Time Synchronization
        time_offset = perform_time_sync(self.output_queue, self.feedback_queue)

        policy = get_transmission(self.config.transmission_policy_name)

        completed_warmups = 0
        if self.num_warmup > 0:
            print(f"[Source] Running {self.num_warmup} warm-up request(s) (not logged/stat).")
            warmup_iter = iter(loader)
            for _ in tqdm(range(self.num_warmup), desc="Warm-up", leave=False):
                try:
                    images, labels = next(warmup_iter)
                except StopIteration:
                    warmup_iter = iter(loader)
                    try:
                        images, labels = next(warmup_iter)
                    except StopIteration:
                        break
                curr_bs = self._current_batch_size(images)
                self._execute_request(policy, images, labels, curr_bs, time_offset)
                completed_warmups += 1
            print(f"[Source] Warm-up complete: {completed_warmups} request(s).")
        
        # Initialize metrics
        total_bytes = 0 
        total_latency = 0.0 
        total_cache_size_bytes = 0
        max_cache_size_bytes = 0
        total_attn_prob_mass_used = 0.0
        total_attn_prob_mass_full = 0.0
        total_token_prune_kept_patch = 0.0
        total_token_prune_full_patch = 0.0
        total_token_prune_kept_residual_mass = 0.0
        total_token_prune_full_residual_mass = 0.0
        total_token_pscore_kept_mass = 0.0
        total_token_pscore_full_mass = 0.0
        total_partial_token_kept_patch = 0.0
        total_partial_token_full_patch = 0.0
        total_partial_token_sample_count = 0.0
        cache_breakdown_accumulator = {}
        
        # Track event statistics
        event_stats_accumulator = {}
        measured_request_count = 0

        print("[Source] Starting Batch Evaluation Loop...")

        pbar = tqdm(enumerate(loader), total=len(loader), leave=False)
        for batch_idx, (images, labels) in pbar:
            curr_bs = self._current_batch_size(images)
            request = self._execute_request(policy, images, labels, curr_bs, time_offset)

            batch_bytes = request['batch_bytes']
            total_bytes += batch_bytes
            all_events = request['all_events']
            group_stats = request['group_stats']
            result = request['result']
            valid_preds = request['valid_preds']
            valid_labels = request['valid_labels']
            latency = request['latency']
            measured_request_count += 1

            # Aggregate Event Stats (Per Request)
            request_latency_map = {}
            for event in all_events:
                etype = self._latency_event_type(event['type'])
                dur_ms = event.get('duration', 0) * 1000.0 # Convert to ms
                
                if etype not in request_latency_map:
                    request_latency_map[etype] = 0.0
                request_latency_map[etype] += dur_ms
                
            # Update Global Accumulator with Per-Request Total
            for etype, total_dur_ms in request_latency_map.items():
                if etype not in event_stats_accumulator:
                    event_stats_accumulator[etype] = {'count': 0, 'sum': 0.0, 'min': float('inf'), 'max': float('-inf')}
                
                stats = event_stats_accumulator[etype]
                stats['count'] += 1
                stats['sum'] += total_dur_ms
                stats['min'] = min(stats['min'], total_dur_ms)
                stats['max'] = max(stats['max'], total_dur_ms)

            total_latency += latency
            cache_size_bytes = getattr(result, 'cache_size_bytes', 0)
            cache_breakdown_bytes = getattr(result, 'cache_breakdown_bytes', {})
            attn_prob_mass_used = getattr(result, 'attn_prob_mass_used', 0.0)
            attn_prob_mass_full = getattr(result, 'attn_prob_mass_full', 0.0)
            token_prune_kept_patch = getattr(result, 'token_prune_kept_patch', 0.0)
            token_prune_full_patch = getattr(result, 'token_prune_full_patch', 0.0)
            token_prune_kept_residual_mass = getattr(result, 'token_prune_kept_residual_mass', 0.0)
            token_prune_full_residual_mass = getattr(result, 'token_prune_full_residual_mass', 0.0)
            token_pscore_kept_mass = getattr(result, 'token_pscore_kept_mass', 0.0)
            token_pscore_full_mass = getattr(result, 'token_pscore_full_mass', 0.0)
            partial_token_kept_patch = getattr(result, 'partial_token_kept_patch', 0.0)
            partial_token_full_patch = getattr(result, 'partial_token_full_patch', 0.0)
            partial_token_sample_count = getattr(result, 'partial_token_sample_count', 0.0)
            total_cache_size_bytes += cache_size_bytes
            max_cache_size_bytes = max(max_cache_size_bytes, cache_size_bytes)
            total_attn_prob_mass_used += attn_prob_mass_used
            total_attn_prob_mass_full += attn_prob_mass_full
            total_token_prune_kept_patch += token_prune_kept_patch
            total_token_prune_full_patch += token_prune_full_patch
            total_token_prune_kept_residual_mass += token_prune_kept_residual_mass
            total_token_prune_full_residual_mass += token_prune_full_residual_mass
            total_token_pscore_kept_mass += token_pscore_kept_mass
            total_token_pscore_full_mass += token_pscore_full_mass
            total_partial_token_kept_patch += partial_token_kept_patch
            total_partial_token_full_patch += partial_token_full_patch
            total_partial_token_sample_count += partial_token_sample_count
            for key, value in cache_breakdown_bytes.items():
                stats = cache_breakdown_accumulator.setdefault(key, {'sum': 0, 'max': 0})
                stats['sum'] += value
                stats['max'] = max(stats['max'], value)
            
            batch_metrics = self.dataset_loader.evaluate_batch(valid_preds, valid_labels)

            # Log Line
            log_entry = {
                'req_id': batch_idx,
                'metrics': batch_metrics,
                'bytes': batch_bytes,
                'cache_size_bytes': cache_size_bytes,
                'cache_breakdown_bytes': cache_breakdown_bytes,
                'attn_prob_mass_used': attn_prob_mass_used,
                'attn_prob_mass_full': attn_prob_mass_full,
                'token_prune_kept_patch': token_prune_kept_patch,
                'token_prune_full_patch': token_prune_full_patch,
                'token_prune_kept_residual_mass': token_prune_kept_residual_mass,
                'token_prune_full_residual_mass': token_prune_full_residual_mass,
                'token_pscore_kept_mass': token_pscore_kept_mass,
                'token_pscore_full_mass': token_pscore_full_mass,
                'partial_token_kept_patch': partial_token_kept_patch,
                'partial_token_full_patch': partial_token_full_patch,
                'partial_token_sample_count': partial_token_sample_count,
                'group_stats': group_stats,
                'events': all_events,
                'labels': self._json_safe_value(valid_labels)
            }
            log_entry['latency'] = latency
            events_file.write(json.dumps(log_entry) + "\n")
            events_file.flush()
            
            # Update pbar description
            pbar_desc = self.dataset_loader.get_pbar_desc()
            avg_kb = total_bytes/1024/(batch_idx*self.loader_batch_size + curr_bs)
            pbar.set_description(f"{pbar_desc} | Avg. Transfer: {avg_kb:.2f} KB/image")
            
            if self.num_requests is not None and measured_request_count >= self.num_requests:
                break

        final_summary = self.dataset_loader.get_summary()
        print(f"[Source] Final Summary: {final_summary}")
        
        # Calculate Latency Breakdown
        latency_breakdown = {}
        print("\n=== Latency Breakdown (ms) ===")
        print(f"{'Event Type':<25} | {'Avg':<8} | {'Min':<8} | {'Max':<8} | {'Count':<6}")
        print("-" * 65)
        
        sorted_stats = sorted(event_stats_accumulator.items(), key=lambda x: x[1]['sum'], reverse=True)
        
        for etype, stats in sorted_stats:
            avg = stats['sum'] / stats['count'] if stats['count'] > 0 else 0
            latency_breakdown[etype] = {
                'avg_ms': avg,
                'min_ms': stats['min'],
                'max_ms': stats['max'],
                'count': stats['count']
            }
            print(f"{etype:<25} | {avg:<8.2f} | {stats['min']:<8.2f} | {stats['max']:<8.2f} | {stats['count']:<6}")
        print("=" * 65 + "\n")

        request_count = max(measured_request_count, 1)
        avg_cache_size_bytes = total_cache_size_bytes / request_count
        avg_attn_prob_coverage_pct = (
            100.0 * total_attn_prob_mass_used / total_attn_prob_mass_full
            if total_attn_prob_mass_full > 0 else 0.0
        )
        attn_col_keep_pct = 100.0 * float(self.config.appcorr_kwargs.get('attn_col_alive_ratio', 1.0))
        avg_token_keep_pct = (
            100.0 * total_token_prune_kept_patch / total_token_prune_full_patch
            if total_token_prune_full_patch > 0 else 100.0
        )
        avg_token_prune_pct = 100.0 - avg_token_keep_pct
        avg_token_residual_mass_keep_pct = (
            100.0 * total_token_prune_kept_residual_mass / total_token_prune_full_residual_mass
            if total_token_prune_full_residual_mass > 0 else 100.0
        )
        avg_token_pscore_coverage_pct = (
            100.0 * total_token_pscore_kept_mass / total_token_pscore_full_mass
            if total_token_pscore_full_mass > 0 else None
        )
        avg_partial_token_keep_pct = (
            100.0 * total_partial_token_kept_patch / total_partial_token_full_patch
            if total_partial_token_full_patch > 0 else 100.0
        )
        avg_partial_token_kept_patch_count = (
            total_partial_token_kept_patch / total_partial_token_sample_count
            if total_partial_token_sample_count > 0 else 0.0
        )
        avg_partial_token_full_patch_count = (
            total_partial_token_full_patch / total_partial_token_sample_count
            if total_partial_token_sample_count > 0 else 0.0
        )
        appcorr_options = normalize_appcorr_kwargs(
            self.config.appcorr_kwargs,
            self.config.transmission_kwargs,
        )
        appcorr_method = appcorr_options.get('method', 'partial_token')
        print("=== Cache Usage ===")
        print(f"Avg cache size per offload: {avg_cache_size_bytes / (1024 ** 2):.2f} MB")
        print(f"Max cache size per offload: {max_cache_size_bytes / (1024 ** 2):.2f} MB")
        cache_breakdown_summary = {}
        if cache_breakdown_accumulator:
            print("Cache breakdown by property:")
            sorted_cache_breakdown = sorted(
                cache_breakdown_accumulator.items(),
                key=lambda item: item[1]['sum'],
                reverse=True,
            )
            small_entry_threshold_bytes = int(0.01 * (1024 ** 2))
            large_entries = []
            other_sum = 0.0
            other_max = 0
            for key, stats in sorted_cache_breakdown:
                avg_value = stats['sum'] / request_count
                if avg_value < small_entry_threshold_bytes:
                    other_sum += stats['sum']
                    other_max = max(other_max, stats['max'])
                    continue
                large_entries.append((key, stats, avg_value))

            for key, stats, avg_value in large_entries:
                print(
                    f"{key:<25} | Avg {avg_value / (1024 ** 2):>7.2f} MB"
                    f" | Max {stats['max'] / (1024 ** 2):>7.2f} MB"
                )
                cache_breakdown_summary[key] = {
                    'avg_bytes': avg_value,
                    'max_bytes': stats['max'],
                }
            if other_sum > 0:
                other_avg = other_sum / request_count
                print(
                    f"{'other':<25} | Avg {other_avg / (1024 ** 2):>7.2f} MB"
                    f" | Max {other_max / (1024 ** 2):>7.2f} MB"
                )
                cache_breakdown_summary['other'] = {
                    'avg_bytes': other_avg,
                    'max_bytes': other_max,
                }
        print("")
        print("=== Attention Stats ===")
        if appcorr_method == 'partial_channel':
            print(f"Configured attention column keep ratio: {attn_col_keep_pct:.2f}%")
            print(f"Avg attention mass covered during V correction: {avg_attn_prob_coverage_pct:.2f}%")
        else:
            print("Selected queries are recomputed with full attention (partial_token).")
            print(
                f"Avg recomputed patch queries per active sample: "
                f"{avg_partial_token_kept_patch_count:.2f} / {avg_partial_token_full_patch_count:.2f}"
            )
        print("")
        print("=== Token Prune Stats ===")
        if appcorr_method == 'partial_channel':
            print(f"Avg patch-token keep ratio during correction: {avg_token_keep_pct:.2f}%")
            print(f"Avg residual mass covered by kept patches: {avg_token_residual_mass_keep_pct:.2f}%")
        else:
            print(f"Avg patch-token keep ratio during correction: {avg_partial_token_keep_pct:.2f}%")
            print(
                f"Avg recomputed patch count per active sample: "
                f"{avg_partial_token_kept_patch_count:.2f}"
            )
            if avg_token_pscore_coverage_pct is None:
                print("Avg combined pscore covered by recomputed patches: N/A (all candidate pscores were zero)")
            else:
                print(f"Avg combined pscore covered by recomputed patches: {avg_token_pscore_coverage_pct:.2f}%")
        print("")

        sample_count_for_bytes = final_summary.get('total_samples', 1) or 1

        # Write Summary
        summary = {
            'exp_id': self.config.exp_id,
            'dataset_summary': final_summary,
            'avg_bytes_per_sample': total_bytes / sample_count_for_bytes,
            'avg_latency_per_batch': total_latency / request_count,
            'num_warmup_requests': completed_warmups,
            'num_measured_requests': measured_request_count,
            'avg_cache_size_bytes_per_offload': avg_cache_size_bytes,
            'max_cache_size_bytes_per_offload': max_cache_size_bytes,
            'attn_col_keep_pct': attn_col_keep_pct,
            'avg_attn_prob_coverage_pct': avg_attn_prob_coverage_pct,
            'avg_token_keep_pct': avg_token_keep_pct,
            'avg_token_prune_pct': avg_token_prune_pct,
            'avg_token_pscore_coverage_pct': avg_token_pscore_coverage_pct,
            'avg_token_residual_mass_keep_pct': avg_token_residual_mass_keep_pct,
            'avg_partial_token_keep_pct': avg_partial_token_keep_pct,
            'avg_partial_token_kept_patch_count': avg_partial_token_kept_patch_count,
            'avg_partial_token_full_patch_count': avg_partial_token_full_patch_count,
            'appcorr_method': appcorr_method,
            'cache_breakdown_bytes_per_offload': cache_breakdown_summary,
            'time_offset_ms': time_offset * 1000,
            'latency_breakdown': latency_breakdown,
            'config': asdict(self.config)
        }
        with open(summary_log_path, "w") as f:
            json.dump(summary, f, indent=4)
            
        events_file.close()
        self.output_queue.put('STOP')
            
