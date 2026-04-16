import multiprocessing
import time
import torch
import numpy as np
from tqdm import tqdm
from dataclasses import asdict

from offload.common import ExperimentConfig
from offload.policies import get_transmission
from offload.mobile.dataset import get_dataset_loader

import os
import json
import datetime
import copy

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

    def __init__(self, output_queue, feedback_queue, config: ExperimentConfig, data_root: str, loader_batch_size: int):
        super().__init__()
        self.output_queue = output_queue
        self.feedback_queue = feedback_queue
        self.config = config
        self.data_root = data_root
        self.loader_batch_size = loader_batch_size

    def _prepare_encode_batch(self, images):
        img_h, img_w, img_c = self.config.image_shape
        server_batch_size = self.config.batch_size

        if isinstance(images, torch.Tensor):
            curr_bs = images.size(0)
            full_batch_np = np.zeros((server_batch_size, img_h, img_w, img_c), dtype=np.uint8)
            real_imgs_np = images.permute(0, 2, 3, 1).cpu().numpy()
            full_batch_np[:curr_bs] = real_imgs_np
            return curr_bs, full_batch_np

        if isinstance(images, (list, tuple)):
            batch_np = []
            for img in images:
                if not isinstance(img, torch.Tensor):
                    raise TypeError(f"Expected image tensor, got {type(img)!r}")
                if img.ndim != 3:
                    raise ValueError(f"Expected CHW image tensor, got shape {tuple(img.shape)}")
                if img.shape[0] == img_c:
                    np_img = img.permute(1, 2, 0).cpu().numpy()
                elif img.shape[-1] == img_c:
                    np_img = img.cpu().numpy()
                else:
                    raise ValueError(f"Expected 3-channel image tensor, got shape {tuple(img.shape)}")
                batch_np.append(np.ascontiguousarray(np_img.astype(np.uint8, copy=False)))
            return len(batch_np), batch_np

        raise TypeError(f"Unsupported image batch type: {type(images)!r}")

    @staticmethod
    def _labels_to_list(labels):
        if isinstance(labels, torch.Tensor):
            return labels.tolist()
        return [label.tolist() if isinstance(label, torch.Tensor) else label for label in labels]

    def run(self):
        dataset_name = getattr(self.config, 'dataset_name', 'imagenet')
        if dataset_name == 'coco2017':
            print(f"[Source] Loading {dataset_name} via FiftyOne (Loader Batch: {self.loader_batch_size})...")
        else:
            print(f"[Source] Loading {dataset_name} from {self.data_root} (Loader Batch: {self.loader_batch_size})...")
        
        try:
            # Initialize Dataset Loader
            dataset_kwargs = dict(getattr(self.config, 'dataset_kwargs', {}))
            # Determine image size from config if possible, or default
            image_size = self.config.image_shape[0] if self.config.image_shape else 256
            pyramid_then_resize = self.config.get_pyramid_resize_order() == 'pyramid_then_resize'
            if pyramid_then_resize and self.config.transmission_policy_name not in {'Laplacian', 'ProgressiveLaplacian'}:
                raise ValueError(
                    "pyramid_then_resize is only supported with Laplacian or ProgressiveLaplacian transmission policies."
                )
            if dataset_name == 'coco2017':
                dataset_kwargs['preserve_original_resolution'] = pyramid_then_resize
            
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
        if hasattr(self.dataset_loader, "set_log_dir"):
            self.dataset_loader.set_log_dir(log_dir)
        
        print(f"[Source] Logs will be saved to {log_dir}")
        events_file = open(events_log_path, "w")

        # Handshake
        self.output_queue.put(('CONFIG', self.config))
        time.sleep(1) # Wait for worker to load model

        # Time Synchronization
        time_offset = perform_time_sync(self.output_queue, self.feedback_queue)

        policy = get_transmission(self.config.transmission_policy_name)
        
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
        cache_breakdown_accumulator = {}
        
        # Track event statistics
        event_stats_accumulator = {}

        print("[Source] Starting Batch Evaluation Loop...")

        pbar = tqdm(enumerate(loader), total=len(loader), leave=False)
        for batch_idx, (images, labels) in pbar:
            # Send Phase
            t_load_start = time.time()
            curr_bs, encode_input = self._prepare_encode_batch(images)
            t_load_end = time.time()
            
            # Encode data
            batch_bytes = 0
            all_patches = []
            local_events = [{'type': 'MOBILE_LOAD', 'timestamp': t_load_start, 'duration': t_load_end - t_load_start}]
            
            # Execute generator pipeline
            encode_gen = policy.encode(encode_input, self.config)
            
            group_idx = 0
            t_pipeline_start = time.time()
            t_send_start = t_pipeline_start # Transmission block start
            
            for group_patches in encode_gen:
                t_decode_end = time.time()
                local_events.append({'type': f'MOBILE_ENCODE_G{group_idx}', 'timestamp': t_pipeline_start, 'duration': t_decode_end - t_pipeline_start})
                
                # Send Patches immediately
                t_tx_start = time.time()
                for p in group_patches:
                    self.output_queue.put(p)
                t_tx_end = time.time()
                
                local_events.append({'type': f'MOBILE_SEND_G{group_idx}', 'timestamp': t_tx_start, 'duration': t_tx_end - t_tx_start})
                
                all_patches.extend(group_patches)
                batch_bytes += sum(len(p.data) for p in group_patches)
                
                group_idx += 1
                t_pipeline_start = time.time() # Start timing next group's encode
                
            total_bytes += batch_bytes
            batch_kb = batch_bytes / 1024.0
            
            # Wait for Response
            result = self.feedback_queue.get()
            
            # Calculate Metrics
            t_result_recv = time.time()
            
            # Server Events
            server_events = result.server_events
            # Adjust timestamps
            for e in server_events:
                e['timestamp'] = e['start'] - time_offset # Map to Mobile Time
                e['duration'] = e['end'] - e['start']
                del e['start'], e['end']
            
            # Final local event addition
            local_events.append({'type': 'MOBILE_RECEIVE', 'timestamp': t_result_recv, 'duration': 0})
            
            # Combine
            all_events = local_events + server_events
            all_events.sort(key=lambda x: x.get('timestamp', 0))
            
            # Group Stats Calculation
            group_stats = {}
            for p in all_patches:
                gid = p.group_id
                if gid not in group_stats:
                    group_stats[gid] = {'count': 0, 'bytes': 0}
                group_stats[gid]['count'] += 1
                group_stats[gid]['bytes'] += len(p.data)

            # Aggregate Event Stats (Per Request)
            request_latency_map = {}
            for event in all_events:
                etype = event['type']
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

            # Calculate metrics via DatasetLoader
            valid_preds = result.output[:curr_bs]
            valid_labels = self._labels_to_list(labels)
            
            latency = t_result_recv - t_send_start # End-to-End approximation
            total_latency += latency
            cache_size_bytes = getattr(result, 'cache_size_bytes', 0)
            cache_breakdown_bytes = getattr(result, 'cache_breakdown_bytes', {})
            attn_prob_mass_used = getattr(result, 'attn_prob_mass_used', 0.0)
            attn_prob_mass_full = getattr(result, 'attn_prob_mass_full', 0.0)
            token_prune_kept_patch = getattr(result, 'token_prune_kept_patch', 0.0)
            token_prune_full_patch = getattr(result, 'token_prune_full_patch', 0.0)
            token_prune_kept_residual_mass = getattr(result, 'token_prune_kept_residual_mass', 0.0)
            token_prune_full_residual_mass = getattr(result, 'token_prune_full_residual_mass', 0.0)
            total_cache_size_bytes += cache_size_bytes
            max_cache_size_bytes = max(max_cache_size_bytes, cache_size_bytes)
            total_attn_prob_mass_used += attn_prob_mass_used
            total_attn_prob_mass_full += attn_prob_mass_full
            total_token_prune_kept_patch += token_prune_kept_patch
            total_token_prune_full_patch += token_prune_full_patch
            total_token_prune_kept_residual_mass += token_prune_kept_residual_mass
            total_token_prune_full_residual_mass += token_prune_full_residual_mass
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
                'group_stats': group_stats,
                'events': all_events,
                'labels': valid_labels
            }
            log_entry['latency'] = latency
            events_file.write(json.dumps(log_entry) + "\n")
            events_file.flush()
            
            # Update pbar description
            pbar_desc = self.dataset_loader.get_pbar_desc()
            avg_kb = total_bytes/1024/(batch_idx*self.loader_batch_size + curr_bs)
            pbar.set_description(f"{pbar_desc} | Avg. Transfer: {avg_kb:.2f} KB/image")
            
            # if (batch_idx+1) == 50:
            #     break # TEMP

        final_summary = self.dataset_loader.get_summary()
        console_summary = copy.deepcopy(final_summary)
        if isinstance(console_summary, dict):
            console_summary.pop("detection_exports", None)
        print(f"[Source] Final Summary: {console_summary}")
        
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

        avg_cache_size_bytes = total_cache_size_bytes / (batch_idx + 1)
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
                avg_value = stats['sum'] / (batch_idx + 1)
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
                other_avg = other_sum / (batch_idx + 1)
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
        print(f"Configured attention column keep ratio: {attn_col_keep_pct:.2f}%")
        print(f"Avg attention mass covered during V correction: {avg_attn_prob_coverage_pct:.2f}%")
        print("")
        print("=== Token Prune Stats ===")
        print(f"Avg patch-token keep ratio during correction: {avg_token_keep_pct:.2f}%")
        print(f"Avg residual mass covered by kept patches: {avg_token_residual_mass_keep_pct:.2f}%")
        print("")

        # Write Summary
        summary = {
            'exp_id': self.config.exp_id,
            'dataset_summary': final_summary,
            'avg_bytes_per_sample': total_bytes / final_summary.get('total_samples', 1),
            'avg_latency_per_batch': total_latency / (batch_idx + 1),
            'avg_cache_size_bytes_per_offload': avg_cache_size_bytes,
            'max_cache_size_bytes_per_offload': max_cache_size_bytes,
            'attn_col_keep_pct': attn_col_keep_pct,
            'avg_attn_prob_coverage_pct': avg_attn_prob_coverage_pct,
            'avg_token_keep_pct': avg_token_keep_pct,
            'avg_token_prune_pct': avg_token_prune_pct,
            'avg_token_residual_mass_keep_pct': avg_token_residual_mass_keep_pct,
            'cache_breakdown_bytes_per_offload': cache_breakdown_summary,
            'time_offset_ms': time_offset * 1000,
            'latency_breakdown': latency_breakdown,
            'config': asdict(self.config)
        }
        with open(summary_log_path, "w") as f:
            json.dump(summary, f, indent=4)
            
        events_file.close()
        self.output_queue.put('STOP')
            
