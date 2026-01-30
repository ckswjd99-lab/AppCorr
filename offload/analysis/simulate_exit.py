
import json
import numpy as np
import argparse
import os

def load_events(log_path):
    data = []
    with open(log_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def simulate_early_exit(events_data, metric, threshold):
    """
    Simulate early exit for a list of requests (batches).
    
    Metrics support:
    - 'max_prob': Top-1 Probability >= Threshold
    - 'entropy': Entropy <= Threshold
    - 'top2_margin': (Top1 - Top2) >= Threshold
    
    Args:
        metric: 'max_prob', 'entropy', 'top2_margin'
        threshold: float
        
    Returns: Global Accuracy, Average Exit Stage, Average Latency (ms), Exit Distribution
    """
    total_samples = 0
    correct_samples = 0
    total_latency = 0.0
    exit_stage_counts = {} # { stage_idx: count }
    
    for batch in events_data:
        labels = batch['labels']
        bs = len(labels)
        total_samples += bs
        
        # 1. Parse Timing Info
        t_send = 0
        t_recv = 0
        
        # Find Mobile Events
        for e in batch['events']:
            if e['type'] == 'MOBILE_SEND':
                t_send = e['timestamp'] # Start of sending
            elif e['type'] == 'MOBILE_RECEIVE':
                t_recv = e['timestamp'] # End of receiving
                
        # Find Last Server Event End Time (to Estimate Downlink)
        # Assuming last event is typically FREE_SESSION or SEND_RESPONSE
        # We look for the max timestamp among server events
        t_server_end = 0
        for e in batch['events']:
            if e['type'] not in ['MOBILE_LOAD', 'MOBILE_ENCODE', 'MOBILE_SEND', 'MOBILE_RECEIVE']:
                end_time = e.get('timestamp', 0) + e.get('duration', 0)
                t_server_end = max(t_server_end, end_time)
                
        # Estimate Downlink Delay (Variable per batch size, but we approximate it as constant RTT tail)
        # RTT_Tail = T_Recv - T_Server_Finish
        downlink_delay = max(0, t_recv - t_server_end)
        
        # Filter HEAD_INFERENCE events & Get their end times
        head_events = []
        for e in batch['events']:
            if e['type'] == 'HEAD_INFERENCE':
                head_events.append(e)
                
        # sort by timestamp just in case
        head_events.sort(key=lambda x: x['timestamp'])
        
        # We need to track per-sample status
        sample_status = [{'exited': False, 'pred': -1, 'exit_stage': -1, 'latency': 0.0} for _ in range(bs)]
        
        active_indices = list(range(bs))
        
        for stage_idx, event in enumerate(head_events):
            if not active_indices:
                break
            
            # Timestamp when this stage finished (decision point)
            t_stage_end = event.get('timestamp', 0) + event.get('duration', 0)
            # Meta contains lists corresponding to the Active Indices of the REAL RUN
            meta = event.get('meta', {})
            if not meta:
                continue
            log_active_indices = meta.get('active_indices')
            top5_probs = meta['top5_probs']
            top5_indices = meta['top5_indices']
            entropies = meta['entropy']
            
            # If log doesn't have active_indices (old logs), fallback to 1-to-1 if sizes match
            if log_active_indices is None:
                 if len(top5_probs) == len(active_indices):
                     log_active_indices = active_indices
                 else:
                     # Mismatch and no index info -> Warning and skip stage
                     # print(f"Warning: Batch {batch['req_id']} Stage {stage_idx} size mismatch & no index info.")
                     continue
            
            # Build Lookup for Current Stage Data
            stage_data_map = {}
            for i, real_idx in enumerate(log_active_indices):
                stage_data_map[real_idx] = {
                    'probs': top5_probs[i],
                    'indices': top5_indices[i],
                    'entropy': entropies[i]
                }
            
            next_active_indices = []
            
            for global_idx in active_indices:
                # Check if we have data for this sample in this stage
                if global_idx not in stage_data_map:
                    # Real Run exited this sample, but Simulation kept it.
                    # We are missing data to make a decision.
                    # Force exit (Conservatice: Exit at previous stage?) 
                    # Or treat as exit at current stage with "Unknown"?
                    # We'll mark as exited but we might lack prediction if we strictly need it from this stage.
                    # Best effort: use last known prediction? Or just ignore/fail sample?
                    # Let's count it as an "Forced Exit due to Missing Data" -> Exit Stage = current
                    # Pred = ?? (We don't have it).
                    # Actually, if Real Run exited, it should have high confidence.
                    # We mark exited, and assume Correct? No, that biases results.
                    # We Mark exited, and try to find prediction from 'final_results' if available?
                    # Simplify: Just mark exited. Pred -1 (Incorrect).
                    sample_status[global_idx]['exited'] = True
                    sample_status[global_idx]['exit_stage'] = stage_idx
                    sample_status[global_idx]['latency'] = (t_stage_end - t_send + downlink_delay) * 1000.0
                    continue

                data = stage_data_map[global_idx]
                
                probs = data['probs']
                indices = data['indices']
                ent = data['entropy']
                
                top1_prob = probs[0]
                top2_prob = probs[1] if len(probs) > 1 else 0.0
                pred_label = indices[0]
                
                # Check Exit Condition
                should_exit = False
                
                if metric == 'max_prob':
                    if top1_prob >= threshold: should_exit = True
                elif metric == 'entropy':
                    if ent <= threshold: should_exit = True
                elif metric == 'top2_margin':
                    if (top1_prob - top2_prob) >= threshold: should_exit = True
                
                # Force exit if this is the last stage
                is_last_stage = (stage_idx == len(head_events) - 1)
                
                if should_exit or is_last_stage:
                    # Finalize
                    sample_status[global_idx]['exited'] = True
                    sample_status[global_idx]['pred'] = pred_label
                    sample_status[global_idx]['exit_stage'] = stage_idx
                    
                    # Latency Calculation
                    lat = (t_stage_end - t_send + downlink_delay) * 1000.0
                    sample_status[global_idx]['latency'] = lat
                else:
                    # Keep active
                    next_active_indices.append(global_idx)
            
            active_indices = next_active_indices
            
        # Per Batch Aggregation
        for i, status in enumerate(sample_status):
            pred = status['pred']
            label = labels[i]
            stage = status['exit_stage']
            lat = status['latency']
            
            if pred == label:
                correct_samples += 1
            
            exit_stage_counts[stage] = exit_stage_counts.get(stage, 0) + 1
            total_latency += lat

    acc = correct_samples / total_samples * 100.0 if total_samples > 0 else 0.0
    avg_lat = total_latency / total_samples if total_samples > 0 else 0.0
    
    # Calc Avg Exit Stage
    total_stages = sum(s * c for s, c in exit_stage_counts.items())
    avg_stage = total_stages / total_samples if total_samples > 0 else 0.0
    
    return acc, avg_stage, avg_lat, exit_stage_counts

def main():
    parser = argparse.ArgumentParser(description="Simulate Early Exit Offline")
    parser.add_argument("--log_path", type=str, required=True, help="Path to events.jsonl")
    parser.add_argument("--metric", type=str, default="max_prob", choices=["max_prob", "entropy", "top2_margin"])
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=1.0)
    parser.add_argument("--step", type=float, default=0.05)
    
    args = parser.parse_args()
    
    events = load_events(args.log_path)
    print(f"Loaded {len(events)} batches.")
    
    print(f"\nSimulation Report: Metric={args.metric}")
    print(f"{'Threshold':<10} | {'Accuracy (%)':<15} | {'Avg Stage':<12} | {'Avg Lat (ms)':<15} | {'Exit Dist.'}")
    print("-" * 85)
    
    import matplotlib.pyplot as plt
    
    thresholds = np.arange(args.start, args.end + args.step/10, args.step)
    
    results = []
    
    for th in thresholds:
        if th == 0.0: continue
        acc, avg_stage, avg_lat, dist = simulate_early_exit(events, args.metric, th)
        dist_str = str(dict(sorted(dist.items())))
        print(f"{th:<10.2f} | {acc:<15.2f} | {avg_stage:<12.2f} | {avg_lat:<15.2f} | {dist_str}")
        results.append((th, acc, avg_lat))
        
    # --- Visualization ---
    ths = [r[0] for r in results]
    accs = [r[1] for r in results]
    lats = [r[2] for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(lats, accs, marker='o', linestyle='-', label=f'{args.metric}', zorder=1)
    
    # 1. Baseline Line
    BASELINE_ACC = 88.11
    plt.axhline(y=BASELINE_ACC, color='gray', linestyle='--', label=f'Baseline ({BASELINE_ACC}%)', zorder=0)
    
    # 2. Highlight Optimal Point (Max Aggression with <1% Loss)
    # Target: Accuracy >= Baseline - 1.0 (87.11%)
    # Goal: Minimize Latency (or Maximize Speedup)
    
    target_acc = BASELINE_ACC - 1.0
    optimal_idx = -1
    min_lat = float('inf')
    
    for i, (th, acc, lat) in enumerate(zip(ths, accs, lats)):
        if acc >= target_acc:
            if lat < min_lat:
                min_lat = lat
                optimal_idx = i
    
    if optimal_idx != -1:
        opt_th = ths[optimal_idx]
        opt_acc = accs[optimal_idx]
        opt_lat = lats[optimal_idx]
        
        # Highlight with Red Circle
        plt.scatter([opt_lat], [opt_acc], color='none', edgecolor='red', s=150, linewidth=2, zorder=2, label='<1% Loss Limit')
        
        # Annotate
        plt.annotate(
            f'Th={opt_th:.2f}\n{opt_acc:.2f}%', 
            (opt_lat, opt_acc), 
            xytext=(10, -20), 
            textcoords='offset points', 
            fontsize=9,
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2")
        )

    plt.title(f'Accuracy vs. Latency Tradeoff\nMetric: {args.metric}, Thresholds: {args.start} - {args.end}')
    plt.xlabel('Average Latency (ms)')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()
    
    img_name = f'acc_latency_tradeoff_{args.metric}.png'
    plt.savefig(img_name)
    print(f"\nPlot saved to {img_name}")

if __name__ == "__main__":
    main()
