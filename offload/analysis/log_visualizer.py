import os
import sys
import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
import re

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize offload events timeline")
    parser.add_argument("log_dir", type=str, help="Directory containing events.jsonl")
    parser.add_argument("--max_sessions", type=int, default=10, help="Maximum number of sessions to analyze (default: 10)")
    return parser.parse_args()

def classify_event(event_name):
    # Known server events
    server_gpu_events = {
        'LOAD_INPUT', 'PREPARE_TOKENS', 'FULL_INFERENCE', 
        'APPROX_FORWARD', 'CORRECT_FORWARD', 'HEAD_INFERENCE', 'DECIDE_EXIT',
        'Preprocess', 'Preprocess::PinMemory', 'Preprocess::ToDevice',
        'Preprocess::Slicing', 'Preprocess::GroupMap', 'Preprocess::Dindices'
    }
    
    # Prefix matchers
    if event_name.startswith('MOBILE_'):
        if event_name.startswith('MOBILE_SEND') or event_name == 'MOBILE_RECV' or event_name == 'MOBILE_RECEIVE':
            return 'Network' # Treated specially later, but base is Network
        return 'Mobile'
    elif event_name == 'SERVER_RECEIVE' or event_name == 'SERVER_SEND':
        return 'Network'
    elif event_name == 'Decode':
        return 'Server (CPU)'
    elif event_name in server_gpu_events or event_name.startswith('Preprocess::'):
        return 'Server (GPU)'
    
    # Default fallback
    parts = event_name.split('_')
    if parts[0] == 'MOBILE': return 'Mobile'
    if parts[0] == 'SERVER': return 'Server (GPU)'
    return 'Server (GPU)' # Most generic ops are server GPU side

def read_events(file_path):
    requests = []
    with open(file_path, 'r') as f:
        for line in f:
            if not line.strip(): continue
            try:
                data = json.loads(line)
                requests.append(data)
            except json.JSONDecodeError:
                continue
    return requests

# Vibrant color palette
COLOR_PALETTE = [
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', 
    '#F06292', '#AED581', '#FFD54F', '#4DB6AC', '#7986CB',
    '#A1887F', '#90A4AE', '#E040FB', '#00BCD4', '#FF9800'
]

def get_color(event_name, color_map):
    if event_name not in color_map:
        color_map[event_name] = COLOR_PALETTE[len(color_map) % len(COLOR_PALETTE)]
    return color_map[event_name]

def plot_timeline(request_index, request_data, output_dir, color_map):
    # Extract events array
    events = request_data if isinstance(request_data, list) else request_data.get('events', [])
    if not events: return
    
    # Filter and extract valid events
    valid_events = []
    for e in events:
        if 'type' not in e:
            continue
            
        start_time = None
        end_time = None
        
        if 'start' in e and 'end' in e:
            start_time = e['start']
            end_time = e['end']
        elif 'timestamp' in e and 'duration' in e:
            start_time = e['timestamp']
            end_time = e['timestamp'] + e['duration']
            
        if start_time is not None and end_time is not None:
            # Normalize to the same format
            new_e = e.copy()
            new_e['start'] = start_time
            new_e['end'] = end_time
            valid_events.append(new_e)
            
    if not valid_events:
        print(f"Warning: No valid events found in request {request_index}")
        return
    
    # Draw transmission lines between send and receive events
    network_spans = []
    
    # Sort events by start time
    valid_events.sort(key=lambda x: x['start'])
    base_time = valid_events[0]['start']
    
    # Y-axis position (highest index = top)
    categories = ['Server (GPU)', 'Server (CPU)', 'Network', 'Mobile']
    y_pos = {cat: i for i, cat in enumerate(categories)}
    
    fig, ax = plt.subplots(figsize=(15, 6))
    
    for event in valid_events:
        name = event['type']
        start = event['start'] - base_time
        end = event['end'] - base_time
        duration = max(end - start, 0.0001) # Minimum visible duration
        
        category = classify_event(name)
        
        # Extract base name for color assignment and legend grouping
        base_name = re.sub(r'_G\d+$', '', name)
        color = get_color(base_name, color_map)
        
        y = y_pos[category]
        
        # Shorten text for display if needed
        display_name = name.replace('Preprocess::', 'Prep:').replace('MOBILE_', '').replace('SERVER_', '')
        
        # Plot event block
        bar = ax.barh(y, width=duration, left=start, height=0.6, align='center', 
                color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Add label if duration is large enough, and clip it within the Axes
        if duration > 0.005:
            # We can use the bounding box of the bar to clip the text
            # But the simplest way in matplotlib is to just add it and set `clip_on=True` 
            # and `clip_box=bar[0].get_bbox()` so it doesn't escape the rectangle.
            t = ax.text(start + duration/2, y, display_name, 
                    ha='center', va='center', rotation=0, fontsize=8, color='black')
            t.set_clip_path(bar[0])
            t.set_clip_on(True)
            
    # Ensure SERVER_RECEIVE events are sorted sequentially by start time
    server_recvs = sorted([e for e in valid_events if e['type'] == 'SERVER_RECEIVE'], key=lambda x: x['start'])
    
    # We will identify MOBILE_SEND_G0, G1, ... and match them to SERVER_RECEIVE chronologically
    mobile_sends = {}
    for e in valid_events:
        if e['type'].startswith('MOBILE_SEND'):
            mobile_sends[e['type']] = e['start'] - base_time
            
    # Draw Uplink spans
    prev_server_recv_time = None
    
    # Try to match G0, G1, G2.. to sequential SERVER_RECEIVE events
    # Or fallback to generic MOBILE_SEND if not progressive
    idx = 0
    while True:
        send_key = f'MOBILE_SEND_G{idx}'
        if send_key not in mobile_sends:
            # Fallback to single SEND if it exists and idx==0
            if idx == 0 and 'MOBILE_SEND' in mobile_sends:
                send_key = 'MOBILE_SEND'
            else:
                break # No more groups
                
        if idx < len(server_recvs):
            mobile_tx_start = mobile_sends[send_key]
            server_rx_end = server_recvs[idx]['end'] - base_time
            
            # Constraint: Transmission start is max(mobile_tx_start, prev_server_recv_time)
            tx_start_eff = mobile_tx_start
            if prev_server_recv_time is not None:
                tx_start_eff = max(mobile_tx_start, prev_server_recv_time)
                
            net_dur = server_rx_end - tx_start_eff
            if net_dur > 0 and tx_start_eff <= server_rx_end:
                label = f"UL {idx}" if 'G' in send_key else "UL"
                ax.barh(y_pos['Network'], width=net_dur, left=tx_start_eff, height=0.3, align='center',
                       color='gray', alpha=0.3, edgecolor='black', linestyle='--')
                
                # Add text label if wide enough
                if net_dur > 0.01:
                    ax.text(tx_start_eff + net_dur/2, y_pos['Network'] - 0.25, label, ha='center', fontsize=7)
                
            prev_server_recv_time = server_rx_end
        idx += 1

    # Formatting
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories)
    ax.set_xlabel("Time (seconds)")
    ax.set_title(f"Request {request_index} Timeline Breakdown")
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)
    
    # Create unified legend
    patches = [mpatches.Patch(color=color_map[k], label=k) for k in sorted(color_map.keys())]
    
    # Place legend below the axes
    ax.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.5, -0.15),
              fancybox=True, shadow=True, ncol=8, fontsize='small')

    # Ensure dir exists
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"request_{request_index:04d}_timeline.png")
    
    # Use bbox_inches='tight' so the legend block is fully enclosed in the saved image
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")

def main():
    args = parse_args()
    log_dir = args.log_dir
    
    events_file = os.path.join(log_dir, 'events.jsonl')
    if not os.path.exists(events_file):
        print(f"Error: {events_file} not found.")
        sys.exit(1)
        
    requests = read_events(events_file)
    original_count = len(requests)
    requests = requests[:args.max_sessions]
    print(f"Loaded {original_count} requests, analyzing first {len(requests)} from {events_file}")
    
    color_map = {}
    
    for idx, req in enumerate(requests):
        plot_timeline(idx, req, log_dir, color_map)
        
    print("All timelines generated successfully.")

if __name__ == "__main__":
    main()
