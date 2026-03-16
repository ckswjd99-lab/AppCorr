import os
import sys
import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize offload events timeline")
    parser.add_argument("log_dir", type=str, help="Directory containing events.jsonl")
    return parser.parse_args()

def classify_event(event_name):
    # Known server events
    server_events = {
        'LOAD_INPUT', 'PREPARE_TOKENS', 'FULL_INFERENCE', 
        'APPROX_FORWARD', 'CORRECT_FORWARD', 'HEAD_INFERENCE', 'DECIDE_EXIT',
        'Decode', 'Preprocess', 'Preprocess::PinMemory', 'Preprocess::ToDevice',
        'Preprocess::Slicing', 'Preprocess::GroupMap', 'Preprocess::Dindices'
    }
    
    # Prefix matchers
    if event_name.startswith('MOBILE_'):
        if event_name == 'MOBILE_SEND' or event_name == 'MOBILE_RECV':
            return 'Network' # Treated specially later, but base is Network
        return 'Mobile'
    elif event_name == 'SERVER_RECEIVE' or event_name == 'SERVER_SEND':
        return 'Network'
    elif event_name in server_events or event_name.startswith('Preprocess::') or event_name.startswith('Decode'):
        return 'Server'
    
    # Default fallback
    parts = event_name.split('_')
    if parts[0] == 'MOBILE': return 'Mobile'
    if parts[0] == 'SERVER': return 'Server'
    return 'Server' # Most generic ops are server side

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
    # request_data is expected to be a dict containing 'events': [...] (if coming from worker directly)
    # usually source.py outputs a list of events per line
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
    
    # Pre-process Network spans: Connect MOBILE_SEND to SERVER_RECEIVE
    # Find sending and receiving pairs to create a 'Transmission' block
    network_spans = []
    
    # Sort events by start time
    valid_events.sort(key=lambda x: x['start'])
    base_time = valid_events[0]['start']
    
    # Group by category
    categories = ['Mobile', 'Network', 'Server']
    y_pos = {cat: i for i, cat in enumerate(categories)}
    
    fig, ax = plt.subplots(figsize=(15, 6))
    
    for event in valid_events:
        name = event['type']
        start = event['start'] - base_time
        end = event['end'] - base_time
        duration = max(end - start, 0.0001) # Minimum visible duration
        
        category = classify_event(name)
        color = get_color(name, color_map)
        
        y = y_pos[category]
        
        # Small vertical offset to prevent exact overlap if events happen concurrently
        # We will just plot them on the exact line for now, alpha=0.8 helps
        ax.barh(y, width=duration, left=start, height=0.6, align='center', 
                color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Optional: Add text if duration is large enough
        if duration > 0.01:
            ax.text(start + duration/2, y, name, 
                    ha='center', va='center', rotation=0, fontsize=8, color='black')
            
    # Draw Network connections manually if exact pair exists
    send_starts = {e['type']: e['start']-base_time for e in valid_events if 'SEND' in e['type']}
    recv_ends = {e['type']: e['end']-base_time for e in valid_events if 'RECEIVE' in e['type'] or 'RECV' in e['type']}
    
    if 'MOBILE_SEND' in send_starts and 'SERVER_RECEIVE' in recv_ends:
        net_start = send_starts['MOBILE_SEND']
        net_end = recv_ends['SERVER_RECEIVE']
        net_dur = net_end - net_start
        if net_dur > 0:
            ax.barh(y_pos['Network'], width=net_dur, left=net_start, height=0.3, align='center',
                   color='gray', alpha=0.3, edgecolor='black', linestyle='--')
            ax.text(net_start + net_dur/2, y_pos['Network'] - 0.2, "Uplink", ha='center', fontsize=8)

    # Formatting
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(categories)
    ax.set_xlabel("Time (seconds)")
    ax.set_title(f"Request {request_index} Timeline Breakdown")
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)
    
    # Create unified legend
    patches = [mpatches.Patch(color=color_map[k], label=k) for k in sorted(color_map.keys())]
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.2, box.width, box.height * 0.8])
    ax.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.5, -0.15),
              fancybox=True, shadow=True, ncol=5, fontsize='small')

    plt.tight_layout()
    # Ensure dir exists
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"request_{request_index:04d}_timeline.png")
    plt.savefig(out_path, dpi=150)
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
    print(f"Loaded {len(requests)} requests from {events_file}")
    
    color_map = {}
    
    for idx, req in enumerate(requests):
        plot_timeline(idx, req, log_dir, color_map)
        
    print("All timelines generated successfully.")

if __name__ == "__main__":
    main()
