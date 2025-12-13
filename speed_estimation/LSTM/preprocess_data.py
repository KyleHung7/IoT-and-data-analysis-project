import pandas as pd
import json
import os

def load_jsonl(file_path):
    """Reads a JSONL file and returns a DataFrame."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return pd.DataFrame(data)

def preprocess_data():
    """
    Loads specific yellow light sensor logs, assigns labels, and merges them.
    """
    # File paths
    go_file = 'sensor_yellow_go.jsonl'
    stop_file = 'sensor_yellow_stop.jsonl'
    output_file = 'ground_truth_dataset.csv'

    print(f"Processing {go_file}...")
    if not os.path.exists(go_file):
        print(f"Error: {go_file} not found.")
        return

    df_go = load_jsonl(go_file)
    # Assign labels for GO sequence
    # 0 = GO, 1 = STOP
    df_go['yellow_light_decision'] = 0 
    df_go['tracker_id'] = 'yellow_go_001' # Unique ID for this sequence
    df_go['frame_index'] = df_go['idx'] # Use original index as frame index
    
    print(f"Loaded 'GO' sequence: {len(df_go)} frames.")

    print(f"Processing {stop_file}...")
    if not os.path.exists(stop_file):
        print(f"Error: {stop_file} not found.")
        return

    df_stop = load_jsonl(stop_file)
    # Assign labels for STOP sequence
    df_stop['yellow_light_decision'] = 1
    df_stop['tracker_id'] = 'yellow_stop_001' # Unique ID for this sequence
    df_stop['frame_index'] = df_stop['idx'] # Use original index as frame index

    print(f"Loaded 'STOP' sequence: {len(df_stop)} frames.")

    # Combine datasets
    df_combined = pd.concat([df_go, df_stop], ignore_index=True)

    # Sort by tracker_id and frame_index to be safe
    df_combined = df_combined.sort_values(by=['tracker_id', 'frame_index'])

    # Select and reorder columns if needed, but keeping all raw data + new labels is best for now
    # Ensure critical columns for training exist
    print("Columns in dataset:", df_combined.columns.tolist())

    # Save to CSV
    df_combined.to_csv(output_file, index=False)
    print(f"Successfully saved combined dataset to {output_file} ({len(df_combined)} rows).")

if __name__ == "__main__":
    preprocess_data()
