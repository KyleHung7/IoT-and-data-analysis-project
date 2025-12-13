"""
Preprocessing script to convert ground_truth_dataset.csv to train_model.py expected format.

This script converts sensor-based ground truth data to the feature format expected by the LSTM model:
- speed_ms
- distance_to_stop_line
- ttc (time to collision)
- distance_to_front_vehicle
- traffic_density
- class_id
- yellow_light_decision
- tracker_id
- frame_index
"""

import pandas as pd
import numpy as np
from pathlib import Path


def estimate_distance_to_stop_line(lat, lon, v_mps, tracker_id):
    """
    Estimate distance to stop line based on GPS position and velocity.
    
    For ground truth data, we estimate based on:
    - Initial distance (estimated from tracker start)
    - Decreasing over time as vehicle approaches
    """
    # Group by tracker_id to track progression
    # Assume max distance is around 100m at start of yellow light
    # Distance decreases as vehicle progresses
    
    # Simple estimation: use velocity integrated over time
    # For now, use a heuristic based on velocity
    # Higher speed = further from stop line initially (to have time to react)
    
    # This is a rough estimate - you may need to calibrate based on your intersection
    if pd.isna(v_mps) or v_mps < 0.1:
        return 50.0  # Default distance if no velocity
    
    # Rough heuristic: faster vehicles start further away
    estimated_distance = min(100.0, max(5.0, v_mps * 5))
    return estimated_distance


def calculate_ttc(speed_ms, distance_to_stop_line):
    """
    Calculate Time To Collision (TTC) with stop line.
    TTC = distance / speed
    """
    if pd.isna(speed_ms) or speed_ms < 0.1:
        return 10.0  # Large value if speed is very low
    
    if pd.isna(distance_to_stop_line) or distance_to_stop_line < 0.1:
        return 0.1  # Very small if at stop line
    
    ttc = distance_to_stop_line / speed_ms
    return min(20.0, ttc)  # Cap at 20 seconds


def preprocess_ground_truth(input_csv, output_csv):
    """
    Convert ground_truth_dataset.csv to train_model.py expected format.
    
    Args:
        input_csv: Path to ground_truth_dataset.csv
        output_csv: Path to output processed CSV
    """
    print(f"Loading ground truth data from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    print(f"Loaded {len(df)} rows with {len(df['tracker_id'].unique())} unique vehicles")
    
    # Create new dataframe with required columns
    processed_df = pd.DataFrame()
    
    # 1. frame_index (already exists)
    processed_df['frame_index'] = df['frame_index']
    
    # 2. tracker_id (convert to numeric)
    # The training script expects numeric tracker_ids for offset calculations
    unique_ids = df['tracker_id'].unique()
    id_map = {tid: i for i, tid in enumerate(unique_ids, start=1)}
    print(f"Mapped {len(unique_ids)} unique string tracker IDs to integers.")
    processed_df['tracker_id'] = df['tracker_id'].map(id_map)
    
    # 3. speed_ms - use v_mps (velocity in m/s)
    processed_df['speed_ms'] = df['v_mps'].fillna(0.0)
    
    # 4. distance_to_stop_line - estimate based on position and velocity
    print("Estimating distance_to_stop_line...")
    
    # Group by tracker_id and calculate progressive distance
    distance_list = []
    for tracker_id, group in df.groupby('tracker_id'):
        # Sort by frame_index
        group = group.sort_values('frame_index')
        
        # Get initial velocity to estimate starting distance
        initial_speed = group['v_mps'].iloc[0] if not pd.isna(group['v_mps'].iloc[0]) else 10.0
        
        # Start with estimated initial distance
        max_distance = min(100.0, max(30.0, initial_speed * 4))
        
        # Calculate progressive distance (decreasing linearly)
        n_frames = len(group)
        distances = np.linspace(max_distance, 5.0, n_frames)
        
        distance_list.extend(distances)
    
    processed_df['distance_to_stop_line'] = distance_list
    
    # 5. ttc - calculate based on speed and distance
    print("Calculating TTC...")
    processed_df['ttc'] = processed_df.apply(
        lambda row: calculate_ttc(row['speed_ms'], row['distance_to_stop_line']), 
        axis=1
    )
    
    # 6. distance_to_front_vehicle - set reasonable default
    # Assume moderate spacing (you can adjust this)
    processed_df['distance_to_front_vehicle'] = 50.0
    
    # 7. traffic_density - set to moderate (0.5 = medium traffic)
    processed_df['traffic_density'] = 0.5
    
    # 8. class_id - default to car (class_id = 2 based on VEHICLE_TYPES in config)
    processed_df['class_id'] = 2
    
    # 9. yellow_light_decision
    # Crucial change: Only set label on the last frame for each vehicle
    # And set traffic light to GREEN so sequence_builder treats this as "before yellow"
    
    processed_df['yellow_light_decision'] = np.nan
    processed_df['vehicle_type'] = 'car'
    processed_df['traffic_light_status'] = 'green'
    processed_df['yellow_light'] = 0
    
    # Apply labels to last frame
    for tracker_id, group in df.groupby('tracker_id'):
        # Get the label from original data (assuming it's consistent for the track)
        label = group['yellow_light_decision'].iloc[0]
        
        # Find index of the last frame for this vehicle
        last_frame_idx = processed_df[processed_df['tracker_id'] == id_map[tracker_id]].index[-1]
        
        # Set label on last frame
        processed_df.at[last_frame_idx, 'yellow_light_decision'] = label
        
        # NOTE: Do NOT set yellow_light=1 or traffic_light_status='yellow' here
        # This prevents identify_yellow_onset_frames from finding a "global" yellow onset
        # and forces it to fall back to using the decision frame for EACH vehicle individually.
        # This is strictly better for independent traces.
        # processed_df.at[last_frame_idx, 'yellow_light'] = 1
        # processed_df.at[last_frame_idx, 'traffic_light_status'] = 'yellow'
    
    # Calculate time_s based on frame_index (assuming ~30fps)
    processed_df['time_s'] = processed_df['frame_index'] / 30.0
    
    # Add speed in km/h for reference
    processed_df['speed_kmh'] = processed_df['speed_ms'] * 3.6
    
    # Reorder columns to match train_model.py expectations
    output_columns = [
        'frame_index',
        'tracker_id',
        'vehicle_type',
        'class_id',
        'time_s',
        'speed_kmh',
        'speed_ms',
        'traffic_light_status',
        'yellow_light',
        'distance_to_stop_line',
        'distance_to_front_vehicle',
        'traffic_density',
        'ttc',
        'yellow_light_decision'
    ]
    
    processed_df = processed_df[output_columns]
    
    # Save to output CSV
    print(f"\nSaving processed data to {output_csv}...")
    processed_df.to_csv(output_csv, index=False)
    
    # Print statistics
    print(f"\nProcessing complete!")
    print(f"Total rows: {len(processed_df)}")
    print(f"Unique vehicles: {processed_df['tracker_id'].nunique()}")
    print(f"\nLabel distribution:")
    print(processed_df['yellow_light_decision'].value_counts())
    print(f"  0 (GO): {(processed_df['yellow_light_decision'] == 0).sum()}")
    print(f"  1 (STOP): {(processed_df['yellow_light_decision'] == 1).sum()}")
    
    print(f"\nSpeed statistics (m/s):")
    print(processed_df['speed_ms'].describe())
    
    print(f"\nDistance to stop line statistics (m):")
    print(processed_df['distance_to_stop_line'].describe())
    
    return processed_df


if __name__ == "__main__":
    # Define input and output paths
    base_dir = Path(__file__).parent
    input_csv = base_dir / "ground_truth_dataset.csv"
    output_csv = base_dir / "ground_truth_processed.csv"
    
    # Check if input file exists
    if not input_csv.exists():
        print(f"Error: Input file not found: {input_csv}")
        print("Please make sure ground_truth_dataset.csv is in the project root directory")
        exit(1)
    
    # Run preprocessing
    processed_df = preprocess_ground_truth(input_csv, output_csv)
    
    print(f"\n✅ Success! Processed data saved to: {output_csv}")
    print(f"\nYou can now train the model using:")
    print(f"python -m LSTM.train_model --csv_path ground_truth_processed.csv")
