"""
Visualization and Analytics Tools

Provides comprehensive visualization tools for model analysis, predictions, and dilemma zone insights.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import warnings

from .config import (
    VISUALIZATION_DIR,
    FIGURE_SIZE,
    DPI,
    PLOT_STYLE,
    FEATURE_COLUMNS
)

warnings.filterwarnings('ignore')

# Set plot style
plt.style.use(PLOT_STYLE)


def plot_training_history(
    train_losses: List[float],
    val_losses: List[float],
    save_path: Path = None
):
    """
    Plot training history (loss curves).
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        save_path: Path to save plot
    """
    plt.figure(figsize=FIGURE_SIZE)
    
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"Training history plot saved to: {save_path}")
    else:
        plt.show()


def plot_prediction_distribution(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    save_path: Path = None
):
    """
    Plot distribution of predictions by true class.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(FIGURE_SIZE[0] * 1.5, FIGURE_SIZE[1]))
    
    # Distribution for GO class
    go_probs = y_pred_proba[y_true == 0]
    axes[0].hist(go_probs, bins=20, alpha=0.7, color='green', edgecolor='black')
    axes[0].set_xlabel('P(STOP)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Prediction Distribution: GO Class')
    axes[0].axvline(0.5, color='red', linestyle='--', label='Decision Threshold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Distribution for STOP class
    stop_probs = y_pred_proba[y_true == 1]
    axes[1].hist(stop_probs, bins=20, alpha=0.7, color='red', edgecolor='black')
    axes[1].set_xlabel('P(STOP)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Prediction Distribution: STOP Class')
    axes[1].axvline(0.5, color='red', linestyle='--', label='Decision Threshold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"Prediction distribution plot saved to: {save_path}")
    else:
        plt.show()


def plot_feature_distributions(
    sequences: List[np.ndarray],
    feature_names: List[str] = FEATURE_COLUMNS,
    normalizer_params: Dict = None,
    labels: List[int] = None,
    save_path: Path = None,
    save_individual: bool = True
):
    """
    Plot distributions of features across sequences.
    Creates separate charts for each feature, colored by GO/STOP decision.
    
    Args:
        sequences: List of sequences (may be normalized)
        feature_names: Names of features
        normalizer_params: Normalization parameters to denormalize features (optional)
        labels: List of labels (0=GO, 1=STOP) for each sequence (optional)
        save_path: Path to save combined plot (optional, if None only saves individual)
        save_individual: If True, save separate chart for each feature
    """
    from .config import NORMALIZATION_METHOD
    
    # Flatten sequences and labels
    # Note: np.vstack preserves order, so we can match labels correctly
    all_features = np.vstack(sequences)
    
    # Flatten labels to match feature rows (each sequence has multiple timesteps)
    if labels is not None and len(labels) > 0:
        # Repeat each label for all timesteps in its sequence
        flattened_labels = []
        for seq_idx, seq in enumerate(sequences):
            label = labels[seq_idx] if seq_idx < len(labels) else 0
            flattened_labels.extend([label] * len(seq))
        flattened_labels = np.array(flattened_labels)
        print(f"Flattened {len(sequences)} sequences with labels into {len(flattened_labels)} feature rows")
        print(f"  GO (0): {np.sum(flattened_labels == 0)}, STOP (1): {np.sum(flattened_labels == 1)}")
    else:
        flattened_labels = None
        print("No labels provided, plotting without GO/STOP distinction")
    
    # Denormalize if parameters are provided
    if normalizer_params:
        if NORMALIZATION_METHOD == "standard":
            mean = np.array(normalizer_params.get('mean', []))
            std = np.array(normalizer_params.get('std', []))
            if len(mean) > 0 and len(std) > 0:
                if len(mean) == all_features.shape[1] and len(std) == all_features.shape[1]:
                    # Denormalize all features at once
                    all_features = (all_features * std) + mean
                    print(f"Denormalized features using standard scaling")
                    print(f"  Feature ranges after denormalization:")
                    for i, feat_name in enumerate(feature_names):
                        feat_data = all_features[:, i]
                        print(f"    {feat_name}: [{feat_data.min():.2f}, {feat_data.max():.2f}]")
                else:
                    print(f"Warning: Normalizer params shape mismatch. Features: {all_features.shape[1]}, Mean: {len(mean)}, Std: {len(std)}")
            else:
                print(f"Warning: Normalizer params empty. Mean: {len(mean)}, Std: {len(std)}")
        elif NORMALIZATION_METHOD == "minmax":
            min_val = np.array(normalizer_params.get('min', []))
            range_val = np.array(normalizer_params.get('range', []))
            if len(min_val) > 0 and len(range_val) > 0:
                if len(min_val) == all_features.shape[1] and len(range_val) == all_features.shape[1]:
                    # Denormalize all features at once
                    all_features = (all_features * range_val) + min_val
                    print(f"Denormalized features using min-max scaling")
                    print(f"  Feature ranges after denormalization:")
                    for i, feat_name in enumerate(feature_names):
                        feat_data = all_features[:, i]
                        print(f"    {feat_name}: [{feat_data.min():.2f}, {feat_data.max():.2f}]")
                else:
                    print(f"Warning: Normalizer params shape mismatch. Features: {all_features.shape[1]}, Min: {len(min_val)}, Range: {len(range_val)}")
            else:
                print(f"Warning: Normalizer params empty. Min: {len(min_val)}, Range: {len(range_val)}")
    else:
        print("Warning: No normalizer_params provided, showing normalized values")
    
    n_features = len(feature_names)
    
    # Helper function to plot a single feature distribution
    def plot_single_feature_distribution(feature_data, feature_name, flattened_labels, ax=None):
        """Plot distribution of a single feature, colored by GO/STOP if labels provided.
        
        Returns:
            xlabel: The label to use for x-axis (may differ from feature_name for speed_ms)
        """
        # Convert speed_ms to km/h for plotting
        if feature_name == 'speed_ms':
            feature_data = feature_data * 3.6  # Convert m/s to km/h
            xlabel = 'speed_kmh'
        else:
            xlabel = feature_name
        
        # Filter out zeros (except for distance_to_stop_line where 0 is valid)
        if feature_name != 'distance_to_stop_line':
            if flattened_labels is not None:
                # Filter both feature data and labels
                non_zero_mask = feature_data != 0
                feature_data_filtered = feature_data[non_zero_mask]
                labels_filtered = flattened_labels[non_zero_mask]
            else:
                non_zero_mask = feature_data != 0
                feature_data_filtered = feature_data[non_zero_mask]
                labels_filtered = None
        else:
            # Keep all data for distance_to_stop_line (0 is valid)
            feature_data_filtered = feature_data
            labels_filtered = flattened_labels
        
        if len(feature_data_filtered) == 0:
            if ax:
                ax.text(0.5, 0.5, 'No non-zero data', ha='center', va='center', transform=ax.transAxes)
            else:
                plt.text(0.5, 0.5, 'No non-zero data', ha='center', va='center', transform=plt.gca().transAxes)
            return xlabel
        
        # Plot with color coding if labels are provided
        if labels_filtered is not None:
            go_mask = labels_filtered == 0
            stop_mask = labels_filtered == 1
            
            go_data = feature_data_filtered[go_mask]
            stop_data = feature_data_filtered[stop_mask]
            
            # Create bins
            data_min = feature_data_filtered.min()
            data_max = feature_data_filtered.max()
            # Use appropriate number of bins based on data range
            n_bins = min(50, int((data_max - data_min) / max(0.1, (data_max - data_min) / 50)) + 1)
            bins = np.linspace(data_min, data_max, n_bins)
            
            # Calculate histograms
            go_counts, go_bin_edges = np.histogram(go_data, bins=bins)
            stop_counts, stop_bin_edges = np.histogram(stop_data, bins=bins)
            
            # Calculate bin centers and width
            bin_centers = (go_bin_edges[:-1] + go_bin_edges[1:]) / 2
            bin_width = (go_bin_edges[1] - go_bin_edges[0]) * 0.5  # Half width for side-by-side bars
            
            # Plot side-by-side bars
            if ax:
                ax.bar(bin_centers - bin_width/2, go_counts, width=bin_width, 
                      alpha=0.8, edgecolor='black', color='green', label='GO')
                ax.bar(bin_centers + bin_width/2, stop_counts, width=bin_width, 
                      alpha=0.8, edgecolor='black', color='red', label='STOP')
                ax.legend()
            else:
                plt.bar(bin_centers - bin_width/2, go_counts, width=bin_width, 
                       alpha=0.8, edgecolor='black', color='green', label='GO')
                plt.bar(bin_centers + bin_width/2, stop_counts, width=bin_width, 
                       alpha=0.8, edgecolor='black', color='red', label='STOP')
                plt.legend()
        else:
            # No labels, use single color
            bins = np.linspace(feature_data_filtered.min(), feature_data_filtered.max(), 50)
            if ax:
                ax.hist(feature_data_filtered, bins=bins, alpha=0.7, edgecolor='black', color='steelblue')
            else:
                plt.hist(feature_data_filtered, bins=bins, alpha=0.7, edgecolor='black', color='steelblue')
        
        return xlabel
    
    # Save individual charts
    if save_individual and save_path:
        save_dir = save_path.parent
        save_stem = save_path.stem
        for i, feature_name in enumerate(feature_names):
            feature_data = all_features[:, i]
            
            # Create individual plot
            plt.figure(figsize=FIGURE_SIZE)
            xlabel = plot_single_feature_distribution(feature_data, feature_name, flattened_labels)
            plt.xlabel(xlabel, fontsize=12, fontweight='bold')
            plt.ylabel('Frequency', fontsize=12, fontweight='bold')
            plt.title(f'Distribution: {xlabel}', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save individual chart
            individual_path = save_dir / f'{save_stem}_{feature_name}.png'
            plt.savefig(individual_path, dpi=DPI, bbox_inches='tight')
            plt.close()
            print(f"Feature distribution plot saved to: {individual_path}")
    
    # Create combined grid plot (optional)
    if save_path:
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(FIGURE_SIZE[0] * 1.5, FIGURE_SIZE[1] * n_rows))
        axes = axes.flatten() if n_features > 1 else [axes]
        
        for i, feature_name in enumerate(feature_names):
            feature_data = all_features[:, i]
            xlabel = plot_single_feature_distribution(feature_data, feature_name, flattened_labels, ax=axes[i])
            axes[i].set_xlabel(xlabel, fontsize=10)
            axes[i].set_ylabel('Frequency', fontsize=10)
            axes[i].set_title(f'Distribution: {xlabel}', fontsize=11)
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_features, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"Combined feature distribution plot saved to: {save_path}")
    else:
        # Show combined plot
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(FIGURE_SIZE[0] * 1.5, FIGURE_SIZE[1] * n_rows))
        axes = axes.flatten() if n_features > 1 else [axes]
        
        for i, feature_name in enumerate(feature_names):
            feature_data = all_features[:, i]
            xlabel = plot_single_feature_distribution(feature_data, feature_name, flattened_labels, ax=axes[i])
            axes[i].set_xlabel(xlabel)
            axes[i].set_ylabel('Frequency')
            axes[i].set_title(f'Distribution: {xlabel}')
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_features, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()


def plot_correlation_matrix(
    sequences: List[np.ndarray],
    feature_names: List[str] = FEATURE_COLUMNS,
    save_path: Path = None
):
    """
    Plot correlation matrix of features.
    
    Args:
        sequences: List of sequences
        feature_names: Names of features
        save_path: Path to save plot
    """
    # Flatten sequences and compute correlation
    all_features = np.vstack(sequences)
    df = pd.DataFrame(all_features, columns=feature_names)
    corr_matrix = df.corr()
    
    plt.figure(figsize=FIGURE_SIZE)
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={'label': 'Correlation'}
    )
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"Correlation matrix plot saved to: {save_path}")
    else:
        plt.show()


def denormalize_features(
    normalized_features: np.ndarray,
    normalizer_params: Dict,
    feature_indices: List[int] = None
) -> np.ndarray:
    """
    Denormalize features back to original scale.
    
    Args:
        normalized_features: Normalized feature array of shape (n_samples, n_features) or (n_samples, 1)
        normalizer_params: Normalization parameters from training
        feature_indices: Optional list of feature indices to denormalize (if None, denormalize all)
        
    Returns:
        Denormalized features
    """
    from .config import NORMALIZATION_METHOD
    
    if not normalizer_params:
        return normalized_features
    
    denormalized = normalized_features.copy()
    
    if NORMALIZATION_METHOD == "standard":
        mean = normalizer_params.get('mean')
        std = normalizer_params.get('std')
        if mean is not None and std is not None:
            mean = np.array(mean)
            std = np.array(std)
            if feature_indices is not None and len(feature_indices) == 1:
                # Single feature: extract the specific feature's mean/std
                idx = feature_indices[0]
                mean = mean[idx] if mean.ndim > 0 else mean
                std = std[idx] if std.ndim > 0 else std
            elif feature_indices is not None:
                # Multiple features: extract subset
                mean = mean[feature_indices]
                std = std[feature_indices]
            # Ensure shapes match for broadcasting
            if denormalized.ndim == 2 and mean.ndim == 0:
                denormalized = (denormalized * std) + mean
            else:
                denormalized = (denormalized * std) + mean
    elif NORMALIZATION_METHOD == "minmax":
        min_val = normalizer_params.get('min')
        max_val = normalizer_params.get('max')
        range_val = normalizer_params.get('range')
        if min_val is not None and range_val is not None:
            min_val = np.array(min_val)
            range_val = np.array(range_val)
            if feature_indices is not None and len(feature_indices) == 1:
                # Single feature: extract the specific feature's min/range
                idx = feature_indices[0]
                min_val = min_val[idx] if min_val.ndim > 0 else min_val
                range_val = range_val[idx] if range_val.ndim > 0 else range_val
            elif feature_indices is not None:
                min_val = min_val[feature_indices]
                range_val = range_val[feature_indices]
            # Ensure shapes match for broadcasting
            if denormalized.ndim == 2 and min_val.ndim == 0:
                denormalized = (denormalized * range_val) + min_val
            else:
                denormalized = (denormalized * range_val) + min_val
    
    return denormalized


def plot_speed_vs_distance_scatter(
    sequences: List[np.ndarray],
    labels: List[int],
    feature_names: List[str] = FEATURE_COLUMNS,
    normalizer_params: Dict = None,
    save_path: Path = None
):
    """
    Plot speed vs distance scatter plot colored by prediction.
    
    Args:
        sequences: List of sequences (may be normalized)
        labels: True labels
        feature_names: Names of features
        normalizer_params: Normalization parameters to denormalize features (optional)
        save_path: Path to save plot
    """
    # Extract last timestep features (most relevant for decision)
    last_features = np.array([seq[-1] for seq in sequences])
    
    speed_idx = feature_names.index('speed_ms') if 'speed_ms' in feature_names else 0
    distance_idx = feature_names.index('distance_to_stop_line') if 'distance_to_stop_line' in feature_names else 1
    
    # Denormalize if parameters are provided
    if normalizer_params:
        from .config import NORMALIZATION_METHOD
        
        # Extract normalized values
        speed_normalized = last_features[:, speed_idx]
        distance_normalized = last_features[:, distance_idx]
        
        # Get normalization parameters (convert to numpy arrays if they're lists)
        if NORMALIZATION_METHOD == "standard":
            mean = np.array(normalizer_params.get('mean', []))
            std = np.array(normalizer_params.get('std', []))
            if len(mean) > speed_idx and len(std) > speed_idx and std[speed_idx] > 0:
                speeds = (speed_normalized * std[speed_idx]) + mean[speed_idx]
                print(f"Denormalized speed: mean={mean[speed_idx]:.2f}, std={std[speed_idx]:.2f}, range=[{speeds.min():.2f}, {speeds.max():.2f}]")
            else:
                speeds = speed_normalized
                print(f"Warning: Could not denormalize speed (idx={speed_idx}, mean_len={len(mean)}, std_len={len(std)})")
            if len(mean) > distance_idx and len(std) > distance_idx and std[distance_idx] > 0:
                distances = (distance_normalized * std[distance_idx]) + mean[distance_idx]
                print(f"Denormalized distance: mean={mean[distance_idx]:.2f}, std={std[distance_idx]:.2f}, range=[{distances.min():.2f}, {distances.max():.2f}]")
            else:
                distances = distance_normalized
                print(f"Warning: Could not denormalize distance (idx={distance_idx}, mean_len={len(mean)}, std_len={len(std)})")
        elif NORMALIZATION_METHOD == "minmax":
            min_val = np.array(normalizer_params.get('min', []))
            range_val = np.array(normalizer_params.get('range', []))
            if len(min_val) > speed_idx and len(range_val) > speed_idx and range_val[speed_idx] > 0:
                speeds = (speed_normalized * range_val[speed_idx]) + min_val[speed_idx]
            else:
                speeds = speed_normalized
            if len(min_val) > distance_idx and len(range_val) > distance_idx and range_val[distance_idx] > 0:
                distances = (distance_normalized * range_val[distance_idx]) + min_val[distance_idx]
            else:
                distances = distance_normalized
        else:
            speeds = speed_normalized
            distances = distance_normalized
    else:
        speeds = last_features[:, speed_idx]
        distances = last_features[:, distance_idx]
        print("Warning: No normalizer_params provided, showing normalized values")
    
    plt.figure(figsize=FIGURE_SIZE)
    
    # Plot by class
    go_mask = np.array(labels) == 0
    stop_mask = np.array(labels) == 1
    
    plt.scatter(speeds[go_mask], distances[go_mask], alpha=0.6, label='GO', color='green', s=50)
    plt.scatter(speeds[stop_mask], distances[stop_mask], alpha=0.6, label='STOP', color='red', s=50)
    
    plt.xlabel('Speed (m/s)')
    plt.ylabel('Distance to Stop Line (m)')
    plt.title('Speed vs Distance: Decision Points')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"Speed vs distance scatter plot saved to: {save_path}")
    else:
        plt.show()


def plot_temporal_feature_evolution(
    sequences: List[np.ndarray],
    feature_name: str,
    feature_names: List[str] = FEATURE_COLUMNS,
    normalizer_params: Dict = None,
    save_path: Path = None
):
    """
    Plot how a feature evolves over time in sequences.
    
    Args:
        sequences: List of sequences (may be normalized)
        feature_name: Name of feature to plot
        feature_names: Names of features
        normalizer_params: Normalization parameters to denormalize features (optional)
        save_path: Path to save plot
    """
    if feature_name not in feature_names:
        raise ValueError(f"Feature {feature_name} not found in feature_names")
    
    feature_idx = feature_names.index(feature_name)
    
    # Extract feature values over time
    feature_evolution = np.array([seq[:, feature_idx] for seq in sequences])
    
    # Denormalize if parameters are provided
    if normalizer_params:
        from .config import NORMALIZATION_METHOD
        
        if NORMALIZATION_METHOD == "standard":
            mean = np.array(normalizer_params.get('mean', []))
            std = np.array(normalizer_params.get('std', []))
            if len(mean) > feature_idx and len(std) > feature_idx:
                # Denormalize: (normalized * std) + mean
                feature_evolution = (feature_evolution * std[feature_idx]) + mean[feature_idx]
        elif NORMALIZATION_METHOD == "minmax":
            min_val = np.array(normalizer_params.get('min', []))
            range_val = np.array(normalizer_params.get('range', []))
            if len(min_val) > feature_idx and len(range_val) > feature_idx:
                # Denormalize: (normalized * range) + min
                feature_evolution = (feature_evolution * range_val[feature_idx]) + min_val[feature_idx]
    
    # Compute mean and std
    mean_evolution = feature_evolution.mean(axis=0)
    std_evolution = feature_evolution.std(axis=0)
    
    plt.figure(figsize=FIGURE_SIZE)
    
    time_steps = range(len(mean_evolution))
    plt.plot(time_steps, mean_evolution, 'b-', linewidth=2, label='Mean')
    plt.fill_between(
        time_steps,
        mean_evolution - std_evolution,
        mean_evolution + std_evolution,
        alpha=0.3,
        label='±1 Std Dev'
    )
    
    plt.xlabel('Time Step (frames before yellow)')
    plt.ylabel(feature_name)
    plt.title(f'Temporal Evolution: {feature_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"Temporal evolution plot saved to: {save_path}")
    else:
        plt.show()


def generate_comprehensive_report(
    sequences: List[np.ndarray],
    labels: List[int],
    y_pred_proba: np.ndarray,
    train_losses: List[float] = None,
    val_losses: List[float] = None,
    normalizer_params: Dict = None,
    output_dir: Path = None
):
    """
    Generate comprehensive visualization report.
    
    Args:
        sequences: List of sequences
        labels: True labels
        y_pred_proba: Predicted probabilities
        train_losses: Training losses (optional)
        val_losses: Validation losses (optional)
        output_dir: Output directory
    """
    if output_dir is None:
        output_dir = VISUALIZATION_DIR
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating comprehensive visualization report...")
    print(f"Output directory: {output_dir}")
    
    # Training history
    if train_losses and val_losses:
        plot_training_history(
            train_losses, val_losses,
            save_path=output_dir / 'training_history.png'
        )
    
    # Prediction distribution
    plot_prediction_distribution(
        np.array(labels), y_pred_proba,
        save_path=output_dir / 'prediction_distribution.png'
    )
    
    # Feature distributions (with denormalization and individual charts)
    plot_feature_distributions(
        sequences,
        normalizer_params=normalizer_params,
        labels=labels,
        save_path=output_dir / 'feature_distributions.png',
        save_individual=True
    )
    
    # Correlation matrix
    plot_correlation_matrix(
        sequences,
        save_path=output_dir / 'correlation_matrix.png'
    )
    
    # Speed vs distance
    plot_speed_vs_distance_scatter(
        sequences, labels,
        normalizer_params=normalizer_params,
        save_path=output_dir / 'speed_vs_distance.png'
    )
    
    # Temporal evolution for key features
    key_features = ['speed_ms', 'distance_to_stop_line', 'ttc']
    for feat_name in key_features:
        if feat_name in FEATURE_COLUMNS:
            plot_temporal_feature_evolution(
                sequences, feat_name,
                normalizer_params=normalizer_params,
                save_path=output_dir / f'temporal_evolution_{feat_name}.png'
            )
    
    print(f"\nComprehensive report generated! Saved to: {output_dir}")


def main():
    """
    Command-line interface for visualization tools.
    """
    import argparse
    from .evaluate_model import load_model, predict
    from .sequence_builder import build_sequences_from_csv
    
    parser = argparse.ArgumentParser(description='Generate Visualization Report')
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    
    # Input data arguments (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--csv_path',
        type=str,
        help='Path to single CSV file with vehicle data'
    )
    input_group.add_argument(
        '--csv_paths',
        type=str,
        nargs='+',
        help='List of paths to multiple CSV files with vehicle data'
    )
    input_group.add_argument(
        '--csv_directory',
        type=str,
        help='Directory containing CSV files to load'
    )
    parser.add_argument(
        '--csv_pattern',
        type=str,
        default='*_speed_log*.csv',
        help='Glob pattern for CSV files when using --csv_directory (default: "*_speed_log*.csv")'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for visualizations'
    )
    parser.add_argument(
        '--plot_type',
        type=str,
        default='all',
        choices=['all', 'training', 'predictions', 'features', 'correlation', 'scatter', 'temporal'],
        help='Type of plot to generate'
    )
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model, normalizer_params, _ = load_model(args.checkpoint_path, device)
    
    # Load sequences - need to use build_sequences_from_dataframe with normalizer params
    from .utils import load_csv_data, load_multiple_csv_files, load_csv_files_from_directory
    from .sequence_builder import build_sequences_from_dataframe
    from .config import SEQUENCE_LENGTH
    
    # Determine which input method to use
    input_count = sum([args.csv_path is not None, args.csv_paths is not None, args.csv_directory is not None])
    if input_count == 0:
        raise ValueError("Must provide one of: csv_path, csv_paths, or csv_directory")
    if input_count > 1:
        raise ValueError("Can only provide one of: csv_path, csv_paths, or csv_directory")
    
    if args.csv_directory:
        df = load_csv_files_from_directory(args.csv_directory, pattern=args.csv_pattern, make_tracker_ids_unique=True)
        print(f"Loaded data from directory: {args.csv_directory}")
    elif args.csv_paths:
        df = load_multiple_csv_files(args.csv_paths, make_tracker_ids_unique=True)
        print(f"Loaded data from {len(args.csv_paths)} CSV files")
    else:
        df = load_csv_data(args.csv_path)
        print(f"Loaded data from: {args.csv_path}")
    
    print(f"Total rows in dataset: {len(df)}")
    print(f"Total unique vehicles: {len(df['tracker_id'].unique())}")
    
    sequences, labels, _ = build_sequences_from_dataframe(
        df,
        sequence_length=SEQUENCE_LENGTH,
        normalize=True,
        fit_normalizer=False,
        normalizer_params=normalizer_params
    )
    
    print(f"Extracted {len(sequences)} sequences with labels")
    print(f"  - GO sequences: {sum(1 for l in labels if l == 0)}")
    print(f"  - STOP sequences: {sum(1 for l in labels if l == 1)}")
    
    # Debug: Check normalizer params
    if normalizer_params:
        print(f"\nNormalizer params available:")
        if 'mean' in normalizer_params:
            mean = np.array(normalizer_params['mean'])
            std = np.array(normalizer_params['std'])
            print(f"  Speed (idx 0): mean={mean[0]:.2f}, std={std[0]:.2f}")
            print(f"  Distance (idx 1): mean={mean[1]:.2f}, std={std[1]:.2f}")
    
    # Get predictions
    y_pred_proba = predict(model, sequences, device)
    
    # Load training history if available
    checkpoint_dir = Path(args.checkpoint_path).parent
    metadata_path = checkpoint_dir / 'training_metadata.json'
    
    train_losses = None
    val_losses = None
    
    if metadata_path.exists():
        import json
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            train_losses = metadata.get('train_losses', [])
            val_losses = metadata.get('val_losses', [])
    
    # Generate visualizations
    if args.plot_type == 'all':
        generate_comprehensive_report(
            sequences, labels, y_pred_proba,
            train_losses, val_losses,
            normalizer_params=normalizer_params,
            output_dir=args.output_dir
        )
    else:
        output_dir = Path(args.output_dir) if args.output_dir else VISUALIZATION_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if args.plot_type == 'training' and train_losses and val_losses:
            plot_training_history(train_losses, val_losses, output_dir / 'training_history.png')
        elif args.plot_type == 'predictions':
            plot_prediction_distribution(np.array(labels), y_pred_proba, output_dir / 'prediction_distribution.png')
        elif args.plot_type == 'features':
            plot_feature_distributions(sequences, normalizer_params=normalizer_params, labels=labels, save_path=output_dir / 'feature_distributions.png', save_individual=True)
        elif args.plot_type == 'correlation':
            plot_correlation_matrix(sequences, save_path=output_dir / 'correlation_matrix.png')
        elif args.plot_type == 'scatter':
            plot_speed_vs_distance_scatter(sequences, labels, normalizer_params=normalizer_params, save_path=output_dir / 'speed_vs_distance.png')
        elif args.plot_type == 'temporal':
            for feat_name in ['speed_ms', 'distance_to_stop_line', 'ttc']:
                if feat_name in FEATURE_COLUMNS:
                    plot_temporal_feature_evolution(sequences, feat_name, normalizer_params=normalizer_params, save_path=output_dir / f'temporal_evolution_{feat_name}.png')


if __name__ == '__main__':
    main()

