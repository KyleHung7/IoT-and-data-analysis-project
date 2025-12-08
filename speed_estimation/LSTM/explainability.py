"""
Explainability Module for Dilemma Zone Model

Provides SHAP analysis, feature importance, temporal attribution, and partial dependence plots.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import warnings

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. Install with: pip install shap")

from .config import (
    SHAP_SAMPLE_SIZE,
    SHAP_BACKGROUND_SIZE,
    SHAP_DIR,
    VISUALIZATION_DIR,
    FEATURE_COLUMNS,
    FEATURE_DIM,
    SEQUENCE_LENGTH,
    FIGURE_SIZE,
    DPI,
    PLOT_STYLE
)
from .model_architecture import DilemmaZoneModel

warnings.filterwarnings('ignore')

# Set plot style
plt.style.use(PLOT_STYLE)


class ModelWrapper:
    """
    Wrapper for PyTorch model to work with SHAP.
    """
    
    def __init__(self, model: DilemmaZoneModel, device: torch.device):
        self.model = model
        self.device = device
        self.model.eval()
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Predict function for SHAP.
        
        Args:
            x: Input array of shape (n_samples, sequence_length, feature_dim) or (n_samples, flattened)
            
        Returns:
            Predictions of shape (n_samples,)
        """
        # Handle both 2D (flattened) and 3D inputs
        if isinstance(x, np.ndarray):
            if x.ndim == 2:
                # Assume flattened: reshape to (n_samples, seq_len, feat_dim)
                n_samples = x.shape[0]
                x = x.reshape(n_samples, SEQUENCE_LENGTH, FEATURE_DIM)
            x = torch.FloatTensor(x)
        
        x = x.to(self.device)
        
        with torch.no_grad():
            predictions = self.model(x)
        
        return predictions.cpu().numpy().flatten()


def compute_shap_values(
    model: DilemmaZoneModel,
    background_data: np.ndarray,
    test_data: np.ndarray,
    device: torch.device,
    explainer_type: str = "deep"
):
    """
    Compute SHAP values for model predictions.
    
    Args:
        model: Trained model
        background_data: Background dataset for SHAP (shape: n_background, seq_len, feat_dim)
        test_data: Test data to explain (shape: n_test, seq_len, feat_dim)
        device: Device to run on
        explainer_type: Type of SHAP explainer ("deep" for DeepExplainer, "kernel" for KernelExplainer)
        
    Returns:
        Tuple of (SHAP values, explainer)
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP is not installed. Install with: pip install shap")
    
    model_wrapper = ModelWrapper(model, device)
    
    if explainer_type == "deep":
        # Use DeepExplainer for neural networks
        # Convert numpy arrays to torch tensors
        background_tensor = torch.FloatTensor(background_data).to(device)
        try:
            explainer = shap.DeepExplainer(model, background_tensor)
            test_tensor = torch.FloatTensor(test_data).to(device)
            # Disable additivity check to avoid errors with sigmoid
            shap_values_raw = explainer.shap_values(test_tensor, check_additivity=False)
            
            # Handle list output (binary classification)
            if isinstance(shap_values_raw, list):
                shap_values_raw = shap_values_raw[0]
            shap_values_array = np.array(shap_values_raw)
            
            # Get base value
            base_val = explainer.expected_value
            if isinstance(base_val, list):
                base_val = base_val[0]
            base_values = np.array([base_val] * len(shap_values_array))
            
            # Create SHAP Explanation object
            shap_values = shap.Explanation(
                values=shap_values_array,
                base_values=base_values,
                data=test_data,
                feature_names=[f"{feat}_t{t}" for t in range(SEQUENCE_LENGTH) for feat in FEATURE_COLUMNS]
            )
        except Exception as e:
            print(f"Warning: DeepExplainer failed ({e}), falling back to KernelExplainer")
            explainer_type = "kernel"
    
    if explainer_type == "kernel":
        # Use KernelExplainer (slower but more general and reliable)
        # Flatten sequences for KernelExplainer
        background_flat = background_data[:min(SHAP_BACKGROUND_SIZE, len(background_data))].reshape(-1, SEQUENCE_LENGTH * FEATURE_DIM)
        test_flat = test_data[:min(SHAP_SAMPLE_SIZE, len(test_data))].reshape(-1, SEQUENCE_LENGTH * FEATURE_DIM)
        
        explainer = shap.KernelExplainer(
            model_wrapper,
            background_flat
        )
        shap_values_flat = explainer.shap_values(test_flat)
        
        # Reshape back to (n_samples, seq_len, feat_dim)
        if isinstance(shap_values_flat, list):
            shap_values_flat = shap_values_flat[0]
        shap_values_flat = np.array(shap_values_flat)
        shap_values = shap_values_flat.reshape(-1, SEQUENCE_LENGTH, FEATURE_DIM)
        
        # Get base value as array
        base_val = explainer.expected_value
        if isinstance(base_val, list):
            base_val = base_val[0]
        base_values = np.array([base_val] * len(shap_values))
        
        # Create SHAP Explanation object (shap already imported at top)
        shap_values = shap.Explanation(
            values=shap_values,
            base_values=base_values,
            data=test_data[:min(SHAP_SAMPLE_SIZE, len(test_data))],
            feature_names=[f"{feat}_t{t}" for t in range(SEQUENCE_LENGTH) for feat in FEATURE_COLUMNS]
        )
    
    return shap_values, explainer


def aggregate_shap_by_feature(shap_values, test_data: np.ndarray) -> shap.Explanation:
    """
    Aggregate SHAP values by feature (average over timesteps).
    
    Args:
        shap_values: SHAP Explanation object with shape (n_samples, seq_len, feat_dim) or numpy array
        test_data: Test data with shape (n_samples, seq_len, feat_dim)
        
    Returns:
        Aggregated SHAP Explanation object with shape (n_samples, feat_dim)
    """
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    
    # Extract values from Explanation object or use directly if numpy array
    if isinstance(shap_values, shap.Explanation):
        shap_array = shap_values.values
        base_values = shap_values.base_values
        if isinstance(base_values, np.ndarray) and base_values.ndim > 0:
            base_value = float(base_values[0]) if len(base_values) > 0 else 0.0
        else:
            base_value = float(base_values) if not isinstance(base_values, (list, np.ndarray)) else 0.0
    else:
        # Assume it's a numpy array
        shap_array = np.array(shap_values)
        base_value = 0.0
    
    # Handle 4D arrays: (n_samples, seq_len, feat_dim, 1) -> (n_samples, seq_len, feat_dim)
    if shap_array.ndim == 4:
        if shap_array.shape[-1] == 1:
            shap_array = shap_array.squeeze(axis=-1)
        else:
            raise ValueError(f"Unexpected 4D SHAP array shape: {shap_array.shape}")
    
    # Squeeze out any remaining singleton dimensions
    shap_array = np.squeeze(shap_array)
    
    # Handle different input shapes
    if shap_array.ndim == 3:
        # (n_samples, seq_len, feat_dim) - average over timesteps
        n_samples = shap_array.shape[0]
        shap_agg = shap_array.mean(axis=1)  # (n_samples, feat_dim)
    elif shap_array.ndim == 2:
        # Check if it's already aggregated or flattened
        if shap_array.shape[1] == FEATURE_DIM:
            # Already aggregated (n_samples, feat_dim)
            shap_agg = shap_array
        else:
            # Flattened (n_samples, seq_len * feat_dim) - reshape first
            n_samples = shap_array.shape[0]
            shap_array = shap_array.reshape(n_samples, SEQUENCE_LENGTH, FEATURE_DIM)
            shap_agg = shap_array.mean(axis=1)  # (n_samples, feat_dim)
    else:
        raise ValueError(f"Unexpected SHAP array shape after squeezing: {shap_array.shape}")
    
    # Average test data over timesteps for feature values
    if test_data.ndim == 3:
        test_agg = test_data.mean(axis=1)  # (n_samples, feat_dim)
    elif test_data.ndim == 2:
        if test_data.shape[1] == FEATURE_DIM:
            # Already aggregated
            test_agg = test_data
        else:
            # Flattened - reshape first
            n_samples = test_data.shape[0]
            test_data = test_data.reshape(n_samples, SEQUENCE_LENGTH, FEATURE_DIM)
            test_agg = test_data.mean(axis=1)  # (n_samples, feat_dim)
    else:
        raise ValueError(f"Unexpected test_data shape: {test_data.shape}")
    
    # Ensure shapes match
    if shap_agg.shape != test_agg.shape:
        raise ValueError(f"Shape mismatch: shap_agg {shap_agg.shape} vs test_agg {test_agg.shape}")
    
    # Create aggregated Explanation object
    shap_agg_explanation = shap.Explanation(
        values=shap_agg,
        base_values=np.array([base_value] * len(shap_agg)),
        data=test_agg,
        feature_names=FEATURE_COLUMNS[:shap_agg.shape[1]]
    )
    
    return shap_agg_explanation


def get_feature_importance(shap_values: shap.Explanation) -> Dict[str, float]:
    """
    Get feature importance from SHAP values.
    
    Args:
        shap_values: SHAP explanation object
        
    Returns:
        Dictionary mapping feature names to importance scores
    """
    # Calculate mean absolute SHAP values per feature
    if isinstance(shap_values, list):
        # Binary classification: use first class (STOP)
        shap_values = shap_values[0]
    
    # Average over samples and time steps
    feature_importance = np.abs(shap_values.values).mean(axis=(0, 1))  # (feature_dim,)
    
    importance_dict = {
        feature_name: float(importance)
        for feature_name, importance in zip(FEATURE_COLUMNS, feature_importance)
    }
    
    # Sort by importance
    importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    
    return importance_dict


def plot_shap_summary(
    shap_values, 
    test_data: np.ndarray = None,
    save_path: Path = None, 
    max_display: int = 20,
    plot_type: str = "dot"
):
    """
    Plot SHAP summary plot (dot plot or bar plot).
    
    Args:
        shap_values: SHAP Explanation object or array
        test_data: Test data corresponding to SHAP values (optional, for dot plot)
        save_path: Path to save plot
        max_display: Maximum number of features to display
        plot_type: Type of plot - "dot" for summary plot, "bar" for bar plot
    """
    if not SHAP_AVAILABLE:
        print("Warning: SHAP not available. Cannot generate summary plot.")
        return
    
    try:
        # Handle list of SHAP values (binary classification)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        # If shap_values is not a SHAP Explanation object, create one
        if not isinstance(shap_values, shap.Explanation):
            # Assume it's a numpy array
            if test_data is None:
                raise ValueError("test_data required when shap_values is not a SHAP Explanation object")
            
            # Determine if it's already flattened or 3D
            if shap_values.ndim == 3:
                # (n_samples, seq_len, feat_dim) - flatten
                n_samples = shap_values.shape[0]
                shap_flat = shap_values.reshape(n_samples, -1)
                test_flat = test_data.reshape(n_samples, -1)
                feature_names = [f"{feat}_t{t}" for t in range(SEQUENCE_LENGTH) for feat in FEATURE_COLUMNS]
            else:
                # Already flattened
                shap_flat = shap_values
                test_flat = test_data
                feature_names = FEATURE_COLUMNS if test_flat.shape[1] == len(FEATURE_COLUMNS) else [f"feature_{i}" for i in range(test_flat.shape[1])]
            
            # Get base value (mean prediction)
            base_value = 0.0  # Default, will be calculated if available
            
            shap_values = shap.Explanation(
                values=shap_flat,
                base_values=np.array([base_value] * len(shap_flat)),
                data=test_flat,
                feature_names=feature_names[:shap_flat.shape[1]]
            )
        
        # Generate summary plot
        if plot_type == "dot":
            # Dot plot (shows feature values colored by SHAP values)
            # Use larger figure size for better visibility
            plt.figure(figsize=(12, max(8, min(max_display * 0.4, 20))))
            shap.summary_plot(
                shap_values, 
                show=False, 
                max_display=max_display,
                plot_type="dot"
            )
        else:
            # Bar plot (shows mean absolute SHAP values)
            plt.figure(figsize=(10, max(6, min(max_display * 0.3, 15))))
            shap.plots.bar(
                shap_values, 
                show=False, 
                max_display=max_display
            )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
            plt.close()
            print(f"SHAP summary plot ({plot_type}) saved to: {save_path}")
        else:
            plt.show()
            
    except Exception as e:
        print(f"Warning: Could not generate SHAP summary plot: {e}")
        import traceback
        traceback.print_exc()


def plot_shap_force_plot(
    shap_values,
    test_data: np.ndarray,
    sample_idx: int,
    save_path: Path
):
    """
    Plot SHAP force plot for a single prediction.
    """
    try:
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        # Get SHAP values for this sample
        if hasattr(shap_values, 'values'):
            shap_values_sample = shap_values.values[sample_idx]
            base_value = shap_values.base_values[sample_idx] if hasattr(shap_values.base_values, '__len__') else shap_values.base_values
        else:
            shap_values_sample = shap_values[sample_idx]
            base_value = 0.0
        
        test_sample = test_data[sample_idx]
        
        # Create explanation object for this sample (flattened for force plot)
        shap_values_flat = shap_values_sample.reshape(-1)
        test_flat = test_sample.reshape(-1)
        feature_names_flat = [f"{feat}_t{t}" for t in range(SEQUENCE_LENGTH) for feat in FEATURE_COLUMNS]
        
        explanation = shap.Explanation(
            values=shap_values_flat,
            base_values=base_value,
            data=test_flat,
            feature_names=feature_names_flat[:len(shap_values_flat)]
        )
        
        # Plot force plot (HTML version, not matplotlib)
        shap.plots.force(explanation, show=False, matplotlib=False)
        # Save as HTML
        html_path = save_path.with_suffix('.html')
        shap.plots.force(explanation, show=False)
        # Note: Force plots are interactive HTML, matplotlib=True has limitations
        print(f"SHAP force plot (HTML) saved to: {html_path}")
        print(f"Note: Force plots are interactive HTML files. Open in browser to view.")
    except Exception as e:
        print(f"Warning: Could not generate SHAP force plot: {e}")


def plot_temporal_attribution(
    shap_values,
    save_path: Path
):
    """
    Plot temporal feature attribution (SHAP values per timestep).
    """
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    
    # Extract values from Explanation object or use directly
    if isinstance(shap_values, shap.Explanation):
        shap_array = shap_values.values
    else:
        shap_array = np.array(shap_values)
    
    # Handle 4D arrays: (n_samples, seq_len, feat_dim, 1) -> (n_samples, seq_len, feat_dim)
    if shap_array.ndim == 4:
        # Squeeze last dimension if it's 1
        if shap_array.shape[-1] == 1:
            shap_array = shap_array.squeeze(axis=-1)
        else:
            raise ValueError(f"Unexpected 4D SHAP array shape: {shap_array.shape}")
    
    # Squeeze out any remaining singleton dimensions
    shap_array = np.squeeze(shap_array)
    
    # Ensure we have the right shape: (n_samples, seq_len, feat_dim)
    if shap_array.ndim == 2:
        # If 2D, assume it's (n_samples, flattened) and reshape
        n_samples = shap_array.shape[0]
        shap_array = shap_array.reshape(n_samples, SEQUENCE_LENGTH, FEATURE_DIM)
    elif shap_array.ndim == 3:
        # Already 3D, but check if first dimension is 1 (single sample)
        if shap_array.shape[0] == 1:
            shap_array = shap_array[0]  # Remove singleton dimension: (seq_len, feat_dim)
            # Average over samples (but we only have one, so just use it)
            temporal_importance = np.abs(shap_array)  # (seq_len, feat_dim)
        else:
            # Average SHAP values over samples, get per timestep
            temporal_importance = np.abs(shap_array).mean(axis=0)  # (seq_len, feat_dim)
    else:
        raise ValueError(f"Unexpected SHAP values shape after squeezing: {shap_array.shape} (original may have been 4D)")
    
    # Ensure 2D for heatmap
    if temporal_importance.ndim != 2:
        raise ValueError(f"Expected 2D array for heatmap, got shape: {temporal_importance.shape}")
    
    plt.figure(figsize=FIGURE_SIZE)
    
    # Plot heatmap - transpose so features are rows, timesteps are columns
    sns.heatmap(
        temporal_importance.T,
        xticklabels=[f"t-{SEQUENCE_LENGTH-i}" for i in range(SEQUENCE_LENGTH)],
        yticklabels=FEATURE_COLUMNS,
        cmap='viridis',
        annot=True,
        fmt='.3f',
        cbar_kws={'label': 'Mean |SHAP Value|'}
    )
    
    plt.xlabel('Time Step (frames before yellow)')
    plt.ylabel('Feature')
    plt.title('Temporal Feature Attribution')
    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Temporal attribution plot saved to: {save_path}")


def plot_partial_dependence(
    model: DilemmaZoneModel,
    background_data: np.ndarray,
    feature_idx: int,
    feature_name: str,
    feature_range: Tuple[float, float],
    n_points: int = 50,
    save_path: Path = None
):
    """
    Plot partial dependence plot for a single feature.
    
    Args:
        model: Trained model
        background_data: Background data for averaging
        feature_idx: Index of feature to vary
        feature_name: Name of feature
        feature_range: (min, max) range for feature values
        n_points: Number of points to evaluate
        save_path: Path to save plot
    """
    device = next(model.parameters()).device
    
    # Create range of values for this feature
    feature_values = np.linspace(feature_range[0], feature_range[1], n_points)
    
    # Use mean of background data as baseline
    baseline = background_data.mean(axis=0, keepdims=True)  # (1, seq_len, feat_dim)
    baseline = np.repeat(baseline, n_points, axis=0)  # (n_points, seq_len, feat_dim)
    
    # Vary the feature of interest
    # Apply variation to all timesteps (or just last timestep)
    baseline[:, -1, feature_idx] = feature_values  # Vary last timestep
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        baseline_tensor = torch.FloatTensor(baseline).to(device)
        predictions = model(baseline_tensor).cpu().numpy().flatten()
    
    # Plot
    plt.figure(figsize=FIGURE_SIZE)
    plt.plot(feature_values, predictions, linewidth=2)
    plt.xlabel(feature_name)
    plt.ylabel('P(STOP)')
    plt.title(f'Partial Dependence: {feature_name}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"Partial dependence plot saved to: {save_path}")
    else:
        plt.show()


def generate_explainability_report(
    model: DilemmaZoneModel,
    background_sequences: List[np.ndarray],
    test_sequences: List[np.ndarray],
    test_labels: List[int],
    device: torch.device,
    output_dir: Path = None,
    explainer_type: str = "deep"
):
    """
    Generate comprehensive explainability report.
    
    Args:
        model: Trained model
        background_sequences: Background sequences for SHAP
        test_sequences: Test sequences to explain
        test_labels: Test labels
        device: Device to run on
        output_dir: Directory to save outputs
        explainer_type: Type of SHAP explainer
    """
    if output_dir is None:
        output_dir = SHAP_DIR
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating explainability report...")
    print(f"Output directory: {output_dir}")
    
    # Convert to numpy arrays
    background_array = np.array(background_sequences)
    test_array = np.array(test_sequences)
    
    # Limit sizes for efficiency
    if len(background_array) > SHAP_BACKGROUND_SIZE:
        background_array = background_array[:SHAP_BACKGROUND_SIZE]
    if len(test_array) > SHAP_SAMPLE_SIZE:
        test_array = test_array[:SHAP_SAMPLE_SIZE]
    
    print(f"Background samples: {len(background_array)}")
    print(f"Test samples: {len(test_array)}")
    
    # Compute SHAP values
    if not SHAP_AVAILABLE:
        print("Warning: SHAP not available. Skipping SHAP analysis.")
        return
    
    print("Computing SHAP values...")
    shap_values, explainer = compute_shap_values(
        model, background_array, test_array, device, explainer_type
    )
    
    # Get feature importance
    print("Calculating feature importance...")
    feature_importance = get_feature_importance(shap_values)
    
    print("\nFeature Importance (from SHAP):")
    print("-" * 50)
    for feature, importance in feature_importance.items():
        print(f"  {feature}: {importance:.4f}")
    
    # Save feature importance
    import json
    importance_path = output_dir / 'feature_importance.json'
    with open(importance_path, 'w') as f:
        json.dump(feature_importance, f, indent=2)
    print(f"\nFeature importance saved to: {importance_path}")
    
    # Generate plots
    print("\nGenerating SHAP plots...")
    
    # Aggregate SHAP values by feature (average over timesteps) for cleaner summary plots
    print("  - Aggregating SHAP values by feature...")
    try:
        shap_values_agg = aggregate_shap_by_feature(shap_values, test_array)
        
        # Summary plots (both dot and bar) - aggregated by feature
        print("  - Generating SHAP summary plot (dot) - aggregated by feature...")
        plot_shap_summary(
            shap_values_agg, 
            test_data=None,  # Already included in Explanation object
            save_path=output_dir / 'shap_summary_dot.png',
            plot_type="dot",
            max_display=len(FEATURE_COLUMNS)
        )
        
        print("  - Generating SHAP summary plot (bar) - aggregated by feature...")
        plot_shap_summary(
            shap_values_agg,
            test_data=None,
            save_path=output_dir / 'shap_summary_bar.png',
            plot_type="bar",
            max_display=len(FEATURE_COLUMNS)
        )
    except Exception as e:
        print(f"Warning: Could not generate aggregated summary plots: {e}")
        import traceback
        traceback.print_exc()
    
    # Also generate full temporal summary plot (all timesteps) - flatten for SHAP
    print("  - Generating SHAP summary plot (bar) - full temporal (flattened)...")
    try:
        # Flatten the temporal data for SHAP summary plot
        if isinstance(shap_values, shap.Explanation):
            shap_flat = shap_values.values.reshape(len(shap_values.values), -1)
            test_flat = shap_values.data.reshape(len(shap_values.data), -1)
            feature_names_flat = [f"{feat}_t{t}" for t in range(SEQUENCE_LENGTH) for feat in FEATURE_COLUMNS]
            
            shap_flat_explanation = shap.Explanation(
                values=shap_flat,
                base_values=shap_values.base_values,
                data=test_flat,
                feature_names=feature_names_flat[:shap_flat.shape[1]]
            )
            
            plot_shap_summary(
                shap_flat_explanation, 
                test_data=None,
                save_path=output_dir / 'shap_summary_temporal_bar.png',
                plot_type="bar",
                max_display=min(30, shap_flat.shape[1])
            )
        else:
            print("  - Skipping temporal summary plot (shap_values not in expected format)")
    except Exception as e:
        print(f"Warning: Could not generate temporal summary plot: {e}")
        import traceback
        traceback.print_exc()
    
    # Temporal attribution
    print("  - Generating temporal attribution plot...")
    plot_temporal_attribution(shap_values, output_dir / 'temporal_attribution.png')
    
    # Force plots for a few samples
    num_force_plots = min(3, len(test_array))
    for i in range(num_force_plots):
        plot_shap_force_plot(
            shap_values, test_array, i,
            output_dir / f'shap_force_plot_sample_{i}.png'
        )
    
    # Partial dependence plots
    print("\nGenerating partial dependence plots...")
    
    # Get feature ranges from background data
    feature_ranges = []
    for feat_idx in range(FEATURE_DIM):
        feat_values = background_array[:, :, feat_idx].flatten()
        feat_min = float(np.percentile(feat_values, 5))
        feat_max = float(np.percentile(feat_values, 95))
        feature_ranges.append((feat_min, feat_max))
    
    # Plot for key features
    key_features = ['speed_ms', 'distance_to_stop_line', 'ttc']
    for feat_name in key_features:
        if feat_name in FEATURE_COLUMNS:
            feat_idx = FEATURE_COLUMNS.index(feat_name)
            plot_partial_dependence(
                model, background_array, feat_idx, feat_name,
                feature_ranges[feat_idx],
                save_path=output_dir / f'partial_dependence_{feat_name}.png'
            )
    
    print(f"\nExplainability report complete! Saved to: {output_dir}")


def main():
    """
    Command-line interface for explainability analysis.
    """
    import argparse
    from .evaluate_model import load_model
    from .sequence_builder import build_sequences_from_dataframe
    from .utils import load_csv_data, load_multiple_csv_files, load_csv_files_from_directory
    
    parser = argparse.ArgumentParser(description='Generate Explainability Report')
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
        help='Output directory for explainability results'
    )
    parser.add_argument(
        '--explainer_type',
        type=str,
        default='deep',
        choices=['deep', 'kernel'],
        help='Type of SHAP explainer'
    )
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model, normalizer_params, model_config = load_model(args.checkpoint_path, device)
    
    # Load and build sequences
    if args.csv_directory:
        df = load_csv_files_from_directory(args.csv_directory, pattern=args.csv_pattern, make_tracker_ids_unique=True)
        print(f"Loaded data from directory: {args.csv_directory}")
    elif args.csv_paths:
        df = load_multiple_csv_files(args.csv_paths, make_tracker_ids_unique=True)
        print(f"Loaded data from {len(args.csv_paths)} CSV files")
    else:
        df = load_csv_data(args.csv_path)
        print(f"Loaded data from: {args.csv_path}")
    
    sequences, labels, _ = build_sequences_from_dataframe(
        df,
        sequence_length=SEQUENCE_LENGTH,
        normalize=True,
        fit_normalizer=False,
        normalizer_params=normalizer_params
    )
    
    print(f"Built {len(sequences)} sequences from data")
    
    # Split into background and test
    n_background = min(SHAP_BACKGROUND_SIZE, len(sequences) // 2)
    background_sequences = sequences[:n_background]
    test_sequences = sequences[n_background:n_background + SHAP_SAMPLE_SIZE]
    test_labels = labels[n_background:n_background + SHAP_SAMPLE_SIZE]
    
    print(f"Using {len(background_sequences)} background samples and {len(test_sequences)} test samples")
    
    # Generate report
    generate_explainability_report(
        model, background_sequences, test_sequences, test_labels,
        device, args.output_dir, args.explainer_type
    )


if __name__ == '__main__':
    main()

