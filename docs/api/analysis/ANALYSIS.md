# Analysis Module Documentation

## Overview

The Analysis module provides comprehensive tools for performance evaluation, error analysis, and statistical assessment of the AprilSLAM system. It includes utilities for covariance analysis, error visualization, and graph-based performance metrics.

## Module Structure

Located in: `src/analysis/`

### Components

- **error_analysis.py**: Error calculation and statistical analysis tools
- **covarience.py**: Covariance matrix analysis and uncertainty quantification
- **graph.py**: Graph visualization and network analysis utilities

### Purpose

The Analysis module serves as the evaluation and assessment toolkit for:
- **Performance Metrics**: Quantitative assessment of SLAM accuracy
- **Error Analysis**: Detailed breakdown of estimation errors
- **Statistical Evaluation**: Uncertainty quantification and confidence intervals
- **Visualization**: Graphical representation of performance data
- **Research Support**: Tools for academic research and development

## Class: GraphVisualizer

Located in: `src/analysis/graph.py`

### Purpose

Provides specialized graph visualization capabilities for SLAM performance analysis and network structure visualization.

### Methods

#### `visualize_slam_graph(self, nodes, connections)`

Creates network visualization of SLAM graph structure.

#### `plot_error_distribution(self, error_data)`

Generates statistical plots of error distributions.

#### `create_performance_dashboard(self, metrics)`

Creates comprehensive performance visualization dashboard.

### Usage Example

```python
from src.analysis.graph import GraphVisualizer

visualizer = GraphVisualizer()
visualizer.visualize_slam_graph(slam_nodes, connections)
visualizer.plot_error_distribution(error_metrics)
```

## Error Analysis Functions

### Translation Error Analysis

#### `calculate_translation_error(estimated_pose, ground_truth_pose)`

Computes Euclidean distance error between estimated and true positions.

#### Parameters
- **estimated_pose** (numpy.ndarray): 4x4 estimated transformation matrix
- **ground_truth_pose** (numpy.ndarray): 4x4 ground truth transformation matrix

#### Returns
- **error** (float): Translation error magnitude in meters

### Rotation Error Analysis

#### `calculate_rotation_error(estimated_rotation, ground_truth_rotation)`

Computes angular error between estimated and true orientations.

#### Parameters
- **estimated_rotation** (numpy.ndarray): 3x3 estimated rotation matrix
- **ground_truth_rotation** (numpy.ndarray): 3x3 ground truth rotation matrix

#### Returns
- **error** (float): Rotation error in degrees

### Statistical Analysis

#### `compute_error_statistics(error_data)`

Calculates comprehensive error statistics.

#### Returns
Dictionary containing:
- **mean_error**: Average error magnitude
- **std_error**: Standard deviation of errors
- **max_error**: Maximum error observed
- **percentiles**: 50th, 95th, 99th percentile errors
- **rmse**: Root mean square error

## Covariance Analysis

### Uncertainty Quantification

#### `estimate_pose_covariance(observations, estimated_pose)`

Estimates pose uncertainty using observation covariance.

#### `propagate_uncertainty(covariance_matrix, transformation)`

Propagates uncertainty through coordinate transformations.

### Confidence Intervals

#### `calculate_confidence_ellipse(covariance_2d, confidence_level=0.95)`

Computes confidence ellipse parameters for 2D uncertainty visualization.

## Performance Metrics

### Accuracy Metrics

- **Absolute Trajectory Error (ATE)**: Overall trajectory accuracy
- **Relative Pose Error (RPE)**: Frame-to-frame accuracy
- **Translation Accuracy**: Position estimation accuracy
- **Rotation Accuracy**: Orientation estimation accuracy

### Precision Metrics

- **Repeatability**: Consistency across multiple runs
- **Convergence Rate**: Speed of algorithm convergence
- **Stability**: Performance under varying conditions

### Example Analysis Pipeline

```python
from src.analysis.error_analysis import (
    calculate_translation_error,
    calculate_rotation_error,
    compute_error_statistics
)
from src.analysis.covarience import estimate_pose_covariance
from src.analysis.graph import GraphVisualizer

# Load data
estimated_poses = load_slam_results()
ground_truth_poses = load_ground_truth()

# Calculate errors
translation_errors = []
rotation_errors = []

for est, gt in zip(estimated_poses, ground_truth_poses):
    trans_err = calculate_translation_error(est, gt)
    rot_err = calculate_rotation_error(est[:3, :3], gt[:3, :3])
    
    translation_errors.append(trans_err)
    rotation_errors.append(rot_err)

# Statistical analysis
trans_stats = compute_error_statistics(translation_errors)
rot_stats = compute_error_statistics(rotation_errors)

# Visualization
visualizer = GraphVisualizer()
visualizer.plot_error_distribution(translation_errors)
visualizer.create_performance_dashboard({
    'translation': trans_stats,
    'rotation': rot_stats
})

# Uncertainty analysis
covariance = estimate_pose_covariance(observations, final_pose)
confidence_ellipse = calculate_confidence_ellipse(covariance[:2, :2])
```

## Integration with DataLogger

### Automated Analysis

```python
from src.simulation.data_logger import DataLogger
from src.analysis.error_analysis import analyze_logged_data

# Load logged data
with DataLogger() as logger:
    # Simulation runs...
    pass

# Analyze results
results = analyze_logged_data('data/csv/slam_simulation_data.csv')
print(f"Mean translation error: {results['mean_translation_error']:.3f}m")
print(f"Mean rotation error: {results['mean_rotation_error']:.1f}°")
```

### Batch Processing

```python
# Analyze multiple simulation runs
batch_results = []
for run_dir in simulation_directories:
    data_file = os.path.join(run_dir, 'slam_simulation_data.csv')
    analysis = analyze_logged_data(data_file)
    batch_results.append(analysis)

# Compare performance across configurations
compare_batch_performance(batch_results)
```

## Visualization Capabilities

### Error Plots

- **Error vs. Time**: Temporal error evolution
- **Error Distribution**: Histogram and statistical plots
- **Trajectory Comparison**: Side-by-side trajectory visualization
- **Heat Maps**: Spatial error distribution

### Performance Dashboards

- **Real-time Metrics**: Live performance monitoring
- **Comparative Analysis**: Multi-configuration comparison
- **Statistical Summary**: Comprehensive metric overview

### Example Visualization Code

```python
import matplotlib.pyplot as plt
from src.analysis.graph import GraphVisualizer

# Create comprehensive analysis plots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Error evolution over time
axes[0, 0].plot(time_stamps, translation_errors)
axes[0, 0].set_title('Translation Error vs Time')
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('Error (m)')

# Error distribution histogram
axes[0, 1].hist(translation_errors, bins=50, alpha=0.7)
axes[0, 1].set_title('Translation Error Distribution')
axes[0, 1].set_xlabel('Error (m)')
axes[0, 1].set_ylabel('Frequency')

# Trajectory comparison
axes[0, 2].plot(gt_trajectory[:, 0], gt_trajectory[:, 1], 'b-', label='Ground Truth')
axes[0, 2].plot(est_trajectory[:, 0], est_trajectory[:, 1], 'r--', label='Estimated')
axes[0, 2].set_title('Trajectory Comparison')
axes[0, 2].legend()

# Similar plots for rotation errors
# ... additional plotting code ...

plt.tight_layout()
plt.show()
```

## Research Applications

### Academic Research Support

- **Benchmark Comparison**: Standard evaluation metrics
- **Statistical Significance**: Hypothesis testing tools
- **Publication Quality Plots**: High-resolution figure generation
- **Reproducible Analysis**: Standardized evaluation protocols

### Algorithm Development

- **Performance Profiling**: Identify bottlenecks and improvements
- **Parameter Optimization**: Guide algorithm tuning
- **Ablation Studies**: Component-wise performance analysis

## Configuration and Customization

### Analysis Parameters

```python
analysis_config = {
    'error_metrics': ['translation', 'rotation', 'trajectory'],
    'statistical_tests': ['normality', 'stationarity'],
    'visualization': {
        'plot_types': ['time_series', 'histogram', 'scatter'],
        'color_scheme': 'viridis',
        'figure_size': (12, 8)
    },
    'export_formats': ['png', 'pdf', 'csv']
}
```

### Custom Metrics

```python
def custom_error_metric(estimated, ground_truth):
    """Define custom error calculation."""
    # Implementation specific to research needs
    return custom_error_value

# Register custom metric
register_error_metric('custom_metric', custom_error_metric)
```

## Output Formats

### Statistical Reports

- **CSV Tables**: Numerical results for further analysis
- **LaTeX Tables**: Publication-ready formatted tables
- **JSON Reports**: Machine-readable analysis results

### Visualization Exports

- **High-resolution Images**: PNG, PDF for publications
- **Interactive Plots**: HTML with plotly for web viewing
- **Vector Graphics**: SVG for scalable figures

## Performance Considerations

### Large Dataset Handling

- **Streaming Analysis**: Process data in chunks
- **Memory Optimization**: Efficient data structures
- **Parallel Processing**: Multi-threaded analysis for speed

### Real-time Analysis

- **Online Statistics**: Incremental metric calculation
- **Sliding Window**: Recent performance tracking
- **Adaptive Thresholds**: Dynamic performance bounds

## Error Handling and Validation

### Data Validation

- **Input Checking**: Verify data format and completeness
- **Outlier Detection**: Identify and handle anomalous data
- **Missing Data**: Interpolation and handling strategies

### Robust Analysis

- **Statistical Robustness**: Handle non-normal distributions
- **Confidence Intervals**: Uncertainty in analysis results
- **Bootstrap Methods**: Sampling-based confidence estimation

## Related Components

- [`DataLogger`](../simulation/DATA_LOGGER.md): Provides data for analysis
- [`SLAM`](../core/SLAM.md): Generates performance data
- [`SLAMVisualizer`](../core/SLAM_VISUALIZER.md): Real-time visualization
- [`Simulation`](../simulation/SIMULATION.md): Ground truth reference 