# DataLogger Documentation

## Overview

The DataLogger module provides comprehensive data collection and logging capabilities for the AprilSLAM simulation system. It handles CSV file management, error tracking, covariance logging, and statistical data collection for performance analysis and research purposes.

## Module: data_logger.py

Located in: `src/simulation/data_logger.py`

### Purpose

The DataLogger module serves as the central data collection hub for:
- **Performance Analysis**: Collecting pose estimation accuracy metrics
- **Research Data**: Generating datasets for algorithm evaluation
- **Debugging Support**: Providing detailed logging for troubleshooting
- **Statistical Analysis**: Gathering data for error analysis and system optimization

### Features

- **Multi-file CSV Management**: Organized output across specialized CSV files
- **Automatic File Handling**: Context manager support for clean resource management
- **Error Metrics Collection**: Comprehensive error analysis and tracking
- **Covariance Logging**: Uncertainty quantification data collection
- **Performance Statistics**: Real-time statistics calculation and reporting
- **Flexible Output Structure**: Configurable output directories and file organization

## Data Classes

### `PoseData`

A dataclass container for pose estimation data with timestamp information.

#### Attributes

- **translation** (numpy.ndarray): 3D translation vector [x, y, z]
- **rotation_matrix** (numpy.ndarray): 3x3 rotation matrix
- **euler_angles** (numpy.ndarray): Euler angles [yaw, pitch, roll] in degrees
- **timestamp** (float): Unix timestamp of the pose measurement

#### Example Usage

```python
pose_data = PoseData(
    translation=np.array([1.2, -0.5, 2.1]),
    rotation_matrix=rotation_matrix,
    euler_angles=np.array([15.0, -5.2, 0.8]),
    timestamp=time.time()
)
```

### `ErrorMetrics`

A dataclass container for error analysis metrics and performance data.

#### Attributes

- **translation_error** (float): Euclidean distance error in meters
- **rotation_error** (float): Angular error in degrees
- **percentage_error** (float): Relative error as percentage
- **node_count** (int): Number of nodes in SLAM graph
- **average_distance** (float): Average distance to detected tags

#### Example Usage

```python
error_metrics = ErrorMetrics(
    translation_error=0.025,
    rotation_error=2.3,
    percentage_error=4.2,
    node_count=5,
    average_distance=1.8
)
```

## Main Class: DataLogger

### Purpose

The DataLogger class manages all data logging operations for the AprilSLAM simulation, providing organized file output and resource management.

### Constructor

#### `__init__(self, output_directory=None)`

Initializes the data logger with optional custom output directory.

#### Parameters

- **output_directory** (str, optional): Custom directory for output files. If None, uses default `data/csv` directory

#### Initialization Process

1. **Directory Setup**: Creates output directory if it doesn't exist
2. **File Creation**: Initializes three CSV files with headers
3. **Writer Setup**: Configures CSV writers for each file type
4. **Counter Initialization**: Sets up data collection counters
5. **Logging Configuration**: Establishes logging for status tracking

#### Files Created

- **slam_simulation_data.csv**: Main simulation data with pose comparisons
- **error_analysis.csv**: Detailed error analysis with coordinate breakdowns
- **covariance_analysis.csv**: Covariance and uncertainty data

## Core Methods

### `log_frame_data(self, estimated_pose, ground_truth_pose, error_metrics)`

Logs main simulation frame data comparing estimated and ground truth poses.

#### Parameters

- **estimated_pose** (PoseData): SLAM-estimated camera pose
- **ground_truth_pose** (PoseData): True camera pose from simulation
- **error_metrics** (ErrorMetrics): Calculated performance metrics

#### Logged Data Columns

| Column | Description | Unit |
|--------|-------------|------|
| Time | Elapsed simulation time | seconds |
| Number_of_Nodes | Count of SLAM graph nodes | count |
| Average_Distance | Mean distance to tags | meters |
| Est_X, Est_Y, Est_Z | Estimated position | meters |
| Est_Roll, Est_Pitch, Est_Yaw | Estimated orientation | degrees |
| GT_X, GT_Y, GT_Z | Ground truth position | meters |
| GT_Roll, GT_Pitch, GT_Yaw | Ground truth orientation | degrees |
| Translation_Difference | Position error magnitude | meters |
| Rotation_Difference | Orientation error magnitude | degrees |

#### Example Usage

```python
logger.log_frame_data(
    estimated_pose=slam_pose_data,
    ground_truth_pose=true_pose_data,
    error_metrics=calculated_metrics
)
```

### `log_error_analysis(self, jump_count, local_pose, world_pose, tag_pose, world_error, local_error, translation_error)`

Logs detailed error analysis data for in-depth performance evaluation.

#### Parameters

- **jump_count** (int): Frame or iteration counter
- **local_pose** (PoseData): Local coordinate pose estimate
- **world_pose** (PoseData): World coordinate pose estimate
- **tag_pose** (PoseData): Individual tag pose estimate
- **world_error** (float): World coordinate error magnitude
- **local_error** (float): Local coordinate error magnitude
- **translation_error** (float): Translation-specific error

#### Purpose

This method provides detailed coordinate system analysis, enabling researchers to:
- Compare local vs. world coordinate accuracy
- Analyze individual tag contribution to overall error
- Track error evolution over time
- Identify systematic biases in estimation

### `log_covariance_data(self, jump_count, tag_pose, translation_error)`

Logs covariance and uncertainty data for statistical analysis.

#### Parameters

- **jump_count** (int): Frame or iteration counter
- **tag_pose** (PoseData): Tag pose estimate with uncertainty
- **translation_error** (float): Associated translation error

#### Purpose

Supports uncertainty quantification research by collecting:
- Pose estimates with associated uncertainties
- Error correlation analysis data
- Statistical performance metrics

## Utility Methods

### `flush_all(self)`

Forces immediate writing of all buffered data to disk.

#### Use Cases

- **Critical Checkpoints**: Ensuring data persistence at important moments
- **Error Recovery**: Guaranteeing data preservation before potential crashes
- **Real-time Monitoring**: Enabling immediate access to logged data

### `get_statistics(self)`

Returns comprehensive statistics about the logging session.

#### Returns

Dictionary containing:
- **frames_logged**: Total number of frames processed
- **entries_logged**: Total number of data entries
- **session_duration**: Time elapsed since initialization
- **average_fps**: Average logging rate
- **file_sizes**: Size information for each CSV file

#### Example Output

```python
stats = logger.get_statistics()
# {
#     'frames_logged': 1250,
#     'entries_logged': 1250,
#     'session_duration': 45.2,
#     'average_fps': 27.6,
#     'file_sizes': {
#         'main_data': '125KB',
#         'error_analysis': '89KB',
#         'covariance': '42KB'
#     }
# }
```

## Helper Methods

### `create_pose_data(self, translation, rotation_matrix, euler_angles=None)`

Creates a PoseData object with automatic timestamp and angle conversion.

#### Parameters

- **translation** (numpy.ndarray): 3D translation vector
- **rotation_matrix** (numpy.ndarray): 3x3 rotation matrix
- **euler_angles** (numpy.ndarray, optional): Pre-calculated Euler angles

#### Returns

- **PoseData**: Fully populated pose data object

#### Features

- **Automatic Timestamping**: Adds current timestamp
- **Euler Conversion**: Converts rotation matrix to Euler angles if not provided
- **Input Validation**: Ensures proper array shapes and types

### `create_error_metrics(self, translation_error, rotation_error, ground_truth_magnitude, node_count, average_distance)`

Creates an ErrorMetrics object with calculated percentage error.

#### Parameters

- **translation_error** (float): Absolute translation error
- **rotation_error** (float): Absolute rotation error
- **ground_truth_magnitude** (float): Ground truth distance for percentage calculation
- **node_count** (int): Current number of SLAM nodes
- **average_distance** (float): Average distance to tags

#### Returns

- **ErrorMetrics**: Fully populated error metrics object

## Resource Management

### Context Manager Support

The DataLogger implements the context manager protocol for automatic resource cleanup.

#### Example Usage

```python
with DataLogger(output_dir="custom/path") as logger:
    # Logging operations
    logger.log_frame_data(pose1, pose2, metrics)
    logger.log_error_analysis(...)
    # Files automatically closed and flushed on exit
```

### Manual Resource Management

#### `close(self)`

Manually closes all files and cleans up resources.

#### `__del__(self)`

Destructor ensures resource cleanup even if close() isn't called explicitly.

## Integration Examples

### With SLAM System

```python
from src.simulation.data_logger import DataLogger
from src.core.slam import SLAM

# Initialize
logger = DataLogger()
slam = SLAM(logger, camera_params)

# In simulation loop
while running:
    # SLAM processing
    detections = slam.detect(frame)
    estimated_pose_matrix = slam.my_pose()
    
    # Get ground truth
    ground_truth_matrix = get_ground_truth()
    
    # Create pose data
    est_pose = logger.create_pose_data(
        estimated_pose_matrix[:3, 3],
        estimated_pose_matrix[:3, :3]
    )
    
    gt_pose = logger.create_pose_data(
        ground_truth_matrix[:3, 3],
        ground_truth_matrix[:3, :3]
    )
    
    # Calculate errors
    translation_error = np.linalg.norm(
        est_pose.translation - gt_pose.translation
    )
    
    error_metrics = logger.create_error_metrics(
        translation_error=translation_error,
        rotation_error=rotation_error,
        ground_truth_magnitude=np.linalg.norm(gt_pose.translation),
        node_count=len(slam.graph.get_nodes()),
        average_distance=slam.average_distance_to_nodes()
    )
    
    # Log data
    logger.log_frame_data(est_pose, gt_pose, error_metrics)
```

### Batch Analysis

```python
# Process multiple simulation runs
results = []
for config in simulation_configs:
    with DataLogger(f"results/run_{config.id}") as logger:
        # Run simulation with logging
        run_simulation(config, logger)
        
        # Collect statistics
        stats = logger.get_statistics()
        results.append(stats)

# Analyze batch results
analyze_batch_performance(results)
```

## File Output Format

### Main Data File (slam_simulation_data.csv)

```csv
Time,Number_of_Nodes,Average_Distance,Est_X,Est_Y,Est_Z,Est_Roll,Est_Pitch,Est_Yaw,GT_X,GT_Y,GT_Z,GT_Roll,GT_Pitch,GT_Yaw,Translation_Difference,Rotation_Difference
0.033,1,1.25,-0.02,0.15,1.25,2.3,-1.2,0.8,0.00,0.00,1.20,0.0,0.0,0.0,0.058,3.1
0.066,2,1.18,0.12,0.08,1.31,1.8,-0.9,1.2,0.10,0.10,1.30,2.0,-1.0,1.0,0.028,1.7
```

### Error Analysis File (error_analysis.csv)

```csv
Number_of_Jumps,Est_X_Local,Est_Y_Local,Est_Z_Local,Est_Roll_Local,Est_Pitch_Local,Est_Yaw_Local,Est_X_World,Est_Y_World,Est_Z_World,Est_Roll_World,Est_Pitch_World,Est_Yaw_World,Tag_Est_X,Tag_Est_Y,Tag_Est_Z,Tag_Est_Roll,Tag_Est_Pitch,Tag_Est_Yaw,Error_World,Error_Local,Translation_Error
1,0.02,0.15,1.25,2.3,-1.2,0.8,0.02,0.15,1.25,2.3,-1.2,0.8,1.05,0.85,0.02,5.2,1.8,-2.1,0.058,0.012,0.058
```

## Performance Considerations

### I/O Optimization

- **Buffered Writing**: CSV writers use buffering for improved performance
- **Periodic Flushing**: Automatic flushing every 10 entries for data safety
- **Minimal Overhead**: Efficient data structure conversion

### Memory Management

- **No Data Retention**: Logger doesn't store historical data in memory
- **Stream Processing**: Data is written immediately after processing
- **Resource Cleanup**: Automatic file handle management

### Scalability

- **Large Dataset Support**: Handles long simulation runs efficiently
- **Multiple File Strategy**: Separates different data types for easier analysis
- **Configurable Output**: Allows custom directory structure for organization

## Error Handling

### File System Errors

- **Directory Creation**: Automatic creation of missing directories
- **Permission Handling**: Graceful handling of permission errors
- **Disk Space**: Detection and reporting of disk space issues

### Data Validation

- **Array Shape Checking**: Validates input array dimensions
- **Type Conversion**: Automatic conversion of compatible data types
- **Missing Data Handling**: Graceful handling of None values

## Related Components

- [`SLAM`](../core/SLAM.md): Main SLAM system that generates data
- [`Simulation`](SIMULATION.md): Simulation engine that provides ground truth
- [`ErrorAnalysis`](../analysis/ANALYSIS.md): Analysis tools for logged data
- [`SLAMGraph`](../core/SLAM_GRAPH.md): Graph structure providing node counts