# SimulationEngine Documentation

## Overview

The `SimulationEngine` class is the main orchestrator for the AprilSLAM simulation system. It coordinates all subsystems including rendering, camera control, SLAM processing, data logging, and ground truth calculations to provide a complete simulation environment.

## Class: SimulationEngine

Located in: `src/simulation/simulation_engine.py`

### Purpose

The SimulationEngine serves as the central coordinator that:
- **Orchestrates Subsystems**: Manages all simulation components in a unified framework
- **Coordinates Processing**: Synchronizes rendering, SLAM, and data logging operations
- **Provides Configuration Management**: Handles simulation settings and parameters
- **Manages System Lifecycle**: Controls initialization, execution, and cleanup
- **Ensures Real-time Performance**: Optimizes processing for interactive simulation

### Features

- **Modular Architecture**: Clean separation of concerns with well-defined interfaces
- **Configuration-driven Setup**: JSON-based configuration for flexible simulation scenarios
- **Error Handling**: Robust error handling with comprehensive logging
- **Context Manager Support**: Automatic resource management and cleanup
- **Performance Monitoring**: Real-time metrics and performance analysis
- **Extensible Design**: Easy integration of new simulation components

### Dependencies

```python
import sys
import os
import time
import logging
import pygame
import cv2
import numpy as np
from termcolor import colored
from typing import Optional, Dict, Any

from src.core.slam import SLAM
from .config_manager import SimulationConfig
from .camera_controller import CameraController
from .renderer import SimulationRenderer
from .data_logger import DataLogger, PoseData, ErrorMetrics
from .ground_truth import GroundTruthCalculator
```

## Constructor

### `__init__(self, config_file, movement_enabled=True)`

Initializes the simulation engine with all required subsystems.

#### Parameters

- **config_file** (str): Path to simulation configuration file
- **movement_enabled** (bool, optional): Whether manual camera movement is enabled. Default: True

#### Initialization Sequence

1. **Logging Setup**: Configures comprehensive logging system
2. **Configuration Loading**: Parses simulation settings from config file
3. **Camera Controller**: Initializes camera movement and input handling
4. **Renderer**: Sets up OpenGL rendering system
5. **Ground Truth Calculator**: Initializes precise pose calculation
6. **Data Logger**: Sets up CSV data collection system
7. **SLAM System**: Initializes SLAM with calculated camera parameters

#### Configuration Requirements

The config file must specify:
- Display settings (resolution, FOV, clipping planes)
- Tag configuration (positions, rotations, textures)
- SLAM parameters (tag sizes, camera settings)
- Simulation bounds and constraints

#### Example Usage

```python
from src.simulation.simulation_engine import SimulationEngine

# Initialize with configuration
engine = SimulationEngine("config/sim_settings.json", movement_enabled=True)

# Run simulation
engine.run()

# Or use as context manager
with SimulationEngine("config/sim_settings.json") as engine:
    engine.run()
```

## Core Methods

### `run(self)`

Main simulation loop that orchestrates all simulation components.

#### Execution Flow

1. **Event Handling**: Processes pygame events and user input
2. **Camera Updates**: Updates camera position and orientation
3. **Frame Rendering**: Renders current scene with OpenGL
4. **Frame Capture**: Captures rendered frame for SLAM processing
5. **SLAM Processing**: Executes tag detection and pose estimation
6. **Performance Logging**: Records metrics and performance data
7. **Display Updates**: Updates visual output and metrics display

#### Loop Control

- **Frame Rate Management**: Controls simulation speed and responsiveness
- **Exit Conditions**: Handles clean shutdown on user request or errors
- **Error Recovery**: Manages exceptions without crashing the simulation

#### Performance Monitoring

Real-time display of:
- SLAM processing metrics
- Rendering performance
- Error statistics
- System resource usage

### `_process_slam(self, frame)`

Processes a captured frame through the SLAM system.

#### Parameters

- **frame** (numpy.ndarray): Captured frame from the renderer

#### Processing Steps

1. **Tag Detection**: Runs AprilTag detection on the frame
2. **Pose Estimation**: Calculates camera pose from detected tags
3. **Graph Updates**: Updates SLAM graph with new observations
4. **Error Calculation**: Compares estimated pose with ground truth
5. **Data Logging**: Records results for analysis

#### Return Values

The method processes detections and updates internal state, logging results for analysis.

### `_log_slam_results(self, slam_pose)`

Logs SLAM processing results with error analysis.

#### Parameters

- **slam_pose** (numpy.ndarray): 4x4 estimated camera pose matrix

#### Logged Information

- **Pose Estimates**: Translation and rotation components
- **Ground Truth**: True camera pose for comparison
- **Error Metrics**: Translation and rotation errors
- **Performance Data**: Processing time and detection statistics

### `_display_performance_metrics(self, error_metrics, estimated_pose, ground_truth_pose)`

Displays real-time performance information on the simulation window.

#### Parameters

- **error_metrics** (ErrorMetrics): Calculated performance metrics
- **estimated_pose** (PoseData): SLAM pose estimate
- **ground_truth_pose** (PoseData): True camera pose

#### Display Elements

- **Error Information**: Translation and rotation errors in user-friendly units
- **Pose Data**: Current estimated and true poses
- **System Status**: Number of detected tags, processing rate
- **Visual Indicators**: Color-coded status information

## Subsystem Integration

### Camera Controller Integration

```python
# Update camera based on input or automation
self._update_camera()

# Get current camera state
position = self.camera.get_position()
rotation = self.camera.get_rotation()
```

### Renderer Integration

```python
# Render current frame
self.renderer.render_frame(camera_position, camera_rotation)

# Capture frame for processing
frame = self.renderer.capture_frame()
```

### SLAM Integration

```python
# Process frame through SLAM
detections = self.slam.detect(frame)
pose_estimate = self.slam.my_pose()

# Update visualization
self.slam.vis_slam(ground_truth_pose)
```

### Data Logger Integration

```python
# Create pose data objects
estimated_pose = self.data_logger.create_pose_data(translation, rotation_matrix)
ground_truth_pose = self.data_logger.create_pose_data(gt_translation, gt_rotation)

# Log performance data
error_metrics = self.data_logger.create_error_metrics(...)
self.data_logger.log_frame_data(estimated_pose, ground_truth_pose, error_metrics)
```

## Configuration Management

### SimulationConfig Integration

```python
# Access configuration parameters
display_size = self.config.display_size
tag_configurations = self.config.tags
camera_parameters = self.config.camera_params

# Runtime configuration updates
self.config.update_parameter("movement_speed", new_speed)
```

### Dynamic Reconfiguration

The engine supports runtime configuration changes for:
- Camera movement parameters
- SLAM algorithm settings
- Data logging options
- Rendering parameters

## Error Handling and Logging

### Comprehensive Logging

```python
# Automatic log file management
log_file = 'data/logs/simulation.log'

# Structured logging with multiple levels
logging.info("🚀 Initializing AprilSLAM Simulation Engine")
logging.warning("⚠️ Camera calibration file not found, using defaults")
logging.error("❌ Failed to initialize OpenGL context")
```

### Exception Management

```python
try:
    # Simulation operations
    self.run()
except KeyboardInterrupt:
    logging.info("🛑 Simulation interrupted by user")
except Exception as e:
    logging.error(f"❌ Simulation error: {e}")
    raise
finally:
    self._cleanup()
```

### Resource Cleanup

```python
def _cleanup(self):
    """Clean up all simulation resources."""
    if self.data_logger:
        self.data_logger.close()
    if self.renderer:
        self.renderer.cleanup()
    pygame.quit()
    logging.info("✓ Simulation resources cleaned up")
```

## Performance Optimization

### Real-time Processing

- **Efficient Frame Processing**: Optimized OpenGL operations
- **SLAM Performance**: Tuned detection and pose estimation
- **Memory Management**: Careful resource allocation and cleanup
- **Thread Safety**: Safe multi-threaded operations where applicable

### Scalability Considerations

- **Large Scenes**: Support for many tags and complex environments
- **Extended Runtime**: Memory-efficient operation for long simulations
- **High Resolution**: Performance scaling with display resolution
- **Data Volume**: Efficient handling of large datasets

## Integration Examples

### Basic Engine Usage

```python
from src.simulation.simulation_engine import SimulationEngine

# Standard simulation run
engine = SimulationEngine("config/sim_settings.json")
engine.run()
```

### Automated Testing

```python
# Disable manual movement for automated testing
engine = SimulationEngine("config/test_config.json", movement_enabled=False)

# Set up Monte Carlo bounds
bounds = np.array([-3, 3, -2, 2, 0.5, 4.0])

# Run automated position sampling
for trial in range(100):
    engine.camera.randomize_position(bounds)
    for frame in range(50):  # 50 frames per position
        engine._process_single_frame()
```

### Custom Event Handling

```python
class CustomSimulationEngine(SimulationEngine):
    def _handle_events(self):
        """Override for custom event handling."""
        super()._handle_events()
        
        # Add custom keyboard shortcuts
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            self._save_current_state()
        if keys[pygame.K_TAB]:
            self._toggle_visualization()
```

### Performance Profiling

```python
import time

class ProfilingEngine(SimulationEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_times = []
        self.slam_times = []
    
    def _process_slam(self, frame):
        start_time = time.time()
        super()._process_slam(frame)
        slam_time = time.time() - start_time
        self.slam_times.append(slam_time)
    
    def get_performance_stats(self):
        return {
            'avg_slam_time': np.mean(self.slam_times),
            'max_slam_time': np.max(self.slam_times),
            'total_frames': len(self.frame_times)
        }
```

## Context Manager Support

### Automatic Resource Management

```python
# Recommended usage pattern
with SimulationEngine("config/sim_settings.json") as engine:
    engine.run()
# Resources automatically cleaned up
```

### Implementation

```python
def __enter__(self):
    return self

def __exit__(self, exc_type, exc_val, exc_tb):
    self._cleanup()
    if exc_type is not None:
        logging.error(f"Simulation exited with exception: {exc_val}")
    return False  # Don't suppress exceptions
```

## System Statistics

### Runtime Metrics

```python
stats = engine.get_simulation_stats()
print(f"Frames processed: {stats['frames_processed']}")
print(f"Average FPS: {stats['average_fps']:.1f}")
print(f"SLAM accuracy: {stats['mean_translation_error']:.2f}mm")
```

### Available Statistics

- **Performance Metrics**: Frame rate, processing times
- **SLAM Statistics**: Detection rates, pose accuracy
- **System Resources**: Memory usage, CPU utilization
- **Error Analysis**: Comprehensive error statistics

## Related Components

- [`Simulation`](SIMULATION.md): High-level simulation interface
- [`CameraController`](CAMERA_CONTROLLER.md): Camera movement and input handling
- [`SimulationRenderer`](RENDERER.md): OpenGL rendering system
- [`DataLogger`](DATA_LOGGER.md): Data collection and analysis
- [`SLAM`](../core/SLAM.md): SLAM system integration