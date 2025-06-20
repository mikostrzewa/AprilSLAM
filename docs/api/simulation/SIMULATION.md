# Simulation Class Documentation

## Overview

The `Simulation` class is the main entry point for the AprilSLAM simulation system. It provides a complete virtual environment with OpenGL rendering, interactive camera controls, SLAM processing, and comprehensive data logging for algorithm testing and evaluation.

## Class: Simulation

Located in: `src/simulation/sim.py`

### Purpose

The Simulation class serves as the high-level interface that:
- **Creates Virtual Environment**: Renders 3D scenes with AprilTags using OpenGL
- **Provides Ground Truth**: Generates accurate pose information for evaluation
- **Integrates SLAM**: Processes simulated camera frames through the SLAM system
- **Logs Performance Data**: Collects comprehensive metrics for analysis
- **Supports User Interaction**: Allows manual camera control for testing scenarios

### Features

- **Real-time 3D Rendering**: OpenGL-based rendering with textured AprilTags
- **Interactive Controls**: Keyboard-based camera movement and rotation
- **SLAM Integration**: Live SLAM processing with pose estimation
- **Monte Carlo Testing**: Automated random position sampling
- **Performance Monitoring**: Real-time display of SLAM accuracy metrics
- **Data Export**: CSV logging for post-simulation analysis
- **Ground Truth Comparison**: Precise error calculation and visualization

### Dependencies

```python
import sys
import os
import time
import json
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import cv2
import csv
import logging
from termcolor import colored
from apriltag import apriltag
from src.core.slam import SLAM
```

## Constructor

### `__init__(self, settings_file, movement_flag=True)`

Initializes the complete simulation environment with all subsystems.

#### Parameters

- **settings_file** (str): Path to JSON configuration file containing simulation parameters
- **movement_flag** (bool, optional): Whether manual camera movement is enabled. Default: True

#### Initialization Process

1. **Configuration Loading**: Parses simulation settings from JSON file
2. **Pygame Setup**: Initializes display and OpenGL context
3. **Texture Loading**: Loads all AprilTag textures into OpenGL
4. **SLAM Initialization**: Sets up SLAM system with calculated camera parameters
5. **Data Logging Setup**: Creates CSV files for performance tracking
6. **Input Handling**: Configures keyboard state management

#### Configuration Parameters

The settings file must contain:
```json
{
    "display_width": 800,
    "display_height": 600,
    "fov_y": 60.0,
    "near_clip": 0.1,
    "far_clip": 100.0,
    "size_scale": 1.0,
    "tag_size_inner": 0.06,
    "tag_size_outer": 0.08,
    "actual_size_in_mm": 60,
    "tags": [
        {
            "id": 0,
            "image": "assets/tags/tag0.png",
            "position": [1.0, 0.0, 2.0],
            "rotation": [0.0, 0.0, 0.0]
        }
    ]
}
```

#### Example Usage

```python
from src.simulation.sim import Simulation

# Create simulation with manual movement
sim = Simulation("config/sim_settings.json", movement_flag=True)
sim.run()

# Create simulation for automated testing
sim_auto = Simulation("config/sim_settings.json", movement_flag=False)
sim_auto.run()
```

## Core Methods

### `run(self)`

Main simulation loop that orchestrates all simulation components.

#### Execution Flow

1. **Event Processing**: Handles pygame events and keyboard input
2. **Movement Update**: Updates camera position based on user input or Monte Carlo
3. **Rendering**: Renders 3D scene with current camera pose
4. **Frame Capture**: Captures rendered frame for SLAM processing
5. **SLAM Processing**: Detects tags and estimates pose
6. **Ground Truth Calculation**: Computes true camera pose
7. **Error Analysis**: Calculates and logs performance metrics
8. **Display Update**: Shows current frame and performance information

#### Performance Monitoring

The simulation displays real-time metrics including:
- Translation error (millimeters)
- Rotation error (degrees)
- Number of detected tags
- Average distance to tags
- Processing frame rate

#### Example Output

```
Time: 45.23s | Nodes: 3 | Avg Dist: 1.85m
Translation Error: 12.4mm | Rotation Error: 2.1°
Estimated Pose: X: 0.123, Y: -0.045, Z: 1.234
Ground Truth:   X: 0.135, Y: -0.052, Z: 1.221
```

### `ground_truth(self, tag_id=0)`

Calculates the true camera pose in the tag's coordinate frame.

#### Parameters

- **tag_id** (int, optional): ID of the tag to use as reference frame. Default: 0

#### Returns

- **transformation_matrix** (numpy.ndarray): 4x4 camera-to-tag transformation matrix

#### Algorithm

1. **Position Calculation**: Adjusts tag position relative to camera
2. **Coordinate Conversion**: Applies OpenGL to computer vision coordinate transforms
3. **Rotation Matrix Creation**: Builds rotation matrix from Euler angles
4. **Transformation Assembly**: Combines rotation and translation into 4x4 matrix

### `world_ground_truth(self, tag_id=0)`

Calculates the true camera pose in the world coordinate frame.

#### Parameters

- **tag_id** (int, optional): Tag ID for reference. Default: 0

#### Returns

- **transformation_matrix** (numpy.ndarray): 4x4 world-to-camera transformation matrix

### `mm_conversion(self, value)`

Converts simulation units to millimeters for display.

#### Parameters

- **value** (float): Value in simulation units

#### Returns

- **value_mm** (float): Value converted to millimeters

### `monte_carlo_position_randomizer(self, bounds)`

Randomizes camera position within specified bounds for automated testing.

#### Parameters

- **bounds** (numpy.ndarray): Array of [x_min, x_max, y_min, y_max, z_min, z_max]

## Input Controls

### Keyboard Mapping

| Key | Action | Description |
|-----|--------|-------------|
| ← → | Strafe | Move camera left/right |
| ↑ ↓ | Forward/Back | Move camera up/down |
| W S | Up/Down | Move camera forward/backward |
| A D | Yaw | Rotate camera left/right |
| Q E | Roll | Roll camera left/right |
| R F | Pitch | Pitch camera up/down |
| ESC | Quit | Exit simulation |

### Movement Parameters

- **Movement Speed**: Configurable translation speed
- **Rotation Speed**: Configurable rotation speed (degrees per frame)
- **Bounds Checking**: Optional movement limits for testing

## Data Logging

### CSV Output Files

The simulation generates three CSV files in `data/csv/`:

#### 1. error_data.csv
Main performance data with pose comparisons:
```csv
Time,Number of Nodes,Avrg Distance,Est_X,Est_Y,Est_Z,Est_Roll,Est_Pitch,Est_Yaw,GT_X,GT_Y,GT_Z,GT_Roll,GT_Pitch,GT_Yaw,Translation Difference,Rotation Difference
```

#### 2. error_params.csv
Detailed error analysis with coordinate breakdowns:
```csv
Number of Jumps,Est_X_Local,Est_Y_Local,Est_Z_Local,Est_Roll_Local,Est_Pitch_Local,Est_Yaw_Local,Est_X_World,Est_Y_World,Est_Z_World,Est_Roll_World,Est_Pitch_World,Est_Yaw_World,Tag_Est_X,Tag_Est_Y,Tag_Est_Z,Tag_Est_Roll,Tag_Est_Pitch,Tag_Est_Yaw,Error_World,Error_Local
```

#### 3. covariance_data.csv
Covariance and uncertainty analysis:
```csv
Number of Jumps,Tag_Est_X,Tag_Est_Y,Tag_Est_Z,Tag_Est_Roll,Tag_Est_Pitch,Tag_Est_Yaw,Translation_Error
```

## Integration Examples

### Basic Simulation Run

```python
from src.simulation.sim import Simulation

# Load configuration and run simulation
simulation = Simulation("config/sim_settings.json")
simulation.run()

# Data will be automatically logged to data/csv/
```

### Automated Testing

```python
# Run simulation without manual movement
simulation = Simulation("config/sim_settings.json", movement_flag=False)

# Use Monte Carlo positioning
bounds = np.array([-2, 2, -1, 1, 0.5, 3.0])  # [x_min, x_max, y_min, y_max, z_min, z_max]

# Run multiple test positions
for i in range(100):
    simulation.monte_carlo_position_randomizer(bounds)
    # Process several frames at this position
    for frame in range(10):
        simulation.run_single_frame()
```

### Custom Configuration

```python
# Custom simulation settings
custom_config = {
    "display_width": 1280,
    "display_height": 720,
    "fov_y": 45.0,
    "tag_size_inner": 0.04,  # Smaller tags for precision testing
    "tags": [
        {"id": 0, "image": "assets/tags/tag0.png", "position": [0.5, 0.0, 1.0], "rotation": [0, 0, 0]},
        {"id": 1, "image": "assets/tags/tag1.png", "position": [-0.5, 0.0, 1.0], "rotation": [0, 45, 0]},
        {"id": 2, "image": "assets/tags/tag2.png", "position": [0.0, 0.5, 1.5], "rotation": [0, 0, 30]}
    ]
}

# Save custom config and use
with open("custom_config.json", "w") as f:
    json.dump(custom_config, f, indent=2)

simulation = Simulation("custom_config.json")
simulation.run()
```

## Performance Considerations

### Rendering Performance

- **OpenGL Optimization**: Efficient texture management and rendering pipeline
- **Frame Rate Control**: Configurable frame rate limiting
- **Resolution Scaling**: Adjustable display resolution for performance/quality balance

### SLAM Performance

- **Real-time Processing**: Optimized for live tag detection and pose estimation
- **Memory Management**: Efficient handling of large datasets
- **Error Calculation**: Fast ground truth comparison algorithms

### Data Logging Efficiency

- **Buffered Writing**: Optimized CSV output for minimal performance impact
- **Selective Logging**: Configurable data collection intervals
- **Memory Usage**: Streaming data processing without accumulation

## Error Handling

### Common Issues

#### Configuration Errors
```python
# Missing required parameters
FileNotFoundError: "Configuration file not found"
KeyError: "Required parameter 'tags' missing from configuration"
```

#### OpenGL Issues
```python
# Graphics initialization problems
RuntimeError: "Failed to initialize OpenGL context"
TextureError: "Failed to load texture file"
```

#### SLAM Integration Issues
```python
# Camera parameter calculation problems
ValueError: "Invalid camera matrix dimensions"
SLAMError: "Failed to initialize SLAM system"
```

### Debugging Tools

#### Logging Output
The simulation provides detailed logging:
```
INFO - Simulation started
INFO - Loaded texture assets/tags/tag0.png with width: 256, height: 256
INFO - SLAM system initialized with camera matrix: [[400, 0, 320], [0, 400, 240], [0, 0, 1]]
INFO - Frame 1250: 3 tags detected, translation error: 15.2mm
```

#### Performance Monitoring
Real-time display of key metrics helps identify issues:
- Detection failure rates
- Pose estimation accuracy
- Processing bottlenecks

## Research Applications

### Algorithm Evaluation

- **Controlled Testing**: Precise ground truth for algorithm validation
- **Parameter Sensitivity**: Easy configuration changes for parameter studies
- **Statistical Analysis**: Comprehensive data collection for statistical evaluation

### Dataset Generation

- **Synthetic Datasets**: Generate large amounts of tagged training data
- **Noise Models**: Add realistic sensor noise and disturbances
- **Scenario Testing**: Test specific challenging scenarios repeatably

## Related Components

- [`SimulationEngine`](SIMULATION_ENGINE.md): Core engine that orchestrates the simulation
- [`CameraController`](CAMERA_CONTROLLER.md): Handles camera movement and input
- [`SimulationRenderer`](RENDERER.md): OpenGL rendering system
- [`DataLogger`](DATA_LOGGER.md): Data collection and CSV output
- [`SLAM`](../core/SLAM.md): SLAM system integration