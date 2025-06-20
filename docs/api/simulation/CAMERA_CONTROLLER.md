# CameraController Documentation

## Overview

The `CameraController` class handles camera movement and keyboard input for the AprilSLAM simulation. It supports both manual control via keyboard input and automatic Monte Carlo position randomization for testing purposes, providing flexible camera management for various simulation scenarios.

## Class: CameraController

Located in: `src/simulation/camera_controller.py`

### Purpose

The CameraController manages all aspects of camera positioning and movement:
- **Position Management**: Tracks and updates 3D camera position
- **Orientation Control**: Manages camera rotation in 3D space
- **Input Processing**: Handles keyboard input for manual control
- **Movement Calculation**: Computes smooth movement and rotation
- **Automated Positioning**: Supports Monte Carlo randomization for testing
- **State Tracking**: Maintains camera state and transformation matrices

### Features

- **6DOF Movement**: Full 6 degrees of freedom for camera positioning
- **Keyboard Input**: Responsive keyboard-based movement controls
- **Configurable Speed**: Adjustable movement and rotation speeds
- **Monte Carlo Support**: Random position generation for automated testing
- **Transformation Matrices**: Automatic calculation of camera transformation matrices
- **State Management**: Clean state tracking and manipulation
- **Bounds Checking**: Optional movement limits and validation

### Dependencies

```python
import numpy as np
import pygame
from typing import Dict, Tuple, Optional
import logging
```

## Constructor

### `__init__(self, movement_enabled=True, movement_speed=1.0, rotation_speed=1.0, size_scale=1.0)`

Initializes the camera controller with specified movement parameters.

#### Parameters

- **movement_enabled** (bool, optional): Whether manual movement is enabled. Default: True
- **movement_speed** (float, optional): Movement speed multiplier. Default: 1.0
- **rotation_speed** (float, optional): Rotation speed in degrees per update. Default: 1.0
- **size_scale** (float, optional): Size scaling factor for movement speed. Default: 1.0

#### Initialization

The constructor sets up:
- Camera position at origin [0, 0, 0]
- Camera rotation at [0, 0, 0] (pitch, yaw, roll)
- Movement speed scaled by size_scale
- Keyboard state tracking for all movement keys

#### Example Usage

```python
from src.simulation.camera_controller import CameraController

# Standard controller with default settings
controller = CameraController()

# High-speed controller for testing
fast_controller = CameraController(
    movement_enabled=True,
    movement_speed=2.0,
    rotation_speed=2.0,
    size_scale=1.5
)

# Automated controller (no manual movement)
auto_controller = CameraController(movement_enabled=False)
```

## Core Methods

### `handle_key_event(self, event)`

Processes pygame keyboard events to update key state tracking.

#### Parameters

- **event** (pygame.event.Event): Pygame event to process

#### Supported Events

- **KEYDOWN**: Records when movement keys are pressed
- **KEYUP**: Records when movement keys are released

#### Key Mapping

The controller tracks the following keys:
- **Translation**: Arrow keys, W/S for forward/backward
- **Rotation**: A/D for yaw, Q/E for roll, R/F for pitch

#### Example Integration

```python
# In main event loop
for event in pygame.event.get():
    if event.type in [pygame.KEYDOWN, pygame.KEYUP]:
        camera_controller.handle_key_event(event)
```

### `update_manual_movement(self)`

Updates camera position and rotation based on current key states.

#### Behavior

Only processes movement if `movement_enabled` is True. Updates are applied continuously while keys are held down.

#### Movement Mapping

| Key Group | Keys | Action | Axis |
|-----------|------|--------|------|
| Translation | ← → | Strafe left/right | X-axis |
| Translation | ↑ ↓ | Move up/down | Y-axis |
| Translation | W S | Move forward/backward | Z-axis |
| Rotation | A D | Yaw left/right | Y-rotation |
| Rotation | Q E | Roll left/right | Z-rotation |
| Rotation | R F | Pitch up/down | X-rotation |

#### Speed Calculation

- Translation speed: `movement_speed * size_scale`
- Rotation speed: `rotation_speed` degrees per update

### `randomize_position(self, bounds)`

Randomizes camera position within specified bounds for Monte Carlo testing.

#### Parameters

- **bounds** (numpy.ndarray): Array of [x_min, x_max, y_min, y_max, z_min, z_max]

#### Validation

- Verifies bounds array contains exactly 6 values
- Raises ValueError if bounds format is incorrect

#### Example Usage

```python
# Define testing bounds
bounds = np.array([-2.0, 2.0, -1.0, 1.0, 0.5, 3.0])

# Randomize position for automated testing
controller.randomize_position(bounds)

# Position is now randomly set within bounds
position = controller.get_position()
print(f"Random position: {position}")
```

## Position and Orientation Management

### `set_position(self, position)`

Sets camera position directly.

#### Parameters

- **position** (numpy.ndarray): New position as [x, y, z]

#### Validation

- Ensures position is a 3D vector
- Raises ValueError if incorrect dimensions

### `set_rotation(self, rotation)`

Sets camera rotation directly.

#### Parameters

- **rotation** (numpy.ndarray): New rotation as [pitch, yaw, roll] in degrees

#### Validation

- Ensures rotation is a 3D vector
- Raises ValueError if incorrect dimensions

### `get_position(self)`

Returns current camera position.

#### Returns

- **position** (numpy.ndarray): Current position as [x, y, z]

#### Notes

Returns a copy to prevent external modification of internal state.

### `get_rotation(self)`

Returns current camera rotation.

#### Returns

- **rotation** (numpy.ndarray): Current rotation as [pitch, yaw, roll] in degrees

#### Notes

Returns a copy to prevent external modification of internal state.

### `get_transformation_matrix(self)`

Calculates and returns the camera transformation matrix.

#### Returns

- **transform** (numpy.ndarray): 4x4 transformation matrix representing camera pose

#### Calculation Process

1. **Angle Conversion**: Converts degrees to radians
2. **Rotation Matrices**: Creates individual rotation matrices for each axis
3. **Matrix Combination**: Combines rotations in proper order (yaw * pitch * roll)
4. **Transformation Assembly**: Creates 4x4 matrix with rotation and translation

#### Matrix Structure

```
[[R11, R12, R13, tx],
 [R21, R22, R23, ty],
 [R31, R32, R33, tz],
 [0,   0,   0,   1 ]]
```

Where R is the 3x3 rotation matrix and [tx, ty, tz] is the translation vector.

## Utility Methods

### `reset(self)`

Resets camera to origin with no rotation.

#### Effects

- Position set to [0, 0, 0]
- Rotation set to [0, 0, 0]
- Logs reset action

### `enable_movement(self, enabled=True)`

Enables or disables manual movement.

#### Parameters

- **enabled** (bool, optional): Whether to enable movement. Default: True

#### Use Cases

- Disable for automated testing
- Enable/disable based on simulation mode
- Temporary movement restrictions

### `set_movement_speed(self, speed)`

Updates movement speed dynamically.

#### Parameters

- **speed** (float): New movement speed multiplier

#### Validation

- Ensures speed is positive
- Raises ValueError for invalid speeds

### `set_rotation_speed(self, speed)`

Updates rotation speed dynamically.

#### Parameters

- **speed** (float): New rotation speed in degrees per update

#### Validation

- Ensures speed is positive
- Raises ValueError for invalid speeds

## Status and Information

### `get_status_info(self)`

Returns comprehensive status information about the camera controller.

#### Returns

Dictionary containing:
- **position**: Current camera position
- **rotation**: Current camera rotation
- **movement_enabled**: Whether manual movement is active
- **movement_speed**: Current movement speed
- **rotation_speed**: Current rotation speed
- **key_states**: Current state of all tracked keys

#### Example Output

```python
status = controller.get_status_info()
# {
#     'position': [1.2, -0.5, 2.0],
#     'rotation': [10.0, 45.0, 0.0],
#     'movement_enabled': True,
#     'movement_speed': 1.5,
#     'rotation_speed': 1.0,
#     'keys_pressed': ['w', 'a']
# }
```

## Integration Examples

### Basic Camera Control

```python
from src.simulation.camera_controller import CameraController
import pygame

# Initialize controller
camera = CameraController()

# Main loop
clock = pygame.time.Clock()
running = True

while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        camera.handle_key_event(event)
    
    # Update camera based on input
    camera.update_manual_movement()
    
    # Get current camera state
    position = camera.get_position()
    rotation = camera.get_rotation()
    transform = camera.get_transformation_matrix()
    
    # Use camera data for rendering...
    
    clock.tick(60)  # 60 FPS
```

### Monte Carlo Testing

```python
# Set up automated camera testing
camera = CameraController(movement_enabled=False)

# Define test bounds
test_bounds = np.array([
    -5.0, 5.0,   # X range
    -3.0, 3.0,   # Y range
     0.5, 4.0    # Z range
])

# Run Monte Carlo simulation
for trial in range(1000):
    # Randomize camera position
    camera.randomize_position(test_bounds)
    
    # Get position for this trial
    position = camera.get_position()
    
    # Run simulation at this position
    run_simulation_trial(position)
    
    # Log results
    print(f"Trial {trial}: Position {position}")
```

### Smooth Movement Animation

```python
import time

class SmoothCameraController(CameraController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_position = np.array([0.0, 0.0, 0.0])
        self.interpolation_speed = 0.1
    
    def set_target_position(self, target):
        """Set target position for smooth movement."""
        self.target_position = np.array(target)
    
    def update_smooth_movement(self):
        """Update position with smooth interpolation."""
        if not np.allclose(self.position, self.target_position):
            direction = self.target_position - self.position
            movement = direction * self.interpolation_speed
            self.position += movement

# Usage
smooth_camera = SmoothCameraController()
smooth_camera.set_target_position([2.0, 1.0, 3.0])

# In update loop
smooth_camera.update_smooth_movement()
```

### Performance Monitoring

```python
class MonitoredCameraController(CameraController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.movement_history = []
        self.max_history = 1000
    
    def update_manual_movement(self):
        """Override to track movement history."""
        old_position = self.position.copy()
        super().update_manual_movement()
        
        # Track movement
        movement = np.linalg.norm(self.position - old_position)
        if movement > 0:
            self.movement_history.append({
                'time': time.time(),
                'movement': movement,
                'position': self.position.copy()
            })
            
            # Limit history size
            if len(self.movement_history) > self.max_history:
                self.movement_history.pop(0)
    
    def get_movement_stats(self):
        """Get movement statistics."""
        if not self.movement_history:
            return {'total_distance': 0, 'avg_speed': 0}
        
        total_distance = sum(h['movement'] for h in self.movement_history)
        time_span = self.movement_history[-1]['time'] - self.movement_history[0]['time']
        avg_speed = total_distance / max(time_span, 1e-6)
        
        return {
            'total_distance': total_distance,
            'avg_speed': avg_speed,
            'num_movements': len(self.movement_history)
        }
```

## Coordinate System

### Camera Coordinate Frame

- **X-axis**: Right direction (positive = right)
- **Y-axis**: Up direction (positive = up)
- **Z-axis**: Forward direction (positive = forward into scene)

### Rotation Conventions

- **Pitch**: Rotation about X-axis (up/down)
- **Yaw**: Rotation about Y-axis (left/right)
- **Roll**: Rotation about Z-axis (twist)

### Transformation Matrix

The transformation matrix converts from world coordinates to camera coordinates:
```
camera_point = transformation_matrix @ world_point
```

## Error Handling

### Input Validation

```python
# Position validation
try:
    camera.set_position([1, 2, 3])
except ValueError as e:
    print(f"Invalid position: {e}")

# Bounds validation
try:
    camera.randomize_position([1, 2, 3])  # Too few values
except ValueError as e:
    print(f"Invalid bounds: {e}")
```

### Robust Operation

The controller handles edge cases gracefully:
- Invalid key events are ignored
- Out-of-range rotations are handled properly
- Matrix calculations use robust numerical methods

## Related Components

- [`SimulationEngine`](SIMULATION_ENGINE.md): Uses CameraController for camera management
- [`SimulationRenderer`](RENDERER.md): Receives camera transformations for rendering
- [`SLAM`](../core/SLAM.md): Compares estimated poses with camera controller ground truth