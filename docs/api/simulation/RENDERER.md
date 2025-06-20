# Renderer Documentation

## Overview

The renderer module provides OpenGL-based rendering capabilities for the AprilSLAM simulation. It handles 3D scene rendering, texture management, tag positioning, and frame capture for SLAM processing, providing a complete visual simulation environment.

## Class: SimulationRenderer

Located in: `src/simulation/renderer.py`

### Purpose

The SimulationRenderer manages all OpenGL rendering operations:
- **3D Scene Rendering**: Creates realistic 3D environments with proper lighting and perspective
- **Texture Management**: Loads and manages AprilTag textures efficiently
- **Camera Integration**: Applies camera transformations for proper view rendering
- **Frame Capture**: Captures rendered frames for SLAM processing
- **Performance Optimization**: Efficient OpenGL operations for real-time rendering

### Features

- **OpenGL Integration**: Full OpenGL rendering pipeline with modern features
- **Texture Loading**: Automatic texture loading from image files
- **Tag Rendering**: Specialized AprilTag rendering with proper positioning
- **Depth Testing**: Proper 3D depth handling for realistic scenes
- **Frame Capture**: High-quality frame capture for computer vision processing
- **Resource Management**: Automatic cleanup and resource management

### Dependencies

```python
import os
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import cv2
import logging
from typing import List, Dict, Tuple, Any, Optional

from .config_manager import SimulationConfig
```

## Constructor

### `__init__(self, config)`

Initializes the OpenGL renderer with the specified configuration.

#### Parameters

- **config** (SimulationConfig): Simulation configuration object containing rendering parameters

#### Initialization Process

1. **Pygame Setup**: Initializes pygame and creates OpenGL window
2. **OpenGL Configuration**: Sets up projection matrices and rendering state
3. **Texture Loading**: Loads all tag textures from configuration
4. **Depth Testing**: Enables proper 3D depth testing
5. **State Validation**: Verifies OpenGL context and capabilities

#### Configuration Requirements

The config object must provide:
- Display dimensions (width, height)
- Field of view and clipping planes
- Tag definitions with images and positions
- Rendering parameters

#### Example Usage

```python
from src.simulation.renderer import SimulationRenderer
from src.simulation.config_manager import SimulationConfig

# Load configuration and create renderer
config = SimulationConfig("config/sim_settings.json")
renderer = SimulationRenderer(config)

# Renderer is now ready for frame rendering
```

## Core Rendering Methods

### `render_frame(self, camera_position, camera_rotation)`

Renders a complete frame with all tags using the specified camera pose.

#### Parameters

- **camera_position** (numpy.ndarray): Camera position as [x, y, z]
- **camera_rotation** (numpy.ndarray): Camera rotation as [pitch, yaw, roll] in degrees

#### Rendering Process

1. **Scene Clearing**: Clears color and depth buffers
2. **Camera Transform**: Applies camera transformation (view matrix)
3. **Tag Sorting**: Sorts tags by depth for proper rendering order
4. **Tag Rendering**: Renders each tag with proper transformation
5. **Buffer Swap**: Presents the rendered frame

#### Performance Considerations

- Depth sorting ensures proper transparency handling
- Efficient OpenGL state management
- Optimized texture binding operations

#### Example Usage

```python
# Set camera pose
camera_pos = np.array([1.0, 0.5, 2.0])
camera_rot = np.array([0.0, 45.0, 0.0])

# Render frame
renderer.render_frame(camera_pos, camera_rot)

# Frame is now displayed and ready for capture
```

### `capture_frame(self)`

Captures the currently rendered frame as a numpy array for SLAM processing.

#### Returns

- **frame** (numpy.ndarray): Captured frame as BGR image array (height, width, 3)

#### Capture Process

1. **OpenGL Read**: Reads pixel data from OpenGL framebuffer
2. **Format Conversion**: Converts from OpenGL RGBA to OpenCV BGR
3. **Array Reshaping**: Reshapes data into proper image dimensions
4. **Coordinate Flip**: Flips Y-axis to match computer vision conventions

#### Performance Notes

- Uses efficient OpenGL pixel operations
- Minimizes memory allocations
- Optimized for real-time capture

#### Example Usage

```python
# Render and capture frame
renderer.render_frame(camera_pos, camera_rot)
frame = renderer.capture_frame()

# Frame is ready for SLAM processing
detections = slam_system.detect(frame)
```

## Camera and Transformation Methods

### `apply_camera_transform(self, camera_position, camera_rotation)`

Applies camera transformation to the OpenGL view matrix.

#### Parameters

- **camera_position** (numpy.ndarray): Camera position [x, y, z]
- **camera_rotation** (numpy.ndarray): Camera rotation [pitch, yaw, roll] in degrees

#### Transformation Details

Applies the inverse camera transform to create the view matrix:
1. **Matrix Reset**: Loads identity matrix
2. **Rotation Application**: Applies inverse rotations (roll, pitch, yaw)
3. **Translation Application**: Applies inverse translation

#### OpenGL Operations

```glsl
glLoadIdentity()
glRotatef(-roll, 0, 0, 1)
glRotatef(-pitch, 1, 0, 0) 
glRotatef(-yaw, 0, 1, 0)
glTranslatef(-x, -y, -z)
```

## Texture Management

### `_load_texture(self, image_path)`

Loads a single texture from an image file.

#### Parameters

- **image_path** (str): Path to image file (relative or absolute)

#### Returns

- **texture_id** (int): OpenGL texture ID

#### Loading Process

1. **Path Resolution**: Resolves relative paths to absolute paths
2. **Image Loading**: Loads image using pygame
3. **Format Conversion**: Converts to OpenGL-compatible format
4. **Texture Creation**: Creates OpenGL texture with proper parameters
5. **Parameter Setting**: Sets filtering and wrapping modes

#### Supported Formats

- PNG (with alpha transparency)
- JPEG/JPG
- BMP
- Other pygame-supported formats

#### Error Handling

```python
try:
    texture_id = renderer._load_texture("assets/tags/tag0.png")
except FileNotFoundError:
    print("Texture file not found")
except Exception as e:
    print(f"Failed to load texture: {e}")
```

### `_load_tag_textures(self)`

Loads all tag textures from the configuration.

#### Process

1. **Configuration Parsing**: Reads tag definitions from config
2. **Texture Loading**: Loads each tag texture individually  
3. **TagData Creation**: Creates TagData objects for each tag
4. **Validation**: Verifies all textures loaded successfully

#### Error Recovery

- Logs individual texture loading failures
- Continues loading remaining textures
- Raises exception only if critical failures occur

## Tag Management

### `get_tag_by_id(self, tag_id)`

Retrieves tag data by ID.

#### Parameters

- **tag_id** (int): ID of the tag to retrieve

#### Returns

- **tag_data** (TagData or None): Tag data object or None if not found

### `get_all_tags(self)`

Returns all loaded tag data objects.

#### Returns

- **tags** (List[TagData]): List of all loaded tag data

### `update_tag_position(self, tag_id, position)`

Updates the position of a specific tag.

#### Parameters

- **tag_id** (int): ID of tag to update
- **position** (numpy.ndarray): New position as [x, y, z]

#### Returns

- **success** (bool): True if tag was found and updated

### `update_tag_rotation(self, tag_id, rotation)`

Updates the rotation of a specific tag.

#### Parameters

- **tag_id** (int): ID of tag to update
- **rotation** (numpy.ndarray): New rotation as [pitch, yaw, roll] in degrees

#### Returns

- **success** (bool): True if tag was found and updated

## Class: TagData

### Purpose

The TagData class represents a single AprilTag with its associated rendering data.

### Constructor

### `__init__(self, tag_id, texture, position, rotation)`

Creates a new TagData object.

#### Parameters

- **tag_id** (int): Unique tag identifier
- **texture** (int): OpenGL texture ID
- **position** (numpy.ndarray): 3D position vector
- **rotation** (numpy.ndarray): 3D rotation vector in degrees

### Methods

### `get_world_z(self)`

Returns the world Z coordinate for depth sorting.

#### Returns

- **z_position** (float): Z coordinate in world space

#### Usage

Used internally by the renderer for proper depth sorting of tags.

## Advanced Rendering Features

### Depth Testing and Sorting

```python
# Tags are automatically sorted by depth for proper rendering
def _render_frame_with_depth_sorting(self):
    # Sort tags by Z distance for proper transparency
    sorted_tags = sorted(self.tags_data, key=lambda tag: tag.get_world_z())
    
    # Render from back to front
    for tag in sorted_tags:
        self._render_tag(tag)
```

### Performance Optimization

```python
# Efficient texture binding
current_texture = None
for tag in self.tags_data:
    if tag.texture != current_texture:
        glBindTexture(GL_TEXTURE_2D, tag.texture)
        current_texture = tag.texture
    self._render_tag_geometry(tag)
```

### Custom Rendering Extensions

```python
class ExtendedRenderer(SimulationRenderer):
    def __init__(self, config):
        super().__init__(config)
        self.enable_shadows = True
        self.enable_lighting = True
    
    def render_frame(self, camera_position, camera_rotation):
        # Apply lighting setup
        if self.enable_lighting:
            self._setup_lighting()
        
        # Render shadows
        if self.enable_shadows:
            self._render_shadows()
        
        # Standard rendering
        super().render_frame(camera_position, camera_rotation)
    
    def _setup_lighting(self):
        """Configure OpenGL lighting."""
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        
        # Set light properties
        light_position = [2.0, 2.0, 2.0, 1.0]
        glLightfv(GL_LIGHT0, GL_POSITION, light_position)
```

## Resource Management

### `cleanup(self)`

Cleans up all OpenGL resources.

#### Process

1. **Texture Cleanup**: Deletes all loaded textures
2. **OpenGL State**: Resets OpenGL state
3. **Memory Cleanup**: Frees associated memory
4. **Logging**: Records cleanup completion

#### Usage

```python
# Always clean up resources
try:
    renderer = SimulationRenderer(config)
    # ... use renderer ...
finally:
    renderer.cleanup()

# Or use context manager pattern
class RendererContext(SimulationRenderer):
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
```

## Integration Examples

### Basic Rendering Loop

```python
from src.simulation.renderer import SimulationRenderer
from src.simulation.config_manager import SimulationConfig

# Initialize renderer
config = SimulationConfig("config/sim_settings.json")
renderer = SimulationRenderer(config)

# Rendering loop
camera_pos = np.array([0.0, 0.0, 3.0])
camera_rot = np.array([0.0, 0.0, 0.0])

try:
    while running:
        # Update camera position
        update_camera_position(camera_pos, camera_rot)
        
        # Render frame
        renderer.render_frame(camera_pos, camera_rot)
        
        # Capture for processing
        frame = renderer.capture_frame()
        
        # Process with SLAM
        process_slam_frame(frame)
        
finally:
    renderer.cleanup()
```

### Multi-Tag Scene Setup

```python
# Configure multiple tags
config_data = {
    "display_width": 800,
    "display_height": 600,
    "fov_y": 60.0,
    "tags": [
        {
            "id": 0,
            "image": "assets/tags/tag0.png", 
            "position": [1.0, 0.0, 2.0],
            "rotation": [0.0, 0.0, 0.0]
        },
        {
            "id": 1,
            "image": "assets/tags/tag1.png",
            "position": [-1.0, 1.0, 2.0],
            "rotation": [0.0, 45.0, 0.0]
        },
        {
            "id": 2,
            "image": "assets/tags/tag2.png",
            "position": [0.0, -1.0, 3.0],
            "rotation": [30.0, 0.0, 0.0]
        }
    ]
}

config = SimulationConfig(config_data)
renderer = SimulationRenderer(config)

# All tags will be rendered automatically
renderer.render_frame(camera_pos, camera_rot)
```

### Performance Monitoring

```python
import time

class PerformanceRenderer(SimulationRenderer):
    def __init__(self, config):
        super().__init__(config)
        self.frame_times = []
        self.capture_times = []
    
    def render_frame(self, camera_position, camera_rotation):
        start_time = time.time()
        super().render_frame(camera_position, camera_rotation)
        render_time = time.time() - start_time
        self.frame_times.append(render_time)
    
    def capture_frame(self):
        start_time = time.time()
        frame = super().capture_frame()
        capture_time = time.time() - start_time
        self.capture_times.append(capture_time)
        return frame
    
    def get_performance_stats(self):
        return {
            'avg_render_time': np.mean(self.frame_times),
            'avg_capture_time': np.mean(self.capture_times),
            'total_frames': len(self.frame_times)
        }
```

## Error Handling

### Common Issues

#### OpenGL Context Problems

```python
try:
    renderer = SimulationRenderer(config)
except Exception as e:
    if "OpenGL" in str(e):
        print("OpenGL initialization failed - check graphics drivers")
    else:
        print(f"Renderer initialization error: {e}")
```

#### Texture Loading Issues

```python
# Handle missing texture files
try:
    renderer._load_tag_textures()
except FileNotFoundError as e:
    print(f"Texture file not found: {e}")
    # Use default/fallback textures
except Exception as e:
    print(f"Texture loading failed: {e}")
```

#### Resource Cleanup

```python
# Ensure proper cleanup even on errors
renderer = None
try:
    renderer = SimulationRenderer(config)
    # ... rendering operations ...
except Exception as e:
    logging.error(f"Rendering error: {e}")
finally:
    if renderer:
        renderer.cleanup()
```

## OpenGL State Management

### Rendering State

The renderer manages several important OpenGL states:

- **Depth Testing**: Enabled for proper 3D rendering
- **Texture Mapping**: Enabled for tag texture rendering
- **Blending**: Configured for transparency support
- **Viewport**: Set to match display dimensions
- **Projection Matrix**: Configured for perspective projection

### State Restoration

```python
def render_with_custom_state(self):
    # Save current state
    glPushAttrib(GL_ALL_ATTRIB_BITS)
    
    try:
        # Apply custom rendering state
        self._apply_custom_state()
        
        # Render with custom state
        self._render_custom_content()
        
    finally:
        # Restore previous state
        glPopAttrib()
```

## Related Components

- [`SimulationEngine`](SIMULATION_ENGINE.md): Uses renderer for frame generation
- [`CameraController`](CAMERA_CONTROLLER.md): Provides camera poses for rendering
- [`TagDetector`](../detection/TAG_DETECTOR.md): Processes rendered frames
- [`DataLogger`](DATA_LOGGER.md): Logs rendering performance metrics