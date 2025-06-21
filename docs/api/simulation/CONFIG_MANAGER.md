# SimulationConfig (ConfigManager) Documentation

## Overview

The `SimulationConfig` class handles loading, validation, and management of simulation configuration settings from JSON files.

## Class: SimulationConfig

Located in: `src/simulation/config_manager.py`

### Purpose

- Loading configuration settings from JSON files
- Validating all configuration parameters
- Providing typed property access to configuration values
- Converting between unit systems (mm to simulation units)
- Managing tag configurations and metadata

### Dependencies

```python
import json
import os
import logging
from typing import Dict, List, Any, Tuple
```

## Constructor

### `__init__(self, config_file: str)`

Initializes the configuration manager and loads settings from JSON file.

#### Parameters
- **config_file** (str): Path to JSON configuration file

#### Raises
- **FileNotFoundError**: If configuration file doesn't exist
- **ValueError**: If configuration is invalid
- **json.JSONDecodeError**: If JSON file is malformed

## Configuration File Format

### Required JSON Structure

```json
{
    "display_width": 1000,
    "display_height": 1000,
    "fov_y": 45,
    "near_clip": 0.1,
    "far_clip": 300.0,
    "size_scale": 2,
    "tag_size_inner": 5,
    "tag_size_outer": 9,
    "actual_size_in_mm": 55.6,
    "tags": [
        {
            "id": 0,
            "image": "tags/tag0.png",
            "position": [0, 0, -50],
            "rotation": [0, 0, 0]
        }
    ]
}
```

## Properties

### Display Configuration

#### `display_width` → int
Display width in pixels.

#### `display_height` → int  
Display height in pixels.

#### `display_size` → Tuple[int, int]
Display dimensions as (width, height).

#### `aspect_ratio` → float
Calculated aspect ratio (width/height).

### Camera Configuration

#### `fov_y` → float
Vertical field of view in degrees.

#### `near_clip` → float
Near clipping plane distance.

#### `far_clip` → float
Far clipping plane distance.

### Tag Configuration

#### `size_scale` → float
Global size scaling factor.

#### `tag_size_inner` → float
Inner tag size (detection area) scaled by size_scale.

#### `tag_size_outer` → float
Outer tag size (visual border) scaled by size_scale.

#### `actual_tag_size_mm` → float
Physical tag size in millimeters.

#### `tags` → List[Dict[str, Any]]
Complete list of tag configurations.

## Methods

### Tag Management

#### `get_tag_by_id(self, tag_id: int) → Dict[str, Any]`

Retrieves tag configuration by ID.

**Parameters:**
- **tag_id** (int): Tag ID to search for

**Returns:**
- Tag configuration dictionary

**Raises:**
- **ValueError**: If tag ID not found

#### `get_tag_count(self) → int`

Returns total number of configured tags.

### Unit Conversion

#### `mm_to_simulation_units(self, value_mm: float) → float`

Converts millimeters to simulation units.

**Formula:**
```
simulation_units = value_mm * tag_size_inner / actual_tag_size_mm
```

#### `simulation_units_to_mm(self, value_sim: float) → float`

Converts simulation units to millimeters.

**Formula:**
```
millimeters = value_sim * actual_tag_size_mm / tag_size_inner
```

## Validation

### Automatic Validation

The configuration manager validates:

- Display dimensions must be positive
- FOV must be between 0 and 180 degrees  
- Near clip must be less than far clip
- Tags array must be non-empty
- Each tag must have required parameters (id, image, position, rotation)

## Usage Examples

### Basic Loading

```python
from src.simulation.config_manager import SimulationConfig

# Load configuration
config = SimulationConfig('config/sim_settings.json')

# Access properties
print(f"Display: {config.display_width}x{config.display_height}")
print(f"FOV: {config.fov_y}°")
print(f"Tags: {config.get_tag_count()}")
```

### Working with Tags

```python
# Get specific tag
tag = config.get_tag_by_id(0)
print(f"Tag 0 position: {tag['position']}")

# Iterate all tags
for tag in config.tags:
    print(f"Tag {tag['id']}: {tag['position']}")
```

### Unit Conversion

```python
# Convert units
sim_units = config.mm_to_simulation_units(100.0)
mm = config.simulation_units_to_mm(10.0)
```

## Error Handling

```python
try:
    config = SimulationConfig('config.json')
except FileNotFoundError:
    print("Configuration file not found")
except ValueError as e:
    print(f"Invalid configuration: {e}")
```

## Integration Notes

- Provides parameters for OpenGL setup
- Tag configurations drive 3D scene construction
- Unit conversion ensures consistent scaling
- Validation prevents runtime errors

---

*Part of AprilSLAM Documentation Suite* 