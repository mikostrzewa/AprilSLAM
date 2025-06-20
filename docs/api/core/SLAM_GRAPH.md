# SLAMGraph and Node Documentation

## Overview

The `SLAMGraph` class manages the spatial relationships between detected AprilTags in a graph-based SLAM system. It maintains coordinate transformations, handles reference frame management, and provides pose estimation capabilities. The `Node` class represents individual tags in the graph structure.

## Class: Node

Located in: `src/core/slam_graph.py`

### Purpose

The Node class represents a single AprilTag in the SLAM graph, storing its spatial relationships and metadata.

### Constructor

#### `__init__(self, local, world, reference, weight=1, updated=True, visible=False)`

#### Parameters

- **local** (numpy.ndarray): 4x4 transformation matrix from camera to tag
- **world** (numpy.ndarray): 4x4 transformation matrix from world origin to tag
- **reference** (int): ID of the reference tag used for world coordinate calculation
- **weight** (int, optional): Reference chain length (distance from world origin). Default: 1
- **updated** (bool, optional): Flag indicating if the node has been recently updated. Default: True
- **visible** (bool, optional): Flag indicating if the tag is currently visible. Default: False

### Attributes and Coordinate Frames

- **local** (numpy.ndarray): **Camera-to-Tag** transformation matrix `T_camera_to_tag`
  - Transforms points from camera frame to tag frame
  - Updated each time the tag is detected
  - Origin: Camera optical center → Tag center
  
- **world** (numpy.ndarray): **World-to-Tag** transformation matrix `T_world_to_tag`
  - Transforms points from world frame to tag frame  
  - Fixed once calculated (unless world frame changes)
  - Origin: World coordinate origin → Tag center
  
- **reference** (int): Reference tag ID used for world coordinate calculation
  - Links this tag to the coordinate frame chain
  - Lower values indicate closer to world origin
  
- **weight** (int): Reference chain length (distance from world origin)
  - Inverse confidence measure: lower weight = higher confidence
  - Weight of 1 = direct reference to world origin
  
- **updated** (bool): Flag indicating if node was recently updated
- **visible** (bool): Flag indicating if tag is currently visible to camera

## Class: SLAMGraph

Located in: `src/core/slam_graph.py`

### Purpose

The SLAMGraph class handles the graph data structure for SLAM operations including:
- Managing spatial relationships between tags
- Coordinate frame transformations and reference management
- Node storage and graph structure maintenance
- Dynamic world coordinate frame selection

**Note**: Pose estimation and analysis logic has been moved to the SLAM class for better separation of concerns.

### Dependencies

```python
import numpy as np
```

## Constructor

### `__init__(self, logger)`

Initializes an empty SLAM graph with coordinate tracking.

#### Parameters

- **logger**: Logging interface for debugging and monitoring

#### Initialized Attributes

- **graph** (dict): Dictionary storing Node objects keyed by tag ID
- **visible_tags** (list): List of currently visible tag IDs
- **coordinate_id** (int): ID of tag serving as world coordinate origin (-1 if none)
- **estimated_pose** (numpy.ndarray): Current camera pose estimate (4x4 matrix)

## Core Methods

### `invert(self, T)`

Inverts a 4x4 transformation matrix using numpy linear algebra.

#### Parameters
- **T** (numpy.ndarray): 4x4 transformation matrix

#### Returns
- **T_inv** (numpy.ndarray): Inverted transformation matrix

### `add_or_update_node(self, tag_id, T, visible_tags)`

Central method for adding new tags or updating existing ones in the graph.

#### Parameters
- **tag_id** (int): ID of the detected tag
- **T** (numpy.ndarray): **Camera-to-Tag** transformation matrix `T_camera_to_tag`
- **visible_tags** (list): List of currently visible tag IDs

#### Coordinate Frame Processing
**Input**: `T_camera_to_tag` from TagDetector.get_pose()
**Storage**: 
- `node.local = T_camera_to_tag` (direct storage)
- `node.world = T_world_to_tag` (calculated via reference chain)

#### Algorithm Logic

1. **First Tag Detection**: If no coordinate frame exists, sets the first tag as world origin
2. **Lower ID Priority**: If a tag with lower ID is detected, switches world coordinate frame
3. **Reference-based Addition**: Uses visible tags as references for new detections
4. **Update Existing**: Updates local coordinates while preserving world coordinates

#### Reference Selection Priority
- Primary: Uses coordinate frame tag if visible
- Secondary: Uses lowest ID visible tag as reference
- Fallback: Chains through existing references

### `find_world(self, reference, T)`

Calculates world coordinates through reference chain transformations.

#### Parameters
- **reference** (int): Reference tag ID
- **T** (numpy.ndarray): **Camera-to-Tag** transformation `T_camera_to_tag`

#### Returns
- **world** (numpy.ndarray): **World-to-Tag** transformation matrix `T_world_to_tag`
- **weight** (int): Updated weight (reference chain length + 1)
- **new_reference** (int): Ultimate reference in the chain

#### Coordinate Frame Calculation
Traces through reference chain to compute world coordinates:
```
T_world_to_tag = T_world_to_ref @ T_ref_to_camera @ T_camera_to_tag
```

### `get_world(self, reference, T)`

Computes world transformation using reference tag's local coordinates.

#### Parameters
- **reference** (int): Reference tag ID  
- **T** (numpy.ndarray): **Camera-to-Tag** transformation `T_camera_to_tag`

#### Returns
- **world_transform** (numpy.ndarray): **World-to-Tag** transformation `T_world_to_tag`

#### Coordinate Frame Formula
```
T_world_to_tag = T_camera_to_ref @ T_camera_to_tag
```
Where `T_camera_to_ref = reference.local` (stored camera-to-reference transformation)

### `update_world(self)`

Placeholder method for global graph optimization (TODO: Implementation needed).

Currently prints "No world update" and does nothing.

## Coordinate Frame Management

### Reference Frame Hierarchy

The SLAMGraph maintains a hierarchical coordinate system:

1. **World Frame**: Established by the lowest-ID detected tag
2. **Reference Frames**: Intermediate coordinate frames for tag chains
3. **Tag Frames**: Individual tag coordinate systems
4. **Camera Frame**: Moving reference frame (estimated by SLAM class)

### Transformation Storage

Each node stores two critical transformations:

#### `node.local` - Camera-to-Tag Transformation
- **Purpose**: Direct observation from camera to tag
- **Update**: Refreshed every time tag is detected
- **Usage**: `point_in_tag = node.local @ point_in_camera`

#### `node.world` - World-to-Tag Transformation  
- **Purpose**: Fixed position of tag in world coordinates
- **Update**: Calculated once, then fixed (unless world frame changes)
- **Usage**: `point_in_tag = node.world @ point_in_world`

### Reference Chain Processing

When a new tag is detected, world coordinates are calculated via reference chain:

```
Tag A (World Origin) ← Reference chain ← Camera ← New Tag B
```

**Mathematical relationship**:
```
T_world_to_tagB = T_world_to_camera @ T_camera_to_tagB
```

Where `T_world_to_camera` is derived from visible reference tags.

## Data Structure Management

The SLAMGraph focuses on maintaining the graph structure and coordinate relationships. Pose estimation and analysis algorithms are handled by the SLAM class.

## Graph Access Methods

### `get_nodes(self)`

Returns the complete graph structure.

#### Returns
- **graph** (dict): Dictionary of all nodes keyed by tag ID

### `get_coordinate_id(self)`

Returns the current world coordinate frame tag ID.

#### Returns
- **coordinate_id** (int): Tag ID serving as world origin, or -1 if none

### `get_estimated_pose(self)`

Returns the most recent camera pose estimate.

#### Returns
- **estimated_pose** (numpy.ndarray): 4x4 camera pose transformation matrix

## Usage Example

```python
import numpy as np
import logging
from src.core.slam_graph import SLAMGraph

# Initialize
logger = logging.getLogger(__name__)
graph = SLAMGraph(logger)

# Simulate tag detections
visible_tags = [1, 2, 3]

# First detection - establishes coordinate frame
T1 = np.array([[1, 0, 0, 0.5],
               [0, 1, 0, 0.0],
               [0, 0, 1, 1.0],
               [0, 0, 0, 1.0]])
graph.add_or_update_node(1, T1, [1])

# Second detection - uses tag 1 as reference
T2 = np.array([[1, 0, 0, -0.5],
               [0, 1, 0, 0.0],
               [0, 0, 1, 1.2],
               [0, 0, 0, 1.0]])
graph.add_or_update_node(2, T2, [1, 2])

# Get current pose estimate
pose = graph.my_pose()
if pose is not None:
    print(f"Camera position: {pose[:3, 3]}")
    print(f"Average distance to tags: {graph.average_distance_to_nodes():.2f}")

# Access graph structure
nodes = graph.get_nodes()
for tag_id, node in nodes.items():
    print(f"Tag {tag_id}: weight={node.weight}, visible={node.visible}")
```

## Coordinate Frame Management

### Dynamic World Frame Selection

The graph automatically selects the world coordinate frame based on tag IDs:

1. **Lowest ID Priority**: Tag with lowest ID becomes world origin
2. **Automatic Switching**: If a lower ID tag is detected, coordinate frame changes
3. **Reference Chain Updates**: All transformations recalculated when frame switches

### Reference Chain Algorithm

```
Camera -> Tag_i -> Tag_ref -> ... -> Tag_world_origin
```

- Each tag stores its reference in the chain
- Weights increase with chain length
- Shorter chains have higher confidence

### Transformation Matrices

- **Local (T_local)**: Camera-to-tag transformation
- **World (T_world)**: World-origin-to-tag transformation  
- **Camera Pose**: Derived from visible tags using weighted average

## Performance Considerations

### Computational Complexity
- Node addition: O(1) for new tags, O(n) for coordinate frame switches
- Pose estimation: O(k) where k is number of visible tags
- Memory usage: O(n) where n is total number of detected tags

### Optimization Opportunities
- **Bundle Adjustment**: Global optimization of all poses (TODO)
- **Loop Closure**: Detection and correction of accumulated drift
- **Covariance Tracking**: Uncertainty quantification for poses

## Error Handling

### Robust Operation
- Handles missing reference tags gracefully
- Continues operation when no tags are visible
- Maintains state during temporary detection failures

### Logging Integration
- Logs world coordinate updates and transformations
- Provides debugging information for reference chain calculations
- Tracks translation vector magnitudes for analysis

## TODO Items

1. **Covariance Matrix Integration**: Add uncertainty quantification to graph nodes
2. **World Update Implementation**: Implement global optimization for `update_world()`
3. **Loop Closure Detection**: Add capability to detect and correct loop closures
4. **Reference Chain Optimization**: Implement algorithms to minimize reference chain lengths

## Integration with SLAM System

The SLAMGraph integrates with the broader SLAM system by:
- Receiving tag detections from SLAM class
- Providing pose estimates for localization
- Supporting visualization through node access methods
- Maintaining spatial consistency across observations

## Related Classes

- [`SLAM`](SLAM.md): Main SLAM interface that uses SLAMGraph
- [`TagDetector`](../detection/TAG_DETECTOR.md): Provides transformation matrices
- [`SLAMVisualizer`](SLAM_VISUALIZER.md): Visualizes graph structure and poses