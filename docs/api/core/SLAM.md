# SLAM Class Documentation

## Overview

The `SLAM` class is the main interface for Simultaneous Localization and Mapping operations in the AprilSLAM system. It coordinates tag detection, pose estimation, graph management, and visualization components to provide a complete SLAM solution.

## Class: SLAM

Located in: `src/core/slam.py`

### Purpose

The SLAM class serves as the central coordinator that:
- Manages AprilTag detection using TagDetector
- Maintains a graph-based representation of spatial relationships
- Provides pose estimation and localization
- Offers visualization capabilities for debugging and analysis
- Integrates all SLAM components into a unified interface

### Dependencies

```python
import numpy as np
from src.detection.tag_detector import TagDetector
from src.core.slam_graph import SLAMGraph
from src.core.slam_visualizer import SLAMVisualizer
```

## Constructor

### `__init__(self, logger, camera_params, tag_type="tagStandard41h12", tag_size=0.06)`

Initializes the SLAM system with all required components.

#### Parameters

- **logger**: Logging interface for debugging and monitoring
- **camera_params** (dict): Camera calibration parameters
  - `camera_matrix`: 3x3 intrinsic camera matrix
  - `dist_coeffs`: Camera distortion coefficients
- **tag_type** (str, optional): AprilTag family type. Default: `"tagStandard41h12"`
- **tag_size** (float, optional): Physical tag size in meters. Default: `0.06`

#### Components Initialized

- **detector**: TagDetector instance for AprilTag detection
- **graph**: SLAMGraph instance for spatial relationship management
- **visualizer**: SLAMVisualizer instance for 3D visualization
- **visible_tags**: List tracking currently visible tag IDs

## Methods

### `detect(self, image)`

Detects AprilTags in the input image and updates visible tags list.

#### Parameters
- **image** (numpy.ndarray): Input BGR image

#### Returns
- **detections** (list): List of detected AprilTags with metadata

#### Side Effects
- Updates `self.visible_tags` with currently detected tag IDs

### `get_pose(self, detection)`

Estimates pose for a detected tag and updates the SLAM graph.

#### Parameters
- **detection** (dict): Detection result from `detect()` method

#### Returns
- **retval** (bool): Success flag from pose estimation
- **rvec** (numpy.ndarray): Rotation vector
- **tvec** (numpy.ndarray): Translation vector

#### Side Effects
- Adds or updates node in SLAM graph if pose estimation succeeds
- Updates spatial relationships between tags

### `my_pose(self)`

Calculates the current camera pose estimate based on visible tags using weighted averaging.

#### Returns
- **pose** (numpy.ndarray or None): 4x4 transformation matrix `T_world_to_camera`, or None if no valid pose can be calculated

#### Coordinate Frame Information
**CRITICAL**: Returns **World-to-Camera** transformation matrix.

**Usage**: `point_in_camera_frame = T_world_to_camera @ point_in_world_frame`

**Matrix Structure**:
```
T_world_to_camera = [[R11, R12, R13, tx],
                     [R21, R22, R23, ty], 
                     [R31, R32, R33, tz],
                     [0,   0,   0,   1 ]]
```
Where `[tx, ty, tz]` is the camera position in world coordinates.

#### Algorithm
- Weighted average of poses from all visible tags
- Uses inverse weights based on reference chain length
- Updates node visibility status
- Returns None if no visible tags are available

### `average_distance_to_nodes(self)`

Calculates the average distance from camera to all detected tags for analysis purposes.

#### Returns
- **distance** (float): Average Euclidean distance to all nodes in the graph

## Direct Component Access

For methods that simply pass through to underlying components, direct access is recommended:

### Tag Detection and Visualization
```python
# For drawing detection visualization
frame = slam.detector.draw(rvec, tvec, corners, frame, tag_id)
```

### Pose Estimation and Analysis

```python
# For getting current pose estimate (SLAM algorithm logic)
current_pose = slam.my_pose()

# For calculating average distance to nodes (analysis logic)
avg_distance = slam.average_distance_to_nodes()
```

### Graph Data Access
```python
# For accessing graph structure (data only)
nodes = slam.graph.get_nodes()
coordinate_id = slam.graph.get_coordinate_id()
```

### Properties

### `coordinate_id`

Read-only property that returns the current world coordinate frame ID.

#### Returns
- **id** (int): Tag ID serving as the world coordinate reference frame

## Visualization Methods

### `slam_graph(self)`

Displays a 2D graph visualization showing the connectivity structure of detected tags.

#### Visualization Elements
- Nodes representing detected tags
- Edges showing reference relationships
- Color coding for tag status (visible, updated, etc.)
- Edge weights indicating reference chain lengths

### `vis_slam(self, ground_truth=None)`

Displays a 3D visualization of the SLAM state.

#### Parameters
- **ground_truth** (numpy.ndarray, optional): Ground truth pose for comparison

#### Visualization Elements
- 3D scatter plot of tag positions
- Color-coded points indicating tag status
- Optional ground truth comparison
- Current estimated camera pose

### `error_graph(self, ground_truth_graph)`

Displays an error analysis graph comparing estimated positions with ground truth.

#### Parameters
- **ground_truth_graph** (dict): Dictionary containing ground truth positions for comparison

#### Visualization Elements
- Error metrics between estimated and true positions
- Color-coded edges indicating error magnitude
- Separate local and world coordinate error visualization

## Usage Example

```python
import cv2
import logging
from src.core.slam import SLAM

# Setup logging
logger = logging.getLogger(__name__)

# Camera parameters
camera_params = {
    'camera_matrix': np.array([[800, 0, 320],
                              [0, 800, 240],
                              [0, 0, 1]], dtype=np.float32),
    'dist_coeffs': np.array([0.1, -0.2, 0, 0, 0], dtype=np.float32)
}

# Initialize SLAM
slam = SLAM(logger, camera_params, tag_size=0.06)

# Process video frames
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect tags
    detections = slam.detect(frame)
    
    # Process each detection
    for detection in detections:
        success, rvec, tvec = slam.get_pose(detection)
        
        if success:
            # Draw visualization
            corners = np.array(detection['lb-rb-rt-lt'], dtype=np.float32)
            frame = slam.detector.draw(rvec, tvec, corners, frame, detection['id'])
    
    # Get current pose estimate
    current_pose = slam.my_pose()
    if current_pose is not None:
        print(f"Current pose: {current_pose[:3, 3]}")  # Translation component
    
    # Visualize SLAM state
    slam.vis_slam()
    slam.slam_graph()
    
    # Display frame
    cv2.imshow('SLAM', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## Integration Notes

### Graph Management
- The SLAM class automatically manages the graph structure through the SLAMGraph component
- Tag poses are continuously refined as new observations become available
- The coordinate frame is dynamically selected based on tag visibility and detection order

### Coordinate Frame Conventions

#### World Coordinate Frame (SLAM Reference)
- **Definition**: Established by the lowest-ID tag initially detected
- **Origin**: Center of the reference tag
- **Axes**: Inherits the tag's coordinate system (X-right, Y-up, Z-out)
- **Usage**: All poses (camera and tags) expressed in this frame
- **Dynamic**: May change if lower-ID tags are detected later

#### Camera Coordinate Frame (Moving Frame)
- **Origin**: Camera optical center
- **Axes**: OpenCV convention (X-right, Y-down, Z-forward)
- **Representation**: 4x4 transformation matrix in world frame
- **Updates**: Continuously estimated via `my_pose()`

#### Tag Coordinate Frames (Landmark Frames)
- **Origin**: Physical center of each AprilTag
- **Axes**: X-right, Y-up, Z-out (when viewing tag)
- **Storage**: Each tag pose stored as world-to-tag transformation
- **Consistency**: All tags referenced to the same world frame

#### Transformation Chain
```
World Frame → Camera Frame → Tag Frame
    ^               ^              ^
    |               |              |
  Fixed       slam.my_pose()  detector.get_pose()
```

**Key Relationships**:
- `T_world_to_camera = slam.my_pose()`
- `T_camera_to_tag = detector.get_pose()[3]`  
- `T_world_to_tag = slam.graph.nodes[tag_id]['transformation']`

### Performance Considerations
- Real-time operation depends on tag detection frequency and complexity
- Graph updates can become computationally expensive with many tags
- Visualization should be used sparingly in production for performance

### Error Handling
- Robust handling of failed pose estimations
- Graceful degradation when no tags are visible
- Logging integration for debugging and monitoring

## TODO Items

- **World Coordinate Flexibility**: Add support for fixed world coordinate systems
- **Graph Optimization**: Implement bundle adjustment for improved accuracy
- **Covariance Integration**: Add uncertainty quantification to pose estimates
- **Loop Closure**: Implement loop closure detection and correction

## Related Classes

- [`TagDetector`](../detection/TAG_DETECTOR.md): Handles AprilTag detection and pose estimation
- [`SLAMGraph`](SLAM_GRAPH.md): Manages spatial relationships and coordinate transformations
- [`SLAMVisualizer`](SLAM_VISUALIZER.md): Provides visualization capabilities 