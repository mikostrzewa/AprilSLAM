# SLAM Architecture Overview

The SLAM system has been refactored to follow the Single Responsibility Principle, splitting the monolithic SLAM class into focused, manageable components.

## Components

### 1. `TagDetector` (`tag_detector.py`)
**Responsibility**: AprilTag detection and pose estimation

**Key Methods**:
- `detect(image)` - Detect AprilTags in an image
- `get_pose(detection)` - Estimate 6DOF pose of a detected tag
- `transformation(rvec, tvec)` - Convert rotation/translation vectors to transformation matrix
- `draw(rvec, tvec, corners, image, tag_id)` - Visualize detection results on image
- `euler_angles(rvec)` - Convert rotation vector to Euler angles

### 2. `SLAMGraph` (`slam_graph.py`)
**Responsibility**: Graph structure management and coordinate transformations

**Key Methods**:
- `add_or_update_node(tag_id, T, visible_tags)` - Add or update nodes in the graph
- `my_pose()` - Calculate current pose estimate from visible tags
- `find_world(reference, T)` - Find world coordinates through reference transformations
- `average_distance_to_nodes()` - Calculate average distance to all nodes
- `get_nodes()` - Access the graph nodes
- `get_coordinate_id()` - Get current coordinate frame ID

**Node Structure**:
- `local` - Local transformation matrix
- `world` - World transformation matrix  
- `reference` - Reference frame ID
- `weight` - Node confidence weight
- `updated` - Update status flag
- `visible` - Visibility status flag

### 3. `SLAMVisualizer` (`slam_visualizer.py`)
**Responsibility**: All visualization including 3D plots and graph visualization

**Key Methods**:
- `vis_slam(graph, estimated_pose, ground_truth)` - 3D visualization of SLAM state
- `slam_graph(graph)` - Graph structure visualization
- `error_graph(graph, ground_truth_graph)` - Error analysis visualization

### 4. `SLAM` (`slam.py`)
**Responsibility**: Main coordinator that orchestrates the components

**Key Methods**:
- `detect(image)` - Delegates to TagDetector
- `get_pose(detection)` - Delegates to TagDetector and updates SLAMGraph
- `my_pose()` - Delegates to SLAMGraph
- `vis_slam()`, `slam_graph()`, `error_graph()` - Delegates to SLAMVisualizer

## Benefits of This Architecture

1. **Single Responsibility**: Each class has one clear purpose
2. **Maintainability**: Easier to understand and modify individual components
3. **Testability**: Each component can be tested independently
4. **Reusability**: Components can be used in different contexts
5. **Extensibility**: Easy to add new features to specific components

## TODO Items

- **SLAMGraph**: Add covariance matrix to the graph for uncertainty quantification
- **SLAMGraph**: Implement the `update_world()` method for coordinate frame updates

## Usage

```python
from slam import SLAM

# Camera parameters
camera_params = {
    'camera_matrix': camera_matrix,
    'dist_coeffs': dist_coeffs
}

# Initialize SLAM system
slam = SLAM(logger, camera_params, tag_size=0.06)

# Process frame
detections = slam.detect(image)
for detection in detections:
    retval, rvec, tvec = slam.get_pose(detection)
    if retval:
        # Process successful detection
        pass

# Get current pose estimate
pose = slam.my_pose()

# Visualize
slam.vis_slam(ground_truth=ground_truth_pose)
slam.slam_graph()
``` 