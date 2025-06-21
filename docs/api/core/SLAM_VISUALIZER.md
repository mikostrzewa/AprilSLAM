# SLAMVisualizer Documentation

## Overview

The `SLAMVisualizer` class provides comprehensive visualization capabilities for the AprilSLAM system. It creates interactive 3D plots for spatial visualization, 2D graph plots for connectivity analysis, and error analysis plots for performance evaluation.

## Class: SLAMVisualizer

Located in: `src/core/slam_visualizer.py`

### Purpose

The SLAMVisualizer class handles all visualization aspects of SLAM including:
- 3D visualization of tag positions and camera poses
- 2D graph visualization showing tag connectivity and references
- Error analysis plots comparing estimated positions with ground truth
- Real-time interactive plotting with matplotlib
- Color-coded status indicators for different tag states

### Dependencies

```python
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
```

## Constructor

### `__init__(self)`

Initializes the visualizer with interactive matplotlib figures and axes.

#### Initialized Components

- **Interactive Mode**: Enables `plt.ion()` for real-time updates
- **Figure Management**: Creates three separate figure windows:
  - `fig_vis`: 3D visualization window
  - `fig_graph`: SLAM graph structure window  
  - `fig_err`: Error analysis window
- **Axes Setup**: Configures appropriate axes for each visualization type

#### Figure Configuration

```python
self.fig_vis = plt.figure("3D Visualization")
self.fig_graph = plt.figure("SLAM Graph")
self.err_graph = plt.figure("Error Graph")
self.ax_vis = self.fig_vis.add_subplot(111, projection='3d')
self.ax_graph = self.fig_graph.add_subplot(111)
self.ax_err = self.err_graph.add_subplot(111)
```

## Visualization Methods

### `vis_slam(self, graph, estimated_pose, ground_truth=None)`

Creates a 3D scatter plot visualization of the SLAM state showing tag positions, camera pose, and optional ground truth.

#### Coordinate Frame
**CRITICAL**: All visualized positions are displayed in **World Coordinates**.

- **Tag Positions**: Extracted from `node.world` transformations (World-to-Tag)
- **Camera Pose**: Estimated position in world frame from `estimated_pose`
- **Ground Truth**: Reference camera position in world frame (if provided)
- **Axes**: X-right, Y-up, Z-out (following world coordinate convention)

#### Parameters

- **graph** (dict): Dictionary of Node objects from SLAMGraph
- **estimated_pose** (numpy.ndarray): 4x4 **World-to-Camera** transformation matrix
- **ground_truth** (numpy.ndarray, optional): Ground truth **World-to-Camera** pose for comparison

#### Visualization Elements

1. **Tag Positions**: 3D points representing detected tags in world coordinates
2. **Color Coding**: Status-based coloring system:
   - **Red**: Not visible tags
   - **Orange**: Not updated tags
   - **Green**: Updated and visible tags
   - **Blue**: Ground truth position (if provided)
   - **Purple**: Estimated camera pose

3. **Dynamic Scaling**: Automatic axis limits based on data extent
4. **Interactive Legend**: Color-coded legend explaining point meanings

#### Algorithm

1. **Data Extraction**: Extracts world positions from graph nodes
2. **Status Analysis**: Determines color for each tag based on visibility and update status
3. **Ground Truth Integration**: Adds ground truth point if provided
4. **Camera Pose Display**: Adds estimated pose as purple point
5. **Axis Configuration**: Sets labels and dynamic limits
6. **Rendering**: Updates display with new data

#### Example Usage

```python
visualizer = SLAMVisualizer()
visualizer.vis_slam(slam_graph.get_nodes(), 
                   slam_graph.get_estimated_pose(), 
                   ground_truth_pose)
```

### `slam_graph(self, graph)`

Creates a 2D network graph visualization showing the connectivity structure between tags and their reference relationships.

#### Coordinate Frame Context
**Note**: This is a **topological graph** showing reference relationships, not spatial positions. The layout is algorithmic (circular) and does not represent physical world coordinates.

#### Parameters

- **graph** (dict): Dictionary of Node objects from SLAMGraph

#### Visualization Elements

1. **Network Structure**: Uses NetworkX for graph layout and rendering
2. **Node Representation**: Circular nodes representing detected tags
3. **Edge Representation**: Lines showing reference relationships
4. **Weight Display**: Edge labels showing reference chain weights
5. **Color Coding**: Same status-based coloring as 3D visualization

#### Algorithm

1. **Graph Construction**: Builds NetworkX graph from SLAM nodes
2. **Edge Creation**: Adds edges based on reference relationships
3. **Layout Calculation**: Uses circular layout for node positioning
4. **Color Assignment**: Applies status-based colors to nodes
5. **Rendering**: Draws graph with labels and edge weights

#### Network Properties

- **Nodes**: Each detected tag becomes a graph node
- **Edges**: Reference relationships create edges with weights
- **Layout**: Circular arrangement for clear visualization
- **Labels**: Tag IDs displayed on nodes, weights on edges

### `error_graph(self, graph, ground_truth_graph)`

Creates an error analysis visualization comparing estimated positions with ground truth data.

#### Coordinate Frame Analysis
**Error Calculations**: Compare **distance magnitudes** between estimated and ground truth positions.

- **World Error**: `|||estimated_world_pos|| - ||true_world_pos|||` - Difference in world position magnitudes
- **Local Error**: `|||||estimated_local_pos|| - ||true_local_pos|||` - Difference in camera-to-tag distance magnitudes
- **Reference**: Compares norms (distances from origin) rather than position vectors directly

#### Parameters

- **graph** (dict): Dictionary of estimated Node objects (with world coordinates)
- **ground_truth_graph** (dict): Dictionary containing ground truth position data (in world coordinates)

#### Visualization Elements

1. **Error Network**: Shows estimation errors as colored edges
2. **Camera Connection**: Displays local coordinate errors to camera
3. **Error Thresholds**: Color-coded edges based on error magnitude:
   - **Green**: Low error (≤ 1.0)
   - **Yellow**: Moderate error (≤ 2.5)
   - **Orange**: High error (≤ 5.0)
   - **Red**: Severe error (> 5.0)

#### Error Metrics

- **World Error**: Absolute difference between estimated and true distance magnitudes from world origin
- **Local Error**: Absolute difference between estimated and true camera-to-tag distance magnitudes  
- **Magnitude Comparison**: Uses L2 norm of position vectors, then compares the norms

#### Algorithm

1. **Graph Construction**: Creates NetworkX graph with camera node
2. **Error Calculation**: Computes position differences for each tag
3. **Edge Creation**: Connects tags to references and camera with error weights
4. **Color Assignment**: Applies error-based color coding to edges
5. **Rendering**: Displays graph with error values and color legend

## Color Coding System

### Node Status Colors

| Color | Meaning | Condition |
|-------|---------|-----------|
| Green | Updated & Visible | `node.visible = True, node.updated = True` |
| Orange | Not Updated | `node.updated = False` |
| Red | Not Visible | `node.visible = False` |
| Purple | Camera Pose | Estimated camera position |
| Blue | Ground Truth | True camera position (if available) |
| Pink | Camera Node | Camera reference in error graph |

### Error Magnitude Colors

| Color | Error Range | Threshold |
|-------|-------------|-----------|
| Green | Low | ≤ 1.0 units |
| Yellow | Moderate | 1.0 - 2.5 units |
| Orange | High | 2.5 - 5.0 units |
| Red | Severe | > 5.0 units |

## Interactive Features

### Real-time Updates

- **Live Plotting**: `plt.ion()` enables real-time visualization updates
- **Canvas Refresh**: Automatic redrawing with `canvas.draw()` and `canvas.flush_events()`
- **Dynamic Scaling**: Axes automatically adjust to data range

### Multiple Windows

- **Independent Figures**: Three separate windows for different visualization types
- **Simultaneous Display**: All visualizations can be shown concurrently
- **Window Management**: Each figure maintains its own state and zoom level

## Usage Examples

### Basic 3D Visualization

```python
from src.core.slam_visualizer import SLAMVisualizer

# Initialize visualizer
visualizer = SLAMVisualizer()

# Visualize SLAM state
nodes = slam_graph.get_nodes()
estimated_pose = slam_graph.get_estimated_pose()
ground_truth = get_ground_truth_pose()  # Your ground truth function

visualizer.vis_slam(nodes, estimated_pose, ground_truth)
```

### Graph Structure Analysis

```python
# Visualize connectivity structure
visualizer.slam_graph(slam_graph.get_nodes())

# This shows:
# - Which tags reference which others
# - Reference chain weights
# - Tag visibility status
```

### Error Analysis

```python
# Compare with ground truth
ground_truth_data = {
    1: {"world": 2.5, "local": 1.8},
    2: {"world": 3.1, "local": 2.2},
    # ... more ground truth data
}

visualizer.error_graph(slam_graph.get_nodes(), ground_truth_data)
```

### Complete Visualization Suite

```python
import matplotlib.pyplot as plt
from src.core.slam_visualizer import SLAMVisualizer

# Initialize
visualizer = SLAMVisualizer()

# In your SLAM loop:
while running:
    # ... SLAM processing ...
    
    # Update all visualizations
    nodes = slam.graph.get_nodes()
    pose = slam.graph.get_estimated_pose()
    
    visualizer.vis_slam(nodes, pose, ground_truth)
    visualizer.slam_graph(nodes)
    
    if ground_truth_available:
        visualizer.error_graph(nodes, ground_truth_data)
    
    # Small delay for smooth animation
    plt.pause(0.01)
```

## Performance Considerations

### Rendering Performance

- **Clear and Redraw**: Each update clears axes and redraws all elements
- **Data Scaling**: Performance scales with number of detected tags
- **Real-time Constraints**: Frequent updates may impact SLAM performance

### Memory Management

- **Figure Persistence**: Figures remain in memory throughout session
- **Data References**: No persistent storage of large datasets
- **Garbage Collection**: Old plot elements automatically cleaned

### Optimization Tips

1. **Update Frequency**: Limit visualization updates for better performance
2. **Selective Visualization**: Only enable needed visualizations
3. **Data Filtering**: Reduce point density for large datasets
4. **Background Processing**: Consider separate thread for visualization

## Error Handling

### Robust Operation

- **Empty Data**: Gracefully handles empty graphs and missing data
- **Axis Limits**: Safe fallbacks when no data points exist
- **Color Assignment**: Handles missing status flags gracefully

### Data Validation

- **Matrix Shapes**: Assumes proper 4x4 transformation matrices
- **Node Structure**: Expects standard Node object attributes
- **Ground Truth Format**: Requires specific dictionary structure for error analysis

## Integration Notes

### SLAM System Integration

- **Passive Visualization**: Does not modify SLAM state or data
- **Real-time Updates**: Designed for live SLAM operation visualization
- **Multi-modal Display**: Supports different visualization needs simultaneously

### External Dependencies

- **NetworkX**: Required for graph layout and rendering
- **Matplotlib**: Core plotting and interactive functionality
- **NumPy**: Array operations for position extraction

## Related Classes

- [`SLAM`](SLAM.md): Main SLAM class that uses visualizer for debugging
- [`SLAMGraph`](SLAM_GRAPH.md): Provides node data for visualization
- [`Node`](SLAM_GRAPH.md#class-node): Individual tag data structure for plotting