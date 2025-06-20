# AprilSLAM Simulation Architecture 

This document provides a comprehensive overview of the AprilSLAM simulation system architecture, explaining how the modular components work together to create a complete SLAM testing environment.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture Principles](#architecture-principles)
- [Core Components](#core-components)
- [Data Flow](#data-flow)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [Extending the System](#extending-the-system)
- [Performance Considerations](#performance-considerations)
- [Troubleshooting](#troubleshooting)

## 🔍 Overview

The AprilSLAM simulation system has been completely refactored from a monolithic design into a modular, well-documented architecture. The system provides:

- **Realistic 3D rendering** using OpenGL
- **Camera control** with keyboard input or Monte Carlo randomization
- **SLAM algorithm integration** with AprilTag detection
- **Ground truth calculation** for performance evaluation
- **Comprehensive data logging** for analysis and research
- **Modular design** for easy extension and maintenance

## 🏛️ Architecture Principles

The new architecture follows these key principles:

### Single Responsibility Principle
Each class has one clear purpose and handles one aspect of the simulation.

### Separation of Concerns
Different aspects of the simulation (rendering, input, logging, etc.) are handled by separate, independent modules.

### Dependency Injection
Components receive their dependencies through constructors, making testing and modification easier.

### Clear Interfaces
Each component exposes a well-defined API with comprehensive documentation.

### Backward Compatibility
Legacy code continues to work through deprecation warnings and wrapper classes.

## 🧩 Core Components

### 1. SimulationEngine 🎮
**File:** `src/simulation/simulation_engine.py`

The main orchestrator that coordinates all subsystems.

**Responsibilities:**
- Initialize and manage all subsystems
- Run the main simulation loop
- Handle high-level event coordination
- Provide cleanup and resource management

**Key Methods:**
```python
def __init__(config_file: str, movement_enabled: bool = True)
def run() -> None
def get_simulation_stats() -> Dict[str, Any]
```

### 2. SimulationConfig 📋
**File:** `src/simulation/config_manager.py`

Manages simulation configuration and settings.

**Responsibilities:**
- Load and validate JSON configuration files
- Provide typed access to configuration parameters
- Handle unit conversions (simulation units ↔ millimeters)
- Validate configuration integrity

**Key Properties:**
```python
@property
def display_size(self) -> Tuple[int, int]
def tag_size_inner(self) -> float
def simulation_units_to_mm(value_sim: float) -> float
```

### 3. CameraController 🎥
**File:** `src/simulation/camera_controller.py`

Handles camera movement and keyboard input.

**Responsibilities:**
- Track camera position and rotation
- Process keyboard input events
- Provide manual and automatic movement modes
- Calculate camera transformation matrices

**Key Methods:**
```python
def handle_key_event(event: pygame.event.Event) -> None
def update_manual_movement() -> None
def randomize_position(bounds: np.ndarray) -> None
def get_transformation_matrix() -> np.ndarray
```

### 4. SimulationRenderer 🎨
**File:** `src/simulation/renderer.py`

Manages OpenGL rendering and texture operations.

**Responsibilities:**
- Initialize OpenGL context and window
- Load and manage AprilTag textures
- Render 3D scenes with proper depth sorting
- Capture frames for SLAM processing
- Handle view matrix transformations

**Key Methods:**
```python
def render_frame(camera_position: np.ndarray, camera_rotation: np.ndarray) -> None
def capture_frame() -> np.ndarray
def get_tag_by_id(tag_id: int) -> Optional[TagData]
```

### 5. DataLogger 📊
**File:** `src/simulation/data_logger.py`

Handles all data logging and CSV file operations.

**Responsibilities:**
- Create and manage CSV files
- Log SLAM performance metrics
- Track error analysis data
- Manage covariance information
- Provide data structures for pose and error data

**Key Methods:**
```python
def log_frame_data(estimated_pose: PoseData, ground_truth_pose: PoseData, error_metrics: ErrorMetrics) -> None
def log_error_analysis(...) -> None
def create_pose_data(translation: np.ndarray, rotation_matrix: np.ndarray) -> PoseData
```

### 6. GroundTruthCalculator 🎯
**File:** `src/simulation/ground_truth.py`

Calculates ground truth poses and transformations.

**Responsibilities:**
- Calculate camera-to-tag transformations
- Handle coordinate frame conversions
- Compute pose estimation errors
- Manage tag-to-tag distance calculations
- Convert between different coordinate systems

**Key Methods:**
```python
def get_camera_to_tag_transform(tag_id: int, camera_position: np.ndarray) -> np.ndarray
def calculate_pose_error(estimated_transform: np.ndarray, ground_truth_transform: np.ndarray) -> Tuple[float, float]
def rotation_matrix_to_euler(rotation_matrix: np.ndarray) -> np.ndarray
```

## 🔄 Data Flow

```
SimulationEngine (Main Orchestrator)
├── SimulationConfig (Configuration Management)
├── CameraController (Movement & Input)  
├── SimulationRenderer (3D Rendering)
├── DataLogger (CSV Logging)
├── GroundTruthCalculator (Reference Calculations)
└── SLAM System (Pose Estimation)

Main Loop Flow:
1. Handle user input → CameraController
2. Update camera position → CameraController  
3. Render 3D scene → SimulationRenderer
4. Capture frame → SimulationRenderer
5. Detect AprilTags → SLAM System
6. Calculate ground truth → GroundTruthCalculator
7. Compute errors → GroundTruthCalculator
8. Log data → DataLogger
9. Update visualizations → SLAM System
```

## 💻 Usage Examples

### Basic Usage

```python
from src.simulation import SimulationEngine

# Create and run simulation
config_file = "config/sim_settings.json"
with SimulationEngine(config_file, movement_enabled=True) as sim:
    sim.run()
```

### Advanced Usage with Custom Configuration

```python
from src.simulation import SimulationEngine, SimulationConfig

# Load and inspect configuration
config = SimulationConfig("config/sim_settings.json")
print(f"Display size: {config.display_size}")
print(f"Tag count: {config.get_tag_count()}")

# Run simulation with specific settings
engine = SimulationEngine("config/sim_settings.json", movement_enabled=False)
stats = engine.get_simulation_stats()
print(f"Simulation stats: {stats}")

try:
    engine.run()
finally:
    engine._cleanup()
```

### Component Usage Examples

```python
from src.simulation import CameraController, DataLogger, GroundTruthCalculator

# Camera control
camera = CameraController(movement_enabled=True)
camera.set_position(np.array([1.0, 2.0, 3.0]))
transform = camera.get_transformation_matrix()

# Data logging
with DataLogger() as logger:
    pose_data = logger.create_pose_data(translation, rotation_matrix)
    error_metrics = logger.create_error_metrics(trans_error, rot_error, gt_magnitude, node_count, avg_dist)
    logger.log_frame_data(estimated_pose, ground_truth_pose, error_metrics)

# Ground truth calculations
gt_calc = GroundTruthCalculator(config, tags_data)
gt_transform = gt_calc.get_camera_to_tag_transform(tag_id=0, camera_position)
translation_error, rotation_error = gt_calc.calculate_pose_error(estimated, ground_truth)
```

## ⚙️ Configuration

### JSON Configuration Structure

```json
{
    "display_width": 800,
    "display_height": 600,
    "fov_y": 45.0,
    "near_clip": 0.1,
    "far_clip": 100.0,
    "size_scale": 1.0,
    "tag_size_inner": 0.8,
    "tag_size_outer": 1.0,
    "actual_size_in_mm": 100.0,
    "tags": [
        {
            "id": 0,
            "image": "tags/tag0.png",
            "position": [5.0, 0.0, 0.0],
            "rotation": [0.0, 0.0, 0.0]
        }
    ]
}
```

### Configuration Validation

The system automatically validates:
- Required parameters are present
- Display dimensions are positive
- Field of view is reasonable (0° < FOV < 180°)
- Near clip < far clip
- Tag configurations are complete

### Unit Conversions

```python
# Convert simulation units to millimeters
real_distance_mm = config.simulation_units_to_mm(simulation_distance)

# Convert millimeters to simulation units  
simulation_distance = config.mm_to_simulation_units(real_distance_mm)
```

## 🔧 Extending the System

### Adding New Renderers

```python
class CustomRenderer(SimulationRenderer):
    def render_frame(self, camera_position, camera_rotation):
        # Custom rendering logic
        pass
    
    def capture_frame(self):
        # Custom frame capture
        pass
```

### Adding New Data Loggers

```python
class CustomDataLogger(DataLogger):
    def __init__(self, output_directory=None, custom_format="json"):
        super().__init__(output_directory)
        self.format = custom_format
    
    def log_custom_data(self, data):
        # Custom logging logic
        pass
```

### Adding New Camera Controllers

```python
class AutomaticCameraController(CameraController):
    def update_automatic_movement(self, trajectory_type="circular"):
        # Implement automatic camera movements
        pass
```

## ⚡ Performance Considerations

### Memory Management
- Textures are loaded once and reused
- CSV files are flushed periodically to prevent memory buildup
- OpenGL resources are properly cleaned up

### Rendering Performance
- Tags are depth-sorted for proper rendering
- Only visible tags are processed for SLAM
- Frame capture is optimized for minimal overhead

### File I/O Optimization
- CSV files use buffered writing
- Batch operations where possible
- Automatic file handle management

### SLAM Performance
- Camera parameters calculated once at initialization
- Efficient ground truth calculations
- Optimized coordinate transformations

## 🔧 Troubleshooting

### Common Issues

#### Import Errors
```bash
# Ensure you're in the project root directory
cd /path/to/AprilSLAM

# Run with proper Python path
python -m src.simulation.simulation_engine
```

#### OpenGL Context Issues
- Ensure graphics drivers are up to date
- Check that display is available (not running headless without X11 forwarding)
- Verify pygame and OpenGL installations

#### AprilTag Library Issues
```bash
# Verify apriltag library path
ls lib/apriltag/build/

# Rebuild if necessary
cd lib/apriltag
make clean && make
```

#### Configuration Errors
- Validate JSON syntax using online validators
- Check that all required fields are present
- Verify file paths are correct (relative to project root)

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run simulation with debug output
engine = SimulationEngine(config_file, movement_enabled=True)
```

### Performance Profiling

```python
import cProfile
import pstats

def profile_simulation():
    engine = SimulationEngine("config/sim_settings.json")
    engine.run()

# Profile the simulation
cProfile.run('profile_simulation()', 'simulation_profile.stats')
stats = pstats.Stats('simulation_profile.stats')
stats.sort_stats('cumulative').print_stats(20)
```

## 🎮 Controls

### Manual Movement Mode
- **Arrow Keys**: Move camera X/Y
- **W/S**: Move camera forward/backward (Z-axis)
- **A/D**: Yaw left/right
- **Q/E**: Roll left/right  
- **R/F**: Pitch up/down
- **ESC/Close Window**: Exit simulation

### Monte Carlo Mode
When movement is disabled, the camera position is randomized within specified bounds for automated testing.

## 📊 Output Files

The simulation generates several CSV files in `data/csv/`:

1. **slam_simulation_data.csv**: Main performance metrics
2. **error_analysis.csv**: Detailed error analysis per tag
3. **covariance_analysis.csv**: Uncertainty quantification data

## 📚 Additional Resources

- **Configuration Examples:** `config/` directory
- **Test Data:** `data/` directory  
- **Asset Files:** `assets/` directory
- **Core SLAM Implementation:** `src/core/slam.py`
- **Detection Systems:** `src/detection/`
- **Analysis Tools:** `src/analysis/`

## 🤝 Contributing

When extending the simulation system:

1. Follow the single responsibility principle
2. Add comprehensive docstrings
3. Include type hints
4. Write unit tests for new components
5. Update this documentation
6. Maintain backward compatibility when possible

## 📄 License

This simulation system is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License. See the LICENSE file for details.

### SLAM Graph Visualization

The simulation provides three types of graph visualizations:

1. **3D Visualization** (`vis_slam`): Shows tag positions in 3D space
2. **SLAM Graph** (`slam_graph`): Shows the connection structure between tags  
3. **Error Graph** (`error_graph`): Shows distance errors between estimated and ground truth positions

#### Error Graph Functionality

The Error Graph is a critical feature that was restored from the original implementation. It visualizes the accuracy of SLAM estimates by showing distance errors between each graph node and ground truth:

**What it shows:**
- **Local Errors**: Distance differences between estimated camera-to-tag and ground truth camera-to-tag
- **World Errors**: Distance differences between estimated tag-to-reference-tag and ground truth tag-to-reference-tag

**Edge Colors:**
- 🟢 **Green**: Low error (≤ 1.0 simulation units)
- 🟡 **Yellow**: Moderate error (1.0 - 2.5 simulation units)  
- 🟠 **Orange**: High error (2.5 - 5.0 simulation units)
- 🔴 **Red**: Severe error (> 5.0 simulation units)

**How it works:**
```python
# Build ground truth reference data
ground_truth_tags = {}
for tag_id, node in slam_graph.items():
    # Calculate ground truth distances
    local_distance = ||ground_truth_camera_to_tag||
    world_distance = ||ground_truth_tag_to_reference_tag||
    
    ground_truth_tags[tag_id] = {
        "local": local_distance,
        "world": world_distance
    }

# Compare with SLAM estimates and visualize errors
error_graph(slam_graph, ground_truth_tags)
```

This visualization helps identify:
- Which tags have the highest estimation errors
- Whether errors are systematic or random
- How error propagates through the SLAM graph
- Performance degradation over time

### Data Logging and Analysis 