# AprilSLAM Documentation

Welcome to the comprehensive documentation for the AprilSLAM system - a real-time Simultaneous Localization and Mapping (SLAM) implementation using AprilTags.

## 📁 Documentation Structure

This documentation is organized into two main categories:

### 🏗️ Architecture Documentation
High-level system design, concepts, and architectural decisions.

- [`SLAM_ARCHITECTURE.md`](architecture/SLAM_ARCHITECTURE.md) - Core SLAM algorithms and design patterns
- [`SIMULATION_ARCHITECTURE.md`](architecture/SIMULATION_ARCHITECTURE.md) - Simulation engine architecture and components

### 📚 API Documentation
Detailed class and module documentation for developers.

#### Core SLAM Components
- [`SLAM.md`](api/core/SLAM.md) - Main SLAM interface and coordinator
- [`SLAM_GRAPH.md`](api/core/SLAM_GRAPH.md) - Graph structure and coordinate management
- [`SLAM_VISUALIZER.md`](api/core/SLAM_VISUALIZER.md) - Visualization and plotting capabilities

#### Detection Module
- [`TAG_DETECTOR.md`](api/detection/TAG_DETECTOR.md) - AprilTag detection and pose estimation
- [`VIDEO_DETECTION.md`](api/detection/VIDEO_DETECTION.md) - Real-time video detection system

#### Simulation Module
- [`SIMULATION.md`](api/simulation/SIMULATION.md) - Main simulation engine
- [`DATA_LOGGER.md`](api/simulation/DATA_LOGGER.md) - Data collection and logging
- [`SIMULATION_ENGINE.md`](api/simulation/SIMULATION_ENGINE.md) - Core simulation components
- [`CAMERA_CONTROLLER.md`](api/simulation/CAMERA_CONTROLLER.md) - Virtual camera management
- [`RENDERER.md`](api/simulation/RENDERER.md) - OpenGL rendering system

#### Calibration Module
- [`CALIBRATION.md`](api/calibration/CALIBRATION.md) - Camera calibration utilities

#### Analysis Module
- [`ANALYSIS.md`](api/analysis/ANALYSIS.md) - Error analysis and performance evaluation

## 🚀 Quick Start

### For Users
1. Read the [SLAM Architecture](architecture/SLAM_ARCHITECTURE.md) for system overview
2. Check [Simulation Architecture](architecture/SIMULATION_ARCHITECTURE.md) for simulation details
3. Follow installation and usage guides in the main [README](../README.md)

### For Developers
1. Start with the [SLAM class](api/core/SLAM.md) for the main interface
2. Understand the [TagDetector](api/detection/TAG_DETECTOR.md) for detection algorithms
3. Explore [SLAMGraph](api/core/SLAM_GRAPH.md) for graph management
4. Check [SLAMVisualizer](api/core/SLAM_VISUALIZER.md) for visualization capabilities

## 🔍 Finding Documentation

### By Component Type
- **Detection**: AprilTag detection, pose estimation, video processing
- **Core**: Main SLAM algorithms, graph management, visualization
- **Simulation**: Virtual environment, rendering, data logging
- **Calibration**: Camera parameter estimation and management
- **Analysis**: Performance evaluation and error analysis

### By Use Case
- **Real-time Detection**: [`TAG_DETECTOR.md`](api/detection/TAG_DETECTOR.md), [`VIDEO_DETECTION.md`](api/detection/VIDEO_DETECTION.md)
- **SLAM Implementation**: [`SLAM.md`](api/core/SLAM.md), [`SLAM_GRAPH.md`](api/core/SLAM_GRAPH.md)
- **Simulation Development**: [`SIMULATION.md`](api/simulation/SIMULATION.md), [`SIMULATION_ENGINE.md`](api/simulation/SIMULATION_ENGINE.md)
- **Data Analysis**: [`DATA_LOGGER.md`](api/simulation/DATA_LOGGER.md), [`ANALYSIS.md`](api/analysis/ANALYSIS.md)
- **Visualization**: [`SLAM_VISUALIZER.md`](api/core/SLAM_VISUALIZER.md), [`RENDERER.md`](api/simulation/RENDERER.md)

## 📊 System Overview

```
AprilSLAM System Architecture
┌─────────────────────────────────────────────────────────────┐
│                        SLAM Core                            │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐│
│  │    SLAM     │ │ SLAMGraph   │ │    SLAMVisualizer       ││
│  │ Controller  │ │ Management  │ │   Visualization         ││
│  └─────────────┘ └─────────────┘ └─────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
           │                              │
           ▼                              ▼
┌─────────────────────────┐    ┌─────────────────────────┐
│     Detection Module    │    │    Simulation Module    │
│ ┌─────────────────────┐ │    │ ┌─────────────────────┐ │
│ │   TagDetector       │ │    │ │   Simulation        │ │
│ │                     │ │    │ │   Engine            │ │
│ └─────────────────────┘ │    │ └─────────────────────┘ │
│ ┌─────────────────────┐ │    │ ┌─────────────────────┐ │
│ │  VideoDetection     │ │    │ │   DataLogger        │ │
│ │                     │ │    │ │                     │ │
│ └─────────────────────┘ │    │ └─────────────────────┘ │
└─────────────────────────┘    └─────────────────────────┘
```

## ⚠️ Critical: Coordinate Frame Conventions

**Understanding coordinate frames is essential for working with AprilSLAM!** Each component uses specific coordinate conventions:

- **TagDetector**: Returns **Camera-to-Tag** transformations `T_camera_to_tag`
- **SLAM**: Estimates **World-to-Camera** poses `T_world_to_camera`  
- **SLAMGraph**: Stores **World-to-Tag** positions `T_world_to_tag`

**Coordinate System Standards**:
- **Camera Frame**: OpenCV convention (X-right, Y-down, Z-forward)
- **Tag Frame**: Standard convention (X-right, Y-up, Z-out)
- **World Frame**: Defined by lowest-ID detected tag

See individual class documentation for detailed coordinate frame information and transformation matrices.

## 🛠️ Development Guidelines

### Code Organization
- **Core Classes**: Fundamental SLAM algorithms and data structures
- **Detection Classes**: Computer vision and tag detection functionality  
- **Simulation Classes**: Virtual environment and testing infrastructure
- **Utility Classes**: Supporting functionality like calibration and analysis

### Design Principles
- **Direct Access**: Simple operations use direct component access rather than wrapper methods
- **Focused Responsibilities**: Each class has a clear, single responsibility
- **Data vs Algorithms**: Clear separation between data structures (SLAMGraph) and algorithms (SLAM)
- **Coordinate Frame Clarity**: All transformations explicitly document their source and target frames

### Documentation Standards
- Each class has comprehensive API documentation
- Usage examples provided for all major components
- Architecture documents explain design decisions
- Integration guides show how components work together

## 🔗 External Dependencies

### Required Libraries
- **OpenCV**: Computer vision and image processing
- **NumPy**: Numerical computations and array operations
- **AprilTag**: Tag detection library
- **Matplotlib**: Plotting and visualization
- **NetworkX**: Graph algorithms and visualization
- **Pygame**: Simulation interface
- **OpenGL**: 3D rendering

### Development Tools
- **Python 3.8+**: Primary development language
- **Git**: Version control
- **Logging**: Debug and monitoring infrastructure

## 📝 Contributing

When adding new classes or modifying existing ones:

1. **Update API Documentation**: Create or modify the appropriate `.md` file in `docs/api/`
2. **Add Usage Examples**: Include practical code examples in documentation
3. **Update This Index**: Add references to new documentation files
4. **Follow Naming Conventions**: Use clear, descriptive class and method names
5. **Add Type Hints**: Include proper type annotations for better documentation

## 📄 License

This documentation is part of the AprilSLAM project and follows the same licensing terms as the main codebase.

---

*Last Updated: 2024*
*Generated for AprilSLAM Version: 1.0* 