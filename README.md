# AprilSLAM 🏷️📍

**A complete Visual SLAM (Simultaneous Localization and Mapping) system using AprilTag fiducial markers**

AprilSLAM is a comprehensive SLAM implementation that uses AprilTag markers as landmarks to perform accurate localization and mapping. The system combines computer vision, robotics algorithms, and 3D simulation to provide a complete solution for visual navigation and mapping.

## 🎯 What Does This Software Do?

AprilSLAM solves the fundamental robotics problem of **"Where am I and what does my environment look like?"** by:

- **🔍 Detecting AprilTag markers** in camera images with sub-pixel precision
- **📐 Estimating camera pose** (position and orientation) relative to detected tags
- **🗺️ Building a map** of tag locations in 3D space
- **📊 Tracking camera movement** through the environment over time
- **🎮 Simulating realistic scenarios** for testing and development
- **📈 Analyzing performance** with comprehensive error metrics

### Key Applications:
- **Robot Navigation**: Autonomous robots navigating indoor environments
- **AR/VR Systems**: Augmented reality applications requiring precise tracking
- **Drone Localization**: UAVs operating in GPS-denied environments  
- **Research & Education**: SLAM algorithm development and testing

## 🚀 Quick Start

### Prerequisites

Make sure you have Python 3.8+ installed with the following packages:

```bash
pip install numpy opencv-python pygame PyOpenGL matplotlib pandas scikit-learn
```

### 🎮 Run the Simulation (Recommended First Step)

The easiest way to see AprilSLAM in action:

```bash
python run_simulation.py
```

This will launch a 3D simulation where you can:
- **Move around** using keyboard controls (WASD + arrow keys)
- **See real-time tag detection** overlaid on the camera view
- **Watch SLAM in action** as the system builds a map
- **Monitor accuracy** with ground truth comparisons

**Controls:**
- `Arrow Keys`: Move camera position
- `W/S`: Move forward/backward  
- `A/D`: Rotate left/right
- `Q/E`: Roll camera
- `R/F`: Pitch up/down

### 📷 Real Camera Detection

To use AprilSLAM with a real camera:

```bash
python src/detection/video_detection.py
```

### 🎯 Camera Calibration

For accurate results with real cameras, calibrate first:

```bash
# 1. Capture calibration images
python src/calibration/take_pics.py

# 2. Run calibration
python src/calibration/calibrate.py
```

## 📊 What You'll See

### Simulation View
- **Purple 3D environment** with AprilTag markers placed in space
- **Real-time camera feed** showing detected tags with overlay information
- **Tag information** including distance, orientation, and ID
- **Console output** with positioning accuracy metrics

### Data Output
- **CSV files** with position estimates, ground truth, and error metrics
- **Log files** with detailed execution information  
- **Visualization plots** showing SLAM performance over time

## 🛠️ Advanced Usage

### Configuration

Customize the simulation by editing `config/sim_settings.json`:

```json
{
    "display_width": 1000,
    "display_height": 1000,
    "fov_y": 45,
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

### Analysis Tools

Analyze SLAM performance:

```bash
# Error analysis and clustering
python src/analysis/error_analysis.py

# Covariance analysis  
python src/analysis/covarience.py

# Log file debugging
python scripts/log_debugging.py
```

### Randomized Testing

Generate randomized scenarios for robust testing:

```bash
python src/simulation/randomize_simulation.py --percentage 0.1
```

## 🏗️ Architecture

AprilSLAM is built with a modular architecture:

```
src/
├── core/          # SLAM algorithms and graph management
├── detection/     # AprilTag detection and computer vision
├── calibration/   # Camera calibration tools
├── simulation/    # 3D simulation environment  
├── analysis/      # Data analysis and visualization
```

**Key Components:**
- **Graph-based SLAM**: Maintains a pose graph for optimization
- **AprilTag Detection**: High-precision marker detection
- **3D Simulation**: OpenGL-based realistic environment
- **Error Analysis**: Comprehensive performance metrics

## 📈 Performance Features

- **Real-time processing** at 30+ FPS
- **Sub-pixel accuracy** in tag detection
- **Robust to lighting conditions** and partial occlusions
- **Scalable** to large environments with many tags
- **Comprehensive logging** for debugging and analysis

## 📁 Project Structure

```
AprilSLAM/
├── 🎮 run_simulation.py        # Main entry point - START HERE!
├── src/                        # Source code modules
├── config/                     # Configuration files
├── data/                       # Generated data and results
├── assets/                     # AprilTag images and patterns
├── lib/                        # External libraries (AprilTag)
└── docs/                       # Documentation and technical details
```

## 🔧 Dependencies

- **OpenCV**: Computer vision and image processing
- **NumPy**: Numerical computations
- **Pygame + OpenGL**: 3D simulation rendering
- **Matplotlib**: Data visualization and plotting
- **Pandas**: Data analysis and CSV handling
- **scikit-learn**: Machine learning for clustering analysis

## 📚 Learn More

- **Technical Details**: See `docs/SLAM_ARCHITECTURE.md` for algorithm details
- **Research Applications**: Perfect for robotics and computer vision research
- **Educational Use**: Great for learning SLAM concepts hands-on

## 🤝 Contributing

This project welcomes contributions! The modular architecture makes it easy to:
- Add new detection algorithms
- Implement different SLAM backends  
- Extend analysis capabilities
- Create new simulation scenarios

## 📄 License

Licensed under the terms in `docs/LICENSE`.

---

**Ready to explore SLAM?** Run `python run_simulation.py` and start mapping! 🗺️✨
