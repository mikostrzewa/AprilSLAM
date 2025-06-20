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

**System Requirements:**
- Python 3.8 or higher
- C++ compiler (for AprilTag library compilation)
- CMake (for building AprilTag)
- Ninja build system (recommended for faster builds)

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd AprilSLAM
```

2. **Build AprilTag library locally:**
```bash
# Navigate to lib directory and clone official repository
cd lib/
git clone https://github.com/AprilRobotics/apriltag.git
cd apriltag

# Build with Ninja (requires cmake and ninja-build)
cmake -B build -GNinja
cmake --build build

# Install Python bindings
cd python
pip install -e .
cd ../../..
```

3. **Install remaining Python dependencies:**
```bash
pip install -r requirements.txt
```

**✅ Note:** AprilTag is intentionally not included in `requirements.txt` since the local build method is more reliable.

**⚠️ Fallback:** If the local build fails, you can try installing AprilTag via pip as a fallback:

**On Ubuntu/Debian:**
```bash
sudo apt-get install cmake build-essential ninja-build
```

**On macOS:**
```bash
brew install cmake ninja
```

**On Windows:**
- Install Visual Studio Build Tools
- Install CMake
- Then run: `pip install apriltag`

4. **Verify installation:**
```bash
# Quick check
python -c "import apriltag; print('AprilTag successfully installed!')"

# Comprehensive verification (recommended)
python scripts/verify_installation.py
```

### 🔧 Troubleshooting Installation

If you encounter issues with the `apriltag` library:

**Common Issues:**

**1. Library Loading Errors (macOS/Linux):**
```
dlopen: libapriltag.dylib not found
```
**Solution:** The apriltag library may need to be rebuilt. Try:
```bash
pip uninstall apriltag
pip install --no-cache-dir apriltag
```

**2. Windows Compilation Issues:**
- **"Microsoft Visual C++ 14.0 is required"**: Install Visual Studio Build Tools
- **"cmake not found"**: Install CMake from cmake.org or via package manager

**3. General Compilation Errors:**
- Ensure you have a working C++ compiler installed
- Try updating pip: `pip install --upgrade pip setuptools wheel`

**Alternative Solutions:**

**Option 1: Build AprilTag Library Locally (Recommended)**
For the most reliable AprilTag installation, build it directly from the [official University of Michigan repository](https://github.com/AprilRobotics/apriltag):

```bash
# Navigate to the lib directory
cd lib/

# Clone the official AprilTag repository
git clone https://github.com/AprilRobotics/apriltag.git

# Build with Ninja (recommended) or make
cd apriltag
cmake -B build -GNinja
cmake --build build

# Install Python bindings
cd python
pip install -e .
```

The simulation will automatically detect and use this locally built version.

**Option 2: Manual AprilTag Build (Alternative Location)**
```bash
# If you prefer to build elsewhere, clone anywhere:
git clone https://github.com/AprilRobotics/apriltag.git
cd apriltag

# Build with make instead of Ninja
mkdir build && cd build
cmake ..
make -j4

# Install Python bindings
cd ../python
pip install -e .
```

**Option 3: Alternative Installation Methods**
```bash
# Try different installation approaches:
pip install apriltag --user
# or
conda install -c conda-forge apriltag
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
- `ESC`: Quit simulation

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

### Design Principles

**Direct Component Access**: AprilSLAM follows a direct access pattern for better maintainability:

```python
# ✅ Preferred: Direct access for simple operations
frame = slam.detector.draw(rvec, tvec, corners, frame, tag_id)
pose = slam.graph.my_pose()
distance = slam.graph.average_distance_to_nodes()

# ✅ Wrapper methods only when they add value (simplify complex calls)
slam.vis_slam(ground_truth=gt_pose)  # Handles complex parameter preparation
slam.slam_graph()  # Abstracts graph data preparation
```

This approach:
- **Reduces complexity** by avoiding unnecessary wrapper layers
- **Improves maintainability** by eliminating duplicate method signatures
- **Keeps interfaces clean** while maintaining component encapsulation
- **Reserves wrappers** for cases where they genuinely simplify usage

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

### Core Dependencies (Required)
- **apriltag**: AprilTag visual fiducial system by [University of Michigan APRIL Robotics Laboratory](https://april.eecs.umich.edu/software/apriltag) - **ESSENTIAL** (built locally from source)
- **opencv-python**: Computer vision and image processing
- **numpy**: Numerical computations and matrix operations
- **pygame**: Window management and input handling
- **PyOpenGL**: 3D rendering and graphics

### Analysis Dependencies
- **matplotlib**: Data visualization and plotting
- **pandas**: Data analysis and CSV handling
- **scikit-learn**: Machine learning for clustering analysis
- **networkx**: Graph visualization for SLAM graph
- **termcolor**: Colored terminal output

### Optional Dependencies
- **seaborn**: Enhanced statistical plotting
- **scipy**: Additional scientific computing tools

**Installation:** Python dependencies are listed in `requirements.txt` for easy installation:
```bash
pip install -r requirements.txt
```
*Note: AprilTag is built separately from source as shown in the installation steps above.*

## 🙏 Acknowledgments

This project builds upon the excellent [AprilTag visual fiducial system](https://april.eecs.umich.edu/software/apriltag) developed by the **APRIL Robotics Laboratory** at the University of Michigan, led by Associate Professor Edwin Olson. 

AprilTag is a robust and flexible visual fiducial system that enables precise 3D position, orientation, and identity detection. The [official AprilTag repository](https://github.com/AprilRobotics/apriltag) is maintained by AprilRobotics and licensed under BSD.

**Key AprilTag Publications:**
- *"AprilTag: A robust and flexible visual fiducial system"* (ICRA 2011)
- *"AprilTag 2: Efficient and robust fiducial detection"* (IROS 2016)
- *"Flexible Layouts for Fiducial Tags"* (under review)

## 📚 Learn More

- **Technical Details**: See `docs/SLAM_ARCHITECTURE.md` for algorithm details
- **Research Applications**: Perfect for robotics and computer vision research
- **Educational Use**: Great for learning SLAM concepts hands-on
- **AprilTag Documentation**: Visit the [official AprilTag website](https://april.eecs.umich.edu/software/apriltag)

## 🤝 Contributing

This project welcomes contributions! The modular architecture makes it easy to:
- Add new detection algorithms
- Implement different SLAM backends  
- Extend analysis capabilities
- Create new simulation scenarios

## 📄 License

AprilSLAM is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International License** (CC BY-NC 4.0).

**What this means:**
- ✅ **Free for research, education, and personal projects**
- ✅ You can share, modify, and distribute the code
- ✅ Must provide attribution to the original authors
- ❌ **Commercial use requires permission**

For **commercial licensing** or questions about usage rights, please contact the project maintainer.

See the full license terms in [`LICENSE`](LICENSE).

---

**Ready to explore SLAM?** Run `python run_simulation.py` and start mapping! 🗺️✨
