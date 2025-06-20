# Video Detection System Documentation

## Overview

The `video_detection.py` module provides a complete real-time AprilTag detection system using webcam input. It demonstrates the practical application of the TagDetector class for live video processing, offering pose estimation, visual overlays, and comprehensive error handling.

## Module: video_detection.py

Located in: `src/detection/video_detection.py`

### Purpose

The video detection module serves as:
- **Real-time Detection Demo**: Live demonstration of AprilTag detection capabilities
- **Integration Example**: Shows how to combine TagDetector with OpenCV video processing
- **Testing Platform**: Interactive environment for testing detection algorithms
- **Educational Tool**: Clear example of computer vision pipeline implementation

### Features

- **Real-time Processing**: Live AprilTag detection from webcam feed
- **6DOF Pose Estimation**: Full position and orientation calculation
- **Visual Feedback**: Interactive overlays with detection information
- **Multi-camera Support**: Automatic fallback to alternative camera sources
- **Robust Error Handling**: Graceful handling of camera and detection failures
- **Performance Monitoring**: Frame rate display and processing statistics

### Dependencies

```python
import numpy as np
import cv2
import sys
import os
from tag_detector import TagDetector
```

## Core Functions

### `load_camera_calibration()`

Loads camera calibration parameters from the saved calibration file.

#### Returns
- **camera_params** (dict): Dictionary containing calibration data
  - `camera_matrix`: 3x3 intrinsic camera matrix
  - `dist_coeffs`: Lens distortion coefficients

#### File Location
Expected calibration file: `data/calibration/camera_calibration_parameters.npz`

#### Error Handling
- **FileNotFoundError**: If calibration file doesn't exist
- **KeyError**: If required calibration parameters are missing

#### Example Usage
```python
try:
    camera_params = load_camera_calibration()
    print("Camera calibration loaded successfully")
except FileNotFoundError:
    print("Please run camera calibration first")
    exit(1)
```

### `initialize_camera(camera_id=0)`

Initializes and configures the camera for optimal video capture.

#### Parameters
- **camera_id** (int, optional): Camera device ID. Default: 0

#### Returns
- **cap** (cv2.VideoCapture): Configured camera capture object

#### Configuration Applied
- **Resolution**: 640x480 pixels for balanced quality and performance
- **Frame Rate**: 30 FPS for smooth real-time operation
- **Auto-fallback**: Attempts alternative camera IDs if primary fails

#### Camera Properties Set
```python
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
```

#### Example Usage
```python
cap = initialize_camera(camera_id=0)
if not cap.isOpened():
    print("Failed to initialize camera")
    exit(1)
```

### `process_detections(detector, detections, frame)`

Processes all detected AprilTags and adds comprehensive visual overlays.

#### Parameters
- **detector** (TagDetector): Configured TagDetector instance
- **detections** (list): List of detected tags from TagDetector.detect()
- **frame** (numpy.ndarray): Video frame to process and annotate

#### Returns
- **frame** (numpy.ndarray): Annotated frame with detection overlays

#### Processing Pipeline

1. **Detection Validation**: Checks if any tags were detected
2. **Pose Estimation**: Calculates 6DOF pose for each detected tag
3. **Visual Annotation**: Adds detection overlays using TagDetector.draw()
4. **Metric Calculation**: Computes distance and orientation angles
5. **Console Output**: Prints detailed detection information
6. **Error Handling**: Manages failed pose estimations gracefully

#### Visualization Elements

- **Tag Outlines**: Colored borders around detected tags
- **Coordinate Axes**: 3D axes showing tag orientation
- **Text Information**: Tag ID, distance, and orientation data
- **Status Indicators**: Different colors for successful/failed detections

#### Console Output Format
```
📍 Tag ID 1:
   Position (x,y,z): (0.123, -0.045, 0.567) m
   Distance: 567.8 mm
   Orientation - Yaw: 12.3°, Pitch: -5.7°, Roll: 0.8°
   Corners: [[100, 150], [200, 150], [200, 250], [100, 250]]
   --------------------------------------------------
```

### `add_info_overlay(frame, fps=None)`

Adds informational overlay with system status and control instructions.

#### Parameters
- **frame** (numpy.ndarray): Video frame to annotate
- **fps** (float, optional): Current frame rate for display

#### Returns
- **frame** (numpy.ndarray): Frame with info overlay added

#### Overlay Elements
- **Semi-transparent Background**: Dark overlay for text readability
- **System Title**: "AprilTag Detection System"
- **Control Instructions**: Key bindings and usage help
- **Performance Metrics**: Frame rate display (if provided)

## Main Application

### `main()`

The primary application entry point that orchestrates the complete detection system.

#### Application Flow

1. **Initialization Phase**:
   - Load camera calibration parameters
   - Initialize camera with error handling
   - Create TagDetector instance
   - Set up performance monitoring

2. **Main Processing Loop**:
   - Capture video frames from camera
   - Detect AprilTags in each frame
   - Process detections and add overlays
   - Display annotated frames
   - Handle user input and controls

3. **Cleanup Phase**:
   - Release camera resources
   - Close display windows
   - Clean exit with resource cleanup

#### User Controls

| Key | Action |
|-----|--------|
| `q` | Quit application |
| `ESC` | Alternative quit method |

#### Performance Monitoring
- Frame rate calculation and display
- Processing time measurement
- Memory usage tracking

## Usage Examples

### Basic Usage

```python
# Run the video detection system
python src/detection/video_detection.py
```

### Integration Example

```python
from src.detection.video_detection import (
    load_camera_calibration, 
    initialize_camera, 
    process_detections
)
from src.detection.tag_detector import TagDetector

# Setup
camera_params = load_camera_calibration()
cap = initialize_camera()
detector = TagDetector(camera_params, tag_size=0.06)

# Processing loop
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect and process
    detections = detector.detect(frame)
    frame = process_detections(detector, detections, frame)
    
    # Display
    cv2.imshow('AprilTag Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
```

### Custom Camera Configuration

```python
import cv2
from src.detection.video_detection import initialize_camera

# Initialize with specific camera
cap = initialize_camera(camera_id=1)

# Custom resolution for high-quality capture
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Custom frame rate for slow motion analysis
cap.set(cv2.CAP_PROP_FPS, 15)
```

## Error Handling and Troubleshooting

### Common Issues

#### Camera Not Found
```
❌ Could not open any camera. Please check camera connection.
```
**Solution**: Verify camera is connected and not used by another application

#### Missing Calibration
```
❌ Camera calibration file not found
```
**Solution**: Run camera calibration first:
```bash
python src/calibration/calibrate.py
```

#### Poor Detection Performance
- **Lighting**: Ensure adequate, even lighting
- **Distance**: Keep tags within 0.5-3 meters from camera
- **Angle**: Avoid extreme viewing angles (>45 degrees)
- **Size**: Use appropriately sized tags for viewing distance

### Performance Optimization

#### For High Frame Rates
```python
# Reduce resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# Disable visualization for processing only
minimal_processing = True
```

#### For High Accuracy
```python
# Increase resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Use smaller tag family for better precision
detector = TagDetector(camera_params, 
                      tag_type="tag36h11", 
                      tag_size=0.04)
```

## Technical Specifications

### Video Processing
- **Input Format**: BGR color images from OpenCV VideoCapture
- **Processing**: Grayscale conversion for detection
- **Output**: Annotated BGR frames with overlays

### Detection Parameters
- **Tag Family**: Configurable (default: tagStandard41h12)
- **Tag Size**: Configurable physical size in meters
- **Detection Range**: Optimal 0.5-3 meters
- **Orientation Range**: ±45 degrees from perpendicular

### Performance Metrics
- **Detection Rate**: Typically 15-30 Hz depending on hardware
- **Latency**: <50ms processing time per frame
- **Accuracy**: ±2mm position, ±2° orientation (optimal conditions)

## Integration Notes

### With SLAM System
```python
from src.core.slam import SLAM

# Initialize SLAM with same camera parameters
slam = SLAM(logging, camera_params, tag_size=0.06)

# In processing loop
detections = slam.detect(frame)
for detection in detections:
    success, rvec, tvec = slam.get_pose(detection)
    # SLAM automatically updates graph
```

### With Data Logging
```python
from src.simulation.data_logger import DataLogger

logger = DataLogger()

# In processing loop
for detection in detections:
    if pose_estimation_successful:
        pose_data = logger.create_pose_data(tvec, rvec)
        logger.log_frame_data(pose_data, ground_truth, error_metrics)
```

## Development Guidelines

### Adding New Features
1. **Maintain Real-time Performance**: Ensure additions don't impact frame rate
2. **Preserve Error Handling**: Maintain robust error handling patterns
3. **Update Documentation**: Document new features and parameters
4. **Test Multiple Cameras**: Verify compatibility across camera types

### Customization Points
- **Detection Parameters**: Tag family, size, detection thresholds
- **Visualization**: Colors, overlay elements, information display
- **Camera Settings**: Resolution, frame rate, exposure
- **Processing Pipeline**: Add preprocessing or postprocessing steps

## Related Components

- [`TagDetector`](TAG_DETECTOR.md): Core detection and pose estimation
- [`SLAM`](../core/SLAM.md): Integration with SLAM system
- [`Camera Calibration`](../calibration/CALIBRATION.md): Required calibration process
- [`Data Logger`](../simulation/DATA_LOGGER.md): Data collection integration 