# TagDetector Documentation

## Overview

The `TagDetector` class is a comprehensive AprilTag detection and pose estimation system designed for computer vision applications in robotics and augmented reality. It provides functionality to detect AprilTags in images, estimate their 3D poses, and visualize the results.

## Class: TagDetector

Located in: `src/detection/tag_detector.py`

### Purpose

The TagDetector class handles:
- AprilTag detection in grayscale images
- 3D pose estimation using camera calibration parameters
- Coordinate frame transformations
- Visualization and debugging tools

### Dependencies

```python
import numpy as np
import cv2
import sys
import os
from apriltag import apriltag
```

**Note**: The class automatically adds the AprilTag library path from `lib/apriltag/build/` to the Python path.

## Constructor

### `__init__(self, camera_params, tag_type="tagStandard41h12", tag_size=0.06)`

Initializes the TagDetector with camera parameters and tag specifications.

#### Parameters

- **camera_params** (dict): Dictionary containing camera calibration data
  - `camera_matrix`: 3x3 numpy array representing the camera's intrinsic matrix
  - `dist_coeffs`: Distortion coefficients array
- **tag_type** (str, optional): AprilTag family type. Default: `"tagStandard41h12"`
- **tag_size** (float, optional): Physical size of the tag in meters. Default: `0.06` (6cm)

#### Example

```python
camera_params = {
    'camera_matrix': np.array([[fx, 0, cx],
                              [0, fy, cy],
                              [0, 0, 1]]),
    'dist_coeffs': np.array([k1, k2, p1, p2, k3])
}

detector = TagDetector(camera_params, tag_type="tagStandard41h12", tag_size=0.06)
```

## Methods

### `detect(self, image)`

Detects AprilTags in the input image.

#### Parameters
- **image** (numpy.ndarray): BGR color image

#### Returns
- **detections** (list): Sorted list of detected tags (by ID), where each detection contains:
  - `id`: Tag ID number
  - `lb-rb-rt-lt`: Corner coordinates in image space
  - Other detection metadata

#### Example
```python
detections = detector.detect(image)
for detection in detections:
    tag_id = detection['id']
    corners = detection['lb-rb-rt-lt']
```

### `get_pose(self, detection)`

Estimates the 3D pose of a detected AprilTag using PnP (Perspective-n-Point) algorithm.

#### Parameters
- **detection** (dict): Single detection result from `detect()` method

#### Returns
- **retval** (bool): Success flag from cv2.solvePnP
- **rvec** (numpy.ndarray): Rotation vector (3x1)
- **tvec** (numpy.ndarray): Translation vector (3x1)
- **T** (numpy.ndarray): 4x4 homogeneous transformation matrix

#### Coordinate Frames and Transformations

**CRITICAL**: The returned transformation matrix represents **Camera-to-Tag** transformation.

##### Tag Coordinate Frame (Target Frame)
- **Origin**: Center of the AprilTag
- **X-axis**: Points to the right when viewing the tag (→)
- **Y-axis**: Points upward when viewing the tag (↑)  
- **Z-axis**: Points out of the tag plane toward the camera (⊙)
- **Convention**: Right-handed coordinate system

##### Camera Coordinate Frame (Source Frame)
- **Origin**: Camera optical center
- **X-axis**: Points right in image plane (→)
- **Y-axis**: Points down in image plane (↓)
- **Z-axis**: Points forward (into the scene) (→)
- **Convention**: OpenCV right-handed coordinate system

##### Transformation Matrix T
```
T_camera_to_tag = [[R11, R12, R13, tx],
                   [R21, R22, R23, ty], 
                   [R31, R32, R33, tz],
                   [0,   0,   0,   1 ]]
```

**Usage**: `point_in_tag_frame = T @ point_in_camera_frame`

**To get Tag-to-Camera transformation**: `T_tag_to_camera = np.linalg.inv(T)`

#### Example
```python
for detection in detections:
    success, rvec, tvec, T = detector.get_pose(detection)
    if success:
        print(f"Tag {detection['id']} pose estimated successfully")
```

### `transformation(self, rvec, tvec)`

Converts rotation and translation vectors to a 4x4 homogeneous transformation matrix.

#### Parameters
- **rvec** (numpy.ndarray): Rotation vector from cv2.solvePnP
- **tvec** (numpy.ndarray): Translation vector from cv2.solvePnP

#### Returns
- **transformation_matrix** (numpy.ndarray): 4x4 homogeneous transformation matrix

### `euler_angles(self, rvec)`

Converts rotation vector to Euler angles representation.

#### Parameters
- **rvec** (numpy.ndarray): Rotation vector

#### Returns
- **angles** (numpy.ndarray): [yaw, pitch, roll] in degrees
  - **Yaw**: Rotation about Y-axis
  - **Pitch**: Rotation about X-axis  
  - **Roll**: Rotation about Z-axis

#### Note
Handles singular cases when sy < 1e-6 to avoid numerical instability.

### `distance(self, tvec)`

Calculates Euclidean distance from the translation vector.

#### Parameters
- **tvec** (numpy.ndarray): Translation vector

#### Returns
- **distance** (float): Euclidean distance magnitude

### `draw(self, rvec, tvec, corners, image, tag_id)`

Draws detection visualization on the image including tag outline, pose information, and coordinate axes.

#### Parameters
- **rvec** (numpy.ndarray): Rotation vector
- **tvec** (numpy.ndarray): Translation vector
- **corners** (numpy.ndarray): Tag corner coordinates
- **image** (numpy.ndarray): Image to draw on
- **tag_id** (int): Tag ID number

#### Returns
- **image** (numpy.ndarray): Image with visualization overlay

#### Visualization Elements
- Green outline around detected tag
- Tag ID and distance information
- Euler angles (yaw, pitch, roll)
- 3D coordinate axes

## Usage Example

```python
import cv2
import numpy as np
from src.detection.tag_detector import TagDetector

# Camera calibration parameters
camera_params = {
    'camera_matrix': np.array([[800, 0, 320],
                              [0, 800, 240],
                              [0, 0, 1]], dtype=np.float32),
    'dist_coeffs': np.array([0.1, -0.2, 0, 0, 0], dtype=np.float32)
}

# Initialize detector
detector = TagDetector(camera_params, tag_size=0.06)

# Load image
image = cv2.imread('test_image.jpg')

# Detect tags
detections = detector.detect(image)

# Process each detection
for detection in detections:
    tag_id = detection['id']
    corners = np.array(detection['lb-rb-rt-lt'], dtype=np.float32)
    
    # Get pose
    success, rvec, tvec, T = detector.get_pose(detection)
    
    if success:
        # Get additional information
        euler = detector.euler_angles(rvec)
        dist = detector.distance(tvec)
        
        print(f"Tag {tag_id}:")
        print(f"  Distance: {dist:.3f} meters")
        print(f"  Euler angles: {euler}")
        print(f"  Translation: {tvec.flatten()}")
        
        # Draw visualization
        image = detector.draw(rvec, tvec, corners, image, tag_id)

# Display result
cv2.imshow('AprilTag Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## Technical Notes

### Coordinate Frame Conventions

#### Image Coordinate System (2D Pixel Space)
- **Origin**: Top-left corner of image
- **u-axis**: Horizontal, pointing right (→)
- **v-axis**: Vertical, pointing down (↓)
- **Units**: Pixels

#### Camera Coordinate System (3D Physical Space)
- **Origin**: Camera optical center (principal point projected to 3D)
- **X-axis**: Right in image plane (→)
- **Y-axis**: Down in image plane (↓)
- **Z-axis**: Forward into scene (→)
- **Units**: Meters (or whatever units tag_size is specified in)
- **Convention**: OpenCV right-handed system

#### Tag Coordinate System (3D Physical Space)
- **Origin**: Physical center of AprilTag
- **X-axis**: Right when viewing tag from front (→)
- **Y-axis**: Up when viewing tag from front (↑)
- **Z-axis**: Out of tag plane toward camera (⊙)
- **Units**: Meters (same as camera frame)
- **Convention**: Standard right-handed system

#### World Coordinate System (SLAM Context)
- **Definition**: Typically defined by first observed tag or manually set
- **Usage**: All tag poses and camera trajectory expressed relative to this frame
- **Transformation**: Camera-to-World = Tag-to-World @ Camera-to-Tag^(-1)

### Performance Considerations
- The detector works on grayscale images for optimal performance
- Tag size parameter must match the physical tag dimensions for accurate pose estimation
- Camera calibration quality directly affects pose estimation accuracy

### Error Handling
- The `get_pose()` method returns a success flag that should be checked
- Singular rotation matrices are handled in `euler_angles()` method
- Invalid detections may occur with poor lighting or image quality

## Integration with SLAM System

The TagDetector is designed to integrate with the broader AprilSLAM system:
- Detections provide landmark observations for SLAM algorithms
- Pose estimates contribute to camera localization
- Tag IDs enable consistent landmark identification across frames
- Transformation matrices facilitate coordinate frame conversions

## Configuration

Tag detection parameters can be tuned by:
- Adjusting the tag family type for different tag designs
- Setting appropriate tag sizes for accurate scale estimation
- Ensuring proper camera calibration for optimal pose accuracy 