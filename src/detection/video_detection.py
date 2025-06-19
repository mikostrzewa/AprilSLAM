#!/usr/bin/env python3
"""
AprilTag Video Detection System

This module provides real-time AprilTag detection and pose estimation using a webcam feed.
It uses the TagDetector class for robust tag detection and displays detected tags with
pose information overlaid on the video stream.

Features:
- Real-time AprilTag detection from webcam
- 6DOF pose estimation (position and orientation)
- Visual overlay with tag information
- Distance calculation and euler angle conversion
- Camera calibration parameter loading
- Clean exit handling

Usage:
    python video_detection.py

Controls:
    - 'q' key: Quit the application
    - ESC key: Alternative quit method

Requirements:
    - Camera calibration file in data/calibration/camera_calibration_parameters.npz
    - Working webcam connected to the system
    - TagDetector module in the same package

Author: Mikolaj Kostrzewa
License: Creative Commons Attribution-NonCommercial 4.0 International
"""

import numpy as np
import cv2
import sys
import os

# Import the TagDetector class from the local module
from tag_detector import TagDetector

def load_camera_calibration():
    """
    Load camera calibration parameters from saved file.
    
    Returns:
        dict: Dictionary containing camera_matrix and dist_coeffs
        
    Raises:
        FileNotFoundError: If calibration file doesn't exist
        KeyError: If required keys are missing from calibration file
    """
    # Construct path to calibration file relative to this script
    calibration_path = os.path.join(
        os.path.dirname(__file__), '..', '..', 
        'data', 'calibration', 'camera_calibration_parameters.npz'
    )
    
    try:
        with np.load(calibration_path) as data:
            camera_params = {
                'camera_matrix': data['camera_matrix'],
                'dist_coeffs': data['dist_coeffs']
            }
        print(f"✓ Loaded camera calibration from: {calibration_path}")
        return camera_params
    except FileNotFoundError:
        print(f"❌ Camera calibration file not found: {calibration_path}")
        print("Please run camera calibration first: python src/calibration/calibrate.py")
        raise
    except KeyError as e:
        print(f"❌ Missing calibration parameter: {e}")
        print("Please re-run camera calibration to generate complete parameters")
        raise


def initialize_camera(camera_id=0):
    """
    Initialize and configure the camera for video capture.
    
    Args:
        camera_id (int): Camera device ID (0 for default camera)
        
    Returns:
        cv2.VideoCapture: Configured camera capture object
        
    Raises:
        RuntimeError: If camera cannot be opened or configured
    """
    print(f"🎥 Initializing camera {camera_id}...")
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        # Try alternative camera IDs
        for alt_id in [1, 2]:
            print(f"   Trying camera {alt_id}...")
            cap = cv2.VideoCapture(alt_id)
            if cap.isOpened():
                print(f"✓ Camera {alt_id} opened successfully")
                break
        else:
            raise RuntimeError("❌ Could not open any camera. Please check camera connection.")
    else:
        print(f"✓ Camera {camera_id} opened successfully")
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    return cap


def process_detections(detector, detections, frame):
    """
    Process all detected AprilTags and add visual overlays to the frame.
    
    Args:
        detector (TagDetector): Configured tag detector instance
        detections (list): List of detected tags from detector
        frame (np.ndarray): Video frame to draw on
        
    Returns:
        np.ndarray: Frame with detection overlays added
    """
    if not detections:
        # Display "No tags detected" message
        cv2.putText(frame, "No AprilTags detected", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame
    
    # Process each detected tag
    for detection in detections:
        tag_id = detection['id']
        
        # Get pose estimation for this tag
        retval, rvec, tvec, transformation_matrix = detector.get_pose(detection)
        
        if retval:  # Pose estimation successful
            # Extract corner points for drawing
            corners = np.array(detection['lb-rb-rt-lt'], dtype=np.float32)
            
            # Draw detection visualization on frame
            frame = detector.draw(rvec, tvec, corners, frame, tag_id)
            
            # Calculate additional metrics
            distance_m = detector.distance(tvec)  # Distance in meters
            distance_mm = distance_m * 1000       # Convert to millimeters
            yaw, pitch, roll = detector.euler_angles(rvec)
            
            # Print detailed information to console
            print(f"📍 Tag ID {tag_id}:")
            print(f"   Position (x,y,z): ({tvec[0][0]:.3f}, {tvec[1][0]:.3f}, {tvec[2][0]:.3f}) m")
            print(f"   Distance: {distance_mm:.1f} mm")
            print(f"   Orientation - Yaw: {yaw:.1f}°, Pitch: {pitch:.1f}°, Roll: {roll:.1f}°")
            print(f"   Corners: {corners.astype(int).tolist()}")
            print("   " + "-" * 50)
        else:
            # Pose estimation failed - still draw basic detection
            corners = np.array(detection['lb-rb-rt-lt'], dtype=np.float32)
            for i in range(4):
                pt1 = tuple(map(int, corners[i]))
                pt2 = tuple(map(int, corners[(i + 1) % 4]))
                cv2.line(frame, pt1, pt2, (0, 0, 255), 2)  # Red for failed pose
            
            cv2.putText(frame, f'ID: {tag_id} (Pose Failed)', 
                       (int(corners[0][0]), int(corners[0][1]) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            print(f"⚠️  Tag ID {tag_id}: Detection OK, but pose estimation failed")
    
    return frame


def add_info_overlay(frame, fps=None):
    """
    Add informational overlay to the video frame.
    
    Args:
        frame (np.ndarray): Video frame to add overlay to
        fps (float, optional): Current FPS to display
        
    Returns:
        np.ndarray: Frame with info overlay added
    """
    height, width = frame.shape[:2]
    
    # Create semi-transparent overlay area
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, height - 100), (300, height - 10), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Add text information
    y_offset = height - 80
    cv2.putText(frame, "AprilTag Detection System", (15, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    y_offset += 20
    cv2.putText(frame, "Press 'q' to quit", (15, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    if fps is not None:
        y_offset += 20
        cv2.putText(frame, f"FPS: {fps:.1f}", (15, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return frame


def main():
    """
    Main function to run the AprilTag video detection system.
    
    This function orchestrates the entire detection pipeline:
    1. Load camera calibration parameters
    2. Initialize the camera and tag detector
    3. Run the main detection loop
    4. Handle cleanup on exit
    """
    try:
        # === INITIALIZATION PHASE ===
        print("🚀 Starting AprilTag Video Detection System")
        print("=" * 50)
        
        # Load camera calibration parameters
        camera_params = load_camera_calibration()
        
        # Initialize camera
        cap = initialize_camera(camera_id=0)  # Try camera 0 first
        
        # Initialize AprilTag detector
        print("🔍 Initializing AprilTag detector...")
        detector = TagDetector(
            camera_params=camera_params,
            tag_type="tagStandard41h12",  # Standard AprilTag family
            tag_size=0.06  # 6cm tags (adjust based on your actual tag size)
        )
        print("✓ TagDetector initialized successfully")
        
        # === DETECTION LOOP ===
        print("\n🎯 Starting detection loop...")
        print("Controls: Press 'q' to quit, ESC for emergency exit")
        print("=" * 50)
        
        frame_count = 0
        fps_timer = cv2.getTickCount()
        
        while True:
            # Capture frame from camera
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to capture frame from camera")
                break
            
            # Detect AprilTags in the current frame
            detections = detector.detect(frame)
            
            # Process detections and add visual overlays
            frame = process_detections(detector, detections, frame)
            
            # Calculate and display FPS every 30 frames
            frame_count += 1
            if frame_count % 30 == 0:
                current_time = cv2.getTickCount()
                fps = 30.0 / ((current_time - fps_timer) / cv2.getTickFrequency())
                fps_timer = current_time
                frame_count = 0
            else:
                fps = None
            
            # Add informational overlay
            frame = add_info_overlay(frame, fps)
            
            # Display the frame
            cv2.imshow('AprilTag Video Detection', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' key or ESC
                print("\n👋 Exiting detection system...")
                break
                
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        raise
    finally:
        # === CLEANUP PHASE ===
        print("🧹 Cleaning up resources...")
        if 'cap' in locals():
            cap.release()
            print("✓ Camera released")
        cv2.destroyAllWindows()
        print("✓ Windows closed")
        print("✅ Cleanup complete")


# Entry point when script is run directly
if __name__ == "__main__":
    main()
