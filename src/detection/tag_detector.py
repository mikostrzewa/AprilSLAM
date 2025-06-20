import numpy as np
import cv2
import sys
import os

# Add the apriltag build directory to Python path
apriltag_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'lib', 'apriltag', 'build'))
if apriltag_path not in sys.path:
    sys.path.insert(0, apriltag_path)

from apriltag import apriltag


class TagDetector:
    """Handles AprilTag detection and pose estimation"""
    
    def __init__(self, camera_params, tag_type="tagStandard41h12", tag_size=0.06):
        self.detector = apriltag(tag_type)
        self.tag_size = tag_size
        self.camera_matrix = camera_params['camera_matrix']
        self.dist_coeffs = camera_params['dist_coeffs']
    
    def detect(self, image):
        """Detect AprilTags in the given image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detections = self.detector.detect(gray)
        detections = sorted(detections, key=lambda x: x['id'])
        return detections
    
    def get_pose(self, detection):
        """Estimate the pose of a detected tag"""
        corners = np.array(detection['lb-rb-rt-lt'], dtype=np.float32)

        # Define the 3D coordinates of the tag's corners in the tag's coordinate frame
        obj_points = np.array([[-self.tag_size / 2, -self.tag_size / 2, 0],
                            [ self.tag_size / 2, -self.tag_size / 2, 0],
                            [ self.tag_size / 2,  self.tag_size / 2, 0],
                            [-self.tag_size / 2,  self.tag_size / 2, 0]], dtype=np.float32)

        # Estimate the pose of the tag
        retval, rvec, tvec = cv2.solvePnP(obj_points, corners, self.camera_matrix, self.dist_coeffs)
        T = self.transformation(rvec, tvec)
        return retval, rvec, tvec, T
    
    def transformation(self, rvec, tvec):
        """Convert rotation vector and translation vector to transformation matrix"""
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix
        # Convert tvec from 2D array (3x1) to 1D array (3,) for proper assignment
        transformation_matrix[:3, 3] = tvec.flatten()
        return transformation_matrix
    
    def euler_angles(self, rvec):
        """Convert rotation vector to Euler angles"""
        rot_matrix, _ = cv2.Rodrigues(rvec)
        sy = np.sqrt(rot_matrix[0, 0] ** 2 + rot_matrix[1, 0] ** 2)
        singular = sy < 1e-6
        if not singular:
            yaw = np.arctan2(rot_matrix[0, 2], rot_matrix[2, 2])  # Yaw (rotation about Y-axis)
            pitch = np.arctan2(-rot_matrix[1, 2], sy)             # Pitch (rotation about X-axis)
            roll = np.arctan2(rot_matrix[1, 0], rot_matrix[1, 1]) # Roll (rotation about Z-axis)
        else:
            yaw = np.arctan2(-rot_matrix[2, 0], rot_matrix[0, 0]) # Yaw (rotation about Y-axis)
            pitch = np.arctan2(-rot_matrix[1, 2], sy)             # Pitch (rotation about X-axis)
            roll = 0                                             # Roll (rotation about Z-axis)

        return np.degrees([yaw, pitch, roll])
    
    def distance(self, tvec):
        """Calculate distance from translation vector"""
        return np.linalg.norm(tvec)
    
    def draw(self, rvec, tvec, corners, image, tag_id):
        """Draw detection results on the image"""
        for i in range(4):
            pt1 = tuple(map(int, corners[i]))
            pt2 = tuple(map(int, corners[(i + 1) % 4]))
            cv2.line(image, pt1, pt2, (0, 255, 0), 2)
        
        yaw, pitch, roll = self.euler_angles(rvec)
        distance_tvec = self.distance(tvec)
        text = f'ID: {tag_id}, Dist: {distance_tvec:.1f} units'
        text2 = f'Yaw: {yaw:.1f}, Pitch: {pitch:.1f}, Roll: {roll:.1f}'
        cv2.putText(image, text, (pt1[0], pt1[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        cv2.putText(image, text2, (pt1[0], pt1[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        cv2.drawFrameAxes(image, self.camera_matrix, self.dist_coeffs, rvec, tvec, 2)
        return image 