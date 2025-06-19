#!/usr/bin/env python3
"""
Ground Truth Calculator for AprilSLAM Simulation

This module provides ground truth pose calculations and coordinate transformations
for evaluating SLAM performance against known true positions and orientations.

Author: Mikolaj Kostrzewa
License: Creative Commons Attribution-NonCommercial 4.0 International
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional

from .config_manager import SimulationConfig
from .renderer import TagData


class GroundTruthCalculator:
    """
    Calculates ground truth poses and transformations for SLAM evaluation.
    
    This class handles:
    - Camera-to-tag transformation calculations
    - World coordinate system transformations
    - Tag-to-tag distance measurements
    - Coordinate frame conversions (OpenGL to SLAM coordinates)
    - Rotation matrix and Euler angle conversions
    """
    
    def __init__(self, config: SimulationConfig, tags_data: List[TagData]):
        """
        Initialize the ground truth calculator.
        
        Args:
            config (SimulationConfig): Simulation configuration
            tags_data (List[TagData]): List of tag data with positions and rotations
        """
        self.config = config
        self.tags_data = tags_data
        
        # Create lookup dictionary for faster tag access
        self.tags_dict = {tag.id: tag for tag in tags_data}
        
        logging.info("✓ Ground truth calculator initialized")
    
    def get_camera_to_tag_transform(self, tag_id: int, camera_position: np.ndarray) -> np.ndarray:
        """
        Calculate transformation matrix from camera to a specific tag.
        
        This represents what the SLAM system should estimate when viewing the tag.
        The transformation accounts for OpenGL coordinate system conversions.
        
        Args:
            tag_id (int): ID of the target tag
            camera_position (np.ndarray): Current camera position [x, y, z]
            
        Returns:
            np.ndarray: 4x4 transformation matrix from camera to tag
            
        Raises:
            ValueError: If tag ID is not found
        """
        if tag_id not in self.tags_dict:
            raise ValueError(f"Tag {tag_id} not found in configuration")
        
        tag = self.tags_dict[tag_id]
        
        # Calculate relative position (tag position relative to camera)
        relative_position = tag.position - camera_position
        
        # Apply OpenGL coordinate system flip (y and z axes)
        relative_position[1:] = -relative_position[1:]
        
        # Create rotation matrix from tag's Euler angles
        rotation_matrix = self._euler_to_rotation_matrix(tag.rotation)
        
        # Apply coordinate system transformation
        flip_matrix = np.array([[1, 0, 0],
                               [0, -1, 0],
                               [0, 0, -1]])
        rotation_matrix = flip_matrix @ rotation_matrix
        
        # Construct 4x4 transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = rotation_matrix
        transform[:3, 3] = relative_position
        
        return transform
    
    def get_tag_world_transform(self, tag_id: int, camera_position: np.ndarray,
                               coordinate_frame_tag_id: int) -> np.ndarray:
        """
        Calculate transformation from a tag to world coordinates.
        
        Args:
            tag_id (int): ID of the target tag
            camera_position (np.ndarray): Current camera position [x, y, z]
            coordinate_frame_tag_id (int): ID of the tag used as coordinate frame origin
            
        Returns:
            np.ndarray: 4x4 transformation matrix from tag to world coordinates
        """
        # Get the base transformation
        tag_transform = self.get_camera_to_tag_transform(tag_id, camera_position)
        
        # Get the coordinate frame transformation
        coord_frame_transform = self.get_camera_to_tag_transform(coordinate_frame_tag_id, camera_position)
        
        # Calculate world transformation
        world_transform = tag_transform @ coord_frame_transform
        
        return world_transform
    
    def get_tag_to_tag_distance(self, tag1_id: int, tag2_id: int, camera_position: np.ndarray) -> float:
        """
        Calculate the distance between two tags as seen from the camera.
        
        Args:
            tag1_id (int): ID of the first tag
            tag2_id (int): ID of the second tag
            camera_position (np.ndarray): Current camera position [x, y, z]
            
        Returns:
            float: Distance between the two tags
            
        Raises:
            ValueError: If either tag ID is not found
        """
        if tag1_id not in self.tags_dict or tag2_id not in self.tags_dict:
            raise ValueError(f"One or both tags not found: {tag1_id}, {tag2_id}")
        
        tag1 = self.tags_dict[tag1_id]
        tag2 = self.tags_dict[tag2_id]
        
        # Calculate relative positions from camera
        tag1_relative = tag1.position - camera_position
        tag2_relative = tag2.position - camera_position
        
        # Calculate Euclidean distance
        distance = np.linalg.norm(tag1_relative - tag2_relative)
        
        return distance
    
    def get_inverse_transform(self, tag_id: int, camera_position: np.ndarray) -> np.ndarray:
        """
        Calculate the inverse transformation (tag to camera).
        
        This represents the camera pose in the tag's coordinate frame.
        
        Args:
            tag_id (int): ID of the reference tag
            camera_position (np.ndarray): Current camera position [x, y, z]
            
        Returns:
            np.ndarray: 4x4 inverse transformation matrix
        """
        if tag_id not in self.tags_dict:
            raise ValueError(f"Tag {tag_id} not found in configuration")
        
        tag = self.tags_dict[tag_id]
        
        # Calculate relative position
        relative_position = tag.position - camera_position
        
        # Apply coordinate system flip
        relative_position[1:] = -relative_position[1:]
        
        # Create rotation matrix
        rotation_matrix = self._euler_to_rotation_matrix(tag.rotation)
        
        # Apply coordinate system transformation
        flip_matrix = np.array([[1, 0, 0],
                               [0, -1, 0],
                               [0, 0, -1]])
        rotation_matrix = flip_matrix @ rotation_matrix
        
        # Calculate inverse transformation
        inverse_rotation = rotation_matrix.T  # Transpose for orthogonal matrices
        inverse_translation = -inverse_rotation @ relative_position
        
        # Construct inverse transformation matrix
        inverse_transform = np.eye(4)
        inverse_transform[:3, :3] = inverse_rotation
        inverse_transform[:3, 3] = inverse_translation
        
        return inverse_transform
    
    def convert_simulation_to_mm(self, value: float) -> float:
        """
        Convert simulation units to millimeters.
        
        Args:
            value (float): Value in simulation units
            
        Returns:
            float: Value in millimeters
        """
        return self.config.simulation_units_to_mm(value)
    
    def convert_mm_to_simulation(self, value_mm: float) -> float:
        """
        Convert millimeters to simulation units.
        
        Args:
            value_mm (float): Value in millimeters
            
        Returns:
            float: Value in simulation units
        """
        return self.config.mm_to_simulation_units(value_mm)
    
    @staticmethod
    def rotation_matrix_to_euler(rotation_matrix: np.ndarray) -> np.ndarray:
        """
        Convert rotation matrix to Euler angles (roll, pitch, yaw).
        
        Args:
            rotation_matrix (np.ndarray): 3x3 rotation matrix
            
        Returns:
            np.ndarray: Euler angles [roll, pitch, yaw] in radians
        """
        R = rotation_matrix
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        
        singular = sy < 1e-6
        
        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])  # roll
            y = np.arctan2(-R[2, 0], sy)      # pitch
            z = np.arctan2(R[1, 0], R[0, 0])  # yaw
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
        
        return np.array([x, y, z])
    
    @staticmethod
    def _euler_to_rotation_matrix(euler_angles: np.ndarray) -> np.ndarray:
        """
        Convert Euler angles to rotation matrix.
        
        Args:
            euler_angles (np.ndarray): Euler angles [roll, pitch, yaw] in degrees
            
        Returns:
            np.ndarray: 3x3 rotation matrix
        """
        # Convert degrees to radians
        angles_rad = np.radians(euler_angles)
        roll, pitch, yaw = angles_rad
        
        # Create individual rotation matrices
        Rx = np.array([[1, 0, 0],
                      [0, np.cos(roll), -np.sin(roll)],
                      [0, np.sin(roll), np.cos(roll)]])
        
        Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                      [0, 1, 0],
                      [-np.sin(pitch), 0, np.cos(pitch)]])
        
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                      [np.sin(yaw), np.cos(yaw), 0],
                      [0, 0, 1]])
        
        # Combined rotation matrix (ZYX order)
        rotation_matrix = Rz @ Ry @ Rx
        
        return rotation_matrix
    
    def calculate_pose_error(self, estimated_transform: np.ndarray, 
                           ground_truth_transform: np.ndarray) -> Tuple[float, float]:
        """
        Calculate pose estimation errors.
        
        Args:
            estimated_transform (np.ndarray): 4x4 estimated transformation matrix
            ground_truth_transform (np.ndarray): 4x4 ground truth transformation matrix
            
        Returns:
            Tuple[float, float]: (translation_error, rotation_error)
        """
        # Extract translation vectors
        est_translation = estimated_transform[:3, 3]
        gt_translation = ground_truth_transform[:3, 3]
        
        # Extract rotation matrices
        est_rotation = estimated_transform[:3, :3]
        gt_rotation = ground_truth_transform[:3, :3]
        
        # Calculate translation error (Euclidean distance)
        translation_error = np.linalg.norm(est_translation - gt_translation)
        
        # Calculate rotation error (Frobenius norm of difference)
        rotation_error = np.linalg.norm(est_rotation - gt_rotation, 'fro')
        
        return translation_error, rotation_error
    
    def get_all_tag_distances_from_camera(self, camera_position: np.ndarray) -> Dict[int, float]:
        """
        Get distances from camera to all tags.
        
        Args:
            camera_position (np.ndarray): Current camera position [x, y, z]
            
        Returns:
            Dict[int, float]: Dictionary mapping tag IDs to distances
        """
        distances = {}
        
        for tag in self.tags_data:
            distance = np.linalg.norm(tag.position - camera_position)
            distances[tag.id] = distance
        
        return distances
    
    def get_closest_tag(self, camera_position: np.ndarray) -> Tuple[int, float]:
        """
        Find the closest tag to the camera.
        
        Args:
            camera_position (np.ndarray): Current camera position [x, y, z]
            
        Returns:
            Tuple[int, float]: (tag_id, distance) of the closest tag
        """
        distances = self.get_all_tag_distances_from_camera(camera_position)
        
        if not distances:
            raise ValueError("No tags available")
        
        closest_tag_id = min(distances.keys(), key=lambda k: distances[k])
        closest_distance = distances[closest_tag_id]
        
        return closest_tag_id, closest_distance
    
    def validate_tag_visibility(self, tag_id: int, camera_position: np.ndarray,
                               max_distance: float = 10.0) -> bool:
        """
        Check if a tag should be visible from the current camera position.
        
        Args:
            tag_id (int): Tag ID to check
            camera_position (np.ndarray): Current camera position [x, y, z]
            max_distance (float): Maximum visibility distance
            
        Returns:
            bool: True if tag should be visible, False otherwise
        """
        if tag_id not in self.tags_dict:
            return False
        
        tag = self.tags_dict[tag_id]
        distance = np.linalg.norm(tag.position - camera_position)
        
        return distance <= max_distance
    
    def __str__(self) -> str:
        """String representation of ground truth calculator."""
        return f"GroundTruthCalculator(tags={len(self.tags_data)})"
    
    def __repr__(self) -> str:
        """Developer representation of ground truth calculator."""
        return f"GroundTruthCalculator(config={self.config}, tags_count={len(self.tags_data)})" 