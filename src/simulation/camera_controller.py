#!/usr/bin/env python3
"""
Camera Controller for AprilSLAM Simulation

This module provides camera movement and keyboard input handling for the simulation.
It supports both manual control via keyboard input and automatic Monte Carlo
position randomization for testing purposes.

Author: Mikolaj Kostrzewa
License: Creative Commons Attribution-NonCommercial 4.0 International
"""

import numpy as np
import pygame
from typing import Dict, Tuple, Optional
import logging


class CameraController:
    """
    Handles camera movement and keyboard input for the simulation.
    
    This class manages:
    - Camera position and rotation state
    - Keyboard input processing
    - Movement and rotation calculations
    - Monte Carlo position randomization
    - Movement bounds checking
    """
    
    def __init__(self, movement_enabled: bool = True, movement_speed: float = 1.0, 
                 rotation_speed: float = 1.0, size_scale: float = 1.0):
        """
        Initialize the camera controller.
        
        Args:
            movement_enabled (bool): Whether manual movement is enabled
            movement_speed (float): Movement speed multiplier
            rotation_speed (float): Rotation speed in degrees per update
            size_scale (float): Size scaling factor for movement speed
        """
        # Camera state
        self.position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.rotation = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # [pitch, yaw, roll]
        
        # Movement configuration
        self.movement_enabled = movement_enabled
        self.movement_speed = movement_speed * size_scale
        self.rotation_speed = rotation_speed
        
        # Keyboard state tracking
        self.key_state: Dict[int, bool] = {
            # Translation keys
            pygame.K_LEFT: False,   pygame.K_RIGHT: False,
            pygame.K_UP: False,     pygame.K_DOWN: False,
            pygame.K_w: False,      pygame.K_s: False,
            # Rotation keys
            pygame.K_a: False,      pygame.K_d: False,  # yaw
            pygame.K_q: False,      pygame.K_e: False,  # roll
            pygame.K_r: False,      pygame.K_f: False   # pitch
        }
        
        logging.info("✓ Camera controller initialized")
    
    def handle_key_event(self, event: pygame.event.Event) -> None:
        """
        Process pygame keyboard events.
        
        Args:
            event (pygame.event.Event): Pygame event to process
        """
        if event.type == pygame.KEYDOWN:
            if event.key in self.key_state:
                self.key_state[event.key] = True
        elif event.type == pygame.KEYUP:
            if event.key in self.key_state:
                self.key_state[event.key] = False
    
    def update_manual_movement(self) -> None:
        """
        Update camera position and rotation based on current key states.
        
        This method processes the current keyboard state and updates the camera
        position and rotation accordingly. Only works if movement_enabled is True.
        """
        if not self.movement_enabled:
            return
        
        # Update position based on arrow keys and W/S
        if self.key_state[pygame.K_LEFT]:   self.position[0] -= self.movement_speed
        if self.key_state[pygame.K_RIGHT]:  self.position[0] += self.movement_speed
        if self.key_state[pygame.K_UP]:     self.position[1] += self.movement_speed
        if self.key_state[pygame.K_DOWN]:   self.position[1] -= self.movement_speed
        if self.key_state[pygame.K_w]:      self.position[2] -= self.movement_speed
        if self.key_state[pygame.K_s]:      self.position[2] += self.movement_speed
        
        # Update rotation based on A/D (yaw), Q/E (roll), R/F (pitch)
        if self.key_state[pygame.K_a]:      self.rotation[1] -= self.rotation_speed  # yaw left
        if self.key_state[pygame.K_d]:      self.rotation[1] += self.rotation_speed  # yaw right
        if self.key_state[pygame.K_q]:      self.rotation[2] -= self.rotation_speed  # roll left
        if self.key_state[pygame.K_e]:      self.rotation[2] += self.rotation_speed  # roll right
        if self.key_state[pygame.K_r]:      self.rotation[0] += self.rotation_speed  # pitch up
        if self.key_state[pygame.K_f]:      self.rotation[0] -= self.rotation_speed  # pitch down
    
    def randomize_position(self, bounds: np.ndarray) -> None:
        """
        Randomize camera position within specified bounds (Monte Carlo).
        
        Args:
            bounds (np.ndarray): Array of [x_min, x_max, y_min, y_max, z_min, z_max]
        """
        if len(bounds) != 6:
            raise ValueError("Bounds must contain exactly 6 values: [x_min, x_max, y_min, y_max, z_min, z_max]")
        
        x_min, x_max, y_min, y_max, z_min, z_max = bounds
        
        self.position[0] = np.random.uniform(x_min, x_max)
        self.position[1] = np.random.uniform(y_min, y_max)
        self.position[2] = np.random.uniform(z_min, z_max)
        
        logging.debug(f"Camera position randomized to: {self.position}")
    
    def set_position(self, position: np.ndarray) -> None:
        """
        Set camera position directly.
        
        Args:
            position (np.ndarray): New position as [x, y, z]
        """
        if len(position) != 3:
            raise ValueError("Position must be a 3D vector")
        self.position = np.array(position, dtype=np.float32)
    
    def set_rotation(self, rotation: np.ndarray) -> None:
        """
        Set camera rotation directly.
        
        Args:
            rotation (np.ndarray): New rotation as [pitch, yaw, roll] in degrees
        """
        if len(rotation) != 3:
            raise ValueError("Rotation must be a 3D vector")
        self.rotation = np.array(rotation, dtype=np.float32)
    
    def get_position(self) -> np.ndarray:
        """
        Get current camera position.
        
        Returns:
            np.ndarray: Current position as [x, y, z]
        """
        return self.position.copy()
    
    def get_rotation(self) -> np.ndarray:
        """
        Get current camera rotation.
        
        Returns:
            np.ndarray: Current rotation as [pitch, yaw, roll] in degrees
        """
        return self.rotation.copy()
    
    def get_transformation_matrix(self) -> np.ndarray:
        """
        Get the camera transformation matrix.
        
        Returns:
            np.ndarray: 4x4 transformation matrix representing camera pose
        """
        # Convert degrees to radians
        pitch_rad, yaw_rad, roll_rad = np.radians(self.rotation)
        
        # Create rotation matrices
        rx = np.array([[1, 0, 0],
                       [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
                       [0, np.sin(pitch_rad), np.cos(pitch_rad)]])
        
        ry = np.array([[np.cos(yaw_rad), 0, np.sin(yaw_rad)],
                       [0, 1, 0],
                       [-np.sin(yaw_rad), 0, np.cos(yaw_rad)]])
        
        rz = np.array([[np.cos(roll_rad), -np.sin(roll_rad), 0],
                       [np.sin(roll_rad), np.cos(roll_rad), 0],
                       [0, 0, 1]])
        
        # Combined rotation matrix (order: yaw * pitch * roll)
        rotation_matrix = ry @ rx @ rz
        
        # Create 4x4 transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = rotation_matrix
        transform[:3, 3] = self.position
        
        return transform
    
    def reset(self) -> None:
        """Reset camera to origin with no rotation."""
        self.position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.rotation = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        logging.info("Camera reset to origin")
    
    def enable_movement(self, enabled: bool = True) -> None:
        """
        Enable or disable manual movement.
        
        Args:
            enabled (bool): Whether to enable manual movement
        """
        self.movement_enabled = enabled
        logging.info(f"Manual movement {'enabled' if enabled else 'disabled'}")
    
    def set_movement_speed(self, speed: float) -> None:
        """
        Set movement speed.
        
        Args:
            speed (float): New movement speed multiplier
        """
        self.movement_speed = speed
        logging.info(f"Movement speed set to: {speed}")
    
    def set_rotation_speed(self, speed: float) -> None:
        """
        Set rotation speed.
        
        Args:
            speed (float): New rotation speed in degrees per update
        """
        self.rotation_speed = speed
        logging.info(f"Rotation speed set to: {speed}")
    
    def get_status_info(self) -> Dict[str, any]:
        """
        Get comprehensive status information about the camera.
        
        Returns:
            Dict[str, any]: Dictionary containing camera status information
        """
        return {
            'position': self.position.tolist(),
            'rotation': self.rotation.tolist(),
            'movement_enabled': self.movement_enabled,
            'movement_speed': self.movement_speed,
            'rotation_speed': self.rotation_speed,
            'active_keys': [key for key, pressed in self.key_state.items() if pressed]
        }
    
    def __str__(self) -> str:
        """String representation of camera state."""
        return (f"Camera(pos={self.position}, rot={self.rotation}, "
                f"movement={'enabled' if self.movement_enabled else 'disabled'})")
    
    def __repr__(self) -> str:
        """Developer representation of camera controller."""
        return (f"CameraController(movement_enabled={self.movement_enabled}, "
                f"movement_speed={self.movement_speed}, rotation_speed={self.rotation_speed})") 