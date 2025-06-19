#!/usr/bin/env python3
"""
OpenGL Renderer for AprilSLAM Simulation

This module handles all OpenGL rendering operations including window management,
texture loading, tag rendering, and camera transformations for the simulation.

Author: Mikolaj Kostrzewa
License: Creative Commons Attribution-NonCommercial 4.0 International
"""

import os
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import cv2
import logging
from typing import List, Dict, Tuple, Any, Optional

from .config_manager import SimulationConfig


class TagData:
    """Represents a single AprilTag with its texture and transform data."""
    
    def __init__(self, tag_id: int, texture: int, position: np.ndarray, rotation: np.ndarray):
        """
        Initialize tag data.
        
        Args:
            tag_id (int): Unique tag identifier
            texture (int): OpenGL texture ID
            position (np.ndarray): 3D position vector
            rotation (np.ndarray): 3D rotation vector in degrees
        """
        self.id = tag_id
        self.texture = texture
        self.position = position.copy()
        self.rotation = rotation.copy()
    
    def get_world_z(self) -> float:
        """Get world Z coordinate for depth sorting."""
        return self.position[2]


class SimulationRenderer:
    """
    Handles all OpenGL rendering operations for the AprilSLAM simulation.
    
    This class manages:
    - OpenGL context initialization
    - Texture loading and management
    - Tag rendering with proper transformations
    - Camera view matrix application
    - Frame capture for SLAM processing
    """
    
    def __init__(self, config: SimulationConfig):
        """
        Initialize the OpenGL renderer.
        
        Args:
            config (SimulationConfig): Simulation configuration object
        """
        self.config = config
        self.display = None
        self.tags_data: List[TagData] = []
        
        self._initialize_pygame()
        self._initialize_opengl()
        self._load_tag_textures()
        
        logging.info("✓ OpenGL renderer initialized")
    
    def _initialize_pygame(self) -> None:
        """Initialize pygame and create the OpenGL display window."""
        pygame.init()
        self.display = self.config.display_size
        pygame.display.set_mode(self.display, DOUBLEBUF | OPENGL)
        pygame.display.set_caption("AprilSLAM Simulation")
        glEnable(GL_TEXTURE_2D)
        logging.info(f"✓ Created OpenGL window: {self.display[0]}x{self.display[1]}")
    
    def _initialize_opengl(self) -> None:
        """Set up OpenGL projection and view matrices."""
        # Set up projection matrix
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(
            self.config.fov_y,
            self.config.aspect_ratio,
            self.config.near_clip,
            self.config.far_clip
        )
        
        # Switch back to model-view matrix
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # Enable depth testing for proper 3D rendering
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        
        logging.info("✓ OpenGL projection matrix configured")
    
    def _load_tag_textures(self) -> None:
        """Load all tag textures from configuration."""
        self.tags_data.clear()
        
        for tag_config in self.config.tags:
            try:
                texture_id = self._load_texture(tag_config["image"])
                position = np.array(tag_config["position"], dtype=np.float32)
                rotation = np.array(tag_config["rotation"], dtype=np.float32)
                
                tag_data = TagData(tag_config["id"], texture_id, position, rotation)
                self.tags_data.append(tag_data)
                
                logging.info(f"✓ Loaded tag {tag_config['id']}: {tag_config['image']}")
            except Exception as e:
                logging.error(f"✗ Failed to load tag {tag_config['id']}: {e}")
                raise
        
        logging.info(f"✓ Loaded {len(self.tags_data)} tag textures")
    
    def _load_texture(self, image_path: str) -> int:
        """
        Load a single texture from an image file.
        
        Args:
            image_path (str): Path to the image file (relative or absolute)
            
        Returns:
            int: OpenGL texture ID
            
        Raises:
            FileNotFoundError: If image file doesn't exist
            Exception: If texture loading fails
        """
        # Resolve image path relative to project root
        if not os.path.isabs(image_path):
            project_root = os.path.join(os.path.dirname(__file__), '..', '..')
            if image_path.startswith('tags/'):
                # Update old tag paths to new assets/tags structure
                resolved_path = os.path.join(project_root, 'assets', image_path)
            else:
                # Handle other relative paths
                resolved_path = os.path.join(project_root, image_path)
        else:
            resolved_path = image_path
        
        if not os.path.exists(resolved_path):
            raise FileNotFoundError(f"Texture file not found: {resolved_path}")
        
        try:
            # Load image with pygame
            texture_surface = pygame.image.load(resolved_path).convert_alpha()
            width = texture_surface.get_width()
            height = texture_surface.get_height()
            
            # Convert to OpenGL format
            texture_data = pygame.image.tostring(texture_surface, "RGBA", True)
            
            # Generate OpenGL texture
            texture_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texture_id)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, texture_data)
            
            # Set texture parameters
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            
            return texture_id
            
        except Exception as e:
            raise Exception(f"Failed to create OpenGL texture from {resolved_path}: {e}")
    
    def apply_camera_transform(self, camera_position: np.ndarray, camera_rotation: np.ndarray) -> None:
        """
        Apply camera transformation (view matrix) to OpenGL.
        
        Args:
            camera_position (np.ndarray): Camera position [x, y, z]
            camera_rotation (np.ndarray): Camera rotation [pitch, yaw, roll] in degrees
        """
        glLoadIdentity()
        
        # Apply inverse camera transform (view matrix)
        # Note: Order and sign are important for proper view transformation
        glRotatef(-camera_rotation[2], 0, 0, 1)  # -roll
        glRotatef(-camera_rotation[0], 1, 0, 0)  # -pitch
        glRotatef(-camera_rotation[1], 0, 1, 0)  # -yaw
        glTranslatef(-camera_position[0], -camera_position[1], -camera_position[2])
    
    def render_frame(self, camera_position: np.ndarray, camera_rotation: np.ndarray) -> None:
        """
        Render a complete frame with all tags.
        
        Args:
            camera_position (np.ndarray): Camera position [x, y, z]
            camera_rotation (np.ndarray): Camera rotation [pitch, yaw, roll] in degrees
        """
        # Clear the frame
        glClearColor(0.5, 0.0, 0.5, 1.0)  # Purple background
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Apply camera transform
        self.apply_camera_transform(camera_position, camera_rotation)
        
        # Sort tags by Z-depth for proper rendering order
        sorted_tags = sorted(self.tags_data, key=lambda tag: tag.get_world_z())
        
        # Render each tag
        for tag in sorted_tags:
            self._render_tag(tag)
        
        # Present the frame
        pygame.display.flip()
    
    def _render_tag(self, tag: TagData) -> None:
        """
        Render a single AprilTag.
        
        Args:
            tag (TagData): Tag data to render
        """
        glPushMatrix()
        
        # Apply tag transformations
        glTranslatef(*tag.position)
        
        # Apply rotations in order: Z, Y, X
        glRotatef(tag.rotation[2], 0, 0, 1)  # roll
        glRotatef(tag.rotation[1], 0, 1, 0)  # yaw
        glRotatef(tag.rotation[0], 1, 0, 0)  # pitch
        
        # Bind tag texture
        glBindTexture(GL_TEXTURE_2D, tag.texture)
        
        # Render tag quad
        half_size = self.config.tag_size_outer / 2
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex3f(-half_size, -half_size, 0)
        glTexCoord2f(1, 0); glVertex3f( half_size, -half_size, 0)
        glTexCoord2f(1, 1); glVertex3f( half_size,  half_size, 0)
        glTexCoord2f(0, 1); glVertex3f(-half_size,  half_size, 0)
        glEnd()
        
        glPopMatrix()
    
    def capture_frame(self) -> np.ndarray:
        """
        Capture the current frame as a numpy array for SLAM processing.
        
        Returns:
            np.ndarray: BGR image array suitable for OpenCV processing
        """
        width, height = self.display
        
        # Read pixels from OpenGL framebuffer
        data = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
        
        # Convert to numpy array and reshape
        image = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
        
        # Flip vertically (OpenGL origin is bottom-left, OpenCV is top-left)
        image = np.flipud(image)
        
        # Convert RGB to BGR for OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        return image
    
    def get_tag_by_id(self, tag_id: int) -> Optional[TagData]:
        """
        Get tag data by ID.
        
        Args:
            tag_id (int): Tag ID to search for
            
        Returns:
            Optional[TagData]: Tag data if found, None otherwise
        """
        for tag in self.tags_data:
            if tag.id == tag_id:
                return tag
        return None
    
    def get_all_tags(self) -> List[TagData]:
        """
        Get all loaded tag data.
        
        Returns:
            List[TagData]: List of all tag data objects
        """
        return self.tags_data.copy()
    
    def update_tag_position(self, tag_id: int, position: np.ndarray) -> bool:
        """
        Update a tag's position.
        
        Args:
            tag_id (int): Tag ID to update
            position (np.ndarray): New position [x, y, z]
            
        Returns:
            bool: True if tag was found and updated, False otherwise
        """
        tag = self.get_tag_by_id(tag_id)
        if tag:
            tag.position = position.copy()
            logging.debug(f"Updated tag {tag_id} position to: {position}")
            return True
        return False
    
    def update_tag_rotation(self, tag_id: int, rotation: np.ndarray) -> bool:
        """
        Update a tag's rotation.
        
        Args:
            tag_id (int): Tag ID to update
            rotation (np.ndarray): New rotation [pitch, yaw, roll] in degrees
            
        Returns:
            bool: True if tag was found and updated, False otherwise
        """
        tag = self.get_tag_by_id(tag_id)
        if tag:
            tag.rotation = rotation.copy()
            logging.debug(f"Updated tag {tag_id} rotation to: {rotation}")
            return True
        return False
    
    def cleanup(self) -> None:
        """Clean up OpenGL resources and pygame."""
        # Delete textures
        if self.tags_data:
            texture_ids = [tag.texture for tag in self.tags_data]
            glDeleteTextures(texture_ids)
            logging.info("✓ Cleaned up OpenGL textures")
        
        # Quit pygame
        pygame.quit()
        logging.info("✓ Cleaned up renderer")
    
    def get_render_stats(self) -> Dict[str, Any]:
        """
        Get rendering statistics.
        
        Returns:
            Dict[str, Any]: Dictionary containing render statistics
        """
        return {
            'display_size': self.display,
            'tag_count': len(self.tags_data),
            'fov_y': self.config.fov_y,
            'aspect_ratio': self.config.aspect_ratio,
            'near_clip': self.config.near_clip,
            'far_clip': self.config.far_clip
        }
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except:
            pass  # Ignore cleanup errors during destruction 