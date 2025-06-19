#!/usr/bin/env python3
"""
Simulation Configuration Manager

This module handles loading and managing simulation configuration settings.
It provides a clean interface for accessing simulation parameters and validates
configuration data to ensure proper simulation setup.

Author: Mikolaj Kostrzewa
License: Creative Commons Attribution-NonCommercial 4.0 International
"""

import json
import os
import logging
from typing import Dict, List, Any, Tuple


class SimulationConfig:
    """
    Manages simulation configuration settings and provides validated access to parameters.
    
    This class is responsible for:
    - Loading configuration from JSON files
    - Validating configuration parameters
    - Providing typed access to configuration values
    - Managing default values and fallbacks
    """
    
    def __init__(self, config_file: str):
        """
        Initialize the configuration manager.
        
        Args:
            config_file (str): Path to the JSON configuration file
            
        Raises:
            FileNotFoundError: If configuration file doesn't exist
            ValueError: If configuration is invalid
        """
        self.config_file = config_file
        self.settings = {}
        self._load_configuration()
        self._validate_configuration()
        
    def _load_configuration(self) -> None:
        """
        Load configuration from the JSON file.
        
        Raises:
            FileNotFoundError: If configuration file doesn't exist
            json.JSONDecodeError: If configuration file is malformed
        """
        try:
            with open(self.config_file, 'r') as f:
                self.settings = json.load(f)
            logging.info(f"✓ Loaded configuration from: {self.config_file}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_file}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")
    
    def _validate_configuration(self) -> None:
        """
        Validate that all required configuration parameters are present and valid.
        
        Raises:
            ValueError: If required parameters are missing or invalid
        """
        required_params = [
            'display_width', 'display_height', 'fov_y', 'near_clip', 'far_clip',
            'size_scale', 'tag_size_inner', 'tag_size_outer', 'actual_size_in_mm', 'tags'
        ]
        
        for param in required_params:
            if param not in self.settings:
                raise ValueError(f"Missing required configuration parameter: {param}")
        
        # Validate display dimensions
        if self.settings['display_width'] <= 0 or self.settings['display_height'] <= 0:
            raise ValueError("Display dimensions must be positive")
        
        # Validate FOV
        if not (0 < self.settings['fov_y'] < 180):
            raise ValueError("Field of view must be between 0 and 180 degrees")
        
        # Validate clipping planes
        if self.settings['near_clip'] >= self.settings['far_clip']:
            raise ValueError("Near clip must be less than far clip")
        
        # Validate tags
        if not isinstance(self.settings['tags'], list) or len(self.settings['tags']) == 0:
            raise ValueError("Tags must be a non-empty list")
        
        for i, tag in enumerate(self.settings['tags']):
            required_tag_params = ['id', 'image', 'position', 'rotation']
            for param in required_tag_params:
                if param not in tag:
                    raise ValueError(f"Tag {i} missing required parameter: {param}")
    
    # Display Configuration
    @property
    def display_width(self) -> int:
        """Get display width in pixels."""
        return self.settings['display_width']
    
    @property
    def display_height(self) -> int:
        """Get display height in pixels."""
        return self.settings['display_height']
    
    @property
    def display_size(self) -> Tuple[int, int]:
        """Get display size as (width, height) tuple."""
        return (self.display_width, self.display_height)
    
    # Camera Configuration
    @property
    def fov_y(self) -> float:
        """Get vertical field of view in degrees."""
        return self.settings['fov_y']
    
    @property
    def near_clip(self) -> float:
        """Get near clipping plane distance."""
        return self.settings['near_clip']
    
    @property
    def far_clip(self) -> float:
        """Get far clipping plane distance."""
        return self.settings['far_clip']
    
    @property
    def aspect_ratio(self) -> float:
        """Calculate and return display aspect ratio."""
        return self.display_width / self.display_height
    
    # Tag Configuration
    @property
    def size_scale(self) -> float:
        """Get global size scaling factor."""
        return self.settings['size_scale']
    
    @property
    def tag_size_inner(self) -> float:
        """Get inner tag size (detection area) scaled by size_scale."""
        return self.settings['tag_size_inner'] * self.size_scale
    
    @property
    def tag_size_outer(self) -> float:
        """Get outer tag size (visual border) scaled by size_scale."""
        return self.settings['tag_size_outer'] * self.size_scale
    
    @property
    def actual_tag_size_mm(self) -> float:
        """Get actual physical tag size in millimeters."""
        return self.settings['actual_size_in_mm']
    
    @property
    def tags(self) -> List[Dict[str, Any]]:
        """Get list of tag configurations."""
        return self.settings['tags']
    
    # Utility Methods
    def get_tag_by_id(self, tag_id: int) -> Dict[str, Any]:
        """
        Get tag configuration by ID.
        
        Args:
            tag_id (int): Tag ID to search for
            
        Returns:
            Dict[str, Any]: Tag configuration dictionary
            
        Raises:
            ValueError: If tag ID not found
        """
        for tag in self.tags:
            if tag['id'] == tag_id:
                return tag
        raise ValueError(f"Tag with ID {tag_id} not found in configuration")
    
    def get_tag_count(self) -> int:
        """Get total number of configured tags."""
        return len(self.tags)
    
    def mm_to_simulation_units(self, value_mm: float) -> float:
        """
        Convert millimeters to simulation units.
        
        Args:
            value_mm (float): Value in millimeters
            
        Returns:
            float: Value in simulation units
        """
        return value_mm * self.tag_size_inner / self.actual_tag_size_mm
    
    def simulation_units_to_mm(self, value_sim: float) -> float:
        """
        Convert simulation units to millimeters.
        
        Args:
            value_sim (float): Value in simulation units
            
        Returns:
            float: Value in millimeters
        """
        return value_sim * self.actual_tag_size_mm / self.tag_size_inner
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return (f"SimulationConfig("
                f"display={self.display_width}x{self.display_height}, "
                f"fov={self.fov_y}°, "
                f"tags={self.get_tag_count()})")
    
    def __repr__(self) -> str:
        """Developer representation of configuration."""
        return f"SimulationConfig(config_file='{self.config_file}')" 