#!/usr/bin/env python3
"""
AprilSLAM Simulation Module

This module provides a complete simulation environment for testing and evaluating
AprilTag-based SLAM algorithms. It includes modular components for rendering,
camera control, data logging, and ground truth calculations.

Author: Mikolaj Kostrzewa
License: Creative Commons Attribution-NonCommercial 4.0 International
"""

# Import main classes for convenient access
from .simulation_engine import SimulationEngine
from .config_manager import SimulationConfig
from .camera_controller import CameraController
from .renderer import SimulationRenderer, TagData
from .data_logger import DataLogger, PoseData, ErrorMetrics
from .ground_truth import GroundTruthCalculator

# Legacy import for backward compatibility
from .sim import Simulation

# Version information
__version__ = "2.0.0"
__author__ = "Mikolaj Kostrzewa"

# Public API
__all__ = [
    # Main simulation engine
    'SimulationEngine',
    
    # Core components
    'SimulationConfig',
    'CameraController', 
    'SimulationRenderer',
    'DataLogger',
    'GroundTruthCalculator',
    
    # Data structures
    'TagData',
    'PoseData', 
    'ErrorMetrics',
    
    # Legacy (deprecated)
    'Simulation',
]
