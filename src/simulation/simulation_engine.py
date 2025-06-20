#!/usr/bin/env python3
"""
AprilSLAM Simulation Engine

This is the main orchestrator for the AprilSLAM simulation system. It coordinates
all subsystems including rendering, camera control, SLAM processing, data logging,
and ground truth calculations to provide a complete simulation environment.

Author: Mikolaj Kostrzewa
License: Creative Commons Attribution-NonCommercial 4.0 International
"""

import sys
import os
import time
import logging
import pygame
import cv2
import numpy as np
from termcolor import colored
from typing import Optional, Dict, Any

# Add apriltag library to path
apriltag_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'lib', 'apriltag', 'build'))
if apriltag_path not in sys.path:
    sys.path.insert(0, apriltag_path)

from src.core.slam import SLAM
from .config_manager import SimulationConfig
from .camera_controller import CameraController
from .renderer import SimulationRenderer
from .data_logger import DataLogger, PoseData, ErrorMetrics
from .ground_truth import GroundTruthCalculator


class SimulationEngine:
    """
    Main simulation engine that orchestrates all components of the AprilSLAM simulation.
    
    This class coordinates:
    - Configuration management
    - OpenGL rendering
    - Camera movement and control
    - SLAM processing
    - Ground truth calculations
    - Data logging and analysis
    - User interface and display
    """
    
    def __init__(self, config_file: str, movement_enabled: bool = True):
        """
        Initialize the simulation engine.
        
        Args:
            config_file (str): Path to simulation configuration file
            movement_enabled (bool): Whether manual camera movement is enabled
        """
        self.running = False
        self.movement_enabled = movement_enabled
        
        # Initialize logging
        self._setup_logging()
        
        # Initialize all subsystems
        logging.info("🚀 Initializing AprilSLAM Simulation Engine")
        
        try:
            # Load configuration
            self.config = SimulationConfig(config_file)
            
            # Initialize camera controller
            self.camera = CameraController(
                movement_enabled=movement_enabled,
                movement_speed=1.0,
                rotation_speed=1.0,
                size_scale=self.config.size_scale
            )
            
            # Initialize renderer
            self.renderer = SimulationRenderer(self.config)
            
            # Initialize ground truth calculator
            self.ground_truth = GroundTruthCalculator(self.config, self.renderer.get_all_tags())
            
            # Initialize data logger
            self.data_logger = DataLogger()
            
            # Initialize SLAM system
            self._initialize_slam()
            
            # Monte Carlo bounds (used when movement is disabled)
            self.monte_carlo_bounds = np.array([-3, 10, -1, 1, -0.25, 3]) * 5
            
            logging.info("✅ Simulation engine initialized successfully")
            
        except Exception as e:
            logging.error(f"❌ Failed to initialize simulation engine: {e}")
            self._cleanup()
            raise
    
    def _setup_logging(self) -> None:
        """Configure logging for the simulation."""
        log_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, 'simulation.log')
        
        # Clear previous log
        if os.path.exists(log_file):
            open(log_file, 'w').close()
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def _initialize_slam(self) -> None:
        """Initialize the SLAM system with camera parameters."""
        # Calculate camera intrinsic parameters from OpenGL settings
        fx = fy = 0.5 * self.config.display_height / np.tan(0.5 * np.radians(self.config.fov_y))
        cx = 0.5 * self.config.display_width
        cy = 0.5 * self.config.display_height
        
        camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        
        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        
        camera_params = {
            'camera_matrix': camera_matrix,
            'dist_coeffs': dist_coeffs
        }
        
        self.slam = SLAM(logging, camera_params, tag_size=self.config.tag_size_inner)
        
        logging.info("✓ SLAM system initialized")
    
    def run(self) -> None:
        """
        Main simulation loop.
        
        This method runs the complete simulation including:
        - Event handling
        - Camera movement
        - Rendering
        - SLAM processing
        - Data logging
        - Performance display
        """
        self.running = True
        logging.info("🎮 Starting simulation main loop")
        
        try:
            while self.running:
                # Handle pygame events
                self._handle_events()
                
                # Update camera position
                self._update_camera()
                
                # Render frame
                self.renderer.render_frame(
                    self.camera.get_position(),
                    self.camera.get_rotation()
                )
                
                # Capture frame for SLAM processing
                frame = self.renderer.capture_frame()
                
                # Process SLAM
                self._process_slam(frame)
                
                # Update display
                self._update_display()
                
                # Control frame rate
                pygame.time.wait(1)
                
        except KeyboardInterrupt:
            logging.info("🛑 Simulation interrupted by user")
        except Exception as e:
            logging.error(f"❌ Simulation error: {e}")
            raise
        finally:
            self._cleanup()
    
    def _handle_events(self) -> None:
        """Handle pygame events including keyboard input and window close."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            else:
                # Pass keyboard events to camera controller
                self.camera.handle_key_event(event)
    
    def _update_camera(self) -> None:
        """Update camera position based on movement mode."""
        if self.movement_enabled:
            # Manual movement via keyboard
            self.camera.update_manual_movement()
        else:
            # Automatic Monte Carlo position randomization
            self.camera.randomize_position(self.monte_carlo_bounds)
    
    def _process_slam(self, frame: np.ndarray) -> None:
        """
        Process SLAM on the current frame and log results.
        
        Args:
            frame (np.ndarray): Current camera frame
        """
        detections = self.slam.detect(frame)
        
        # Process detections and update SLAM graph
        for detection in detections:
            retval, rvec, tvec = self.slam.get_pose(detection)
            if retval:
                corners = np.array(detection['lb-rb-rt-lt'], dtype=np.float32)
                frame = self.slam.detector.draw(rvec, tvec, corners, frame, detection['id'])
        
        # Display processed frame
        cv2.imshow('AprilTag Detection', frame)
        
        # Get current SLAM pose estimate
        slam_pose = self.slam.my_pose()
        
        if slam_pose is not None:
            self._log_slam_results(slam_pose)
        
        # Update SLAM graph
        self.slam.slam_graph()
    
    def _log_slam_results(self, slam_pose: np.ndarray) -> None:
        """
        Log SLAM results and calculate errors against ground truth.
        
        Args:
            slam_pose (np.ndarray): Current SLAM pose estimate
        """
        try:
            # Extract pose data
            est_translation = slam_pose[:3, 3]
            est_rotation_matrix = slam_pose[:3, :3]
            est_euler = self.ground_truth.rotation_matrix_to_euler(est_rotation_matrix)
            
            # Get ground truth pose
            camera_pos = self.camera.get_position()
            gt_transform = self.ground_truth.get_inverse_transform(
                self.slam.coordinate_id, camera_pos
            )
            
            gt_translation = gt_transform[:3, 3]
            gt_rotation_matrix = gt_transform[:3, :3]
            gt_euler = self.ground_truth.rotation_matrix_to_euler(gt_rotation_matrix)
            
            # Calculate errors
            translation_error, rotation_error = self.ground_truth.calculate_pose_error(
                slam_pose, gt_transform
            )
            
            # Create data objects
            estimated_pose = self.data_logger.create_pose_data(
                est_translation, est_rotation_matrix, est_euler
            )
            
            ground_truth_pose = self.data_logger.create_pose_data(
                gt_translation, gt_rotation_matrix, gt_euler
            )
            
            gt_magnitude = np.linalg.norm(gt_translation)
            error_metrics = self.data_logger.create_error_metrics(
                translation_error, rotation_error, gt_magnitude,
                len(self.slam.graph.get_nodes()),
                self.slam.average_distance_to_nodes()
            )
            
            # Log to CSV
            self.data_logger.log_frame_data(estimated_pose, ground_truth_pose, error_metrics)
            
            # Log detailed node analysis
            self._log_node_analysis()
            
            # Generate and display error graph (this was missing from the modular implementation)
            self._generate_error_graph()
            
            # Display results
            self._display_performance_metrics(error_metrics, estimated_pose, ground_truth_pose)
            
            # Update SLAM visualizer
            self.slam.vis_slam(ground_truth=gt_transform)
            
        except Exception as e:
            logging.error(f"Error logging SLAM results: {e}")
    
    def _log_node_analysis(self) -> None:
        """Log detailed analysis of SLAM graph nodes."""
        try:
            camera_pos = self.camera.get_position()
            
            for tag_id, node in self.slam.graph.get_nodes().items():
                if not node.visible:
                    continue
                
                # Calculate ground truth for this tag
                gt_local = self.ground_truth.get_camera_to_tag_transform(tag_id, camera_pos)
                gt_world_distance = self.ground_truth.get_tag_to_tag_distance(
                    tag_id, self.slam.coordinate_id, camera_pos
                )
                
                # Extract poses
                local_translation = node.local[:3, 3]
                local_rotation = node.local[:3, :3]
                local_euler = self.ground_truth.rotation_matrix_to_euler(local_rotation)
                
                world_translation = node.world[:3, 3]
                world_rotation = node.world[:3, :3]
                world_euler = self.ground_truth.rotation_matrix_to_euler(world_rotation)
                
                gt_translation = gt_local[:3, 3]
                gt_rotation = gt_local[:3, :3]
                gt_euler = self.ground_truth.rotation_matrix_to_euler(gt_rotation)
                
                # Calculate errors
                local_error = abs(np.linalg.norm(local_translation) - np.linalg.norm(gt_translation))
                world_error = abs(np.linalg.norm(world_translation) - gt_world_distance)
                translation_error = np.linalg.norm(local_translation - gt_translation)
                
                # Create pose data objects
                local_pose = self.data_logger.create_pose_data(
                    local_translation, local_rotation, local_euler
                )
                world_pose = self.data_logger.create_pose_data(
                    world_translation, world_rotation, world_euler
                )
                tag_pose = self.data_logger.create_pose_data(
                    gt_translation, gt_rotation, gt_euler
                )
                
                # Log error analysis
                self.data_logger.log_error_analysis(
                    node.weight, local_pose, world_pose, tag_pose,
                    world_error, local_error, translation_error
                )
                
                # Log covariance data
                self.data_logger.log_covariance_data(node.weight, local_pose, translation_error)
                
        except Exception as e:
            logging.error(f"Error in node analysis: {e}")
    
    def _generate_error_graph(self) -> None:
        """
        Generate and display the error graph showing distance errors between SLAM nodes and ground truth.
        
        This method builds a ground truth reference structure for each tag with:
        - "local": Ground truth distance from camera to tag
        - "world": Ground truth distance from tag to coordinate reference tag
        
        Then passes this to the SLAM visualizer's error_graph method to display
        distance errors between estimated and ground truth positions.
        """
        try:
            camera_pos = self.camera.get_position()
            ground_truth_tags = {}
            
            # Build ground truth reference data for each node in the SLAM graph
            for tag_id, node in self.slam.graph.get_nodes().items():
                # Calculate ground truth local distance (camera to tag)
                gt_local_transform = self.ground_truth.get_camera_to_tag_transform(tag_id, camera_pos)
                local_distance = np.linalg.norm(gt_local_transform[:3, 3])
                
                # Calculate ground truth world distance (tag to coordinate reference tag)
                world_distance = self.ground_truth.get_tag_to_tag_distance(
                    tag_id, self.slam.coordinate_id, camera_pos
                )
                
                # Store ground truth distances for this tag
                ground_truth_tags[tag_id] = {
                    "local": local_distance,
                    "world": world_distance
                }
                
                # Log ground truth vs estimated comparisons
                estimated_local = np.linalg.norm(node.local[:3, 3])
                estimated_world = np.linalg.norm(node.world[:3, 3])
                
                logging.debug(f"Tag {tag_id} - GT Local: {local_distance:.4f}, Est Local: {estimated_local:.4f}")
                logging.debug(f"Tag {tag_id} - GT World: {world_distance:.4f}, Est World: {estimated_world:.4f}")
            
            # Call the SLAM error_graph method to visualize the errors
            self.slam.error_graph(ground_truth_tags)
            
        except Exception as e:
            logging.error(f"Error generating error graph: {e}")
    
    def _display_performance_metrics(self, error_metrics: ErrorMetrics, 
                                   estimated_pose: PoseData, ground_truth_pose: PoseData) -> None:
        """
        Display performance metrics in the console.
        
        Args:
            error_metrics (ErrorMetrics): Calculated error metrics
            estimated_pose (PoseData): Estimated pose data
            ground_truth_pose (PoseData): Ground truth pose data
        """
        # Convert to millimeters for display
        est_distance_mm = self.ground_truth.convert_simulation_to_mm(
            np.linalg.norm(estimated_pose.translation)
        )
        gt_distance_mm = self.ground_truth.convert_simulation_to_mm(
            np.linalg.norm(ground_truth_pose.translation)
        )
        error_distance_mm = self.ground_truth.convert_simulation_to_mm(error_metrics.translation_error)
        
        # Scale units for better readability
        def scale_units(value_mm):
            if value_mm >= 1000:
                return value_mm / 1000, 'm'
            elif value_mm >= 10:
                return value_mm / 10, 'cm'
            else:
                return value_mm, 'mm'
        
        est_scaled, est_unit = scale_units(est_distance_mm)
        gt_scaled, gt_unit = scale_units(gt_distance_mm)
        error_scaled, error_unit = scale_units(error_distance_mm)
        
        # Clear terminal and display metrics
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print(colored("🎯 AprilSLAM Simulation Performance", 'cyan', attrs=['bold']))
        print(colored("=" * 50, 'cyan'))
        print(colored(f"📐 Estimated Distance: {est_scaled:.4f} {est_unit}", 'green'))
        print(colored(f"🎯 Ground Truth Distance: {gt_scaled:.4f} {gt_unit}", 'yellow'))
        print(colored(f"📏 Translation Error: {error_scaled:.4f} {error_unit}", 'cyan'))
        print(colored(f"🔄 Rotation Error: {error_metrics.rotation_error:.4f}", 'magenta'))
        print(colored(f"📊 Percentage Error: {error_metrics.percentage_error:.2f}%", 'red'))
        print(colored(f"🏷️  SLAM Nodes: {error_metrics.node_count}", 'blue'))
        print(colored(f"📍 Average Distance: {error_metrics.average_distance:.2f}", 'white'))
        
        # Display keyboard controls
        if self.movement_enabled:
            print(colored("\n🎮 Controls:", 'white', attrs=['bold']))
            print(colored("  Arrow Keys: Move X/Y  |  W/S: Move Z", 'white'))
            print(colored("  A/D: Yaw  |  Q/E: Roll  |  R/F: Pitch", 'white'))
            print(colored("  ESC: Quit Simulation", 'yellow'))
    
    def _update_display(self) -> None:
        """Update display and handle window events."""
        # pygame display is updated in renderer.render_frame()
        # Handle OpenCV window
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            self.running = False
    
    def _cleanup(self) -> None:
        """Clean up all resources."""
        logging.info("🧹 Cleaning up simulation resources")
        
        try:
            # Close data logger
            if hasattr(self, 'data_logger'):
                self.data_logger.close()
            
            # Clean up renderer
            if hasattr(self, 'renderer'):
                self.renderer.cleanup()
            
            # Close OpenCV windows
            cv2.destroyAllWindows()
            
            # Quit pygame
            pygame.quit()
            
            logging.info("✅ Cleanup completed successfully")
            
        except Exception as e:
            logging.error(f"❌ Error during cleanup: {e}")
    
    def get_simulation_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive simulation statistics.
        
        Returns:
            Dict[str, Any]: Dictionary containing simulation statistics
        """
        stats = {
            'config': str(self.config),
            'camera': self.camera.get_status_info(),
            'renderer': self.renderer.get_render_stats(),
            'data_logger': self.data_logger.get_statistics(),
            'movement_enabled': self.movement_enabled,
            'running': self.running
        }
        
        if hasattr(self.slam, 'graph'):
            stats['slam_nodes'] = len(self.slam.graph.get_nodes())
        
        return stats
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self._cleanup()


def main():
    """Main entry point for the simulation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='AprilSLAM Simulation Engine')
    parser.add_argument('--config', '-c', 
                       default=os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'sim_settings.json'),
                       help='Path to simulation configuration file')
    parser.add_argument('--no-movement', action='store_true',
                       help='Disable manual movement (use Monte Carlo randomization)')
    
    args = parser.parse_args()
    
    try:
        with SimulationEngine(args.config, movement_enabled=not args.no_movement) as sim:
            sim.run()
    except KeyboardInterrupt:
        print("\n🛑 Simulation interrupted by user")
    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main() 