#!/usr/bin/env python3
"""
Data Logger for AprilSLAM Simulation

This module handles all data logging operations including CSV file management,
error tracking, covariance logging, and statistical data collection for
performance analysis and research purposes.

Author: Mikolaj Kostrzewa
License: Creative Commons Attribution-NonCommercial 4.0 International
"""

import os
import csv
import time
import logging
import numpy as np
from typing import Dict, List, Optional, Any, TextIO
from dataclasses import dataclass


@dataclass
class PoseData:
    """Container for pose estimation data."""
    translation: np.ndarray
    rotation_matrix: np.ndarray
    euler_angles: np.ndarray
    timestamp: float


@dataclass
class ErrorMetrics:
    """Container for error analysis metrics."""
    translation_error: float
    rotation_error: float
    percentage_error: float
    node_count: int
    average_distance: float


class DataLogger:
    """
    Handles all data logging operations for the AprilSLAM simulation.
    
    This class manages:
    - CSV file creation and writing
    - Error metrics collection
    - Covariance data logging
    - Performance statistics
    - File cleanup and resource management
    """
    
    def __init__(self, output_directory: Optional[str] = None):
        """
        Initialize the data logger.
        
        Args:
            output_directory (Optional[str]): Directory for output files.
                                             If None, uses default data/csv directory.
        """
        self.start_time = time.time()
        self.output_dir = self._setup_output_directory(output_directory)
        
        # File handles and writers
        self.main_data_file: Optional[TextIO] = None
        self.error_params_file: Optional[TextIO] = None
        self.covariance_file: Optional[TextIO] = None
        
        self.main_writer: Optional[csv.writer] = None
        self.error_writer: Optional[csv.writer] = None
        self.covariance_writer: Optional[csv.writer] = None
        
        # Data collection counters
        self.frame_count = 0
        self.logged_entries = 0
        
        self._initialize_csv_files()
        logging.info("✓ Data logger initialized")
    
    def _setup_output_directory(self, output_directory: Optional[str]) -> str:
        """
        Set up the output directory for CSV files.
        
        Args:
            output_directory (Optional[str]): Custom output directory
            
        Returns:
            str: Path to the output directory
        """
        if output_directory is None:
            # Default to data/csv directory relative to project root
            project_root = os.path.join(os.path.dirname(__file__), '..', '..')
            output_directory = os.path.join(project_root, 'data', 'csv')
        
        # Create directory if it doesn't exist
        os.makedirs(output_directory, exist_ok=True)
        logging.info(f"✓ Output directory: {output_directory}")
        
        return output_directory
    
    def _initialize_csv_files(self) -> None:
        """Initialize all CSV files with appropriate headers."""
        try:
            # Main simulation data file
            main_data_path = os.path.join(self.output_dir, 'slam_simulation_data.csv')
            self.main_data_file = open(main_data_path, 'w', newline='')
            self.main_writer = csv.writer(self.main_data_file)
            
            # Write main data header
            self.main_writer.writerow([
                'Time', 'Number_of_Nodes', 'Average_Distance',
                'Est_X', 'Est_Y', 'Est_Z',
                'Est_Roll', 'Est_Pitch', 'Est_Yaw',
                'GT_X', 'GT_Y', 'GT_Z',
                'GT_Roll', 'GT_Pitch', 'GT_Yaw',
                'Translation_Difference', 'Rotation_Difference'
            ])
            
            # Error parameters file
            error_params_path = os.path.join(self.output_dir, 'error_analysis.csv')
            self.error_params_file = open(error_params_path, 'w', newline='')
            self.error_writer = csv.writer(self.error_params_file)
            
            # Write error parameters header
            self.error_writer.writerow([
                'Number_of_Jumps',
                'Est_X_Local', 'Est_Y_Local', 'Est_Z_Local',
                'Est_Roll_Local', 'Est_Pitch_Local', 'Est_Yaw_Local',
                'Est_X_World', 'Est_Y_World', 'Est_Z_World',
                'Est_Roll_World', 'Est_Pitch_World', 'Est_Yaw_World',
                'Tag_Est_X', 'Tag_Est_Y', 'Tag_Est_Z',
                'Tag_Est_Roll', 'Tag_Est_Pitch', 'Tag_Est_Yaw',
                'Error_World', 'Error_Local', 'Translation_Error'
            ])
            
            # Covariance data file
            covariance_path = os.path.join(self.output_dir, 'covariance_analysis.csv')
            self.covariance_file = open(covariance_path, 'w', newline='')
            self.covariance_writer = csv.writer(self.covariance_file)
            
            # Write covariance header
            self.covariance_writer.writerow([
                'Number_of_Jumps',
                'Tag_Est_X', 'Tag_Est_Y', 'Tag_Est_Z',
                'Tag_Est_Roll', 'Tag_Est_Pitch', 'Tag_Est_Yaw',
                'Translation_Error'
            ])
            
            logging.info("✓ CSV files initialized with headers")
            
        except Exception as e:
            logging.error(f"✗ Failed to initialize CSV files: {e}")
            self._cleanup_files()
            raise
    
    def log_frame_data(self, estimated_pose: PoseData, ground_truth_pose: PoseData,
                      error_metrics: ErrorMetrics) -> None:
        """
        Log main simulation frame data.
        
        Args:
            estimated_pose (PoseData): Estimated camera pose
            ground_truth_pose (PoseData): Ground truth camera pose
            error_metrics (ErrorMetrics): Calculated error metrics
        """
        if not self.main_writer:
            logging.warning("Main data writer not initialized")
            return
        
        current_time = time.time() - self.start_time
        
        try:
            self.main_writer.writerow([
                current_time,
                error_metrics.node_count,
                error_metrics.average_distance,
                estimated_pose.translation[0], estimated_pose.translation[1], estimated_pose.translation[2],
                estimated_pose.euler_angles[0], estimated_pose.euler_angles[1], estimated_pose.euler_angles[2],
                ground_truth_pose.translation[0], ground_truth_pose.translation[1], ground_truth_pose.translation[2],
                ground_truth_pose.euler_angles[0], ground_truth_pose.euler_angles[1], ground_truth_pose.euler_angles[2],
                error_metrics.translation_error,
                error_metrics.rotation_error
            ])
            
            self.frame_count += 1
            self.logged_entries += 1
            
            # Flush periodically to ensure data is written
            if self.logged_entries % 10 == 0:
                self.main_data_file.flush()
                
        except Exception as e:
            logging.error(f"✗ Failed to log frame data: {e}")
    
    def log_error_analysis(self, jump_count: int, local_pose: PoseData, world_pose: PoseData,
                          tag_pose: PoseData, world_error: float, local_error: float,
                          translation_error: float) -> None:
        """
        Log error analysis data for detailed performance evaluation.
        
        Args:
            jump_count (int): Number of SLAM graph jumps/updates
            local_pose (PoseData): Local coordinate pose estimate
            world_pose (PoseData): World coordinate pose estimate
            tag_pose (PoseData): Tag pose estimate
            world_error (float): World coordinate error
            local_error (float): Local coordinate error
            translation_error (float): Translation error magnitude
        """
        if not self.error_writer:
            logging.warning("Error analysis writer not initialized")
            return
        
        try:
            self.error_writer.writerow([
                jump_count,
                local_pose.translation[0], local_pose.translation[1], local_pose.translation[2],
                local_pose.euler_angles[0], local_pose.euler_angles[1], local_pose.euler_angles[2],
                world_pose.translation[0], world_pose.translation[1], world_pose.translation[2],
                world_pose.euler_angles[0], world_pose.euler_angles[1], world_pose.euler_angles[2],
                tag_pose.translation[0], tag_pose.translation[1], tag_pose.translation[2],
                tag_pose.euler_angles[0], tag_pose.euler_angles[1], tag_pose.euler_angles[2],
                world_error, local_error, translation_error
            ])
            
        except Exception as e:
            logging.error(f"✗ Failed to log error analysis: {e}")
    
    def log_covariance_data(self, jump_count: int, tag_pose: PoseData, translation_error: float) -> None:
        """
        Log covariance analysis data for uncertainty quantification.
        
        Args:
            jump_count (int): Number of SLAM graph jumps/updates
            tag_pose (PoseData): Tag pose estimate
            translation_error (float): Translation error magnitude
        """
        if not self.covariance_writer:
            logging.warning("Covariance writer not initialized")
            return
        
        try:
            self.covariance_writer.writerow([
                jump_count,
                tag_pose.translation[0], tag_pose.translation[1], tag_pose.translation[2],
                tag_pose.euler_angles[0], tag_pose.euler_angles[1], tag_pose.euler_angles[2],
                translation_error
            ])
            
        except Exception as e:
            logging.error(f"✗ Failed to log covariance data: {e}")
    
    def flush_all(self) -> None:
        """Flush all open file buffers to ensure data is written to disk."""
        try:
            if self.main_data_file:
                self.main_data_file.flush()
            if self.error_params_file:
                self.error_params_file.flush()
            if self.covariance_file:
                self.covariance_file.flush()
            logging.debug("✓ Flushed all data buffers")
        except Exception as e:
            logging.error(f"✗ Failed to flush buffers: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get logging statistics.
        
        Returns:
            Dict[str, Any]: Dictionary containing logging statistics
        """
        runtime = time.time() - self.start_time
        
        return {
            'runtime_seconds': runtime,
            'frames_logged': self.frame_count,
            'total_entries': self.logged_entries,
            'average_fps': self.frame_count / runtime if runtime > 0 else 0,
            'output_directory': self.output_dir,
            'files_open': {
                'main_data': self.main_data_file is not None,
                'error_analysis': self.error_params_file is not None,
                'covariance': self.covariance_file is not None
            }
        }
    
    def _cleanup_files(self) -> None:
        """Close all open files safely."""
        files_to_close = [
            (self.main_data_file, "main data"),
            (self.error_params_file, "error parameters"),
            (self.covariance_file, "covariance")
        ]
        
        for file_handle, name in files_to_close:
            if file_handle:
                try:
                    file_handle.close()
                    logging.info(f"✓ Closed {name} file")
                except Exception as e:
                    logging.error(f"✗ Error closing {name} file: {e}")
        
        # Reset file handles
        self.main_data_file = None
        self.error_params_file = None
        self.covariance_file = None
        self.main_writer = None
        self.error_writer = None
        self.covariance_writer = None
    
    def create_pose_data(self, translation: np.ndarray, rotation_matrix: np.ndarray,
                        euler_angles: Optional[np.ndarray] = None) -> PoseData:
        """
        Create a PoseData object with proper formatting.
        
        Args:
            translation (np.ndarray): Translation vector [x, y, z]
            rotation_matrix (np.ndarray): 3x3 rotation matrix
            euler_angles (Optional[np.ndarray]): Euler angles [roll, pitch, yaw].
                                               If None, computed from rotation matrix.
        
        Returns:
            PoseData: Formatted pose data object
        """
        if euler_angles is None:
            euler_angles = self._rotation_matrix_to_euler(rotation_matrix)
        
        return PoseData(
            translation=translation.copy(),
            rotation_matrix=rotation_matrix.copy(),
            euler_angles=euler_angles.copy(),
            timestamp=time.time()
        )
    
    def create_error_metrics(self, translation_error: float, rotation_error: float,
                           ground_truth_magnitude: float, node_count: int,
                           average_distance: float) -> ErrorMetrics:
        """
        Create an ErrorMetrics object with calculated percentage error.
        
        Args:
            translation_error (float): Translation error magnitude
            rotation_error (float): Rotation error magnitude
            ground_truth_magnitude (float): Ground truth translation magnitude
            node_count (int): Number of SLAM graph nodes
            average_distance (float): Average distance to nodes
        
        Returns:
            ErrorMetrics: Formatted error metrics object
        """
        percentage_error = (translation_error / ground_truth_magnitude * 100) if ground_truth_magnitude > 0 else 0
        
        return ErrorMetrics(
            translation_error=translation_error,
            rotation_error=rotation_error,
            percentage_error=percentage_error,
            node_count=node_count,
            average_distance=average_distance
        )
    
    @staticmethod
    def _rotation_matrix_to_euler(rotation_matrix: np.ndarray) -> np.ndarray:
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
    
    def close(self) -> None:
        """Close all files and clean up resources."""
        self.flush_all()
        self._cleanup_files()
        
        stats = self.get_statistics()
        logging.info(f"✓ Data logger closed. Runtime: {stats['runtime_seconds']:.2f}s, "
                    f"Frames: {stats['frames_logged']}, Avg FPS: {stats['average_fps']:.2f}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.close()
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.close()
        except:
            pass  # Ignore cleanup errors during destruction 