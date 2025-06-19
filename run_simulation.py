#!/usr/bin/env python3

"""
AprilSLAM Simulation Runner
==========================

This script serves as the main entry point for running the AprilSLAM simulation.
It handles the proper path setup and imports to run the simulation from the
reorganized directory structure.

Usage:
    python run_simulation.py [--movement]

Options:
    --movement    Enable camera movement during simulation (default: True)
"""

import sys
import os
import argparse

# Add the src directory to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

def main():
    parser = argparse.ArgumentParser(description="Run the AprilSLAM simulation")
    parser.add_argument("--no-movement", action="store_true", 
                       help="Disable camera movement during simulation")
    args = parser.parse_args()
    
    # Import and run the simulation
    from src.simulation.sim import Simulation
    
    config_path = os.path.join(project_root, 'config', 'sim_settings.json')
    movement_flag = not args.no_movement
    
    print(f"Starting AprilSLAM simulation...")
    print(f"Config file: {config_path}")
    print(f"Movement enabled: {movement_flag}")
    
    sim = Simulation(config_path, movment_flag=movement_flag)
    
    try:
        sim.run()
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
        sim.csvfile.close()
        sim.error_file.close()
        sim.covariance_file.close()
        import pygame
        pygame.quit()
        sys.exit()

if __name__ == '__main__':
    main() 