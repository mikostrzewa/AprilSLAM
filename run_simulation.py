#!/usr/bin/env python3

"""
AprilSLAM Simulation Runner v2.0
================================

Main entry point for running the AprilSLAM simulation. This script initializes
the new modular simulation environment and handles the main execution loop.

The simulation now uses a modular architecture with the following components:
- SimulationEngine: Main orchestrator
- SimulationConfig: Configuration management  
- CameraController: Movement and input handling
- SimulationRenderer: OpenGL rendering
- DataLogger: CSV data logging
- GroundTruthCalculator: Reference calculations

Usage:
    python run_simulation.py [--config CONFIG_FILE] [--no-movement] [--debug]

Author: Mikolaj Kostrzewa
License: Creative Commons Attribution-NonCommercial 4.0 International
"""

import sys
import os
import argparse
import logging

# Add project root to Python path for imports
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the new modular simulation system
from src.simulation import SimulationEngine


def setup_logging(debug=False):
    """Configure logging for the simulation runner."""
    log_dir = os.path.join(project_root, 'data', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, 'simulation_runner.log')
    
    level = logging.DEBUG if debug else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='AprilSLAM Simulation Runner - Modular Architecture v2.0',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_simulation.py                           # Run with default settings
    python run_simulation.py --config my_config.json  # Use custom config
    python run_simulation.py --no-movement            # Auto movement mode
    python run_simulation.py --debug                   # Enable debug logging

Controls (Manual Movement Mode):
    Arrow Keys: Move camera X/Y
    W/S: Move camera forward/backward
    A/D: Yaw left/right
    Q/E: Roll left/right
    R/F: Pitch up/down
    ESC/Close Window: Exit simulation

The simulation uses a modular architecture providing:
    • Realistic 3D rendering with OpenGL
    • Camera control with keyboard or Monte Carlo
    • SLAM algorithm integration with AprilTags
    • Ground truth calculation for evaluation
    • Comprehensive CSV data logging
    • Modular design for easy extension
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        default=os.path.join(project_root, 'config', 'sim_settings.json'),
        help='Path to simulation configuration file (default: config/sim_settings.json)'
    )
    
    parser.add_argument(
        '--no-movement',
        action='store_true',
        help='Disable manual movement (use automatic Monte Carlo positioning)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    parser.add_argument(
        '--legacy',
        action='store_true',
        help='Use legacy simulation class (deprecated, shows warnings)'
    )
    
    return parser.parse_args()


def main():
    """Main simulation runner."""
    print("🚀 AprilSLAM Simulation Runner v2.0")
    print("📦 Modular Architecture Edition")
    print("=" * 50)
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up logging
    setup_logging(args.debug)
    logger = logging.getLogger(__name__)
    
    # Validate configuration file
    if not os.path.exists(args.config):
        logger.error(f"❌ Configuration file not found: {args.config}")
        print(f"❌ Configuration file not found: {args.config}")
        sys.exit(1)
    
    logger.info(f"📋 Using configuration: {args.config}")
    logger.info(f"🎮 Manual movement: {'disabled' if args.no_movement else 'enabled'}")
    
    if args.legacy:
        logger.warning("⚠️ Using legacy simulation class (deprecated)")
        print("⚠️ WARNING: Using deprecated legacy simulation class")
        print("Please switch to the new modular system for better performance")
        from src.simulation.sim import Simulation
        simulation_class = Simulation
        init_args = (args.config, not args.no_movement)
        init_kwargs = {}
    else:
        logger.info("✨ Using new modular simulation engine")
        simulation_class = SimulationEngine
        init_args = (args.config,)
        init_kwargs = {'movement_enabled': not args.no_movement}
    
    try:
        # Create simulation
        logger.info("🔧 Initializing simulation components...")
        if args.legacy:
            simulation = simulation_class(*init_args, **init_kwargs)
        else:
            simulation = simulation_class(*init_args, **init_kwargs)
        
        logger.info("🎯 Starting AprilSLAM simulation...")
        print("🎯 Starting AprilSLAM simulation...")
        print("Press Ctrl+C to stop the simulation")
        
        if not args.no_movement:
            print("\n🎮 Manual Movement Controls:")
            print("  📐 Arrow Keys: Move X/Y  |  W/S: Move Z")
            print("  🔄 A/D: Yaw  |  Q/E: Roll  |  R/F: Pitch")
            print("  🚪 ESC/Close Window: Exit")
        else:
            print("\n🎲 Monte Carlo Mode: Camera position will be randomized")
        
        print("=" * 50)
        
        # Run the simulation with context manager if available
        if hasattr(simulation, '__enter__'):
            with simulation:
                simulation.run()
        else:
            simulation.run()
        
    except KeyboardInterrupt:
        logger.info("🛑 Simulation interrupted by user")
        print("\n🛑 Simulation stopped by user")
    
    except FileNotFoundError as e:
        logger.error(f"❌ File not found: {e}")
        print(f"❌ Error: File not found - {e}")
        print("💡 Check that all required files exist and paths are correct")
        sys.exit(1)
    
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        print(f"❌ Import Error: {e}")
        print("💡 Check that all dependencies are installed:")
        print("   pip install pygame PyOpenGL opencv-python numpy termcolor")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"❌ Simulation failed: {e}", exc_info=True)
        print(f"❌ Simulation failed: {e}")
        if not args.debug:
            print("💡 Try running with --debug for more information")
        sys.exit(1)
    
    finally:
        logger.info("🧹 Cleaning up...")
        print("✅ Simulation completed")


if __name__ == "__main__":
    main() 