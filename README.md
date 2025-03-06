# AprilTag SLAM Simulation

This project is a simulation for testing and visualizing SLAM (Simultaneous Localization and Mapping) using AprilTags as landmarks. The simulation is built in Python and utilizes `sim.py` for rendering the environment and `slam.py` for processing AprilTag-based localization.

## Features

- **AprilTag-based Localization**: Uses fiducial markers for precise position tracking.
- **Customizable Simulation**: Adjustable display settings, field of view, and clipping planes.
- **Tag Management**: Define AprilTag properties such as position, rotation, and scale via `sim_settings.json`.
- **Scalable Environment**: Modify `size_scale` and other parameters to test various scenarios.

## Installation

Ensure you have Python installed and the necessary dependencies:

```bash
pip install numpy opencv-python matplotlib
```

## Usage

1. Modify `sim_settings.json` to adjust the environment settings.
2. Run the simulation:
   
   ```bash
   python sim.py
   ```

3. Execute the SLAM processing:

   ```bash
   python slam.py
   ```

## Configuration

Modify `sim_settings.json` to change simulation parameters:

```json
{
    "display_width": 1000,
    "display_height": 1000,
    "fov_y": 45,
    "near_clip": 0.1,
    "far_clip": 300.0,
    "size_scale": 2,
    "tag_size_inner": 5,
    "tag_size_outer": 9,
    "tags": [
        {
            "id": 0,
            "image": "tags/tag0.png",
            "position": [0, 0, -50],
            "rotation": [0, 0, 0]
        }
    ]
}
```

## File Structure

- **`sim.py`** - Handles the visualization of the AprilTag environment.
- **`slam.py`** - Processes AprilTags for localization and mapping.
- **`sim_settings.json`** - Stores simulation parameters such as tag placement and rendering options.

## Future Enhancements

- Implement real-time visualization for SLAM updates.
- Add noise modeling for more realistic testing.
- Integrate robot movement simulation.

## License

This project is licensed under the MIT License.

## Acknowledgments

- AprilTag detection library
- OpenCV for image processing
- Matplotlib for visualization
