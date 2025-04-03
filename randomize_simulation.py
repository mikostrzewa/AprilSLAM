import json
import random
import argparse
import os

def randomize_value(val, pct):
    # For non-zero values: new = val * (1 + random offset)
    # For zero, apply a small absolute offset based on pct (using 1 as base)
    if val != 0:
        return val * (1 + random.uniform(-pct, pct))
    else:
        return random.uniform(-pct, pct)

def main():
    parser = argparse.ArgumentParser(description="Randomize sim settings by a given percentage.")
    parser.add_argument("--percentage", type=float, default=0.1, help="Allowed randomization percentage (e.g., 0.1 for 10%%)")
    args = parser.parse_args()
    
    # Define file paths
    base_path = "/Users/astra/Documents/AprilSLAM"
    input_file = os.path.join(base_path, "sim_settings.json")
    output_file = os.path.join(base_path, "sim_settings_randomized.json")
    
    # Load sim settings
    with open(input_file, "r") as f:
        settings = json.load(f)
    
    # Randomize positions and rotations of each tag
    for tag in settings.get("tags", []):
        tag["position"] = [randomize_value(coord, args.percentage) for coord in tag.get("position", [])]
        tag["rotation"] = [randomize_value(angle, args.percentage) for angle in tag.get("rotation", [])]
    
    # Write randomized settings to new file
    with open(output_file, "w") as f:
        json.dump(settings, f, indent=4)
    
    print(f"Randomized settings written to {output_file}")

if __name__ == "__main__":
    main()
