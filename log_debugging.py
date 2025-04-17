import re
from collections import defaultdict

import matplotlib.pyplot as plt

# Function to parse the log file and extract data
def parse_log_file(file_path):
    tag_data = defaultdict(list)
    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(r"Tag ID: (\d+), Tag to World GT Distance: ([\d.]+)", line)
            if match:
                tag_id = int(match.group(1))
                gt_distance = float(match.group(2))
                tag_data[tag_id].append(gt_distance)
    return tag_data

# Function to plot the change of GT distance over time for each tag ID
def plot_gt_distance(tag_data):
    for tag_id, distances in tag_data.items():
        plt.plot(distances, label=f"Tag ID {tag_id}")
    plt.xlabel("Time (arbitrary units)")
    plt.ylabel("GT Distance")
    plt.title("Change of GT Distance Over Time")
    plt.legend()
    plt.show()

# Main function
def main():
    log_file_path = "last_run.log"  # Replace with the actual log file path
    tag_data = parse_log_file(log_file_path)
    plot_gt_distance(tag_data)

if __name__ == "__main__":
    main()