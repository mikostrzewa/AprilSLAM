import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D


class SLAMVisualizer:
    """Handles all SLAM visualization including 3D plots and graph visualization"""
    
    def __init__(self):
        plt.ion()  # Turn on interactive mode
        self.fig_vis = plt.figure("3D Visualization")
        self.fig_graph = plt.figure("SLAM Graph")
        self.err_graph = plt.figure("Error Graph")
        self.ax_vis = self.fig_vis.add_subplot(111, projection='3d')
        self.ax_graph = self.fig_graph.add_subplot(111)
        self.ax_err = self.err_graph.add_subplot(111)
    
    def vis_slam(self, graph, estimated_pose, ground_truth=None):
        """Visualize the 3D SLAM state"""
        self.ax_vis.cla()

        xs = []
        ys = []
        zs = []
        colors = []

        for node in graph.values():
            world_pos = node.world[:3, 3]  # Extract the translation vector (world position)
            xs.append(world_pos[0])
            ys.append(world_pos[1])
            zs.append(world_pos[2])
            if not node.visible:
                colors.append('red')
            elif not node.updated:
                colors.append('orange')
            else:
                colors.append('green')

        if ground_truth is not None:
            ground_truth_pos = ground_truth[:3, 3]
            xs.append(ground_truth_pos[0])
            ys.append(ground_truth_pos[1])
            zs.append(ground_truth_pos[2])
            colors.append('blue')

        # Add the estimated pose as a purple point
        estimated_pos = estimated_pose[:3, 3]
        xs.append(estimated_pos[0])
        ys.append(estimated_pos[1])
        zs.append(estimated_pos[2])
        colors.append('purple')
        
        self.ax_vis.set_xlabel('X')
        self.ax_vis.set_ylabel('Y')
        self.ax_vis.set_zlabel('Z')

        # Set the limits of the axes based on the data
        if xs and ys and zs:  # Check if we have data points
            self.ax_vis.set_xlim(min(xs), max(xs))
            self.ax_vis.set_ylim(min(ys), max(ys))
            self.ax_vis.set_zlim(min(zs), max(zs))

        # Render the points with increased size
        self.ax_vis.scatter(xs, ys, zs, c=colors, s=50)  # Adjust 's' to change the size of the points

        # Add legend
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Not Visible', markersize=10, markerfacecolor='red'),
            Line2D([0], [0], marker='o', color='w', label='Not Updated', markersize=10, markerfacecolor='orange'),
            Line2D([0], [0], marker='o', color='w', label='Updated', markersize=10, markerfacecolor='green'),
            Line2D([0], [0], marker='o', color='w', label='Ground Truth', markersize=10, markerfacecolor='blue'),
            Line2D([0], [0], marker='o', color='w', label='Estimated Pose', markersize=10, markerfacecolor='purple')
        ]
        self.ax_vis.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.5, 1))

        self.fig_vis.canvas.draw()
        self.fig_vis.canvas.flush_events()

    def slam_graph(self, graph):
        """Visualize the SLAM graph structure"""
        G = nx.Graph()
        for node_id, node in graph.items():
            G.add_node(node_id)
        for node_id, node in graph.items():
            if node.reference in graph:
                G.add_edge(node_id, node.reference, weight=node.weight)
        
        self.ax_graph.cla()  # Clear the current axes
        pos = nx.circular_layout(G)
        node_colors = [
            "red" if not node.visible else
            "orange" if not node.updated else
            "green" for node in graph.values()
        ]
        nx.draw(
            G, pos, with_labels=True, node_size=700, node_color=node_colors,
            font_size=10, font_color="black", font_weight="bold", ax=self.ax_graph
        )
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=self.ax_graph)

        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Not Visible', markersize=10, markerfacecolor='red'),
            Line2D([0], [0], marker='o', color='w', label='Not Updated', markersize=10, markerfacecolor='orange'),
            Line2D([0], [0], marker='o', color='w', label='Updated', markersize=10, markerfacecolor='green')
        ]
        self.ax_graph.legend(handles=legend_elements, loc='upper right')
        
        self.fig_graph.canvas.draw()
        self.fig_graph.canvas.flush_events()

    def error_graph(self, graph, ground_truth_graph):
        """Visualize the error graph with ground truth comparisons"""
        G = nx.Graph()
        
        G.add_node("Camera")

        for node_id, node in graph.items():
            G.add_node(node_id)
        for node_id, node in graph.items():
            if node.reference in graph:
                diff_world = abs(np.linalg.norm(node.world[:3, 3]) - ground_truth_graph[node_id]["world"])

                G.add_edge(node_id, node.reference, weight=round(diff_world, 3))

                diff_local = abs(np.linalg.norm(node.local[:3, 3]) - ground_truth_graph[node_id]["local"])
                G.add_edge(node_id, "Camera", weight=round(diff_local, 3))

        self.ax_err.cla()  # Clear the current axes
        pos = nx.planar_layout(G)
        
        # Define thresholds for edge colors
        GREEN_THRESHOLD = 1
        YELLOW_THRESHOLD = 2.5
        ORANGE_THRESHOLD = 5

        edge_colors = []
        for _, _, d in G.edges(data=True):
            weight = d['weight']
            if weight <= GREEN_THRESHOLD:
                edge_colors.append('green')
            elif weight <= YELLOW_THRESHOLD:
                edge_colors.append('yellow')
            elif weight <= ORANGE_THRESHOLD:
                edge_colors.append('orange')
            else:
                edge_colors.append('red')

        node_colors = ["pink"]  # for the "Camera" node
        node_colors.extend(
            "red" if not node.visible else
            "orange" if not node.updated else
            "green"
            for node in graph.values()
        )

        nx.draw(
            G, pos, with_labels=True, node_size=700, node_color=node_colors,
            font_size=10, font_color="black", font_weight="bold", edge_color=edge_colors, ax=self.ax_err
        )
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=self.ax_err)
        
        # Add legend
        legend_elements = [
            Line2D([0], [0], color='green', lw=2, label='Low Error'),
            Line2D([0], [0], color='yellow', lw=2, label='Moderate Error'),
            Line2D([0], [0], color='orange', lw=2, label='High Error'),
            Line2D([0], [0], color='red', lw=2, label='Severe Error')
        ]
        self.ax_err.legend(handles=legend_elements, loc='upper right')
        
        self.err_graph.canvas.draw()
        self.err_graph.canvas.flush_events() 