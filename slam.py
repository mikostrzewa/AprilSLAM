import numpy as np
import cv2
from apriltag import apriltag
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from time import sleep

# TODO: Add flexibility with the world tag

class Node:
    def __init__(self, local, world, reference, weight=1, updated=True, visible=False):
        self.local = local
        self.world = world
        self.reference = reference
        self.weight = weight
        self.updated = updated
        self.visible = visible


class SLAM:
    def __init__(self,logger, camera_params, tag_type="tagStandard41h12", tag_size=0.06):
        self.logger = logger
        self.logger.info("Initializing SLAM")
        self.detector = apriltag(tag_type)
        self.tag_size = tag_size
        self.camera_matrix = camera_params['camera_matrix']
        self.dist_coeffs = camera_params['dist_coeffs']
        self.graph = {}
        self.visible_tags = []
        self.coordinate_id = -1
        plt.ion()  # Turn on interactive mode
        self.fig_vis = plt.figure("3D Visualization")
        self.fig_graph = plt.figure("SLAM Graph")
        self.err_graph = plt.figure("Error Graph")
        self.ax_vis = self.fig_vis.add_subplot(111, projection='3d')
        self.ax_graph = self.fig_graph.add_subplot(111)
        self.ax_err = self.err_graph.add_subplot(111)
        self.estimated_pose = np.zeros((4, 4))  

    def transformation(self, rvec, tvec):
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, 3] = tvec.flatten()
        return transformation_matrix

    def invert(self, T):
        return np.linalg.inv(T)
    
    def detect(self, image):
        self.visible_tags = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detections = self.detector.detect(gray)
        detections = sorted(detections, key=lambda x: x['id'])
        self.visible_tags = [detection['id'] for detection in detections]
        return detections
    
    def get_pose(self, detection):
        corners = np.array(detection['lb-rb-rt-lt'], dtype=np.float32)

        # Define the 3D coordinates of the tag's corners in the tag's coordinate frame
        obj_points = np.array([[-self.tag_size / 2, -self.tag_size / 2, 0],
                            [ self.tag_size / 2, -self.tag_size / 2, 0],
                            [ self.tag_size / 2,  self.tag_size / 2, 0],
                            [-self.tag_size / 2,  self.tag_size / 2, 0]], dtype=np.float32)

        # Estimate the pose of the tag
        retval, rvec, tvec = cv2.solvePnP(obj_points, corners, self.camera_matrix, self.dist_coeffs)
        T = self.transformation(rvec, tvec)
        if self.coordinate_id == -1 or self.coordinate_id == detection['id']:
            self.coordinate_id = detection['id']
            self.graph[self.coordinate_id] = Node(self.invert(T), np.eye(4), self.coordinate_id)
        elif detection['id'] < self.coordinate_id:
            self.coordinate_id = detection['id']
            self.graph[self.coordinate_id] = Node(self.invert(T), np.eye(4), self.coordinate_id)
            self.update_world() #This updates all transformations for the new world coordinate
        else:
            reference = min(self.visible_tags) #This might be the issue 
            if(reference == self.coordinate_id):
                # If the reference is is visibale
                self.graph[detection['id']] = Node(self.invert(T), self.get_world(reference,T), self.coordinate_id) # Let's assume this is correct for now
            elif(detection['id'] in self.graph and self.graph[detection['id']].reference == self.coordinate_id):
                # If the detection is already in the graph and its reference is the coordinate_id
                self.logger.info(f"World not updated! Detection ID: {detection['id']}, Node World: {self.graph[detection['id']].world}")
                node = self.graph[detection['id']]
                self.graph[detection['id']] = Node(self.invert(T), node.world, self.coordinate_id, weight=node.weight, updated=False)
            elif(reference != detection['id'] and reference in self.graph):
                world,weight,new_reference = self.find_world(reference,T)
                self.graph[detection['id']] = Node(self.invert(T), world, new_reference, weight, updated=self.graph[reference].updated)
            else:
                print("Cannot find world reference")

       
        return retval, rvec, tvec
    
    def find_world(self, reference, T):
        world = np.matmul(self.get_world(reference,T),self.graph[reference].world)
        weight = self.graph[reference].weight+1
        new_reference = self.graph[reference].reference
        return world,weight,new_reference
    
    def euler_angles(self, rvec):
        rot_matrix, _ = cv2.Rodrigues(rvec)
        sy = np.sqrt(rot_matrix[0, 0] ** 2 + rot_matrix[1, 0] ** 2)
        singular = sy < 1e-6
        if not singular:
            yaw = np.arctan2(rot_matrix[0, 2], rot_matrix[2, 2])  # Yaw (rotation about Y-axis)
            pitch = np.arctan2(-rot_matrix[1, 2], sy)             # Pitch (rotation about X-axis)
            roll = np.arctan2(rot_matrix[1, 0], rot_matrix[1, 1]) # Roll (rotation about Z-axis)
        else:
            yaw = np.arctan2(-rot_matrix[2, 0], rot_matrix[0, 0]) # Yaw (rotation about Y-axis)
            pitch = np.arctan2(-rot_matrix[1, 2], sy)             # Pitch (rotation about X-axis)
            roll = 0                                             # Roll (rotation about Z-axis)

        return np.degrees([yaw, pitch, roll])
    
    def distance(self, tvec):
        return np.linalg.norm(tvec)
    
    def draw(self, rvec, tvec, corners, image, tag_id):
        for i in range(4):
            pt1 = tuple(map(int, corners[i]))
            pt2 = tuple(map(int, corners[(i + 1) % 4]))
            cv2.line(image, pt1, pt2, (0, 255, 0), 2)
        
        yaw, pitch, roll = self.euler_angles(rvec)
        distance_tvec = self.distance(tvec)
        text = f'ID: {tag_id}, Dist: {distance_tvec:.1f} units'
        text2 = f'Yaw: {yaw:.1f}, Pitch: {pitch:.1f}, Roll: {roll:.1f}'
        cv2.putText(image, text, (pt1[0], pt1[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(image, text2, (pt1[0], pt1[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.drawFrameAxes(image, self.camera_matrix, self.dist_coeffs, rvec, tvec, 2)
        return image
    
    def my_pose(self):
        if not self.visible_tags:
            return None
        
        T_sum = np.zeros((4, 4))
        count = 0

        for node in self.graph.values():
            node.visible = False
        for tag_id in self.visible_tags:
            node = self.graph.get(tag_id)
            node.visible = True
            T = np.matmul(node.local, node.world)
            if T is not None:
                T_sum += T/node.weight
                count += 1/node.weight
        
        if count == 0:
            return None
        
        T_avg = T_sum / count
        self.estimated_pose = T_avg 
        return T_avg
    
    def get_world(self,reference,T):
        return np.matmul(T,self.graph[reference].local) #this is the issue, as the reference local is wrong
    def update_world(self):
        print("No world update")
        # TODO: Implement the update_world method
        pass

    def average_distance_to_nodes(self):
        total_distance = 0
        node_count = 0

        for node in self.graph.values():
            translation = node.local[:3, 3]
            distance = np.linalg.norm(translation)
            total_distance += distance
            node_count += 1

        if node_count == 0:
            return 0

        average_distance = total_distance / node_count
        return average_distance

    def slam_graph(self):
        G = nx.Graph()
        for node_id, node in self.graph.items():
            G.add_node(node_id)
        for node_id, node in self.graph.items():
            if node.reference in self.graph:
                G.add_edge(node_id, node.reference, weight=node.weight)
        self.ax_graph.cla()  # Clear the current axes
        pos = nx.circular_layout(G)
        node_colors = [
            "red" if not node.visible else
            "orange" if not node.updated else
            "green" for node in self.graph.values()
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

    def vis_slam(self, ground_truth=None):
        self.ax_vis.cla()

        xs = []
        ys = []
        zs = []
        colors = []

        for node in self.graph.values():
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
        estimated_pos = self.estimated_pose[:3, 3]
        xs.append(estimated_pos[0])
        ys.append(estimated_pos[1])
        zs.append(estimated_pos[2])
        colors.append('purple')
        
        self.ax_vis.set_xlabel('X')
        self.ax_vis.set_ylabel('Y')
        self.ax_vis.set_zlabel('Z')

        # Set the limits of the axes based on the data
        self.ax_vis.set_xlim(min(xs), max(xs))
        self.ax_vis.set_ylim(min(ys), max(ys))
        self.ax_vis.set_zlim(min(zs), max(zs))

        # Render the points with increased size
        self.ax_vis.scatter(xs, ys, zs, c=colors, s=50)  # Adjust 's' to change the size of the points

        # Add legend
        from matplotlib.lines import Line2D
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

    def error_graph(self, ground_truth_graph):
        G = nx.Graph()
        
        G.add_node("Camera")

        for node_id, node in self.graph.items():
            G.add_node(node_id)
        for node_id, node in self.graph.items():
            if node.reference in self.graph:
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
            for node in self.graph.values()
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







