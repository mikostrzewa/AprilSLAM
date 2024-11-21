import numpy as np
import cv2
from apriltag import apriltag
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Node:
    def __init__(self, local, world, reference, weight=1, updated=True, visible=False):
        self.local = local
        self.world = world
        self.reference = reference
        self.weight = weight
        self.updated = updated
        self.visible = visible


class SLAM:
    def __init__(self, camera_params, tag_type="tagStandard41h12", tag_size=0.06):
        self.detector = apriltag(tag_type)
        self.tag_size = tag_size
        self.camera_matrix = camera_params['camera_matrix']
        self.dist_coeffs = camera_params['dist_coeffs']
        self.graph = {}
        self.visible_tags = []
        self.coordinate_id = -1
        plt.ion()  # Turn on interactive mode
        self.fig_vis = plt.figure()
        self.fig_graph = plt.figure()
        self.ax = self.fig_vis.add_subplot(111, projection='3d')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
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
            reference = min(self.visible_tags)
            if(reference == self.coordinate_id):
                self.graph[detection['id']] = Node(self.invert(T), self.get_world(reference,T), self.coordinate_id)
            elif(detection['id'] in self.graph and self.graph[detection['id']].reference == self.coordinate_id):
                node = self.graph[detection['id']]
                self.graph[detection['id']] = Node(self.invert(T), node.world, self.coordinate_id, weight=node.weight, updated=False)
            elif(reference != detection['id'] and reference in self.graph):
                print(f"Reference: {reference}, Detection: {detection['id']}")
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
        return np.matmul(T,self.graph[reference].local)
    def update_world(self):
        print("No world update")
        # TODO: Implement the update_world method
        pass
    def slam_graph(self):
        G = nx.Graph()
        for node_id, node in self.graph.items():
            G.add_node(node_id)
        for node_id, node in self.graph.items():
            if node.reference in self.graph:
                G.add_edge(node_id, node.reference, weight=node.weight)
        self.fig_graph.clf()  # Clear the current figure
        pos = nx.circular_layout(G)
        node_colors = [
            "red" if not node.visible else
            "orange" if not node.updated else
            "green" for node in self.graph.values()
        ]
        nx.draw(
            G, pos, with_labels=True, node_size=700, node_color=node_colors,
            font_size=10, font_color="black", font_weight="bold"
        )
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Not Visible', markersize=10, markerfacecolor='red'),
            Line2D([0], [0], marker='o', color='w', label='Not Updated', markersize=10, markerfacecolor='orange'),
            Line2D([0], [0], marker='o', color='w', label='Updated', markersize=10, markerfacecolor='green')
        ]
        self.fig_graph.legend(handles=legend_elements, loc='upper right')
        
        self.fig_graph.canvas.draw()
        self.fig_graph.canvas.flush_events()

    def vis_slam(self, ground_truth=None):
        self.ax.cla()

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
        
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        # Set the limits of the axes based on the data
        self.ax.set_xlim(min(xs), max(xs))
        self.ax.set_ylim(min(ys), max(ys))
        self.ax.set_zlim(min(zs), max(zs))

        # Render the points with increased size
        self.ax.scatter(xs, ys, zs, c=colors, s=50)  # Adjust 's' to change the size of the points

        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Not Visible', markersize=10, markerfacecolor='red'),
            Line2D([0], [0], marker='o', color='w', label='Not Updated', markersize=10, markerfacecolor='orange'),
            Line2D([0], [0], marker='o', color='w', label='Updated', markersize=10, markerfacecolor='green'),
            Line2D([0], [0], marker='o', color='w', label='Ground Truth', markersize=10, markerfacecolor='blue'),
            Line2D([0], [0], marker='o', color='w', label='Estimated Pose', markersize=10, markerfacecolor='purple')
        ]
        self.ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.5, 1))

        self.fig_vis.canvas.draw()
        self.fig_vis.canvas.flush_events()




