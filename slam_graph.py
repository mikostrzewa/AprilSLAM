import numpy as np

# TODO: Add covariance matrix to the graph

class Node:
    def __init__(self, local, world, reference, weight=1, updated=True, visible=False):
        self.local = local
        self.world = world
        self.reference = reference
        self.weight = weight
        self.updated = updated
        self.visible = visible


class SLAMGraph:
    """Handles the SLAM graph structure and coordinate transformations"""
    
    def __init__(self, logger):
        self.logger = logger
        self.graph = {}
        self.visible_tags = []
        self.coordinate_id = -1
        self.estimated_pose = np.zeros((4, 4))
    
    def invert(self, T):
        """Invert a transformation matrix"""
        return np.linalg.inv(T)
    
    def add_or_update_node(self, tag_id, T, visible_tags):
        """Add a new node to the graph or update an existing one"""
        self.visible_tags = visible_tags
        
        if self.coordinate_id == -1 or self.coordinate_id == tag_id:
            self.coordinate_id = tag_id
            self.graph[self.coordinate_id] = Node(self.invert(T), np.eye(4), self.coordinate_id)
        elif tag_id < self.coordinate_id:
            self.coordinate_id = tag_id
            self.graph[self.coordinate_id] = Node(self.invert(T), np.eye(4), self.coordinate_id)
            self.update_world()  # This updates all transformations for the new world coordinate
        else:
            reference = min(self.visible_tags)  # This might be the issue 
            if reference == self.coordinate_id:
                # If the reference is visible
                self.graph[tag_id] = Node(self.invert(T), self.get_world(reference, T), self.coordinate_id)
                # Log the result of get_world's translational vector length
                world_transform = self.get_world(reference, T)
                translation_vector = world_transform[:3, 3]
                vector_length = np.linalg.norm(translation_vector)
                self.logger.info(f"Tag ID {tag_id} (reference: {reference}): World transform translation length = {vector_length}")
            elif tag_id in self.graph and self.graph[tag_id].reference == self.coordinate_id:
                # If the detection is already in the graph and its reference is the coordinate_id
                self.logger.info(f"World not updated! Detection ID: {tag_id}, Node World: {self.graph[tag_id].world}")
                node = self.graph[tag_id]
                self.graph[tag_id] = Node(self.invert(T), node.world, self.coordinate_id, weight=node.weight, updated=False)
            elif reference != tag_id and reference in self.graph:
                world, weight, new_reference = self.find_world(reference, T)
                self.graph[tag_id] = Node(self.invert(T), world, new_reference, weight, updated=self.graph[reference].updated)
            else:
                print("Cannot find world reference")
    
    def find_world(self, reference, T):
        """Find world coordinates through reference transformations"""
        world = np.matmul(self.graph[reference].world, self.get_world(reference, T))
        weight = self.graph[reference].weight + 1
        new_reference = self.graph[reference].reference
        return world, weight, new_reference
    
    def get_world(self, reference, T):
        """Get world transformation from reference"""
        return self.graph[reference].local @ T
    
    def update_world(self):
        """Update world coordinates (TODO: Implement)"""
        print("No world update")
        # TODO: Implement the update_world method
        pass
    
    def my_pose(self):
        """Calculate the current pose estimate from visible tags"""
        if not self.visible_tags:
            return None
        
        T_sum = np.zeros((4, 4))
        count = 0

        for node in self.graph.values():
            node.visible = False
        
        for tag_id in self.visible_tags:
            if tag_id in self.graph:
                node = self.graph[tag_id]
                node.visible = True
                T = np.matmul(node.world, node.local)
                if T is not None:
                    T_sum += T / node.weight
                    count += 1 / node.weight
        
        if count == 0:
            return None
        
        T_avg = T_sum / count
        self.estimated_pose = T_avg 
        return T_avg
    
    def average_distance_to_nodes(self):
        """Calculate average distance to all nodes"""
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
    
    def get_nodes(self):
        """Get all nodes in the graph"""
        return self.graph
    
    def get_coordinate_id(self):
        """Get the current coordinate frame ID"""
        return self.coordinate_id
    
    def get_estimated_pose(self):
        """Get the current estimated pose"""
        return self.estimated_pose 