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
    

    
    def get_nodes(self):
        """Get all nodes in the graph"""
        return self.graph
    
    def get_coordinate_id(self):
        """Get the current coordinate frame ID"""
        return self.coordinate_id
    
    def get_estimated_pose(self):
        """Get the current estimated pose"""
        return self.estimated_pose 