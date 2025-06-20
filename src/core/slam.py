import numpy as np
from src.detection.tag_detector import TagDetector
from src.core.slam_graph import SLAMGraph
from src.core.slam_visualizer import SLAMVisualizer

# TODO: Add flexibility with the world tag


class SLAM:
    def __init__(self, logger, camera_params, tag_type="tagStandard41h12", tag_size=0.06):
        self.logger = logger
        self.logger.info("Initializing SLAM")
        
        # Initialize components
        self.detector = TagDetector(camera_params, tag_type, tag_size)
        self.graph = SLAMGraph(logger)
        self.visualizer = SLAMVisualizer()
        
        self.visible_tags = []  

    def detect(self, image):
        """Detect AprilTags in the image"""
        detections = self.detector.detect(image)
        self.visible_tags = [detection['id'] for detection in detections]
        return detections
    
    def get_pose(self, detection):
        """Get the pose of a detected tag and update the graph"""
        retval, rvec, tvec, T = self.detector.get_pose(detection)
        if retval:
            self.graph.add_or_update_node(detection['id'], T, self.visible_tags)
        return retval, rvec, tvec
    

    
    def my_pose(self):
        """Calculate the current pose estimate from visible tags"""
        if not self.visible_tags:
            return None
        
        T_sum = np.zeros((4, 4))
        count = 0

        # Update node visibility
        for node in self.graph.get_nodes().values():
            node.visible = False
        
        for tag_id in self.visible_tags:
            if tag_id in self.graph.get_nodes():
                node = self.graph.get_nodes()[tag_id]
                node.visible = True
                T = np.matmul(node.world, node.local)
                if T is not None:
                    T_sum += T / node.weight
                    count += 1 / node.weight
        
        if count == 0:
            return None
        
        T_avg = T_sum / count
        # Update estimated pose in graph for reference
        self.graph.estimated_pose = T_avg 
        return T_avg
    
    def average_distance_to_nodes(self):
        """Calculate average distance to all nodes"""
        total_distance = 0
        node_count = 0

        for node in self.graph.get_nodes().values():
            translation = node.local[:3, 3]
            distance = np.linalg.norm(translation)
            total_distance += distance
            node_count += 1

        if node_count == 0:
            return 0

        average_distance = total_distance / node_count
        return average_distance

    @property
    def coordinate_id(self):
        """Get the current coordinate frame ID"""
        return self.graph.get_coordinate_id()

    def slam_graph(self):
        """Visualize the SLAM graph structure"""
        self.visualizer.slam_graph(self.graph.get_nodes())

    def vis_slam(self, ground_truth=None):
        """Visualize the 3D SLAM state"""
        self.visualizer.vis_slam(self.graph.get_nodes(), self.graph.get_estimated_pose(), ground_truth)

    def error_graph(self, ground_truth_graph):
        """Visualize the error graph"""
        self.visualizer.error_graph(self.graph.get_nodes(), ground_truth_graph)







