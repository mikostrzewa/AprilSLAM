import numpy as np
from tag_detector import TagDetector
from slam_graph import SLAMGraph
from slam_visualizer import SLAMVisualizer

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
    
    def draw(self, rvec, tvec, corners, image, tag_id):
        """Draw detection visualization on image"""
        return self.detector.draw(rvec, tvec, corners, image, tag_id)
    
    def my_pose(self):
        """Get the current pose estimate"""
        return self.graph.my_pose()

    def average_distance_to_nodes(self):
        """Get average distance to all nodes"""
        return self.graph.average_distance_to_nodes()
    
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







