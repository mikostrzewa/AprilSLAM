import numpy as np
import cv2
from apriltag import apriltag

class SLAM:
    def __init__(self, camera_params, tag_type="tagStandard41h12", tag_size=0.06):
        self.detector = apriltag(tag_type)
        self.tag_size = tag_size
        self.camera_matrix = camera_params['camera_matrix']
        self.dist_coeffs = camera_params['dist_coeffs']
        self.tag_graph = {}
        self.visible_tags = []
        self.coordinate_id = -1

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

        self.tag_graph[detection['id']] = self.invert(T)
        self.visible_tags.append(detection['id'])   
        return retval, rvec, tvec
    
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
    
    def get_coordinate_id(self):
        if self.tag_graph:
            self.coordinate_id = min(self.tag_graph.keys())
        return self.coordinate_id
    
    def my_pose(self):
        if not self.visible_tags:
            return None
        
        T_sum = np.zeros((4, 4))
        count = 0
        
        for tag_id in self.visible_tags:
            T = self.tag_graph.get(tag_id)
            if T is not None:
                T_sum += T
                count += 1
        
        if count == 0:
            return None
        
        T_avg = T_sum / count
        return T_avg

    
    def graph_global(self):
        for tag_id, T in self.tag_graph.items():
            if self.coordinate_id != -1 and tag_id != self.coordinate_id:
                new_T = np.matmul(self.invert(self.tag_graph[self.coordinate_id]), T)
                self.tag_graph[tag_id] = new_T