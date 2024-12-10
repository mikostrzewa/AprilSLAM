import sys
import json
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import cv2
from apriltag import apriltag
from slam import SLAM
import csv
import time

#TODO: Add covarince matrix to the graph
class Simulation:
    def __init__(self, settings_file):
        self.load_settings(settings_file)
        self.init_pygame()
        self.load_textures()
        self.init_opengl()
        self.init_slam()
        self.camera_position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.movement_speed = 1 * self.size_scale
        self.key_state = {pygame.K_LEFT: False, pygame.K_RIGHT: False, pygame.K_UP: False, pygame.K_DOWN: False, pygame.K_w: False, pygame.K_s: False}
        self.start_time = time.time()
        self.csvfile = open('error_data.csv', 'w', newline='')
        self.csvwriter = csv.writer(self.csvfile)
        # Write the header
        self.csvwriter.writerow([
            'Time','Number of Nodes',"Avrg Distance",
            'Est_X', 'Est_Y', 'Est_Z',
            'Est_Roll', 'Est_Pitch', 'Est_Yaw',
            'GT_X', 'GT_Y', 'GT_Z',
            'GT_Roll', 'GT_Pitch', 'GT_Yaw','Translation Difference','Rotation Difference'
        ])
        
    def rotation_matrix_to_euler_angles(self,R):
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0

        return np.array([x, y, z])
    def load_settings(self, settings_file):
        with open(settings_file, 'r') as f:
            self.settings = json.load(f)
        self.display_width = self.settings["display_width"]
        self.display_height = self.settings["display_height"]
        self.fov_y = self.settings["fov_y"]
        self.near_clip = self.settings["near_clip"]
        self.far_clip = self.settings["far_clip"]
        self.size_scale = self.settings["size_scale"]
        self.tag_size_inner = self.settings["tag_size_inner"] * self.size_scale
        self.tag_size_outer = self.settings["tag_size_outer"] * self.size_scale
        self.tags = self.settings["tags"]

    def init_pygame(self):
        pygame.init()
        self.display = (self.display_width, self.display_height)
        pygame.display.set_mode(self.display, DOUBLEBUF | OPENGL)
        glEnable(GL_TEXTURE_2D)

    def load_textures(self):
        self.tags_data = []
        for tag in self.tags:
            texture, _, _ = self.load_texture(tag["image"])
            position = np.array(tag["position"], dtype=np.float32)
            rotation = np.array(tag["rotation"], dtype=np.float32)
            self.tags_data.append({"id": tag["id"], "texture": texture, "position": position, "rotation": rotation})

    def load_texture(self, image):
        textureSurface = pygame.image.load(image).convert_alpha()
        width = textureSurface.get_width()
        height = textureSurface.get_height()
        textureData = pygame.image.tostring(textureSurface, "RGBA", True)
        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, textureData)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        return texture, width, height

    def init_opengl(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(self.fov_y, (self.display_width / self.display_height), self.near_clip, self.far_clip)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def init_slam(self):
        # Calculate camera parameters
        fx = fy = 0.5 * self.display_height / np.tan(0.5 * np.radians(self.fov_y))
        cx = 0.5 * self.display_width
        cy = 0.5 * self.display_height
        camera_matrix = np.array([[fx, 0, cx],
                                  [0, fy, cy],
                                  [0, 0, 1]])
        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        camera_params = {'camera_matrix': camera_matrix, 'dist_coeffs': dist_coeffs}
        self.slam = SLAM(camera_params, tag_size=self.tag_size_inner)

    def ground_truth(self,tag_id=0):
        # Find the tag with the smallest id
        tag = self.tags_data[tag_id]
        
        # Adjust tag position by subtracting camera position
        position = tag["position"] - self.camera_position

        # Flip y and z axes (OpenGL coordinate system adjustment)
        position[1:] = -position[1:]  # Flip y and z axes
        
        # Extract rotation
        rotation = np.radians(tag["rotation"])
        
        # Create rotation matrix from Euler angles (using the ZYX convention)
        rz = np.array([[np.cos(rotation[2]), -np.sin(rotation[2]), 0],
                       [np.sin(rotation[2]), np.cos(rotation[2]), 0],
                       [0, 0, 1]])
        
        ry = np.array([[np.cos(rotation[1]), 0, np.sin(rotation[1])],
                       [0, 1, 0],
                       [-np.sin(rotation[1]), 0, np.cos(rotation[1])]])
        
        rx = np.array([[1, 0, 0],
                       [0, np.cos(rotation[0]), -np.sin(rotation[0])],
                       [0, np.sin(rotation[0]), np.cos(rotation[0])]])
        
        # Combined rotation matrix (rotation order: ZYX)
        rotation_matrix = rz @ ry @ rx

        # Flip axes to match the coordinate system
        flip_x = np.array([[1, 0, 0],
                           [0, -1, 0],
                           [0, 0, -1]])
        
        rotation_matrix = flip_x @ rotation_matrix
        
        # Invert the rotation matrix (transpose for orthogonal matrices)
        inverse_rotation_matrix = rotation_matrix.T
        
        # Invert the translation vector using the inverse rotation matrix
        inverted_translation = -inverse_rotation_matrix @ position
        
        # Combine rotation matrix and inverted translation to form the transformation matrix
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = inverse_rotation_matrix
        transformation_matrix[:3, 3] = inverted_translation
        
        return transformation_matrix

    def ground_truth_difference(self, tag1_id, tag2_id):
        tag1 = self.tags_data[tag1_id]
        tag2 = self.tags_data[tag2_id]
        trans_diff = np.linalg.norm(tag1["position"] - tag2["position"])

        return trans_diff

        pass
    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key in self.key_state:
                        self.key_state[event.key] = True
                elif event.type == pygame.KEYUP:
                    if event.key in self.key_state:
                        self.key_state[event.key] = False

            if self.key_state[pygame.K_LEFT]:
                self.camera_position[0] -= self.movement_speed
            if self.key_state[pygame.K_RIGHT]:
                self.camera_position[0] += self.movement_speed
            if self.key_state[pygame.K_UP]:
                self.camera_position[1] += self.movement_speed
            if self.key_state[pygame.K_DOWN]:
                self.camera_position[1] -= self.movement_speed
            if self.key_state[pygame.K_w]:
                self.camera_position[2] -= self.movement_speed
            if self.key_state[pygame.K_s]:
                self.camera_position[2] += self.movement_speed

            glClearColor(0.5, 0.0, 0.5, 1.0)  # Clear to purple background
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            for tag in self.tags_data:
                glLoadIdentity()
                tag_position = tag["position"] - self.camera_position
                glTranslatef(*tag_position)
                rotation = tag["rotation"]
                glRotatef(rotation[0], 1, 0, 0)  # Rotate around x-axis
                glRotatef(rotation[1], 0, 1, 0)  # Rotate around y-axis
                glRotatef(rotation[2], 0, 0, 1)  # Rotate around z-axis
                glBindTexture(GL_TEXTURE_2D, tag["texture"])
                # Render textured quad with size based on tag_size
                size = (self.tag_size_outer) / 2
                glBegin(GL_QUADS)
                glTexCoord2f(0, 0); glVertex3f(-size, -size, 0)
                glTexCoord2f(1, 0); glVertex3f(size, -size, 0)
                glTexCoord2f(1, 1); glVertex3f(size, size, 0)
                glTexCoord2f(0, 1); glVertex3f(-size, size, 0)
                glEnd()

            # Capture the rendered image
            width, height = self.display
            data = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
            image = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
            image = np.flipud(image)  # Flip the image vertically
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Detect AprilTags in the frame
            detections = self.slam.detect(image)

            # Process each detected tag
            for detection in detections:
                retval, rvec, tvec = self.slam.get_pose(detection)
                if retval:
                    corners = np.array(detection['lb-rb-rt-lt'], dtype=np.float32)
                    image = self.slam.draw(rvec, tvec, corners, image, detection['id'])

            # Display the resulting image with detections
            cv2.imshow('AprilTag Detection', image)

            pose = self.slam.my_pose()
            if pose is not None:
                translation_vector = pose[:3, 3]
                rotation_matrix = pose[:3, :3]
                print("Estimated Translation Vector:", translation_vector)
                print("Estimated Rotation Matrix:\n", rotation_matrix)

                ground_truth_tags = [{} for _ in range(len(self.slam.graph))]
                for tag_id, node in self.slam.graph.items():
                    true_pose = self.ground_truth(tag_id)
                    translation_dist = np.linalg.norm(true_pose[:3, 3])
                    ground_truth_tags[tag_id]["local"] = translation_dist

                    tag_to_world = np.linalg.norm(self.ground_truth_difference(self.slam.coordinate_id, tag_id))
                    ground_truth_tags[tag_id]["world"] = tag_to_world

                
                self.slam.error_graph(ground_truth_tags)

                # Compare with ground truth
                ground_truth_pose = self.ground_truth()
                gt_translation_vector = ground_truth_pose[:3, 3]
                gt_rotation_matrix = ground_truth_pose[:3, :3]
                print("Ground Truth Translation Vector:", gt_translation_vector)
                print("Ground Truth Rotation Matrix:\n", gt_rotation_matrix)

                # Calculate differences
                translation_diff = np.linalg.norm(translation_vector - gt_translation_vector)
                rotation_diff = np.linalg.norm(rotation_matrix - gt_rotation_matrix)
                print("Translation Difference:", translation_diff)
                print("Rotation Difference:", rotation_diff)
                self.slam.vis_slam(ground_truth=ground_truth_pose)

                current_time = time.time() - self.start_time
                 # Estimated pose
                translation_vector = pose[:3, 3]
                rotation_matrix = pose[:3, :3]
                est_angles = self.rotation_matrix_to_euler_angles(rotation_matrix)

                # Ground truth pose
                ground_truth_pose = self.ground_truth()
                gt_translation_vector = ground_truth_pose[:3, 3]
                gt_rotation_matrix = ground_truth_pose[:3, :3]
                gt_angles = self.rotation_matrix_to_euler_angles(gt_rotation_matrix)

                # Write data to CSV
                
                self.csvwriter.writerow([
                    current_time,len(self.slam.graph),self.slam.average_distance_to_nodes(),
                    translation_vector[0], translation_vector[1], translation_vector[2],
                    est_angles[0], est_angles[1], est_angles[2],
                    gt_translation_vector[0], gt_translation_vector[1], gt_translation_vector[2],
                    gt_angles[0], gt_angles[1], gt_angles[2],translation_diff,rotation_diff
                ])
                

            self.slam.slam_graph()
            
            pygame.display.flip()
            pygame.time.wait(1)

    
if __name__ == '__main__':
    sim = Simulation('sim_settings.json')
    try:
        sim.run()
    except KeyboardInterrupt:
        sim.csvfile.close()
        pygame.quit()
        sys.exit()
