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
import logging
import os
from termcolor import colored

#TODO: Add covarince matrix to the graph

logging.basicConfig(
    filename='last_run.log',  # name of the file
    level=logging.DEBUG,          # level of messages to capture
    format='%(asctime)s - %(levelname)s - %(message)s'
)
class Simulation:
    def __init__(self, settings_file, movment_flag = True):
        # Clear the log file at the start of the simulation
        if os.path.exists('last_run.log'):
            open('last_run.log', 'w').close()
            logging.info("Simulation started")
        self.load_settings(settings_file)
        self.init_pygame()
        self.load_textures()
        self.init_opengl()
        self.init_slam()
        self.camera_position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.movement_flag = movment_flag
        self.movement_speed = 1 * self.size_scale
        self.key_state = {pygame.K_LEFT: False, pygame.K_RIGHT: False, pygame.K_UP: False, pygame.K_DOWN: False, pygame.K_w: False, pygame.K_s: False}
        self.start_time = time.time()
        self.csvfile = open('error_data.csv', 'w', newline='')
        self.error_file = open('error_params.csv', 'w', newline='')
        self.csvwriter = csv.writer(self.csvfile)
        self.csvwriter_errors = csv.writer(self.error_file)
        self.csvfile = open('error_data.csv', 'w', newline='')
        self.error_file = open('error_params.csv', 'w', newline='')
        self.csvwriter = csv.writer(self.csvfile)
        self.csvwriter_errors = csv.writer(self.error_file)
        self.covariance_file = open('covariance_data.csv', 'w', newline='')
        self.csvwriter_covariance = csv.writer(self.covariance_file)
        # Write the header
        self.csvwriter.writerow([
            'Time','Number of Nodes',"Avrg Distance",
            'Est_X', 'Est_Y', 'Est_Z',
            'Est_Roll', 'Est_Pitch', 'Est_Yaw',
            'GT_X', 'GT_Y', 'GT_Z',
            'GT_Roll', 'GT_Pitch', 'GT_Yaw','Translation Difference','Rotation Difference'
        ])
        
        self.csvwriter_errors.writerow([
            'Number of Jumps',
            'Est_X_Local', 'Est_Y_Local', 'Est_Z_Local',
            'Est_Roll_Local', 'Est_Pitch_Local', 'Est_Yaw_Local',
            'Est_X_World', 'Est_Y_World', 'Est_Z_World',
            'Est_Roll_World', 'Est_Pitch_World', 'Est_Yaw_World',
            'Tag_Est_X', 'Tag_Est_Y', 'Tag_Est_Z',
            'Tag_Est_Roll', 'Tag_Est_Pitch', 'Tag_Est_Yaw',
            'Error_World', 'Error_Local',
        ])

        self.csvwriter_covariance.writerow([
            'Number of Jumps',
            'Tag_Est_X', 'Tag_Est_Y', 'Tag_Est_Z',
            'Tag_Est_Roll', 'Tag_Est_Pitch', 'Tag_Est_Yaw',
            'Translation_Error',
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
        self.actual_tag_size = self.settings["actual_size_in_mm"]
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
        logging.info(f"Loaded texture with width: {width}, height: {height}")
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
        self.slam = SLAM(logging, camera_params, tag_size=self.tag_size_inner)

    def world_ground_truth(self, tag_id=0):
        Tw = self.ground_truth(self.slam.coordinate_id)

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
        # Combined rotation matrix (rotation order: XYZ)
        rotation_matrix = rz @ ry @ rx

        # Flip axes to match the coordinate system
        flip_x = np.array([[1, 0, 0],
                           [0, -1, 0],
                           [0, 0, -1]])
        
        rotation_matrix = flip_x @ rotation_matrix
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, 3] = position

        return transformation_matrix @ Tw

    def ground_truth(self,tag_id=0):
        tag = self.tags_data[tag_id]
    

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
        # Combined rotation matrix (rotation order: XYZ)
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
        trans_diff = np.linalg.norm((tag1["position"]-self.camera_position) - (tag2["position"]-self.camera_position))

        return trans_diff

    def mm_conversion(self, value):
        return value * self.actual_tag_size/self.tag_size_inner

    def monte_carlo_position_randomizer(self, bounds):
        x_min, x_max, y_min, y_max, z_min, z_max = bounds
        self.camera_position[0] = np.random.uniform(x_min, x_max)
        self.camera_position[1] = np.random.uniform(y_min, y_max)
        self.camera_position[2] = np.random.uniform(z_min, z_max)
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

                    
            if self.movement_flag:
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
            else:
                bounds = np.array([-3, 10, -1, 1, -0.25, 3])*5 #x_min, x_max, y_min, y_max, z_min, z_max
                self.monte_carlo_position_randomizer(bounds)



            glClearColor(0.5, 0.0, 0.5, 1.0)  # Clear to purple background
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            # Sort tags by z position from lowest to highest
            tags_with_z = []
            for tag in self.tags_data:
                tag_position = tag["position"] - self.camera_position
                z_pos = tag_position[2]
                tags_with_z.append((z_pos, tag))

            tags_sorted = sorted(tags_with_z, key=lambda x: x[0])

            for z_pos, tag in tags_sorted:
                glLoadIdentity()
                tag_position = tag["position"] - self.camera_position
                glTranslatef(*tag_position)
                rotation = tag["rotation"]
                glRotatef(rotation[2], 0, 0, 1)  # Rotate around global z-axis
                glRotatef(rotation[1], 0, 1, 0)  # Rotate around global y-axis
                glRotatef(rotation[0], 1, 0, 0)  # Rotate around global x-axis
                glBindTexture(GL_TEXTURE_2D, tag["texture"])
                # Render textured quad with size based on tag_size
                size = (self.tag_size_outer) / 2
                logging.debug(f"Tag size (outer/2): {size}")
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
            # Compare with ground truth
            ground_truth_pose = self.ground_truth()
            gt_translation_vector = ground_truth_pose[:3, 3]
            gt_rotation_matrix = ground_truth_pose[:3, :3]
            if pose is not None:
                translation_vector = pose[:3, 3]
                rotation_matrix = pose[:3, :3]
                logging.info("Estimated Translation Vector: %s", translation_vector)
                logging.info("Estimated Rotation Matrix:\n%s", rotation_matrix)

                ground_truth_tags = [{} for _ in range(len(self.slam.graph))]
                try:
                    for tag_id, node in self.slam.graph.items():
                        # Calculate the ground truth translation distance
                        true_pose = self.ground_truth(tag_id)
                        translation_dist = np.linalg.norm(true_pose[:3, 3])
                        ground_truth_tags[tag_id]["local"] = translation_dist

                        #tag_to_world = np.linalg.norm(self.ground_truth(tag_id)[:3, 3] - self.ground_truth(self.slam.coordinate_id)[:3, 3])
                        #tag_to_world = np.linalg.norm(self.world_ground_truth(tag_id)[:3, 3])
                        tag_to_world = self.ground_truth_difference(tag_id, self.slam.coordinate_id)
                        # Calculate and log the difference between the two ground truth methods
                        logging.info(f"Tag ID: {tag_id}, Tag to World GT Distance: {tag_to_world}")
                        logging.info(f"Tag ID: {tag_id}, Tag to World Distance: {np.linalg.norm(node.world[:3, 3])}")

                        # Log the ground truth distance, node.local distance, and their difference
                        local_dist = np.linalg.norm(node.local[:3, 3])
                        logging.info(f"Tag ID: {tag_id}, Ground Truth Local Distance: {translation_dist}")
                        logging.info(f"Tag ID: {tag_id}, Node Local Distance: {local_dist}")
                        logging.info(f"Tag ID: {tag_id}, Local Distance Difference: {abs(local_dist - translation_dist)}")
                        # Log both ground truth and local transformation matrices side by side
                        ground_truth_matrix = self.ground_truth(tag_id)
                        local_matrix = node.local
                        
                        logging.info(f"Tag ID: {tag_id}, Ground Truth Matrix vs Local Matrix:")
                        for i in range(4):
                            gt_row = ' '.join([f"{val:8.4f}" for val in ground_truth_matrix[i]])
                            local_row = ' '.join([f"{val:8.4f}" for val in local_matrix[i]])
                            logging.info(f"GT: [{gt_row}]    Local: [{local_row}]")
                        ground_truth_tags[tag_id]["world"] = tag_to_world

                        diff_world = abs(np.linalg.norm(node.world[:3, 3]) - tag_to_world)
                        logging.info(f"Tag ID: {tag_id}, World Distance Difference: {diff_world}")

                        diff_local = abs(np.linalg.norm(node.local[:3, 3]) - translation_dist)
                        
                        translation_local = node.local[:3, 3]
                        angles_local = self.rotation_matrix_to_euler_angles(node.local[:3, :3])

                        translation_world = node.world[:3, 3]
                        angles_world = self.rotation_matrix_to_euler_angles(node.world[:3, :3])
                        if(node.visible):
                            tag_translation = true_pose[:3, 3]
                            tag_rotation = true_pose[:3, :3]
                            tag_angles = self.rotation_matrix_to_euler_angles(tag_rotation)
                            translation_error = np.linalg.norm(translation_vector-gt_translation_vector)

                            self.csvwriter_covariance.writerow([
                                node.weight,  # Replace with actual number of jumps if available
                                *translation_local,
                                *angles_local,
                                translation_error,
                            ])

                        self.csvwriter_errors.writerow([
                            node.weight,  # Replace with actual number of jumps if available
                            *translation_local,
                            *angles_local,
                            *translation_world,
                            *angles_world,
                            *tag_translation,
                            *tag_rotation,
                            diff_world,
                            diff_local,
                            translation_error,
                        ])
                except Exception as e:
                    logging.error(f"Error processing graph nodes: {e}")
                    logging.debug(f"Graph: {self.slam.graph}")
                    logging.debug(f"Camera Position: {self.camera_position}")
                    continue
                
                self.slam.error_graph(ground_truth_tags)

                logging.info("Ground Truth Translation Vector: %s", gt_translation_vector)
                logging.info("Ground Truth Rotation Matrix:\n%s", gt_rotation_matrix)

                # Calculate differences
                translation_diff = np.linalg.norm(translation_vector - gt_translation_vector)
                rotation_diff = np.linalg.norm(rotation_matrix - gt_rotation_matrix)
                logging.info("Translation Difference: %s", translation_diff)
                logging.info("Rotation Difference: %s", rotation_diff)

                # Print the translation and rotation differences in a nice format
                def scale_units(value_mm):
                    if value_mm >= 1000:
                        return value_mm / 1000, 'm'
                    elif value_mm >= 10:
                        return value_mm / 10, 'cm'
                    else:
                        return value_mm, 'mm'

                est_translation_mm = self.mm_conversion(np.linalg.norm(translation_vector))
                gt_translation_mm = self.mm_conversion(np.linalg.norm(gt_translation_vector))
                translation_diff_mm = self.mm_conversion(translation_diff)
                # Calculate and print percentage difference
                if gt_translation_mm != 0:
                    translation_percentage_diff = (translation_diff_mm / gt_translation_mm) * 100
                else:
                    translation_percentage_diff = 0

                est_translation_scaled, est_unit = scale_units(est_translation_mm)
                gt_translation_scaled, gt_unit = scale_units(gt_translation_mm)
                translation_diff_scaled, diff_unit = scale_units(translation_diff_mm)

                # Clear the terminal
                os.system('cls' if os.name == 'nt' else 'clear')

                print(colored(f"Estimated Translation Vector Distance: {est_translation_scaled:.4f} {est_unit}", 'green'))
                print(colored(f"Ground Truth Translation Vector Distance: {gt_translation_scaled:.4f} {gt_unit}", 'yellow'))
                print(colored(f"Translation Difference: {translation_diff_scaled:.4f} {diff_unit}", 'cyan'))
                print(colored(f"Rotation Difference: {rotation_diff:.4f}", 'magenta'))
                print(colored(f"Translation Percentage Difference: {translation_percentage_diff:.2f}%", 'red'))
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
    sim = Simulation('sim_settings.json',movment_flag=False)
    try:
        sim.run()
    except KeyboardInterrupt:
        sim.csvfile.close()
        pygame.quit()
        sys.exit()
