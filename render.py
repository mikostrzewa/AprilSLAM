import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from PIL import Image
import numpy as np
import cv2
from apriltag import apriltag

def load_texture(image_name):
    # Load image using PIL
    img = Image.open(image_name)
    img_data = img.convert("RGBA").tobytes()
    width, height = img.size

    # Generate texture ID and bind it
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)

    # Set texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    # Upload texture data
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height,
                 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
    return texture_id

def draw_quad(position, texture_id):
    x, y, z = position
    glPushMatrix()
    glTranslatef(x, y, z)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glBegin(GL_QUADS)
    glTexCoord2f(0, 0); glVertex3f(-1, -1, 0)
    glTexCoord2f(1, 0); glVertex3f(1, -1, 0)
    glTexCoord2f(1, 1); glVertex3f(1, 1, 0)
    glTexCoord2f(0, 1); glVertex3f(-1, 1, 0)
    glEnd()
    glPopMatrix()

def capture_scene(width, height):
    # Read pixels from the OpenGL buffer
    glPixelStorei(GL_PACK_ALIGNMENT, 1)
    data = glReadPixels(0, 0, width, height, GL_BGR, GL_UNSIGNED_BYTE)
    # Convert to numpy array
    image = np.frombuffer(data, dtype=np.uint8)
    image = image.reshape((height, width, 3))
    # Flip the image vertically
    image = np.flipud(image)
    return image

def main():
    # Initialize Pygame and set up the OpenGL context
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    # Set up the camera
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, display[0] / display[1], 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    # Enable textures and depth test
    glEnable(GL_TEXTURE_2D)
    glEnable(GL_DEPTH_TEST)

    # Set the background color to blue
    glClearColor(1.0, 1.0, 1.0, 1.0);


    # Load the PNG images as textures (ensure these images contain AprilTags)
    texture_ids = [
        load_texture('tag0.png'),
        load_texture('tag1.png'),
        load_texture('tag2.png')
    ]

    # Define positions for the three images
    positions = [
        (-2.0, 0.0, -10.0),  # Left
        (0.0, 0.0, -20.0),   # Center
        (7.0, 0.0, -15.0)    # Right
    ]

    # Print world positions relative to the camera
    print("World Positions relative to the camera:")
    for i, pos in enumerate(positions):
        print(f"Image {i+1}: x={pos[0]}, y={pos[1]}, z={pos[2]}")

    # Initialize the AprilTag detector
    detector = apriltag("tagStandard41h12")

    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Clear the screen and depth buffer with the background color
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # Draw each image at its position
        for texture_id, position in zip(texture_ids, positions):
            draw_quad(position, texture_id)

        # Update the display
        pygame.display.flip()
        pygame.time.wait(10)

        # Capture the rendered scene
        image = capture_scene(*display)

        # Convert the image to grayscale for AprilTag detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect AprilTags in the frame
        detections = detector.detect(gray)

        # Draw detections
        for detection in detections:
            corners = np.array(detection['lb-rb-rt-lt'], dtype=np.int32)
            cv2.polylines(image, [corners], True, (0, 255, 0), 2)
            tag_id = detection['id']
            center = tuple(corners.mean(axis=0).astype(int))
            cv2.putText(image, f"ID: {tag_id}", center, cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 0), 2)

        # Display the image with detections (optional)
        cv2.imshow('AprilTag Detection', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False

    # Clean up
    pygame.quit()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
