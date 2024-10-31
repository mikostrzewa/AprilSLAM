import sys
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

def loadTexture(image, padding):
    textureSurface = pygame.image.load(image).convert_alpha()
    width = textureSurface.get_width()
    height = textureSurface.get_height()
    # Create a new surface with white background for padding
    paddedWidth = width + padding * 2
    paddedHeight = height + padding * 2
    paddedSurface = pygame.Surface((paddedWidth, paddedHeight), pygame.SRCALPHA)
    paddedSurface.fill((255, 255, 255, 255))  # White background
    # Blit the original image onto the padded surface
    paddedSurface.blit(textureSurface, (padding, padding))
    textureData = pygame.image.tostring(paddedSurface, "RGBA", True)
    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, paddedWidth, paddedHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, textureData)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    return texture, paddedWidth, paddedHeight

def main():
    pygame.init()
    display = (1000, 1000)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    glEnable(GL_TEXTURE_2D)
    padding = 50  # Amount of padding in pixels

    # Load textures for tag0.png, tag1.png, and tag2.png
    texture0, _, _ = loadTexture('tag0.png', padding)
    texture1, _, _ = loadTexture('tag1.png', padding)
    texture2, _, _ = loadTexture('tag2.png', padding)
    
    # Set up a 3D perspective projection
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, (display[0] / display[1]), 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    
    # Define positions for each tag in 3D space
    tag_positions = [
        (texture0, (0, 0, -20)),    # tag0 at center
        (texture1, (-2, 0, -10)), # tag1 to the left
        (texture2, (5, 0, -15)),  # tag2 to the right
    ]
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
    
        glClearColor(0.5, 0.0, 0.5, 1.0)  # Clear to purple background
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
       
        for texture, position in tag_positions:
            glLoadIdentity()
            glTranslatef(*position)
            glBindTexture(GL_TEXTURE_2D, texture)
            # Render textured quad
            glBegin(GL_QUADS)
            glTexCoord2f(0, 0); glVertex3f(-1, -1, 0)
            glTexCoord2f(1, 0); glVertex3f(1, -1, 0)
            glTexCoord2f(1, 1); glVertex3f(1, 1, 0)
            glTexCoord2f(0, 1); glVertex3f(-1, 1, 0)
            glEnd()
        
        pygame.display.flip()
        pygame.time.wait(10)
    
if __name__ == '__main__':
    main()
