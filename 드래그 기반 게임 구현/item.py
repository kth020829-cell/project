import pygame
import random

class Item:
    def __init__(self, x, y, type):
        self.type = type
        self.pos = [x, y]
        self.speed = 1

        if self.type == "bone":
            self.image = pygame.image.load("bone.png").convert_alpha()
            self.image = pygame.transform.scale(self.image, (100, 100))

        elif self.type == "house":
            self.image = pygame.image.load("house.png").convert_alpha()
            self.image = pygame.transform.scale(self.image, (100, 100))

    def update_and_draw(self, dt, screen):
        self.pos[1] += self.speed

        if self.pos[1] > screen.get_height() + 50:
            return False

        rect = self.image.get_rect(center=(int(self.pos[0]), int(self.pos[1])))
        screen.blit(self.image, rect)

        return True
