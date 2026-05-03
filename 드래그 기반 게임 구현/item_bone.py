import pygame
import random as rnd
import math

class Bone_Project:
    def __init__(self, x, y, angle=None):  ### angle 입력 가능하도록 추가
        self.pos = [x, y]

        # angle이 없으면 랜덤, 있으면 그 방향으로 발사
        if angle is None:
            angle = rnd.uniform(0, math.pi * 2)

        self.dir_x = math.cos(angle)
        self.dir_y = math.sin(angle)
        self.speed = 5
        self.radius = 25

        # 생존 시간
        self.lifetime = 2000
        self.spawn_time = pygame.time.get_ticks()

    def update_and_draw(self, dt, screen):

        # 이동
        self.pos[0] += self.dir_x * self.speed
        self.pos[1] += self.dir_y * self.speed

        # 생존 시간 끝
        if pygame.time.get_ticks() - self.spawn_time > self.lifetime:
            return False

        # 화면 밖이면 삭제
        if (self.pos[0] < -60 or self.pos[0] > screen.get_width() + 60 or
            self.pos[1] < -60 or self.pos[1] > screen.get_height() + 60):
            return False

        # 뼈 표시
        pygame.draw.circle(screen, (255, 255, 180), (int(self.pos[0]), int(self.pos[1])), self.radius)

        return True
