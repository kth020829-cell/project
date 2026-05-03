import pygame
import random as rnd
import math

class Villain:
    def __init__(self, x, y):
        self.pos = [x, y]
        self.speed = rnd.uniform(1.0, 2.0)

        # 각 악당마다 고유한 추적 오프셋 생성
        self.offset_x = rnd.randint(-100, 100)
        self.offset_y = rnd.randint(-120, 120)

        # villain 이미지 불러오기
        self.image = pygame.image.load("villain.png").convert_alpha()
        self.image = pygame.transform.scale(self.image, (100, 130))  # 크기 조절

        # 이미지 중심 위치 맞추기 위해 반지름
        self.width = self.image.get_width()
        self.height = self.image.get_height()

    def update_and_draw(self, dt, screen, player_pos):
        # 목표는 플레이어 + 랜덤 오프셋
        target_x = player_pos[0] + self.offset_x
        target_y = player_pos[1] + self.offset_y

        dx = target_x - self.pos[0]
        dy = target_y - self.pos[1]

        dist = math.hypot(dx, dy)
        if dist != 0:
            dx /= dist
            dy /= dist

        # 아래로 떨어지며 20%만 추적
        self.pos[0] += dx * self.speed * 0.2
        self.pos[1] += self.speed

        # 화면 밖이면 삭제
        if self.pos[1] > screen.get_height() + 50:
            return False

        # 🟢 이미지 그리기
        screen.blit(self.image, (self.pos[0] - self.width // 2, self.pos[1] - self.height // 2))
        return True
