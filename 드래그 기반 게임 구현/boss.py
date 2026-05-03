import pygame, random as rnd, math

class Boss:
    def __init__(self, x, y):
        self.pos = [x, y]

        # HP
        self.max_hp = 3
        self.hp = self.max_hp

        # 이동 설정
        self.speed = rnd.uniform(2.0, 4.0)
        self.radius = 80

        img = pygame.image.load("bossicon.png").convert_alpha()
        self.image = pygame.transform.scale(img, (self.radius * 2, self.radius * 2))

        # 이동 목표
        self.target = [x, y]

        # 생성 시각 기록
        self.spawn_time = pygame.time.get_ticks()

    def update(self, dt):
        # 랜덤하게 이동 목표 변경
        if rnd.random() < 0.01:
            self.target = [rnd.randint(60, 540), rnd.randint(80, 660)]

        dx = self.target[0] - self.pos[0]
        dy = self.target[1] - self.pos[1]
        dist = math.hypot(dx, dy)

        if dist != 0:
            dx /= dist
            dy /= dist

        self.pos[0] += dx * self.speed * dt * 0.06
        self.pos[1] += dy * self.speed * dt * 0.06

    def draw(self, screen):

        rect = self.image.get_rect(center=(int(self.pos[0]), int(self.pos[1])))
        screen.blit(self.image, rect)

        # ===== HP 바 =====
        total_width = 160
        bar_x = self.pos[0] - total_width // 2
        bar_y = self.pos[1] - self.radius - 36

        ratio = max(0, self.hp / self.max_hp)
        green_width = int(total_width * ratio)

        # 바 배경
        pygame.draw.rect(screen, (50, 50, 50), (bar_x, bar_y, total_width, 14))
        # 체력 부분
        pygame.draw.rect(screen, (50, 220, 50), (bar_x, bar_y, green_width, 14))
        # 잃은 체력 부분
        pygame.draw.rect(
            screen,
            (220, 50, 50),
            (bar_x + green_width, bar_y, total_width - green_width, 14)
        )
