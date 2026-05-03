import pygame

class Player:
    def __init__(self, x, y):
        self.image = pygame.image.load(r"융융_기본얼굴.png")
        self.image = pygame.transform.scale(self.image, (100, 80))
        self.pos = [x, y]
        self.to = [0, 0]
        self.angle = 0

    # 마우스 이동은 위치 직접 세팅이므로 이 부분을 조건부로 바꿔줌
    def update(self, dt, screen):
        width, height = screen.get_size()
        # 화면 밖으로 못 나가도록
        self.pos[0] = min(max(self.pos[0], 32), width - 32)
        self.pos[1] = min(max(self.pos[1], 32), height - 32)

    def draw(self, screen):
        # 방향에 따라 회전
        if self.to == [-1, -1]: self.angle = 45
        elif self.to == [-1, 0]: self.angle = 90
        elif self.to == [-1, 1]: self.angle = 135
        elif self.to == [0, 1]: self.angle = 180
        elif self.to == [1, 1]: self.angle = -135
        elif self.to == [1, 0]: self.angle = -90
        elif self.to == [1, -1]: self.angle = -45
        elif self.to == [0, -1]: self.angle = 0

        rotated = pygame.transform.rotate(self.image, self.angle)
        calib_pos = (
            self.pos[0] - rotated.get_width() / 2,
            self.pos[1] - rotated.get_height() / 2
        )
        screen.blit(rotated, calib_pos)

