import pygame
import sys

class Menu:
    def __init__(self, screen, width, height):
        self.screen = screen
        self.WIDTH = width
        self.HEIGHT = height
        self.font = pygame.font.Font("Galmuri14.ttf", 40)
        self.small_font = pygame.font.Font("Galmuri14.ttf", 25)

        # 버튼 영역 정의
        self.start_rect = pygame.Rect(width//2 - 100, height//2 - 30, 200, 60)
        self.quit_rect = pygame.Rect(width//2 - 100, height//2 + 50, 200, 60)

    def draw_button(self, rect, text):
        pygame.draw.rect(self.screen, (255, 255, 255), rect, border_radius=10)
        pygame.draw.rect(self.screen, (0, 0, 0), rect, 3, border_radius=10)

        label = self.small_font.render(text, True, (0, 0, 0))
        text_rect = label.get_rect(center=rect.center)
        self.screen.blit(label, text_rect)

    def show(self):
        while True:
            self.screen.fill((20, 20, 20))

            title = self.font.render("융융이 살아남기", True, (255, 255, 255))
            title_rect = title.get_rect(center=(self.WIDTH//2, self.HEIGHT//2 - 120))
            self.screen.blit(title, title_rect)

            # 버튼 그리기
            self.draw_button(self.start_rect, "START")
            self.draw_button(self.quit_rect, "QUIT")

            # 이벤트 처리
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = pygame.mouse.get_pos()

                    if self.start_rect.collidepoint(mx, my):
                        return "start"

                    if self.quit_rect.collidepoint(mx, my):
                        pygame.quit()
                        sys.exit()

            pygame.display.update()
