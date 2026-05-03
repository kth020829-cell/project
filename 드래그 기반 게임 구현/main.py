import pygame, sys, random as rnd, time, math
from player import Player
from villain import Villain
from item import Item
from item_bone import Bone_Project
from menu import Menu
from boss import Boss 


class Game:
    def __init__(self):
        pygame.init()
        
        self.WIDTH, self.HEIGHT = 600, 750
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("융융이 살아남기")

        self.clock = pygame.time.Clock()
        self.FPS = 60

        # 배경 이미지 2장 (스크롤)
        self.bg_image = pygame.image.load("background.jpg").convert()
        self.bg_image = pygame.transform.scale(self.bg_image, (self.WIDTH, self.HEIGHT))

        self.bg_y1 = 0
        self.bg_y2 = -self.HEIGHT
        self.scroll_speed = 0.2

        # bgm
        pygame.mixer.music.load("평상시노래.mp3")
        pygame.mixer.music.play(-1)    
        self.current_bgm = "normal" #현재 노래 상태

        # 객체 상태
        self.player = Player(self.WIDTH//2, self.HEIGHT//2)
        self.villains = [Villain(rnd.randint(0, self.WIDTH), -20) for _ in range(10)]
        self.items = []
        self.bones = []

        # 게임 상태
        self.gameover = False
        self.start_time = time.time()
        self.score = 0
        self.kill_count = 0

        # 타이머
        self.item_timer = 0
        self.villain_timer = 0

        # 집 아이템
        self.doghouse_active = False
        self.doghouse_timer = 0
        self.doghouse_pos = [0, 0]

        # 보스 + 피버
        self.boss = None
        self.boss_active = False
        self.fever_active = False
        self.fever_timer = 0
        self.next_boss_time = 30  # 30초마다 보스 등장

        # 폰트
        self.font = pygame.font.Font("Galmuri14.ttf", 30)

    # ------------------------- 기본 기능 -------------------------

    def draw_text(self, txt, size, pos, color, center=False, bold=False):
        font = pygame.font.Font("Galmuri14.ttf", size)
        font.set_bold(bold)
        r = font.render(txt, True, color)
        if center:
            rect = r.get_rect(center=pos)
            self.screen.blit(r, rect)
        else:
            self.screen.blit(r, pos)

    def scroll_background(self, dt):
        self.bg_y1 += self.scroll_speed * dt
        self.bg_y2 += self.scroll_speed * dt

        if self.bg_y1 >= self.HEIGHT:
            self.bg_y1 = -self.HEIGHT
        if self.bg_y2 >= self.HEIGHT:
            self.bg_y2 = -self.HEIGHT

        self.screen.blit(self.bg_image, (0, self.bg_y1))
        self.screen.blit(self.bg_image, (0, self.bg_y2))

    def reset(self):
        self.player = Player(self.WIDTH//2, self.HEIGHT//2)
        self.villains = [Villain(rnd.randint(0, self.WIDTH), -20) for _ in range(10)]
        self.items = []
        self.bones = []
        self.score = 0
        self.kill_count = 0
        self.start_time = time.time()
        self.gameover = False
        self.doghouse_active = False
        self.doghouse_timer = 0

        # 보스 리셋
        self.boss = None
        self.boss_active = False
        self.fever_active = False
        self.fever_timer = 0
        self.next_boss_time = 30

    # ------------------------- 업데이트 -------------------------

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    pygame.quit()
                    sys.exit()
        # 드래그 이동
        if pygame.mouse.get_pressed()[0]:
            mx, my = pygame.mouse.get_pos()
            dx = mx - self.player.pos[0]
            dy = my - self.player.pos[1]
            self.player.pos[0] += dx * 0.15
            self.player.pos[1] += dy * 0.15

        # 재시작
        if self.gameover:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_a]:
                self.reset()

    def update_villains(self, dt):
        new_list = []
        for v in self.villains:
            if v.update_and_draw(dt, self.screen, self.player.pos):
                new_list.append(v)
        self.villains = new_list

    def update_items(self, dt):
        new_list = []
        for it in self.items:
            if it.update_and_draw(dt, self.screen):
                new_list.append(it)
        self.items = new_list

    def update_bones(self, dt):
        new_list = []
        for bp in self.bones:
            if bp.update_and_draw(dt, self.screen):
                new_list.append(bp)
        self.bones = new_list

    # ------------------------- 충돌 처리 -------------------------

    def check_player_collision(self):
        # 빌런 충돌 (보호존이면 무시)
        if not self.doghouse_active:
            for v in self.villains:
                dist = math.hypot(v.pos[0] - self.player.pos[0], v.pos[1] - self.player.pos[1])
                if dist < 25:
                    self.gameover = True

            # 보스 충돌 체크 (보호존이면 무시)
            if self.boss_active and self.boss is not None:
                dist = math.hypot(self.boss.pos[0] - self.player.pos[0],
                                  self.boss.pos[1] - self.player.pos[1])
                if dist < self.boss.radius + 20:
                    self.gameover = True

    def check_bone_hits(self):

        # 빌런 맞음
        for bp in list(self.bones):
            for v in list(self.villains):
                if math.hypot(bp.pos[0] - v.pos[0], bp.pos[1] - v.pos[1]) < 40:
                    try:
                        self.villains.remove(v)
                        self.kill_count += 1
                    except:
                        pass

            # 보스 맞음
            if self.boss_active and self.boss is not None:
                if math.hypot(bp.pos[0] - self.boss.pos[0], bp.pos[1] - self.boss.pos[1]) < self.boss.radius + 20:
                    # 보스 hp 감소 (최소 0)
                    self.boss.hp = max(0, self.boss.hp - 1)
                    try:
                        if bp in self.bones:
                            self.bones.remove(bp)
                    except:
                        pass

    def pickup_items(self):
        for it in list(self.items):
            dist = math.hypot(self.player.pos[0] - it.pos[0], self.player.pos[1] - it.pos[1])
            if dist < 30:
                if it.type == "bone":
                    # 총알을 여러방향으로 발사
                    bullet_count = 12
                    angle_step = 360 / bullet_count

                    for i in range(bullet_count):
                        angle = math.radians(i * angle_step)
                        self.bones.append(Bone_Project(self.player.pos[0], self.player.pos[1], angle))
                elif it.type == "house":
                    self.doghouse_active = True
                    self.doghouse_timer = 5000
                self.items.remove(it)
                break

    def doghouse_effect(self, dt):
        if not self.doghouse_active:
            return

        self.doghouse_timer -= dt
        self.doghouse_pos = [self.player.pos[0], self.player.pos[1]]

        pygame.draw.circle(
            self.screen, (70, 200, 255),
            (int(self.doghouse_pos[0]), int(self.doghouse_pos[1])),
            60, 3
        )

        new_list = []
        for v in self.villains:
            dist = math.hypot(v.pos[0] - self.doghouse_pos[0], v.pos[1] - self.doghouse_pos[1])
            if dist > 60:
                new_list.append(v)
            else:
                self.kill_count += 1
        self.villains = new_list

        if self.doghouse_timer <= 0:
            self.doghouse_active = False


    # ------------------------- 스폰 -------------------------

    def spawn_logic(self, dt):

        #
        if self.fever_active:
            item_interval = 1000   # 피버때 1초
        else:
            item_interval = 4000   # 평소 4초

        # 빌런 추가
        self.villain_timer += dt
        if self.villain_timer > 1000:
            self.villains.append(Villain(rnd.randint(0, self.WIDTH), -20))
            self.villain_timer = 0

        # 아이템 추가
        self.item_timer += dt
        if self.item_timer > item_interval:
            spawn_x = rnd.randint(50, self.WIDTH - 50)
            item_type = rnd.choice(["bone", "house"])
            self.items.append(Item(spawn_x, -20, item_type))
            self.item_timer = 0

    # ------------------------- 보스맵 처리 -------------------------

    def boss_logic(self, dt):
        elapsed = time.time() - self.start_time

        # 보스 조건: 보스가 아예 없을 때만 생성
        if (self.boss is None) and (elapsed >= self.next_boss_time):

            self.boss_active = True
            self.fever_active = True
            self.fever_timer = 8000

            self.boss = Boss(self.WIDTH // 2, 120)
            self.boss.spawn_time = pygame.time.get_ticks()
            self.boss.max_hp = 10
            self.boss.hp = self.boss.max_hp

            pygame.mixer.music.load("보스등장곡.mp3")
            pygame.mixer.music.play(-1)
            self.current_bgm = "boss"

        # 보스 없으면 아래 로직 필요 없음
        if self.boss is None:
            return

        # 보스 움직임
        self.boss.update(dt)
        self.boss.draw(self.screen)

        if self.fever_active:
            self.fever_timer -= dt
            if self.fever_timer <= 0:
                self.fever_active = False

        # 보스 처치
        if self.boss.hp <= 0:
            self.boss_active = False
            self.fever_active = False
            self.boss = None

            pygame.mixer.music.load("평상시노래.mp3")
            pygame.mixer.music.play(-1)
            self.current_bgm = "normal"

            self.next_boss_time = elapsed + 30


    # ------------------------- 메인 루프 -------------------------

    def run(self):
        while True:
            dt = self.clock.tick(self.FPS)

            self.handle_events()
            self.scroll_background(dt)

            if not self.gameover:

                self.player.update(dt, self.screen)
                self.player.draw(self.screen)

                self.update_villains(dt)
                self.update_items(dt)
                self.update_bones(dt)

                # 보호존 여부는 check_player_collision 내부에서 확인하므로 그냥 호출
                self.check_player_collision()

                self.check_bone_hits()
                self.pickup_items()
                self.doghouse_effect(dt)

                self.spawn_logic(dt)

                # 보스맵 기능 실행
                self.boss_logic(dt)

                # 점수 표시 (피버 시 색상 변경)
                if self.fever_active:
                    self.draw_text(f"FEVER!!  Time: {self.score:.1f}  Kill: {self.kill_count}", 30, (10, 10), (255, 200, 0))
                else:
                    self.draw_text(f"Time: {self.score:.1f}  Kill: {self.kill_count}", 30, (10, 10), (255,255,255))

                self.score = time.time() - self.start_time

            else:
                self.draw_text("GAME OVER", 80, (self.WIDTH/2, self.HEIGHT/2 - 80), (255,255,255), center=True)
                self.draw_text(f"Score: {self.score:.1f}", 50, (self.WIDTH/2, self.HEIGHT/2), (255,255,255), center=True)
                self.draw_text(f"Killed: {self.kill_count}", 50, (self.WIDTH/2, self.HEIGHT/2 + 50), (255,255,0), center=True)
                self.draw_text("Press A to Restart", 30, (self.WIDTH/2, self.HEIGHT/2 + 120), (200,200,255), center=True)
                self.draw_text("Press Q to Quit", 28, (self.WIDTH/2, self.HEIGHT/2 + 160), (255,180,180), center=True)
                
            pygame.display.update()


if __name__ == "__main__":
    pygame.init()
    WIDTH, HEIGHT = 600, 750
    screen = pygame.display.set_mode((WIDTH, HEIGHT))

    # 메뉴 실행
    menu = Menu(screen, WIDTH, HEIGHT)
    choice = menu.show()

    if choice == "start":
        Game().run()
