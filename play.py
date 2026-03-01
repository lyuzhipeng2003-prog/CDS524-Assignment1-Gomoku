import pygame
import sys
import numpy as np
from gomoku_game import GomokuEnv, BOARD_SIZE, CELL_SIZE, WINDOW_WIDTH, WINDOW_HEIGHT
from dqn_agent import DQNAgent

def show_start_screen(screen):
    font_title = pygame.font.Font(None, 74)
    font_text = pygame.font.Font(None, 36)
    title_text = font_title.render("Gomoku vs DQN AI", True, (255,255,255))
    controls = [
        "Click on a cell to place your black stone",
        "AI plays white stones",
        "First to five in a row wins",
        "Press Y to start, ESC to quit"
    ]
    control_surfaces = [font_text.render(line, True, (200,200,200)) for line in controls]

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_y:
                    return
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
        screen.fill((0,0,0))
        screen.blit(title_text, title_text.get_rect(center=(WINDOW_WIDTH//2, 100)))
        y = 200
        for surf in control_surfaces:
            screen.blit(surf, surf.get_rect(center=(WINDOW_WIDTH//2, y)))
            y += 40
        pygame.display.flip()
        pygame.time.wait(50)

def show_game_over_screen(screen, winner):
    font_big = pygame.font.Font(None, 74)
    font_med = pygame.font.Font(None, 48)
    if winner == 1:
        result = "You Win!"
    elif winner == 2:
        result = "AI Wins!"
    else:
        result = "Draw!"
    game_over = font_big.render("Game Over", True, (255,0,0))
    result_surf = font_med.render(result, True, (255,255,255))
    restart = font_med.render("Press R to restart, ESC to quit", True, (200,200,200))

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    return
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
        screen.fill((0,0,0))
        screen.blit(game_over, game_over.get_rect(center=(WINDOW_WIDTH//2, 150)))
        screen.blit(result_surf, result_surf.get_rect(center=(WINDOW_WIDTH//2, 250)))
        screen.blit(restart, restart.get_rect(center=(WINDOW_WIDTH//2, 350)))
        pygame.display.flip()
        pygame.time.wait(50)

def get_clicked_cell(pos):
    x, y = pos
    row = y // CELL_SIZE
    col = x // CELL_SIZE
    # 边界检查
    if row >= BOARD_SIZE or col >= BOARD_SIZE:
        return None, None
    return row, col

def play_with_human(agent):
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT+80))  # 增加高度容纳累计奖励
    pygame.display.set_caption("Gomoku vs AI")
    clock = pygame.time.Clock()

    while True:
        show_start_screen(screen)
        env = GomokuEnv()
        state = env.reset()  # 展平向量
        state_img = state.reshape(3, BOARD_SIZE, BOARD_SIZE)
        done = False

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if event.type == pygame.MOUSEBUTTONDOWN and env.turn == 1 and not done:
                    r, c = get_clicked_cell(pygame.mouse.get_pos())
                    if r is not None and c is not None:
                        action = r * BOARD_SIZE + c
                        legal = env.get_legal_actions()
                        if action in legal:
                            # 合法落子：执行动作+更新状态
                            next_state, reward, done = env.step(action)
                            state_img = next_state.reshape(3, BOARD_SIZE, BOARD_SIZE)
                        else:
                            # ========== 核心修复：非法落子主动触发渲染 ==========
                            # 执行非法动作（更新奖励值）
                            env.step(action)
                            # 强制刷新UI，显示-10奖励
                            env.render(screen)
                            # ==================================================

            # AI回合
            if env.turn == 2 and not done:
                legal = env.get_legal_actions()
                if len(legal) > 0:
                    action = agent.choose_action(state_img, legal, eval_mode=True)
                    next_state, reward, done = env.step(action)
                    state_img = next_state.reshape(3, BOARD_SIZE, BOARD_SIZE)
                else:
                    done = True   # 无合法位置，平局

            # 持续渲染UI（确保每帧都刷新）
            env.render(screen)
            clock.tick(10)

        show_game_over_screen(screen, env.winner)

if __name__ == '__main__':
    agent = DQNAgent(input_channels=3, board_size=BOARD_SIZE, action_dim=BOARD_SIZE*BOARD_SIZE)
    try:
        agent.load('dqn_gomoku.pth')
        print("Model loaded successfully!")
    except FileNotFoundError:
        print("Warning: Model file not found! Playing with untrained AI.")
    play_with_human(agent)