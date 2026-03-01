import numpy as np
import pygame
from pygame.locals import *

# ====================== 状态/行动空间数学定义（核心标注） ======================
"""
状态空间与行动空间数学定义：
1. 状态空间 S：S ∈ R^(3×9×9)
   - 通道0：黑子位置（1表示有黑子，0表示无）
   - 通道1：白子位置（1表示有白子，0表示无）
   - 通道2：当前回合标志（1表示玩家回合，0表示AI回合）
   总状态数：2^(9×9×2) × 2 = 2^163（实际通过CNN降维学习，避免维度灾难）

2. 行动空间 A：A = {0,1,2,...,80}
   - 共81个离散行动，对应9×9棋盘的每个落子位置
   - 行动i的坐标转换：行 = i // 9，列 = i % 9

3. 合法行动 A_legal：A_legal ⊂ A
   - 仅包含棋盘空白位置对应的行动，AI仅能从合法行动中选择
"""
# ============================================================================

BOARD_SIZE = 9
CELL_SIZE = 60
WINDOW_WIDTH = BOARD_SIZE * CELL_SIZE
WINDOW_HEIGHT = BOARD_SIZE * CELL_SIZE
MARGIN = 20  # 用于显示文字

# 颜色
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BROWN = (150, 75, 0)
LIGHT_BROWN = (200, 180, 140)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)  # 新增：正奖励绿色

class GomokuEnv:
    def __init__(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)  # 0空,1黑子(玩家),2白子(AI)
        self.turn = 1  # 1表示玩家(黑), 2表示AI(白)
        self.done = False
        self.winner = None
        self.current_reward = 0  # 核心新增：实时奖励值，用于UI显示
        self.total_reward = 0    # 可选新增：累计奖励值

    def reset(self):
        """重置游戏状态，包括奖励值"""
        self.board.fill(0)
        self.turn = 1
        self.done = False
        self.winner = None
        self.current_reward = 0  # 重置实时奖励
        self.total_reward = 0    # 重置累计奖励
        return self._get_state()

    def _get_state(self):
        """返回3通道状态（符合状态空间S的定义）"""
        state = np.zeros((3, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        # 通道0：黑子
        state[0, self.board == 1] = 1
        # 通道1：白子
        state[1, self.board == 2] = 1
        # 通道2：当前玩家标志 (全部为1表示轮到黑方，0表示轮到白方)
        if self.turn == 1:
            state[2, :, :] = 1
        else:
            state[2, :, :] = 0
        return state.flatten()   # 展平为243维向量

    def get_legal_actions(self):
        """返回所有合法行动（符合A_legal的定义）"""
        return np.where(self.board.flatten() == 0)[0]

    def check_win(self, player):
        """检查指定玩家是否获胜"""
        # 获取该玩家所有棋子位置
        positions = np.argwhere(self.board == player)
        if len(positions) < 5:
            return False
        # 检查四个方向：横、竖、正斜、反斜
        directions = [(1,0), (0,1), (1,1), (1,-1)]
        for (x, y) in positions:
            for dx, dy in directions:
                count = 1
                # 正向检查
                for step in range(1, 5):
                    nx, ny = x + step*dx, y + step*dy
                    if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and self.board[nx, ny] == player:
                        count += 1
                    else:
                        break
                # 反向检查（避免重复遍历）
                for step in range(1, 5):
                    nx, ny = x - step*dx, y - step*dy
                    if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and self.board[nx, ny] == player:
                        count += 1
                    else:
                        break
                if count >= 5:
                    return True
        return False

    def step(self, action):
        """
        执行动作 action (符合行动空间A的定义)
        返回 (next_state, reward, done)
        """
        if self.done:
            return self._get_state(), 0, True

        # 将动作索引转为棋盘坐标（行动空间A的坐标转换）
        row = action // BOARD_SIZE
        col = action % BOARD_SIZE

        # 检查是否合法
        if self.board[row, col] != 0:
            # 非法动作：负奖励-10
            self.current_reward = -10  # 更新实时奖励
            self.total_reward += self.current_reward  # 累加累计奖励
            print(f"Illegal action: ({row}, {col}) - already occupied!")
            return self._get_state(), self.current_reward, False

        # 落子
        self.board[row, col] = self.turn

        # 检查胜负
        if self.check_win(self.turn):
            self.done = True
            self.winner = self.turn
            # 正奖励（AI获胜+100）/ 负奖励（玩家获胜-100）
            self.current_reward = 100 if self.turn == 2 else -100
            self.total_reward += self.current_reward  # 累加累计奖励
            return self._get_state(), self.current_reward, True

        # 检查平局
        if len(self.get_legal_actions()) == 0:
            self.done = True
            self.winner = 0  # 平局
            self.current_reward = 0  # 平局无奖励
            self.total_reward += self.current_reward  # 累加累计奖励
            return self._get_state(), self.current_reward, True

        # 切换玩家
        self.turn = 3 - self.turn   # 1<->2

        # 正常落子：中立奖励0
        self.current_reward = 0
        self.total_reward += self.current_reward  # 累加累计奖励
        return self._get_state(), self.current_reward, False

    def render(self, screen):
        """渲染游戏界面，包含实时奖励和累计奖励显示"""
        screen.fill(LIGHT_BROWN)
        # 绘制网格线
        for i in range(BOARD_SIZE):
            pygame.draw.line(screen, BLACK, (i*CELL_SIZE, 0), (i*CELL_SIZE, WINDOW_HEIGHT))
            pygame.draw.line(screen, BLACK, (0, i*CELL_SIZE), (WINDOW_WIDTH, i*CELL_SIZE))
        # 绘制棋子
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self.board[r, c] == 1:
                    pygame.draw.circle(screen, BLACK, (c*CELL_SIZE + CELL_SIZE//2, r*CELL_SIZE + CELL_SIZE//2), CELL_SIZE//2 - 2)
                elif self.board[r, c] == 2:
                    pygame.draw.circle(screen, WHITE, (c*CELL_SIZE + CELL_SIZE//2, r*CELL_SIZE + CELL_SIZE//2), CELL_SIZE//2 - 2)
        
        # 绘制基础文字
        font = pygame.font.Font(None, 36)
        turn_text = font.render("Your Turn (Black)" if self.turn == 1 else "AI Turn (White)", True, RED)
        screen.blit(turn_text, (10, WINDOW_HEIGHT + 10))
        
        # ====================== 核心新增：实时奖励显示（带颜色区分） ======================
        # 根据奖励正负选择颜色：正奖励绿色，负奖励红色，零奖励黑色
        if self.current_reward > 0:
            reward_color = GREEN
        elif self.current_reward < 0:
            reward_color = RED
        else:
            reward_color = BLACK
        
        # 绘制实时奖励
        current_reward_text = font.render(f"Current Reward: {self.current_reward}", True, reward_color)
        screen.blit(current_reward_text, (WINDOW_WIDTH - 280, WINDOW_HEIGHT + 10))
        
        # 绘制累计奖励（可选，增强展示效果）
        total_reward_text = font.render(f"Total Reward: {self.total_reward}", True, BLUE)
        screen.blit(total_reward_text, (10, WINDOW_HEIGHT + 50))
        # =================================================================================
        
        pygame.display.flip()