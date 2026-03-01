import pygame
import numpy as np
import matplotlib.pyplot as plt
import logging
from gomoku_game import GomokuEnv, BOARD_SIZE, CELL_SIZE, WINDOW_WIDTH, WINDOW_HEIGHT
from dqn_agent import DQNAgent

# ====================== 状态/行动空间数学定义（训练代码标注） ======================
"""
训练过程核心参数（基于状态/行动空间定义）：
- 状态输入维度：3×9×9（对应状态空间S的3通道定义）
- 行动输出维度：81（对应行动空间A的81个离散行动）
- 合法行动过滤：每次选择行动前过滤非法位置，仅从A_legal中选择
"""
# ==================================================================================

# 配置日志
logging.basicConfig(filename='train.log', level=logging.INFO, format='%(asctime)s - %(message)s')

def train(episodes=5000, render_every=200):
    env = GomokuEnv()
    # 初始化智能体：输入通道3（状态空间），行动维度81（行动空间）
    agent = DQNAgent(input_channels=3, board_size=BOARD_SIZE, action_dim=BOARD_SIZE*BOARD_SIZE)

    episode_rewards = []
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT+40))  # 多出40用于文字
    clock = pygame.time.Clock()

    for ep in range(episodes):
        state = env.reset()  # 返回展平的243维向量，但我们需要3×9×9形状
        # 将状态重塑为3×9×9（符合状态空间S的维度定义）
        state_img = state.reshape(3, BOARD_SIZE, BOARD_SIZE)
        total_reward = 0
        done = False

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            legal_actions = env.get_legal_actions()  # 获取合法行动A_legal
            if env.turn == 1:  # 玩家回合（训练时采用随机）
                if legal_actions.size > 0:
                    action = np.random.choice(legal_actions)
                else:
                    action = 0
                next_state, reward, done = env.step(action)
                # 修复：更新玩家回合后的状态
                state_img = next_state.reshape(3, BOARD_SIZE, BOARD_SIZE)
            else:  # AI回合
                # 基于ε-贪心策略选择行动（Q-Learning核心策略）
                action = agent.choose_action(state_img, legal_actions)
                next_state, reward, done = env.step(action)
                agent.store_transition(state_img, action, reward, next_state.reshape(3, BOARD_SIZE, BOARD_SIZE), done)
                agent.update()  # Q-Learning核心更新逻辑
                total_reward += reward
                state_img = next_state.reshape(3, BOARD_SIZE, BOARD_SIZE)

            # 渲染
            if ep % render_every == 0:
                env.render(screen)
                clock.tick(10)

        # 动态衰减epsilon（Q-Learning探索率衰减）
        agent.decay_epsilon(ep, episodes)
        episode_rewards.append(total_reward)
        
        # 打印并记录日志
        if ep % 100 == 0:
            log_msg = f"Episode {ep}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}"
            print(log_msg)
            logging.info(log_msg)

        # 提前终止训练：连续500局胜率>90%
        if ep > 500 and np.mean(episode_rewards[-500:]) > 90:
            print(f"提前终止训练：连续500局平均奖励达到{np.mean(episode_rewards[-500:]):.2f}")
            break

    pygame.quit()
    # 绘制训练曲线
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Progress (Q-Learning DQN)')  # 标注算法类型
    plt.savefig('training_curve.png')
    plt.show()
    # 保存模型
    agent.save('dqn_gomoku.pth')
    return agent

if __name__ == '__main__':
    trained_agent = train(episodes=4000, render_every=200)