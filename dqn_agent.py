import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class DQN(nn.Module):
    """
    深度Q网络（DQN）：Q-Learning的深度扩展版本
    核心思想：用卷积神经网络近似Q-Learning的Q值表，解决高维状态空间无法用表格存储的问题
    输入：3×9×9的棋盘状态（黑子位置、白子位置、当前回合标志）
    输出：81个行动的Q值（对应9×9棋盘每个位置的落子价值）
    """
    def __init__(self, input_channels=3, board_size=9, action_dim=81):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)  # 池化层减少参数
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # 池化层减少参数
        
        # 重新计算全连接层输入维度（9x9经过两次池化后为2x2）
        fc1_input = 64 * (board_size//4) * (board_size//4)  # 9//4=2
        self.fc1 = nn.Linear(fc1_input, 512)
        self.fc2 = nn.Linear(512, action_dim)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)   # 展平
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class ReplayBuffer:
    """
    经验回放池：Q-Learning的优化手段，打破样本相关性，提升训练稳定性
    存储格式：(state, action, reward, next_state, done)
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states = np.array([item[0] for item in batch])
        actions = np.array([item[1] for item in batch])
        rewards = np.array([item[2] for item in batch])
        next_states = np.array([item[3] for item in batch])
        dones = np.array([item[4] for item in batch])
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """
    DQN智能体：基于Q-Learning核心思想实现，覆盖Q-Learning所有核心要素：
    1. ε-贪心探索策略（epsilon）：平衡探索（随机落子）和利用（选最优Q值落子）
    2. 学习率（lr）：通过Adam优化器实现，控制Q值更新步长
    3. 折扣因子（gamma）：权衡即时奖励（当前落子的奖惩）和未来奖励（最终胜负的奖惩）
    4. 目标网络（target_net）：固定目标Q值，避免Q-Learning的过估计问题
    5. 核心目标：学习从棋盘状态到落子行动的最优策略，最大化累积奖励
    """
    def __init__(self, input_channels=3, board_size=9, action_dim=81, lr=0.001, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995,
                 buffer_size=10000, batch_size=32, target_update=100):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dim = action_dim
        self.gamma = gamma  # Q-Learning折扣因子，权衡即时/未来奖励
        self.epsilon = epsilon  # Q-Learningε-贪心探索率
        self.epsilon_min = epsilon_min  # 最小探索率（保证一定探索）
        self.epsilon_decay = epsilon_decay  # 探索率衰减率
        self.batch_size = batch_size
        self.target_update = target_update
        self.update_counter = 0

        self.policy_net = DQN(input_channels, board_size, action_dim).to(self.device)
        self.target_net = DQN(input_channels, board_size, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)  # Q-Learning学习率
        self.memory = ReplayBuffer(buffer_size)

    def choose_action(self, state, legal_actions, eval_mode=False):
        """
        Q-Learningε-贪心策略实现：
        - 探索（ε概率）：随机选择合法行动
        - 利用（1-ε概率）：选择Q值最大的合法行动
        """
        # 随机选择（探索）
        if not eval_mode and np.random.rand() < self.epsilon:
            if legal_actions is not None and len(legal_actions) > 0:
                return np.random.choice(legal_actions)
            return 0  # 无合法动作返回0
        
        # 最优选择（利用）
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor).cpu().numpy().flatten()
        
        # 将非法动作的Q值设为极小，避免选中
        mask = np.ones(self.action_dim) * -1e9
        mask[legal_actions] = 0
        q_values += mask
        
        return np.argmax(q_values)

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def update(self):
        """
        Q-Learning核心更新公式实现：
        Q(s,a) ← Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]
        其中：
        - α：学习率（Adam优化器自动调整）
        - r：即时奖励
        - γ：折扣因子
        - max(Q(s',a'))：下一状态的最大Q值
        """
        if len(self.memory) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # 转换为tensor并移到设备
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # 计算当前Q值 Q(s,a)
        q_values = self.policy_net(states).gather(1, actions)
        # 计算目标Q值（双DQN优化）r + γ·max(Q(s',a'))
        next_q_actions = self.policy_net(next_states).argmax(1, keepdim=True)
        next_q_values = self.target_net(next_states).gather(1, next_q_actions).detach()
        target = rewards + self.gamma * next_q_values * (1 - dones)

        # 计算损失并优化（对应Q-Learning的更新公式）
        loss = nn.MSELoss()(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新目标网络
        self.update_counter += 1
        if self.update_counter % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self, ep=None, episodes=None):
        """动态epsilon衰减：前80%快速衰减，后20%保持最小值"""
        if ep is None or episodes is None:
            # 兼容原有逻辑
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        else:
            decay_rate = self.epsilon_decay if ep < episodes * 0.8 else 1.0
            self.epsilon = max(self.epsilon_min, self.epsilon * decay_rate)

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())