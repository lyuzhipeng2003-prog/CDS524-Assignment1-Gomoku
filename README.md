# CDS524-Assignment1-Gomoku
Gomoku game based on Q-Learning/DQN with Pygame UI


# CDS524-Assignment1-Gomoku
A Gomoku (Five in a Row) game based on Q-Learning/DQN, with Pygame interactive UI.

## Project Overview
This project implements a human-AI Gomoku game, where the AI is trained by DQN (an extended version of Q-Learning). The game supports real-time reward display, legal/illegal move detection, and win/draw judgment.

## Environment Dependencies
Install required libraries before running:
```bash
pip install pygame torch numpy matplotlib

HOW TO RUN
1、Clone the repository to your local machine:
git clone https://github.com/Lyuzhipeng2003/CDS524-Assignment1-Gomoku.git
cd CDS524-Assignment1-Gomoku

2、Train the AI model:
python train.py

3、Play against the AI:
python play.py

File Structure
gomoku_game.py: Core game environment (board management, reward calculation, UI rendering)
dqn_agent.py: DQN agent implementation (Q-Learning logic, experience replay, model update)
train.py: Model training script (training loop, reward curve visualization)
play.py: Human-AI battle script (interactive UI, mouse click control)
Gomoku_QLearning.ipynb: Colab/Jupyter Notebook (code + inline explanations)
training_curve.png: Training reward curve screenshot
game_screenshot.png: Game UI screenshot
