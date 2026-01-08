"""
train_mcts.py - MCTS Agent 参数优化脚本

功能：
- 优化 MCTSAgent 的超参数（如 UCB 常数 c_puct）
- 由于 MCTS 搜索开销大，我们使用较少的评估局数
"""

import sys
import os
# 添加父目录到 path 以便导入 agent 和 poolenv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from bayes_opt import BayesianOptimization
from poolenv import PoolEnv
from agent import BasicAgent
from mcts_agent import MCTSAgent

# 评估局数 (MCTS 较慢，设小一点)
N_EVAL_GAMES = 5

def evaluate_params(c_puct):
    """
    评估函数：输入 c_puct，返回 MCTSAgent 的胜率
    """
    # 模拟预算固定为 20 (根据性能调整)
    agent_learner = MCTSAgent(simulation_budget=20, c_puct=c_puct)
    agent_opponent = BasicAgent()
    
    env = PoolEnv()
    players = [agent_opponent, agent_learner]
    target_ball_choice = ['solid', 'solid', 'stripe', 'stripe']
    
    total_score = 0
    
    for i in range(N_EVAL_GAMES):
        env.reset(target_ball=target_ball_choice[i % 4])
        while True:
            player = env.get_curr_player()
            obs = env.get_observation(player)
            
            if player == 'A':
                action = players[i % 2].decision(*obs)
            else:
                action = players[(i + 1) % 2].decision(*obs)
                
            env.take_shot(action)
            done, info = env.get_done()
            
            if done:
                learner_role = 'B' if i % 2 == 0 else 'A'
                winner = info['winner']
                
                if winner == learner_role:
                    total_score += 1.0
                elif winner == 'SAME':
                    total_score += 0.5
                break
                
    return total_score / N_EVAL_GAMES

def run_optimization():
    # 搜索 c_puct (通常在 0.5 到 5.0 之间)
    pbounds = {
        'c_puct': (0.5, 5.0)
    }
    
    optimizer = BayesianOptimization(
        f=evaluate_params,
        pbounds=pbounds,
        random_state=42,
        verbose=2
    )
    
    print("开始训练 MCTS Agent 参数...")
    print(f"每轮评估局数: {N_EVAL_GAMES}")
    
    optimizer.maximize(
        init_points=3,
        n_iter=5
    )
    
    print("\n训练结束！")
    print("最佳参数组合:")
    print(optimizer.max)

if __name__ == "__main__":
    run_optimization()
