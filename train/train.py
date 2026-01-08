"""
train.py - 启发式权重优化脚本

功能：
- 使用贝叶斯优化 (Bayesian Optimization) 自动调整 NewAgent 的启发式权重
- 通过与 BasicAgent 对战来评估每一组参数的性能
- 输出最佳参数组合

使用方法：
    python train.py
"""

import sys
import os
# 添加父目录到 path 以便导入 agent 和 poolenv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from bayes_opt import BayesianOptimization
from poolenv import PoolEnv
from agent import BasicAgent, NewAgent

# 设置评估局数 (为了速度，每组参数只打 10 局，正式训练可增加)
N_EVAL_GAMES = 10

def evaluate_params(w_cut_angle, w_distance, w_safety_penalty, w_cushion_penalty, w_position, w_safety_quality, w_lookahead):
    """
    评估函数：输入一组权重，返回 NewAgent 的平均得分或胜率
    """
    weights = {
        'w_cut_angle': w_cut_angle,
        'w_distance': w_distance,
        'w_safety_penalty': w_safety_penalty,
        'w_cushion_penalty': w_cushion_penalty,
        'w_position': w_position,
        'w_safety_quality': w_safety_quality,
        'w_lookahead': w_lookahead
    }
    
    env = PoolEnv()
    agent_opponent = BasicAgent()
    agent_learner = NewAgent(weights=weights)
    
    # NewAgent 始终作为 Agent B (方便统计)
    players = [agent_opponent, agent_learner]
    target_ball_choice = ['solid', 'solid', 'stripe', 'stripe']
    
    total_score = 0
    wins = 0
    
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
                # 统计 NewAgent (Agent B) 的胜负
                # 注意：evaluate.py 中有复杂的轮换逻辑，这里简化处理
                # 我们只关心 learner 是否获胜
                
                # 确定本局 learner 是 A 还是 B
                # i=0: A=Opponent, B=Learner (Learner is B)
                # i=1: A=Learner, B=Opponent (Learner is A)
                
                learner_role = 'B' if i % 2 == 0 else 'A'
                winner = info['winner']
                
                if winner == learner_role:
                    wins += 1
                    total_score += 1.0
                elif winner == 'SAME':
                    total_score += 0.5
                
                break
                
    # 返回胜率作为优化目标
    return total_score / N_EVAL_GAMES

def run_optimization():
    # 定义搜索空间
    pbounds = {
        'w_cut_angle': (0.5, 5.0),       # 切球角度惩罚权重
        'w_distance': (5.0, 20.0),       # 距离惩罚权重
        'w_safety_penalty': (10.0, 100.0), # 防守方案的基础负分
        'w_cushion_penalty': (0.0, 20.0),  # 贴库惩罚权重
        'w_position': (0.0, 10.0),         # 走位权重
        'w_safety_quality': (0.0, 5.0),    # 防守质量权重
        'w_lookahead': (0.0, 5.0)          # 前瞻权重
    }
    
    optimizer = BayesianOptimization(
        f=evaluate_params,
        pbounds=pbounds,
        random_state=42,
        verbose=2
    )
    
    print("开始训练 (贝叶斯优化)...")
    print(f"每轮评估局数: {N_EVAL_GAMES}")
    
    # 初始探索 5 次，优化 10 次
    optimizer.maximize(
        init_points=5,
        n_iter=10
    )
    
    print("\n训练结束！")
    print("最佳参数组合:")
    print(optimizer.max)

if __name__ == "__main__":
    run_optimization()
