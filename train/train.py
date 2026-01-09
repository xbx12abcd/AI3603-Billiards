"""
train.py - 训练价值网络

流程：
1. 数据收集 (Data Collection): 
   - 让 Agent (NewAgent) 自我对弈。
   - 记录 (State, Winner) 对。
2. 训练 (Training):
   - 使用收集的数据训练 BilliardValueNet。
"""

import os
import sys
# 将 ../eval 加入 path 以导入 poolenv 和 agents
sys.path.append(os.path.join(os.path.dirname(__file__), '../eval'))

import torch
import torch.nn as nn
import torch.optim as optim
from poolenv import PoolEnv
from agents.new_agent import NewAgent
from agents.basic_agent_pro import BasicAgentPro
from agents.network import BilliardValueNet, state_to_tensor
import random
import numpy as np

def collect_data(n_games=50):
    """收集对战数据"""
    env = PoolEnv()
    env.enable_noise = False
    env.MAX_HIT_COUNT = 20
    
    # 使用 NewAgent 与 BasicAgentPro 对战
    # agent_a: NewAgent (Learner)
    # agent_b: BasicAgentPro (Teacher/Opponent)
    agent_a = NewAgent()
    agent_b = BasicAgentPro() 
    
    data_buffer = [] # [(state_tensor, winner)]
    
    print(f"Starting data collection for {n_games} games...")
    
    for i in range(n_games):
        if (i+1) % 5 == 0:
            print(f"Collecting Game {i+1}/{n_games}...")
            
        env.reset(target_ball=['solid', 'stripe'][i%2])
        
        episode_states = [] # [(state_tensor, player_id)]
        
        while True:
            player = env.get_curr_player()
            obs = env.get_observation(player)
            balls, my_targets, table = obs
            
            # 记录当前状态
            state_tensor = state_to_tensor(balls, my_targets, table.w, table.l)
            episode_states.append((state_tensor, player))
            
            # 决策
            if player == 'A':
                action = agent_a.decision(*obs)
            else:
                action = agent_b.decision(*obs)
                
            env.take_shot(action)
            done, info = env.get_done()
            
            if done:
                winner = info['winner'] # 'A', 'B', 'SAME'
                
                # 回溯标记价值
                # 如果 Winner 是 A，则 A 的状态价值为 1，B 的状态价值为 -1
                for s_tensor, p_id in episode_states:
                    target_val = 0.0
                    if winner == 'SAME':
                        target_val = 0.0
                    elif winner == p_id:
                        target_val = 1.0
                    else:
                        target_val = -1.0
                    
                    data_buffer.append((s_tensor, torch.tensor([target_val])))
                break
                
    return data_buffer

def train_model(data, model, epochs=5):
    """训练模型"""
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    print(f"Training on {len(data)} samples...")
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        random.shuffle(data)
        
        # Mini-batch training (Batch size 32)
        batch_size = 32
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            inputs = torch.stack([item[0] for item in batch])
            targets = torch.stack([item[1] for item in batch])
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}, Loss: {total_loss / (len(data)/batch_size):.4f}")

if __name__ == "__main__":
    # 1. 初始化模型
    # 模型文件位于 ../eval/billiard_value_net.pth
    model_path = os.path.join(os.path.dirname(__file__), "../eval/billiard_value_net.pth")
    model = BilliardValueNet(input_dim=61, hidden_dim=128)
    
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path))
            print(f"Loaded existing model from {model_path}")
        except:
            print("Failed to load existing model, starting from scratch.")
    
    # 2. 收集数据
    print("Step 1: Collecting Data...")
    training_data = collect_data(n_games=50) 
    
    # 3. 训练
    print("Step 2: Training...")
    train_model(training_data, model, epochs=10)
    
    # 4. 保存
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
