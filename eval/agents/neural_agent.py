"""
neural_agent.py - 基于神经网络评估的 Agent

继承自 NewAgent，但使用 BilliardValueNet 来评估候选动作产生的局面。
"""

import copy
import numpy as np
import torch
import pooltool as pt
from .new_agent import NewAgent
from .network import BilliardValueNet, state_to_tensor
from .agent import simulate_with_timeout

class NeuralAgent(NewAgent):
    def __init__(self, model_path=None):
        super().__init__()
        self.model = BilliardValueNet(input_dim=61, hidden_dim=128)
        
        if model_path:
            try:
                self.model.load_state_dict(torch.load(model_path))
                self.model.eval()
                print(f"NeuralAgent loaded model from {model_path}")
            except Exception as e:
                print(f"Failed to load model: {e}, using random initialization.")
        
    def decision(self, balls=None, my_targets=None, table=None):
        """决策逻辑"""
        if balls is None or table is None:
            return self._random_action()
            
        # 1. 使用 NewAgent 的逻辑生成候选动作
        # 为了复用代码，我们暂时复制部分逻辑或直接调用父类方法如果结构允许
        # 但父类 decision 直接返回了动作，我们需要拦截。
        # 因此，我们需要重写 decision，但复用 _generate_candidates (如果父类有拆分的话)
        # 遗憾的是 NewAgent.decision 是一个大函数。
        # 我们这里简化处理：复制 NewAgent.decision 的前半部分（生成候选），
        # 然后用 Network 评分替代 heuristic_score。
        
        cue_ball = balls.get('cue')
        if not cue_ball: return self._random_action()
        ball_radius = cue_ball.params.R
        cue_pos = cue_ball.state.rvw[0]
        
        remaining_targets = [bid for bid in my_targets if balls[bid].state.s != 4]
        if not remaining_targets: remaining_targets = ['8']
        
        candidates = []
        
        # --- 候选生成 (与 NewAgent 相同) ---
        for target_id in remaining_targets:
            target_ball = balls[target_id]
            target_pos = target_ball.state.rvw[0]
            
            for pocket_id, pocket in table.pockets.items():
                pocket_pos = pocket.center
                phi, cut_angle, dist_cue_ghost = self._calculate_cut_angle(cue_pos, target_pos, pocket_pos, ball_radius)
                
                if cut_angle > 80: continue
                if not self._is_path_clear(target_pos, pocket_pos, balls, [target_id, 'cue'], ball_radius): continue
                
                target_vec = np.array(target_pos[:2])
                pocket_vec = np.array(pocket_pos[:2])
                tp_vec = pocket_vec - target_vec
                tp_dir = tp_vec / np.linalg.norm(tp_vec)
                ghost_vec = target_vec - tp_dir * (2 * ball_radius)
                
                if not self._is_path_clear(cue_pos, list(ghost_vec) + [0], balls, [target_id, 'cue'], ball_radius): continue
                
                base_v = 1.5 + dist_cue_ghost * 1.5 + (cut_angle / 90) * 2.0
                base_v = min(base_v, 6.0)
                
                candidates.append({
                    'phi': phi, 'V0': base_v, 'type': 'attack',
                    'heuristic_score': 100 # Placeholder
                })
                
        # 防守候选
        safety_shots = self._find_best_safety_shot(balls, my_targets, table, cue_pos, ball_radius)
        candidates.extend(safety_shots)
        
        if not candidates:
             for i in range(5):
                 candidates.append({'phi': np.random.uniform(0, 360), 'V0': np.random.uniform(1.0, 5.0), 'type': 'random'})

        # --- 模拟与神经网络评估 ---
        best_shot = None
        best_value = -float('inf')
        
        # 只评估前 20 个候选（如果有启发式排序最好，这里简单全评估或随机）
        # 为了体现 NN 的作用，我们应该评估更多候选
        eval_candidates = candidates[:20]
        
        for cand in eval_candidates:
            # 细化力度
            test_V0s = [cand['V0']]
            if cand.get('type') == 'attack':
                test_V0s = [cand['V0']*0.9, cand['V0'], cand['V0']*1.1]
            
            for v in test_V0s:
                v = np.clip(v, 0.5, 8.0)
                phi = cand['phi']
                
                # 模拟
                sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
                sim_table = copy.deepcopy(table)
                cue = pt.Cue(cue_ball_id="cue")
                cue.set_state(V0=v, phi=phi, theta=0, a=0, b=0)
                shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
                
                if simulate_with_timeout(shot):
                    # 获取模拟后的状态
                    next_balls = shot.balls
                    
                    # 转换状态为 Tensor
                    # 注意：my_targets 在下一时刻是否变化？
                    # 如果进球了，target 列表应该减少。我们需要手动推断一下。
                    # 这里简化处理：直接用当前 my_targets，虽然不太准确（如果进球了，NN应该学会看到球进了就是好）
                    state_tensor = state_to_tensor(next_balls, my_targets, table.w, table.l)
                    
                    # 神经网络打分
                    with torch.no_grad():
                        value = self.model(state_tensor).item()
                    
                    # 简单的规则辅助：如果犯规（白球进袋），给极低分
                    if next_balls['cue'].state.s == 4:
                        value = -1.0
                        
                    if value > best_value:
                        best_value = value
                        best_shot = {'V0': v, 'phi': phi, 'theta': 0, 'a': 0, 'b': 0}
                        
        if best_shot:
            print(f"[NeuralAgent] 选定动作: Value={best_value:.4f}, phi={best_shot['phi']:.1f}")
            return best_shot
        else:
            return self._random_action()
