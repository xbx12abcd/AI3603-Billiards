"""
new_agent.py - 基于启发式搜索的智能 Agent

功能：
- 提供 NewAgent 类
- 包含 _calculate_cut_angle 等几何计算方法
"""

from .agent import Agent, analyze_shot_for_reward, simulate_with_timeout
import numpy as np
import copy
import random
import pooltool as pt
import torch
import os
from .network import BilliardValueNet, state_to_tensor

class NewAgent(Agent):
    """基于启发式搜索的智能 Agent"""
    
    def __init__(self, weights=None, model_path=None):
        super().__init__()
        self.enable_noise = False
        self.noise_std = {
            'V0': 0.1, 'phi': 0.1, 'theta': 0.1, 'a': 0.003, 'b': 0.003
        }
        
        # 加载神经网络模型 (如果有)
        self.model = None
        # 尝试加载默认路径
        if model_path is None:
             # 假设模型在 eval 目录下，或者当前工作目录
             potential_paths = ["billiard_value_net.pth", "eval/billiard_value_net.pth", "../billiard_value_net.pth"]
             for p in potential_paths:
                 if os.path.exists(p):
                     model_path = p
                     break
        
        if model_path and os.path.exists(model_path):
            try:
                self.model = BilliardValueNet(input_dim=61, hidden_dim=128)
                self.model.load_state_dict(torch.load(model_path))
                self.model.eval()
                print(f"[NewAgent] Loaded value network from {model_path}")
            except Exception as e:
                print(f"[NewAgent] Failed to load model: {e}")
                self.model = None

        # 默认启发式权重
        self.weights = {
            'w_cut_angle': 1.5,
            'w_distance': 10.0,
            'w_safety_penalty': 30.0,
            'w_cushion_penalty': 5.0,
            'w_position': 5.0,      # 走位权重：越小越好 (距离下一球越近越好)
            'w_safety_quality': 2.0, # 防守质量：越大越好 (距离对手球越远越好)
            'w_lookahead': 2.0       # 2步走位评分权重 (越大越倾向于好走位)
        }
        if weights:
            self.weights.update(weights)
        
    def _calculate_cut_angle(self, cue_pos, target_pos, pocket_pos, ball_radius):
        """计算切球角度
        
        参数:
            cue_pos: 白球位置 [x, y, z]
            target_pos: 目标球位置 [x, y, z]
            pocket_pos: 袋口位置 [x, y, z]
            ball_radius: 球半径
            
        返回:
            phi: 击球角度 (度)
            cut_angle: 切球角度 (度, 0=直球, 90=极薄)
            distance: 白球到幽灵球的距离
        """
        # 转换为 numpy 数组 (忽略 z 轴)
        cue = np.array(cue_pos[:2])
        target = np.array(target_pos[:2])
        pocket = np.array(pocket_pos[:2])
        
        # 1. 计算目标球到袋口的向量
        target_to_pocket = pocket - target
        dist_target_pocket = np.linalg.norm(target_to_pocket)
        if dist_target_pocket < 1e-5:
            return 0, 0, 0 # 已经在袋口
            
        # 单位向量
        u_target_pocket = target_to_pocket / dist_target_pocket
        
        # 2. 计算幽灵球位置 (Ghost Ball)
        # 幽灵球位于目标球沿 (pocket-target) 反方向 2R 处
        ghost_pos = target - u_target_pocket * (2 * ball_radius)
        
        # 3. 计算白球到幽灵球的向量 (这是击球方向)
        cue_to_ghost = ghost_pos - cue
        dist_cue_ghost = np.linalg.norm(cue_to_ghost)
        
        if dist_cue_ghost < 1e-5:
            phi = 0
        else:
            # 计算角度 phi (相对于 x 轴)
            phi_rad = np.arctan2(cue_to_ghost[1], cue_to_ghost[0])
            phi = np.degrees(phi_rad) % 360
            
        # 4. 计算切球角度 (Cut Angle)
        # 即 cue_to_ghost 和 target_to_pocket 之间的夹角
        # cos(theta) = (u_cue_ghost . u_target_pocket)
        if dist_cue_ghost > 1e-5:
            u_cue_ghost = cue_to_ghost / dist_cue_ghost
            cos_theta = np.dot(u_cue_ghost, u_target_pocket)
            # 限制范围 [-1, 1]
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            cut_angle = np.degrees(np.arccos(cos_theta))
        else:
            cut_angle = 0
            
        return phi, cut_angle, dist_cue_ghost

    def _is_path_clear(self, start_pos, end_pos, balls, ignore_ids, ball_radius):
        """检查路径是否被阻挡 (简单的圆柱体检测)"""
        start = np.array(start_pos[:2])
        end = np.array(end_pos[:2])
        path_vec = end - start
        path_len = np.linalg.norm(path_vec)
        if path_len < 1e-5:
            return True
            
        u_path = path_vec / path_len
        
        # 检查所有球
        for bid, ball in balls.items():
            if bid in ignore_ids or ball.state.s == 4: # 忽略自己和已进袋的球
                continue
                
            pos = np.array(ball.state.rvw[0][:2])
            
            # 计算球心到路径线段的距离
            # 投影长度
            proj = np.dot(pos - start, u_path)
            
            # 如果投影在路径范围内
            if 0 < proj < path_len:
                # 垂直距离
                closest_point = start + u_path * proj
                dist = np.linalg.norm(pos - closest_point)
                
                # 如果距离小于 2R (两个球的半径和)，则认为阻挡
                if dist < 2 * ball_radius * 0.95: # 0.95 是容差
                    return False
        
        return True

    def _evaluate_position_quality(self, balls, table, my_targets, cue_ball_id='cue'):
        """评估当前白球位置对下一杆的有利程度 (Lookahead)"""
        cue_ball = balls.get(cue_ball_id)
        if not cue_ball: return 0
        
        cue_pos = cue_ball.state.rvw[0]
        ball_radius = cue_ball.params.R
        
        remaining_targets = [bid for bid in my_targets if balls[bid].state.s != 4]
        if not remaining_targets:
            remaining_targets = ['8']
            
        max_opportunity_score = 0
        
        for target_id in remaining_targets:
            target_ball = balls[target_id]
            target_pos = target_ball.state.rvw[0]
            
            # Check all pockets
            for pocket in table.pockets.values():
                pocket_pos = pocket.center
                
                phi, cut_angle, dist = self._calculate_cut_angle(cue_pos, target_pos, pocket_pos, ball_radius)
                
                # Filter impossible shots
                if cut_angle > 75: continue
                
                # Check path
                if not self._is_path_clear(target_pos, pocket_pos, balls, [target_id, 'cue'], ball_radius):
                    continue
                # Note: ghost ball pos approx check
                # 这里用 target_pos 作为幽灵球近似，实际上 cue_pos 到 ghost_pos
                # 更精确的计算需要 ghost_pos
                target_vec = np.array(target_pos[:2])
                pocket_vec = np.array(pocket_pos[:2])
                tp_vec = pocket_vec - target_vec
                tp_dir = tp_vec / np.linalg.norm(tp_vec)
                ghost_vec = target_vec - tp_dir * (2 * ball_radius)
                
                if not self._is_path_clear(cue_pos, list(ghost_vec) + [0], balls, [target_id, 'cue'], ball_radius):
                    continue
                    
                # Score this opportunity
                # Closer distance is better, smaller cut angle is better
                # Max score around 100
                opp_score = 100 - cut_angle - dist * 10
                if opp_score > max_opportunity_score:
                    max_opportunity_score = opp_score
                    
        return max_opportunity_score

    def _find_best_safety_shot(self, balls, my_targets, table, cue_pos, ball_radius):
        """寻找最佳防守/解球方案 (Kick Shot)
        目标：合法击中自己的球，并尽量让白球停在难打的位置
        """
        safety_candidates = []
        
        remaining_targets = [bid for bid in my_targets if balls[bid].state.s != 4]
        if not remaining_targets:
            remaining_targets = ['8']

        # 1. 尝试直接击打 (虽然不能进球，但至少不犯规)
        for tid in remaining_targets:
            target_pos = balls[tid].state.rvw[0]
            # 计算方向
            vec = np.array(target_pos[:2]) - np.array(cue_pos[:2])
            dist = np.linalg.norm(vec)
            if dist < 1e-5: continue
            
            phi_rad = np.arctan2(vec[1], vec[0])
            phi = np.degrees(phi_rad) % 360
            
            # 检查路径是否被阻挡 (如果是被阻挡才需要解球，但这里也可以作为保底)
            # 如果直接路径通畅，就轻推
            if self._is_path_clear(cue_pos, target_pos, balls, [tid, 'cue'], ball_radius):
                 # 力度控制：刚好碰到球并弹开一点
                 # 粗略估算：滚动摩擦减速。
                 # 假设我们要打到球，速度不需要太大。
                 v_safety = 1.0 + dist * 0.5 
                 safety_candidates.append({'phi': phi, 'V0': v_safety, 'type': 'direct_safety'})

        # 2. 尝试单库解球 (Kick Shot)
        # 简化模型：寻找库边镜像点
        # 针对每个目标球，计算其关于四个库边的镜像
        # 库边位置 (Pooltool 默认): x: [0, w], y: [0, l]
        # table.w, table.l
        table_w = table.w
        table_l = table.l
        
        cushions = [
            {'name': 'right', 'mirror_axis': 'x', 'val': table_w},
            {'name': 'left', 'mirror_axis': 'x', 'val': 0},
            {'name': 'top', 'mirror_axis': 'y', 'val': table_l},
            {'name': 'bottom', 'mirror_axis': 'y', 'val': 0},
        ]

        for tid in remaining_targets:
            t_pos = balls[tid].state.rvw[0]
            
            for cush in cushions:
                # 计算镜像点
                mirror_pos = list(t_pos)
                if cush['mirror_axis'] == 'x':
                    mirror_pos[0] = 2 * cush['val'] - t_pos[0]
                else:
                    mirror_pos[1] = 2 * cush['val'] - t_pos[1]
                
                # 计算白球到镜像点的方向
                vec = np.array(mirror_pos[:2]) - np.array(cue_pos[:2])
                dist_mirror = np.linalg.norm(vec)
                phi_rad = np.arctan2(vec[1], vec[0])
                phi = np.degrees(phi_rad) % 360
                
                # 简单的路径检查 (检查白球到库边撞击点)
                # 撞击点计算
                # 白球射线: cue_pos + t * vec
                # 求与库边的交点
                # 这比较麻烦，做个简化：只要白球周围没球挡着就行
                # 或者直接加入候选，让模拟器去验证
                
                v_kick = 2.5 + dist_mirror * 0.8 # 解球通常需要大一点力
                safety_candidates.append({'phi': phi, 'V0': v_kick, 'type': 'kick_shot'})
        
        return safety_candidates

    def decision(self, balls=None, my_targets=None, table=None):
        """决策逻辑"""
        if balls is None or table is None:
            return self._random_action()
            
        # 获取白球
        cue_ball = balls.get('cue')
        if not cue_ball:
            return self._random_action()
            
        ball_radius = cue_ball.params.R
        cue_pos = cue_ball.state.rvw[0]
        
        # 确定实际目标球列表
        remaining_targets = [bid for bid in my_targets if balls[bid].state.s != 4]
        if not remaining_targets:
            remaining_targets = ['8'] # 清台后打黑8
            
        best_shot = None
        best_score = -float('inf')
        
        # 候选击球方案列表
        candidates = []
        
        # 1. 进攻策略：寻找直接进球机会
        for target_id in remaining_targets:
            target_ball = balls[target_id]
            target_pos = target_ball.state.rvw[0]
            
            for pocket_id, pocket in table.pockets.items():
                pocket_pos = pocket.center
                
                # 计算几何参数
                phi, cut_angle, dist_cue_ghost = self._calculate_cut_angle(
                    cue_pos, target_pos, pocket_pos, ball_radius
                )
                
                # 过滤高难度球
                if cut_angle > 80: # 角度太大，很难打
                    continue
                    
                # 路径检查
                # 1. 目标球到袋口
                if not self._is_path_clear(target_pos, pocket_pos, balls, [target_id, 'cue'], ball_radius):
                    continue
                
                # 2. 重新精确计算 ghost_pos 用于路径检测
                target_vec = np.array(target_pos[:2])
                pocket_vec = np.array(pocket_pos[:2])
                tp_vec = pocket_vec - target_vec
                tp_dir = tp_vec / np.linalg.norm(tp_vec)
                ghost_vec = target_vec - tp_dir * (2 * ball_radius)
                
                # 检查白球路径
                if not self._is_path_clear(cue_pos, list(ghost_vec) + [0], balls, [target_id, 'cue'], ball_radius):
                    continue
                    
                # 这是一个候选方案
                # 估算力度: 距离越远需要力度越大，切角越大需要力度越大
                # 基础力度 2.0 m/s
                base_v = 1.5 + dist_cue_ghost * 1.5 + (cut_angle / 90) * 2.0
                base_v = min(base_v, 6.0)
                
                candidates.append({
                    'phi': phi,
                    'V0': base_v,
                    'target_id': target_id,
                    'heuristic_score': 100 - cut_angle - dist_cue_ghost * 10, # 基础分 100，优先进攻
                    'type': 'attack'
                })
        
        # 2. 防守/解球策略：如果没有好的进攻机会，或者为了保底，加入解球候选
        # 获取解球候选
        safety_shots = self._find_best_safety_shot(balls, my_targets, table, cue_pos, ball_radius)
        # 给解球方案较低的分数，确保只有在没进攻机会时才选
        for s in safety_shots:
            s['heuristic_score'] = -self.weights.get('w_safety_penalty', 50.0) # 负分，作为备选
            candidates.append(s)

        # 按启发式分数排序，取前N个进行模拟验证
        # 增加候选数量，包含防守策略
        candidates.sort(key=lambda x: x['heuristic_score'], reverse=True)
        top_candidates = candidates[:10] # 增加到前10个
        
        # 如果依然没有候选 (极端情况)，随机打几个
        if not top_candidates:
             for i in range(5):
                 top_candidates.append({
                     'phi': random.uniform(0, 360), 
                     'V0': random.uniform(1.0, 5.0),
                     'type': 'random'
                 })
        
        # 3. 模拟验证选出最佳方案
        last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        
        # 增加候选数量到 15
        top_candidates = candidates[:15]
        
        for cand in top_candidates:
            if cand.get('type') == 'attack':
                test_V0s = [cand['V0'], cand['V0']*0.8, cand['V0']*1.2]
            elif cand.get('type') == 'kick_shot':
                test_V0s = [cand['V0'], cand['V0']*0.9, cand['V0']*1.1] # 解球对力度敏感
            else:
                test_V0s = [cand['V0']]
            
            for v in test_V0s:
                v = np.clip(v, 0.5, 8.0)
                phi = cand['phi']
                
                # 构建 shot
                sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
                sim_table = copy.deepcopy(table)
                cue = pt.Cue(cue_ball_id="cue")
                cue.set_state(V0=v, phi=phi, theta=0, a=0, b=0)
                
                shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
                
                if simulate_with_timeout(shot):
                    score = analyze_shot_for_reward(shot, last_state_snapshot, my_targets)
                    
                    # --- 增强评分逻辑 (Agent 内部优化) ---
                    # 获取模拟后的球状态
                    sim_cue = shot.balls['cue']
                    new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and last_state_snapshot[bid].state.s != 4]
                    own_pocketed = [bid for bid in new_pocketed if bid in my_targets]
                    
                    # 判断下一杆是谁的球权
                    # 规则：打进己方球且不犯规 -> 继续
                    # 犯规 -> 对手
                    # 没打进 -> 对手
                    # 简单判断犯规 (analyze_shot_for_reward 内部逻辑比较复杂，这里简化判断)
                    # 如果 score 扣了大分，说明犯规了
                    # 更好的方式是看 analyze_shot_for_reward 里的标志位，但这里没法直接获取
                    # 我们通过 score 和 own_pocketed 来推断
                    
                    is_foul = score < -10 # 粗略判断
                    is_my_turn = (len(own_pocketed) > 0) and (not is_foul)
                    
                    # --- 神经网络价值评估 ---
                    if self.model:
                        try:
                            sim_table = shot.table
                            # 确定用于评估的目标球
                            if is_my_turn:
                                eval_targets = [bid for bid in my_targets if bid not in own_pocketed]
                                if not eval_targets: eval_targets = ['8']
                                # 评估 V(s', my_targets) -> 越大越好
                                s_tensor = state_to_tensor(shot.balls, eval_targets, sim_table.w, sim_table.l)
                                with torch.no_grad():
                                    val = self.model(s_tensor.unsqueeze(0)).item() # [1, input_dim]
                                score += val * 50.0 # 权重 50，相当于进一颗球
                                
                            else:
                                # 对手回合
                                opp_targets = self._get_opponent_targets(shot.balls, my_targets)
                                # 评估 V(s', opp_targets) -> 越小越好 (对手胜率低)
                                s_tensor = state_to_tensor(shot.balls, opp_targets, sim_table.w, sim_table.l)
                                with torch.no_grad():
                                    val = self.model(s_tensor.unsqueeze(0)).item()
                                score -= val * 50.0 
                        except Exception as e:
                            print(f"Model eval error: {e}")

                    # 1. 走位逻辑 (Position Play)
                    # 如果进球了，考虑白球停在哪
                    if own_pocketed and sim_cue.state.s != 4:
                        # 简单的距离奖励 (保留作为基础)
                        # score -= min_dist_next * self.weights.get('w_position', 5.0) 
                        
                        # 使用更高级的 Lookahead 评分
                        # 如果有下一杆机会，给予大幅奖励
                        next_shot_quality = self._evaluate_position_quality(shot.balls, table, my_targets)
                        
                        # 奖励 = 质量分 * 权重
                        # 质量分 max 100 左右
                        score += next_shot_quality * self.weights.get('w_lookahead', 0.5)
                        
                        # 如果 Lookahead 发现是死球 (0分)，则启用备用的距离惩罚
                        if next_shot_quality < 5:
                             # 寻找下一杆的最佳目标 (距离)
                            remaining_after_shot = [bid for bid in remaining_targets if bid not in own_pocketed]
                            if not remaining_after_shot:
                                remaining_after_shot = ['8']
                            
                            min_dist_next = float('inf')
                            cue_final_pos = sim_cue.state.rvw[0]
                            for next_bid in remaining_after_shot:
                                next_ball_pos = sim_balls[next_bid].state.rvw[0]
                                d = np.linalg.norm(next_ball_pos[:2] - cue_final_pos[:2])
                                if d < min_dist_next:
                                    min_dist_next = d
                            if min_dist_next != float('inf'):
                                score -= min_dist_next * self.weights.get('w_position', 5.0)

                    # 2. 防守逻辑 (Safety Quality)
                    # 如果没进球且没犯规，考虑白球是否安全
                    # 简单的安全定义：白球距离所有对手球越远越好
                    elif score > -50 and sim_cue.state.s != 4: # -50 是大概的犯规分界线，这里粗略判断没犯大错
                        # 假设对手目标是剩下的非己方球
                        # 这里简化为：距离刚才的目标球越远越好 (假设没打进)
                        # 或者距离所有潜在目标越远越好
                        
                        # 既然不知道对手确切目标 (虽然 poolenv 知道，但 agent 接口里 my_targets 只有自己的)
                        # 假设对手打的是非 my_targets 的球
                        enemy_potential_targets = [bid for bid in balls if bid not in my_targets and bid != 'cue' and bid != '8' and balls[bid].state.s != 4]
                        if not enemy_potential_targets:
                            enemy_potential_targets = ['8']
                            
                        min_dist_enemy = float('inf')
                        cue_final_pos = sim_cue.state.rvw[0]
                        
                        for eb_id in enemy_potential_targets:
                             eb_pos = sim_balls[eb_id].state.rvw[0]
                             d = np.linalg.norm(eb_pos[:2] - cue_final_pos[:2])
                             if d < min_dist_enemy:
                                 min_dist_enemy = d
                        
                        # 如果能算出来，就奖励距离
                        if min_dist_enemy != float('inf'):
                             score += min_dist_enemy * self.weights.get('w_safety_quality', 2.0)
                    
                    # 3. 简单的白球位置惩罚 (避免贴库)
                    if sim_cue.state.s != 4: # 白球未进袋
                        cx, cy = sim_cue.state.rvw[0][:2]
                        # 检查是否贴库 (假设库边区域为 0.1m)
                        margin = 0.1
                        if cx < margin or cx > table.w - margin or cy < margin or cy > table.l - margin:
                            score -= self.weights['w_cushion_penalty'] # 使用权重
                            
                    if score > best_score:
                        best_score = score
                        best_shot = {
                            'V0': v, 'phi': phi, 'theta': 0, 'a': 0, 'b': 0
                        }
                        
        if best_shot:
            print(f"[NewAgent] 选择方案: score={best_score}, phi={best_shot['phi']:.1f}")
            return best_shot
        else:
            print("[NewAgent] 无可行方案，随机击球")
            return self._random_action()
