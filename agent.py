"""
agent.py - Agent 决策模块

定义 Agent 基类和具体实现：
- Agent: 基类，定义决策接口
- BasicAgent: 基于贝叶斯优化的参考实现
- NewAgent: 学生自定义实现模板
- analyze_shot_for_reward: 击球结果评分函数
"""

import math
import pooltool as pt
import numpy as np
from pooltool.objects import PocketTableSpecs, Table, TableType
import copy
import os
from datetime import datetime
import random
import signal
# from poolagent.pool import Pool as CuetipEnv, State as CuetipState
# from poolagent import FunctionAgent

from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

# ============ 超时安全模拟机制 ============
class SimulationTimeoutError(Exception):
    """物理模拟超时异常"""
    pass

def _timeout_handler(signum, frame):
    """超时信号处理器"""
    raise SimulationTimeoutError("物理模拟超时")

def simulate_with_timeout(shot, timeout=3):
    """带超时保护的物理模拟 (Windows 兼容版 - 无超时)
    
    参数：
        shot: pt.System 对象
        timeout: 忽略
    
    返回：
        bool: True 表示模拟成功
    """
    try:
        pt.simulate(shot, inplace=True)
        return True
    except Exception as e:
        print(f"[WARNING] 物理模拟出错: {e}")
        return False

# ============================================



def analyze_shot_for_reward(shot: pt.System, last_state: dict, player_targets: list):
    """
    分析击球结果并计算奖励分数（完全对齐台球规则）
    
    参数：
        shot: 已完成物理模拟的 System 对象
        last_state: 击球前的球状态，{ball_id: Ball}
        player_targets: 当前玩家目标球ID，['1', '2', ...] 或 ['8']
    
    返回：
        float: 奖励分数
            +50/球（己方进球）, +100（合法黑8）, +10（合法无进球）
            -100（白球进袋）, -150（非法黑8/白球+黑8）, -30（首球/碰库犯规）
    
    规则核心：
        - 清台前：player_targets = ['1'-'7'] 或 ['9'-'15']，黑8不属于任何人
        - 清台后：player_targets = ['8']，黑8成为唯一目标球
    """
    
    # 1. 基本分析
    new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and last_state[bid].state.s != 4]
    
    # 根据 player_targets 判断进球归属（黑8只有在清台后才算己方球）
    own_pocketed = [bid for bid in new_pocketed if bid in player_targets]
    enemy_pocketed = [bid for bid in new_pocketed if bid not in player_targets and bid not in ["cue", "8"]]
    
    cue_pocketed = "cue" in new_pocketed
    eight_pocketed = "8" in new_pocketed

    # 2. 分析首球碰撞（定义合法的球ID集合）
    first_contact_ball_id = None
    foul_first_hit = False
    valid_ball_ids = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'}
    
    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, 'ids') else []
        if ('cushion' not in et) and ('pocket' not in et) and ('cue' in ids):
            # 过滤掉 'cue' 和非球对象（如 'cue stick'），只保留合法的球ID
            other_ids = [i for i in ids if i != 'cue' and i in valid_ball_ids]
            if other_ids:
                first_contact_ball_id = other_ids[0]
                break
    
    # 首球犯规判定：完全对齐 player_targets
    if first_contact_ball_id is None:
        # 未击中任何球（但若只剩白球和黑8且已清台，则不算犯规）
        if len(last_state) > 2 or player_targets != ['8']:
            foul_first_hit = True
    else:
        # 首次击打的球必须是 player_targets 中的球
        if first_contact_ball_id not in player_targets:
            foul_first_hit = True
    
    # 3. 分析碰库
    cue_hit_cushion = False
    target_hit_cushion = False
    foul_no_rail = False
    
    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, 'ids') else []
        if 'cushion' in et:
            if 'cue' in ids:
                cue_hit_cushion = True
            if first_contact_ball_id is not None and first_contact_ball_id in ids:
                target_hit_cushion = True

    if len(new_pocketed) == 0 and first_contact_ball_id is not None and (not cue_hit_cushion) and (not target_hit_cushion):
        foul_no_rail = True
        
    # 4. 计算奖励分数
    score = 0
    
    # 白球进袋处理
    if cue_pocketed and eight_pocketed:
        score -= 150  # 白球+黑8同时进袋，严重犯规
    elif cue_pocketed:
        score -= 100  # 白球进袋
    elif eight_pocketed:
        # 黑8进袋：只有清台后（player_targets == ['8']）才合法
        if player_targets == ['8']:
            score += 100  # 合法打进黑8
        else:
            score -= 150  # 清台前误打黑8，判负
            
    # 首球犯规和碰库犯规
    if foul_first_hit:
        score -= 30
    if foul_no_rail:
        score -= 30
        
    # 进球得分（own_pocketed 已根据 player_targets 正确分类）
    score += len(own_pocketed) * 50
    score -= len(enemy_pocketed) * 20
    
    # 合法无进球小奖励
    if score == 0 and not cue_pocketed and not eight_pocketed and not foul_first_hit and not foul_no_rail:
        score = 10
        
    return score

class Agent():
    """Agent 基类"""
    def __init__(self):
        pass
    
    def decision(self, *args, **kwargs):
        """决策方法（子类需实现）
        
        返回：dict, 包含 'V0', 'phi', 'theta', 'a', 'b'
        """
        pass
    
    def _random_action(self,):
        """生成随机击球动作
        
        返回：dict
            V0: [0.5, 8.0] m/s
            phi: [0, 360] 度
            theta: [0, 90] 度
            a, b: [-0.5, 0.5] 球半径比例
        """
        action = {
            'V0': round(random.uniform(0.5, 8.0), 2),   # 初速度 0.5~8.0 m/s
            'phi': round(random.uniform(0, 360), 2),    # 水平角度 (0°~360°)
            'theta': round(random.uniform(0, 90), 2),   # 垂直角度
            'a': round(random.uniform(-0.5, 0.5), 3),   # 杆头横向偏移（单位：球半径比例）
            'b': round(random.uniform(-0.5, 0.5), 3)    # 杆头纵向偏移
        }
        return action



class BasicAgent(Agent):
    """基于贝叶斯优化的智能 Agent"""
    
    def __init__(self, target_balls=None):
        """初始化 Agent
        
        参数：
            target_balls: 保留参数，暂未使用
        """
        super().__init__()
        
        # 搜索空间
        self.pbounds = {
            'V0': (0.5, 8.0),
            'phi': (0, 360),
            'theta': (0, 90), 
            'a': (-0.5, 0.5),
            'b': (-0.5, 0.5)
        }
        
        # 优化参数
        self.INITIAL_SEARCH = 20
        self.OPT_SEARCH = 10
        self.ALPHA = 1e-2
        
        # 模拟噪声（可调整以改变训练难度）
        self.noise_std = {
            'V0': 0.1,
            'phi': 0.1,
            'theta': 0.1,
            'a': 0.003,
            'b': 0.003
        }
        self.enable_noise = False
        
        print("BasicAgent (Smart, pooltool-native) 已初始化。")

    
    def _create_optimizer(self, reward_function, seed):
        """创建贝叶斯优化器
        
        参数：
            reward_function: 目标函数，(V0, phi, theta, a, b) -> score
            seed: 随机种子
        
        返回：
            BayesianOptimization对象
        """
        gpr = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=self.ALPHA,
            n_restarts_optimizer=10,
            random_state=seed
        )
        
        bounds_transformer = SequentialDomainReductionTransformer(
            gamma_osc=0.8,
            gamma_pan=1.0
        )
        
        optimizer = BayesianOptimization(
            f=reward_function,
            pbounds=self.pbounds,
            random_state=seed,
            verbose=0,
            bounds_transformer=bounds_transformer
        )
        optimizer._gp = gpr
        
        return optimizer


    def decision(self, balls=None, my_targets=None, table=None):
        """使用贝叶斯优化搜索最佳击球参数
        
        参数：
            balls: 球状态字典，{ball_id: Ball}
            my_targets: 目标球ID列表，['1', '2', ...]
            table: 球桌对象
        
        返回：
            dict: 击球动作 {'V0', 'phi', 'theta', 'a', 'b'}
                失败时返回随机动作
        """
        if balls is None:
            print(f"[BasicAgent] Agent decision函数未收到balls关键信息，使用随机动作。")
            return self._random_action()
        try:
            
            # 保存一个击球前的状态快照，用于对比
            last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}

            remaining_own = [bid for bid in my_targets if balls[bid].state.s != 4]
            if len(remaining_own) == 0:
                my_targets = ["8"]
                print("[BasicAgent] 我的目标球已全部清空，自动切换目标为：8号球")

            # 1.动态创建“奖励函数” (Wrapper)
            # 贝叶斯优化器会调用此函数，并传入参数
            def reward_fn_wrapper(V0, phi, theta, a, b):
                # 创建一个用于模拟的沙盒系统
                sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
                sim_table = copy.deepcopy(table)
                cue = pt.Cue(cue_ball_id="cue")

                shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
                
                try:
                    if self.enable_noise:
                        V0_noisy = V0 + np.random.normal(0, self.noise_std['V0'])
                        phi_noisy = phi + np.random.normal(0, self.noise_std['phi'])
                        theta_noisy = theta + np.random.normal(0, self.noise_std['theta'])
                        a_noisy = a + np.random.normal(0, self.noise_std['a'])
                        b_noisy = b + np.random.normal(0, self.noise_std['b'])
                        
                        V0_noisy = np.clip(V0_noisy, 0.5, 8.0)
                        phi_noisy = phi_noisy % 360
                        theta_noisy = np.clip(theta_noisy, 0, 90)
                        a_noisy = np.clip(a_noisy, -0.5, 0.5)
                        b_noisy = np.clip(b_noisy, -0.5, 0.5)
                        
                        shot.cue.set_state(V0=V0_noisy, phi=phi_noisy, theta=theta_noisy, a=a_noisy, b=b_noisy)
                    else:
                        shot.cue.set_state(V0=V0, phi=phi, theta=theta, a=a, b=b)
                    
                    # 关键：使用带超时保护的物理模拟（3秒上限）
                    if not simulate_with_timeout(shot, timeout=3):
                        return 0  # 超时是物理引擎问题，不惩罚agent
                except Exception as e:
                    # 模拟失败，给予极大惩罚
                    return -500
                
                # 使用我们的“裁判”来打分
                score = analyze_shot_for_reward(
                    shot=shot,
                    last_state=last_state_snapshot,
                    player_targets=my_targets
                )


                return score

            print(f"[BasicAgent] 正在为 Player (targets: {my_targets}) 搜索最佳击球...")
            
            seed = np.random.randint(1e6)
            optimizer = self._create_optimizer(reward_fn_wrapper, seed)
            optimizer.maximize(
                init_points=self.INITIAL_SEARCH,
                n_iter=self.OPT_SEARCH
            )
            
            best_result = optimizer.max
            best_params = best_result['params']
            best_score = best_result['target']

            if best_score < 10:
                print(f"[BasicAgent] 未找到好的方案 (最高分: {best_score:.2f})。使用随机动作。")
                return self._random_action()
            action = {
                'V0': float(best_params['V0']),
                'phi': float(best_params['phi']),
                'theta': float(best_params['theta']),
                'a': float(best_params['a']),
                'b': float(best_params['b']),
            }

            print(f"[BasicAgent] 决策 (得分: {best_score:.2f}): "
                  f"V0={action['V0']:.2f}, phi={action['phi']:.2f}, "
                  f"θ={action['theta']:.2f}, a={action['a']:.3f}, b={action['b']:.3f}")
            return action

        except Exception as e:
            print(f"[BasicAgent] 决策时发生严重错误，使用随机动作。原因: {e}")
            import traceback
            traceback.print_exc()
            return self._random_action()

class NewAgent(Agent):
    """基于启发式搜索的智能 Agent"""
    
    def __init__(self):
        super().__init__()
        self.enable_noise = False
        self.noise_std = {
            'V0': 0.1, 'phi': 0.1, 'theta': 0.1, 'a': 0.003, 'b': 0.003
        }
        
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
            s['heuristic_score'] = -50 # 负分，作为备选
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
        
        for cand in top_candidates:
            # 针对不同类型调整微调策略
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
                    # 1. 如果没有犯规且没有进球 (防守成功)，给予额外奖励
                    # 原有 score 对于无进球无犯规只有 10 分
                    # 我们希望优先选择不犯规的
                    
                    # 2. 简单的白球位置惩罚 (避免贴库)
                    # 获取模拟后的白球位置
                    sim_cue = shot.balls['cue']
                    if sim_cue.state.s != 4: # 白球未进袋
                        cx, cy = sim_cue.state.rvw[0][:2]
                        # 检查是否贴库 (假设库边区域为 0.1m)
                        margin = 0.1
                        if cx < margin or cx > table.w - margin or cy < margin or cy > table.l - margin:
                            score -= 5 # 贴库惩罚
                            
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