"""
agent.py - Agent 决策模块

定义 Agent 基类和具体实现：

- Agent: 基类，定义决策接口
- BasicAgent: 基于贝叶斯优化的参考实现
- NewAgent: 学生自定义实现模板（增强版）
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
    """带超时保护的物理模拟
    
    参数：
        shot: pt.System 对象
        timeout: 超时时间（秒），默认3秒
    
    返回：
        bool: True 表示模拟成功，False 表示超时或失败
    
    说明：
        在 Windows 上不支持 signal.SIGALRM，因此直接运行模拟。
    """
    try:
        if hasattr(signal, 'SIGALRM'):
            # 设置超时信号处理器 (Linux/Unix)
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(timeout)  # 设置超时时间
            
            try:
                pt.simulate(shot, inplace=True)
                signal.alarm(0)  # 取消超时
                return True
            except SimulationTimeoutError:
                print(f"[WARNING] 物理模拟超时（>{timeout}秒），跳过此次模拟")
                return False
            finally:
                signal.signal(signal.SIGALRM, old_handler)  # 恢复原处理器
        else:
            # Windows 系统直接运行
            pt.simulate(shot, inplace=True)
            return True
            
    except Exception as e:
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)  # 取消超时
        # print(f"[WARNING] 物理模拟出错: {e}")
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
            
            # 1.动态创建"奖励函数" (Wrapper)
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
                
                # 使用我们的"裁判"来打分
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


class EnhancedNewAgent:
    """基于几何计算和物理模拟的增强版台球智能体（核心实现）"""
    
    def __init__(self):
        # 物理参数
        self.ball_radius = 0.02625  # 台球半径（米）
        self.table_width = 0.99     # 球桌宽度
        self.table_length = 1.98    # 球桌长度
        
        # 策略参数
        self.offensive_weight = 0.6
        self.defensive_weight = 0.4
        self.risk_tolerance = 0.3
        
        # 搜索参数
        self.max_attack_candidates = 10
        self.max_defense_candidates = 8
        self.simulation_count = 15
        
        # 击球参数选项
        self.vertical_angles = [0, 2, 5]  # 垂直角度
        self.side_spins = [-0.3, -0.15, 0, 0.15, 0.3]  # 横向偏移
        self.top_spins = [-0.2, 0, 0.2]  # 纵向偏移
        
        # 缓存
        self.shot_cache = {}
        self.cache_size = 300
        
        print("EnhancedNewAgent 已初始化")
    
    def _random_action(self):
        """随机动作生成器"""
        return {
            'V0': round(random.uniform(0.5, 8.0), 2),
            'phi': round(random.uniform(0, 360), 2),
            'theta': round(random.uniform(0, 90), 2),
            'a': round(random.uniform(-0.5, 0.5), 3),
            'b': round(random.uniform(-0.5, 0.5), 3)
        }
    
    def decision(self, balls=None, my_targets=None, table=None):
        """主决策函数"""
        if balls is None or table is None:
            return self._random_action()
        
        try:
            # 获取球桌尺寸
            self.table_width = table.w
            self.table_length = table.l
            
            # 1. 分析当前局势
            situation = self._analyze_situation(balls, my_targets, table)
            
            # 2. 根据局势生成候选动作
            if situation == "OFFENSIVE":
                # 进攻机会多，优先尝试进球
                candidates = self._generate_offensive_candidates(balls, my_targets, table)
            elif situation == "DEFENSIVE":
                # 需要防守，制造麻烦
                candidates = self._generate_defensive_candidates(balls, my_targets, table)
            else:  # BALANCED
                # 平衡策略，两者都考虑
                offensive = self._generate_offensive_candidates(balls, my_targets, table)
                defensive = self._generate_defensive_candidates(balls, my_targets, table)
                candidates = offensive[:8] + defensive[:8]  # 各取前8个
            
            if not candidates:
                print("[EnhancedAgent] 没有生成候选动作，使用保守策略")
                return self._conservative_safety_shot(balls, my_targets, table)
            
            # 3. 模拟验证选择最佳
            best_action = None
            best_score = -float('inf')
            
            last_state = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
            
            for cand in candidates[:self.simulation_count]:
                # 模拟击球
                sim_result = self._simulate_shot(last_state, my_targets, table, cand)
                
                if sim_result is None:
                    continue
                
                sim_state, score = sim_result
                
                # 增强评分：考虑白球最终位置
                if score > -100:  # 不是严重犯规
                    cue_ball = sim_state.get('cue')
                    if cue_ball and cue_ball.state.s != 4:
                        position_score = self._evaluate_cue_position(
                            cue_ball.state.rvw[0], balls, my_targets, table
                        )
                        score += position_score * 10
                
                # 如果是防守动作，额外考虑防守效果
                if cand.get('type') == 'defensive' and score > 0:
                    defense_quality = self._evaluate_defense_quality(
                        sim_state, balls, my_targets, table
                    )
                    score += defense_quality * 20
                
                if score > best_score:
                    best_score = score
                    best_action = cand
            
            # 4. 如果没有好动作，使用保守策略
            if best_action is None or best_score < 0:
                print("[EnhancedAgent] 所有候选得分都小于0，使用保守策略")
                return self._conservative_safety_shot(balls, my_targets, table)
            
            # 5. 添加执行噪声（模拟真实误差）
            final_action = self._add_execution_noise(best_action)
            
            print(f"[EnhancedAgent] 选择动作: 得分={best_score:.1f}, "
                  f"V0={final_action['V0']:.2f}, phi={final_action['phi']:.1f}, "
                  f"类型={best_action.get('type', 'unknown')}")
            
            return final_action
            
        except Exception as e:
            print(f"[EnhancedAgent] 决策出错: {e}")
            import traceback
            traceback.print_exc()
            return self._random_action()
    
    def _analyze_situation(self, balls, my_targets, table):
        """分析游戏局势"""
        cue_pos = balls['cue'].state.rvw[0]
        
        # 统计剩余球数
        my_remaining = [
            bid for bid in my_targets 
            if bid != '8' and balls[bid].state.s != 4
        ]
        
        # 检查直接进球机会
        pocket_opportunities = 0
        for target_id in my_remaining:
            target_pos = balls[target_id].state.rvw[0]
            for pocket_id, pocket in table.pockets.items():
                if self._has_direct_pocket_line(cue_pos, target_pos, pocket.center, balls, target_id):
                    pocket_opportunities += 1
                    break  # 一个球有一个机会就行
        
        # 局势判断
        if len(my_remaining) <= 2:
            return "OFFENSIVE"  # 清台阶段
        elif pocket_opportunities >= 2:
            return "OFFENSIVE"  # 进攻机会多
        elif pocket_opportunities == 0:
            return "DEFENSIVE"  # 没有进球机会
        else:
            return "BALANCED"
    
    def _generate_offensive_candidates(self, balls, my_targets, table):
        """生成进攻候选动作"""
        candidates = []
        cue_pos = balls['cue'].state.rvw[0]
        
        # 确定要打的目标球
        target_ids = [bid for bid in my_targets if bid != '8' and balls[bid].state.s != 4]
        if not target_ids:
            if '8' in my_targets and balls['8'].state.s != 4:
                target_ids = ['8']
            else:
                return candidates  # 没有目标球
        
        for target_id in target_ids:
            target_ball = balls[target_id]
            target_pos = target_ball.state.rvw[0]
            
            # 对每个袋口尝试
            for pocket_id, pocket in table.pockets.items():
                pocket_pos = pocket.center
                
                # 计算几何参数
                shot_params = self._calculate_shot_parameters(
                    cue_pos, target_pos, pocket_pos
                )
                
                if shot_params is None:
                    continue
                
                # 检查路径是否畅通
                if not self._is_path_clear(
                    cue_pos, target_pos, balls, 
                    ignore_ids=['cue', target_id]
                ):
                    # 尝试解球路径
                    kick_params = self._calculate_kick_shot(
                        cue_pos, target_pos, pocket_pos, balls, table, target_id
                    )
                    if kick_params:
                        candidates.append({
                            **kick_params,
                            'target_id': target_id,
                            'pocket_id': pocket_id,
                            'type': 'offensive_kick',
                            'heuristic_score': 60
                        })
                    continue
                
                # 计算启发式分数
                heuristic_score = self._calculate_heuristic_score(
                    cue_pos, target_pos, pocket_pos, balls, target_id
                )
                
                # 生成不同的击球方式
                base_action = {
                    'V0': shot_params['V0'],
                    'phi': shot_params['phi'],
                    'theta': 0,
                    'a': 0,
                    'b': 0,
                    'target_id': target_id,
                    'pocket_id': pocket_id,
                    'type': 'offensive',
                    'heuristic_score': heuristic_score
                }
                
                # 添加击球变化
                candidates.extend(self._add_shot_variations(base_action))
        
        # 按启发式分数排序
        candidates.sort(key=lambda x: x.get('heuristic_score', 0), reverse=True)
        return candidates[:self.max_attack_candidates]
    
    def _generate_defensive_candidates(self, balls, my_targets, table):
        """生成防守候选动作"""
        candidates = []
        cue_pos = balls['cue'].state.rvw[0]
        opponent_targets = self._get_opponent_targets(my_targets)
        
        # 1. 主动防守：阻挡对手进球路线
        for opp_id in opponent_targets:
            opp_ball = balls.get(opp_id)
            if not opp_ball or opp_ball.state.s == 4:
                continue
            
            opp_pos = opp_ball.state.rvw[0]
            
            # 找到对手最容易进的袋口
            best_pocket = None
            best_angle = 180
            
            for pocket_id, pocket in table.pockets.items():
                pocket_pos = pocket.center
                # 简单的角度计算
                vec = np.array(pocket_pos[:2]) - np.array(opp_pos[:2])
                if np.linalg.norm(vec) > 0:
                    angle = abs(math.degrees(math.atan2(vec[1], vec[0])))
                    if angle < best_angle:
                        best_angle = angle
                        best_pocket = (pocket_id, pocket_pos)
            
            if best_pocket and best_angle < 60:
                # 计算阻挡位置
                block_pos = self._calculate_block_position(
                    opp_pos, best_pocket[1], self.ball_radius * 3
                )
                
                # 计算如何让白球到达阻挡位置
                action = self._calculate_safety_shot(
                    cue_pos, block_pos, table
                )
                
                if action:
                    candidates.append({
                        **action,
                        'defense_type': 'block',
                        'target_id': opp_id,
                        'type': 'defensive',
                        'heuristic_score': 70
                    })
        
        # 2. 解球防守：确保碰到自己的球
        safety_shots = self._generate_safety_shots(balls, my_targets, table)
        for shot in safety_shots:
            candidates.append({
                **shot,
                'defense_type': 'safety',
                'type': 'defensive',
                'heuristic_score': 50
            })
        
        # 3. 制造混乱：轻推让局面复杂化
        if len(candidates) < 5:
            chaos_shots = self._generate_chaos_shots(balls, table)
            candidates.extend(chaos_shots)
        
        return candidates[:self.max_defense_candidates]
    
    def _calculate_shot_parameters(self, cue_pos, target_pos, pocket_pos):
        """计算击球参数（基于几何算法）"""
        try:
            # 转换为2D坐标
            cue = np.array(cue_pos[:2])
            target = np.array(target_pos[:2])
            pocket = np.array(pocket_pos[:2])
            
            # 1. 计算目标球到袋口方向
            target_to_pocket = pocket - target
            dist_target_pocket = np.linalg.norm(target_to_pocket)
            
            if dist_target_pocket < 1e-5:
                return None
            
            u_target_pocket = target_to_pocket / dist_target_pocket
            
            # 2. 计算幽灵球位置
            ghost_pos = target - u_target_pocket * (2 * self.ball_radius)
            
            # 3. 计算白球到幽灵球的方向
            cue_to_ghost = ghost_pos - cue
            dist_cue_ghost = np.linalg.norm(cue_to_ghost)
            
            if dist_cue_ghost < 1e-5:
                return None
            
            # 4. 计算角度
            phi_rad = np.arctan2(cue_to_ghost[1], cue_to_ghost[0])
            phi = np.degrees(phi_rad) % 360
            
            # 5. 计算切球角度
            u_cue_ghost = cue_to_ghost / dist_cue_ghost
            cos_theta = np.dot(u_cue_ghost, u_target_pocket)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            cut_angle = np.degrees(np.arccos(cos_theta))
            
            # 6. 计算所需力度
            # 基础力度 + 距离修正 + 角度修正
            base_v = 2.0
            distance_factor = dist_cue_ghost * 0.8
            angle_factor = (cut_angle / 90) * 1.5
            
            V0 = base_v + distance_factor + angle_factor
            V0 = np.clip(V0, 1.0, 6.0)
            
            return {
                'V0': V0,
                'phi': phi,
                'cut_angle': cut_angle,
                'distance': dist_cue_ghost
            }
            
        except Exception as e:
            print(f"[EnhancedAgent] 计算击球参数出错: {e}")
            return None
    
    def _is_path_clear(self, start_pos, end_pos, balls, ignore_ids):
        """检查路径是否畅通"""
        start = np.array(start_pos[:2])
        end = np.array(end_pos[:2])
        path_vec = end - start
        path_len = np.linalg.norm(path_vec)
        
        if path_len < 1e-5:
            return True
        
        u_path = path_vec / path_len
        
        for bid, ball in balls.items():
            if bid in ignore_ids or ball.state.s == 4:
                continue
            
            pos = np.array(ball.state.rvw[0][:2])
            ball_vec = pos - start
            
            # 投影长度
            proj = np.dot(ball_vec, u_path)
            
            # 如果投影在路径范围内
            if -self.ball_radius < proj < path_len + self.ball_radius:
                # 垂直距离
                closest_point = start + u_path * proj
                dist = np.linalg.norm(pos - closest_point)
                
                # 如果距离小于两个球的半径和，则认为阻挡
                if dist < 2 * self.ball_radius * 0.95:
                    return False
        
        return True
    
    def _calculate_kick_shot(self, cue_pos, target_pos, pocket_pos, balls, table, target_id):
        """计算解球路径（单库反弹）"""
        # 简单解球：尝试四个库边的镜像点
        cushion_points = [
            np.array([0, 0]),  # 左下角
            np.array([table.w, 0]),  # 右下角
            np.array([0, table.l]),  # 左上角
            np.array([table.w, table.l]),  # 右上角
            np.array([table.w/2, 0]),  # 下中
            np.array([table.w/2, table.l]),  # 上中
            np.array([0, table.l/2]),  # 左中
            np.array([table.w, table.l/2]),  # 右中
        ]
        
        for point in cushion_points:
            # 检查白球到库边的路径
            if not self._is_path_clear(cue_pos, point, balls, ignore_ids=['cue']):
                continue
            
            # 检查库边到目标球的路径
            if not self._is_path_clear(point, target_pos, balls, ignore_ids=[target_id]):
                continue
            
            # 计算击球方向
            direction = point - np.array(cue_pos[:2])
            dist = np.linalg.norm(direction)
            
            if dist < 1e-5:
                continue
            
            phi_rad = np.arctan2(direction[1], direction[0])
            phi = np.degrees(phi_rad) % 360
            
            # 计算力度（解球需要稍大力）
            V0 = min(5.0, dist * 1.2 + 1.5)
            
            return {
                'V0': V0,
                'phi': phi,
                'theta': 3.0,  # 小角度
                'a': 0,
                'b': 0
            }
        
        return None
    
    def _calculate_block_position(self, target_pos, pocket_pos, distance):
        """计算阻挡位置"""
        target = np.array(target_pos[:2])
        pocket = np.array(pocket_pos[:2])
        
        direction = pocket - target
        dist = np.linalg.norm(direction)
        
        if dist < 1e-5:
            return target + np.array([0.1, 0.1])
        
        direction = direction / dist
        
        # 在目标球后方
        block_pos = target - direction * distance
        
        # 确保在桌面内
        block_pos[0] = np.clip(block_pos[0], self.ball_radius, self.table_width - self.ball_radius)
        block_pos[1] = np.clip(block_pos[1], self.ball_radius, self.table_length - self.ball_radius)
        
        return block_pos
    
    def _calculate_safety_shot(self, cue_pos, target_pos, table):
        """计算安全击球参数"""
        cue = np.array(cue_pos[:2])
        target = np.array(target_pos[:2])
        
        direction = target - cue
        dist = np.linalg.norm(direction)
        
        if dist < 1e-5:
            return None
        
        phi_rad = np.arctan2(direction[1], direction[0])
        phi = np.degrees(phi_rad) % 360
        
        # 安全击球：轻推为主
        V0 = min(3.0, dist * 0.8 + 0.5)
        
        return {
            'V0': V0,
            'phi': phi,
            'theta': 2.0,  # 小角度
            'a': 0,
            'b': 0
        }
    
    def _generate_safety_shots(self, balls, my_targets, table):
        """生成安全击球（确保碰到自己的球）"""
        shots = []
        cue_pos = balls['cue'].state.rvw[0]
        
        # 找到自己的球
        my_balls = [bid for bid in my_targets if bid != '8' and balls[bid].state.s != 4]
        if not my_balls and '8' in my_targets and balls['8'].state.s != 4:
            my_balls = ['8']
        
        for ball_id in my_balls:
            ball_pos = balls[ball_id].state.rvw[0]
            
            # 直接击打
            action = self._calculate_safety_shot(cue_pos, ball_pos, table)
            if action:
                shots.append(action)
        
        return shots
    
    def _generate_chaos_shots(self, balls, table):
        """生成制造混乱的击球"""
        shots = []
        cue_pos = balls['cue'].state.rvw[0]
        
        # 随机轻推不同方向
        for _ in range(3):
            # 随机选择一个区域
            target_x = random.uniform(self.ball_radius, self.table_width - self.ball_radius)
            target_y = random.uniform(self.ball_radius, self.table_length - self.ball_radius)
            
            action = self._calculate_safety_shot(
                cue_pos, 
                [target_x, target_y, 0], 
                table
            )
            
            if action:
                action['V0'] = random.uniform(1.0, 2.5)  # 轻推
                shots.append(action)
        
        return shots
    
    def _add_shot_variations(self, base_action):
        """为基本动作添加击球变化"""
        variations = []
        
        base_V0 = base_action['V0']
        base_phi = base_action['phi']
        
        # 不同的击球方式
        shot_types = [
            {'name': 'center', 'theta': 0, 'a': 0, 'b': 0, 'V0_factor': 1.0},
            {'name': 'follow', 'theta': 5, 'a': 0, 'b': 0.2, 'V0_factor': 0.9},  # 高杆
            {'name': 'draw', 'theta': 5, 'a': 0, 'b': -0.2, 'V0_factor': 1.1},   # 低杆
            {'name': 'left', 'theta': 0, 'a': 0.2, 'b': 0, 'V0_factor': 1.0},    # 左塞
            {'name': 'right', 'theta': 0, 'a': -0.2, 'b': 0, 'V0_factor': 1.0},  # 右塞
        ]
        
        for shot in shot_types:
            variation = base_action.copy()
            variation['V0'] = base_V0 * shot['V0_factor']
            variation['theta'] = shot['theta']
            variation['a'] = shot['a']
            variation['b'] = shot['b']
            variation['shot_type'] = shot['name']
            
            # 根据击球类型调整分数
            if shot['name'] in ['follow', 'draw']:
                variation['heuristic_score'] = base_action.get('heuristic_score', 0) * 1.05
            elif shot['name'] in ['left', 'right']:
                variation['heuristic_score'] = base_action.get('heuristic_score', 0) * 0.95
            else:
                variation['heuristic_score'] = base_action.get('heuristic_score', 0)
            
            variations.append(variation)
        
        return variations
    
    def _simulate_shot(self, last_state, player_targets, table, action):
        """模拟击球并返回结果"""
        try:
            # 创建模拟系统
            sim_balls = {bid: copy.deepcopy(ball) for bid, ball in last_state.items()}
            sim_table = copy.deepcopy(table)
            cue = pt.Cue(cue_ball_id="cue")
            
            shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
            
            # 应用动作
            shot.cue.set_state(
                V0=action['V0'],
                phi=action['phi'],
                theta=action.get('theta', 0),
                a=action.get('a', 0),
                b=action.get('b', 0)
            )
            
            # 模拟
            if simulate_with_timeout(shot):
                score = analyze_shot_for_reward(shot, last_state, player_targets)
                return shot.balls, score
            else:
                return None, -50  # 模拟失败
        
        except Exception as e:
            print(f"[EnhancedAgent] 模拟出错: {e}")
            return None, -100
    
    def _evaluate_cue_position(self, cue_pos, balls, my_targets, table):
        """评估白球位置质量"""
        score = 0.0
        
        # 1. 是否贴库（负分）
        margin = 0.1
        if (cue_pos[0] < margin or cue_pos[0] > self.table_width - margin or
            cue_pos[1] < margin or cue_pos[1] > self.table_length - margin):
            score -= 0.5
        
        # 2. 是否在开球区（正分）
        if (self.table_width/4 < cue_pos[0] < 3*self.table_width/4 and
            self.table_length/4 < cue_pos[1] < 3*self.table_length/4):
            score += 0.3
        
        # 3. 离自己目标球的平均距离（越小越好）
        total_distance = 0
        count = 0
        
        for target_id in my_targets:
            if target_id == '8':
                continue
                
            target_ball = balls.get(target_id)
            if target_ball and target_ball.state.s != 4:
                target_pos = target_ball.state.rvw[0]
                dist = np.linalg.norm(np.array(cue_pos[:2]) - np.array(target_pos[:2]))
                total_distance += dist
                count += 1
        
        if count > 0:
            avg_distance = total_distance / count
            if avg_distance < 0.5:
                score += 0.4
            elif avg_distance < 1.0:
                score += 0.2
        
        return score
    
    def _evaluate_defense_quality(self, sim_state, original_balls, my_targets, table):
        """评估防守质量"""
        quality = 0.0
        
        # 获取对手目标球
        opponent_targets = self._get_opponent_targets(my_targets)
        
        # 检查白球是否阻挡了对手的进球路线
        cue_ball = sim_state.get('cue')
        if not cue_ball or cue_ball.state.s == 4:
            return quality
        
        cue_pos = cue_ball.state.rvw[0]
        
        for opp_id in opponent_targets:
            opp_ball = original_balls.get(opp_id)
            if not opp_ball or opp_ball.state.s == 4:
                continue
            
            opp_pos = opp_ball.state.rvw[0]
            
            # 检查白球是否在对手球和袋口之间
            for pocket_id, pocket in table.pockets.items():
                pocket_pos = pocket.center
                
                # 计算白球是否在连线上
                if self._is_point_between(cue_pos[:2], opp_pos[:2], pocket_pos[:2]):
                    quality += 0.3
                    break
        
        return min(quality, 1.0)
    
    def _is_point_between(self, point, start, end):
        """检查点是否在线段上"""
        point = np.array(point)
        start = np.array(start)
        end = np.array(end)
        
        # 检查点是否在线段附近
        line_vec = end - start
        line_len = np.linalg.norm(line_vec)
        
        if line_len < 1e-5:
            return False
        
        point_vec = point - start
        
        # 投影长度
        proj = np.dot(point_vec, line_vec) / line_len
        
        # 如果投影在线段范围内
        if -self.ball_radius < proj < line_len + self.ball_radius:
            # 垂直距离
            closest_point = start + (line_vec / line_len) * proj
            dist = np.linalg.norm(point - closest_point)
            
            # 如果距离很近，认为在线上
            if dist < self.ball_radius * 2:
                return True
        
        return False
    
    def _conservative_safety_shot(self, balls, my_targets, table):
        """保守安全击球：确保不犯规"""
        cue_pos = balls['cue'].state.rvw[0]
        
        # 找到离白球最近的自方球
        closest_ball = None
        min_distance = float('inf')
        
        for target_id in my_targets:
            if target_id == '8' and len([bid for bid in my_targets if bid != '8' and balls[bid].state.s != 4]) > 0:
                continue
                
            ball = balls.get(target_id)
            if ball and ball.state.s != 4:
                ball_pos = ball.state.rvw[0]
                distance = np.linalg.norm(np.array(cue_pos[:2]) - np.array(ball_pos[:2]))
                
                if distance < min_distance:
                    min_distance = distance
                    closest_ball = (target_id, ball_pos)
        
        if closest_ball:
            # 轻推最近的自方球
            ball_id, ball_pos = closest_ball
            direction = np.array(ball_pos[:2]) - np.array(cue_pos[:2])
            dist = np.linalg.norm(direction)
            
            if dist > 1e-5:
                phi_rad = np.arctan2(direction[1], direction[0])
                phi = np.degrees(phi_rad) % 360
                
                return {
                    'V0': min(2.5, dist * 0.5 + 0.8),
                    'phi': phi,
                    'theta': 1.0,
                    'a': 0,
                    'b': 0
                }
        
        # 如果找不到球，随机轻推
        return {
            'V0': 1.8,
            'phi': random.uniform(0, 360),
            'theta': 1.0,
            'a': 0,
            'b': 0
        }
    
    def _add_execution_noise(self, action):
        """添加执行噪声"""
        noisy_action = action.copy()
        
        # 力度噪声 (±5%)
        noisy_action['V0'] = action['V0'] * random.uniform(0.95, 1.05)
        noisy_action['V0'] = np.clip(noisy_action['V0'], 0.5, 8.0)
        
        # 角度噪声 (±2度)
        noisy_action['phi'] = (action['phi'] + random.uniform(-2, 2)) % 360
        
        # 垂直角度噪声 (±1度)
        if 'theta' in action:
            noisy_action['theta'] = max(0, action['theta'] + random.uniform(-1, 1))
            noisy_action['theta'] = min(noisy_action['theta'], 90)
        
        # 偏移噪声 (±0.05)
        if 'a' in action:
            noisy_action['a'] = np.clip(action['a'] + random.uniform(-0.05, 0.05), -0.5, 0.5)
        
        if 'b' in action:
            noisy_action['b'] = np.clip(action['b'] + random.uniform(-0.05, 0.05), -0.5, 0.5)
        
        return noisy_action
    
    def _has_direct_pocket_line(self, cue_pos, target_pos, pocket_pos, balls, target_id):
        """检查是否有直接进球路线"""
        # 计算切球角度
        shot_params = self._calculate_shot_parameters(cue_pos, target_pos, pocket_pos)
        if not shot_params:
            return False
        
        # 角度太大不考虑
        if shot_params['cut_angle'] > 60:
            return False
        
        # 检查路径
        if not self._is_path_clear(cue_pos, target_pos, balls, ignore_ids=['cue', target_id]):
            return False
        
        # 检查目标球到袋口路径
        if not self._is_path_clear(target_pos, pocket_pos, balls, ignore_ids=[target_id]):
            return False
        
        return True
    
    def _calculate_heuristic_score(self, cue_pos, target_pos, pocket_pos, balls, target_id):
        """计算启发式分数"""
        score = 100.0
        
        # 距离惩罚
        cue_to_target = np.linalg.norm(np.array(cue_pos[:2]) - np.array(target_pos[:2]))
        score -= cue_to_target * 5
        
        target_to_pocket = np.linalg.norm(np.array(target_pos[:2]) - np.array(pocket_pos[:2]))
        score -= target_to_pocket * 8
        
        # 角度惩罚
        shot_params = self._calculate_shot_parameters(cue_pos, target_pos, pocket_pos)
        if shot_params:
            score -= shot_params['cut_angle'] * 0.5
        
        return max(score, 10)
    
    def _get_opponent_targets(self, my_targets):
        """获取对手目标球"""
        all_balls = {str(i) for i in range(1, 16)}
        my_set = set(my_targets)
        
        if '8' in my_set:
            my_set.remove('8')
        
        opponent_set = all_balls - my_set
        return list(opponent_set)


class NewAgent(Agent):
    """增强版智能体（完整实现）"""
    
    def __init__(self):
        super().__init__()
        self.enhanced_agent = EnhancedNewAgent()
    
    def decision(self, balls=None, my_targets=None, table=None):
        """决策方法
        
        参数：
            observation: (balls, my_targets, table)
        
        返回：
            dict: {'V0', 'phi', 'theta', 'a', 'b'}
        """
        return self.enhanced_agent.decision(balls, my_targets, table)
    
    def _random_action(self):
        """生成随机击球动作"""
        return self.enhanced_agent._random_action()