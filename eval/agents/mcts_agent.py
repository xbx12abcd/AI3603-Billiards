"""
mcts_agent.py - 基于蒙特卡洛树搜索 (MCTS) 的 Agent

该模块实现了基于 MCTS 的决策逻辑。
由于台球环境的物理模型已知且精确 (pooltool)，我们采用了 AlphaZero 风格的 MCTS，
直接使用模拟器作为环境模型，而不是像 MuZero 那样学习隐式模型。

核心组件：
1. MCTSNode: 树节点，存储状态、访问次数、价值估算等
2. MCTSAgent: Agent 实现，执行搜索和决策
"""

import math
import pooltool as pt
import numpy as np
import copy
import random
import time
from .agent import Agent, analyze_shot_for_reward, simulate_with_timeout
from .new_agent import NewAgent

class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        """
        初始化树节点
        
        参数:
            state: 包含 'balls', 'table', 'my_targets' 的字典 (环境快照)
            parent: 父节点
            action: 导致此节点的动作 (from parent)
        """
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value_sum = 0.0
        self.untried_actions = None # 将在 expand 时填充
        
    @property
    def value(self):
        if self.visits == 0:
            return 0
        return self.value_sum / self.visits

    def is_fully_expanded(self):
        return self.untried_actions is not None and len(self.untried_actions) == 0

    def is_terminal(self):
        # 简化判断：如果游戏结束（例如黑8进袋或白球进袋导致犯规等），视为终局
        # 这里暂时通过外部逻辑判断，或者检查 state 中的 done 标志
        # 在台球中，通常根据是否还有目标球来判断
        balls = self.state['balls']
        my_targets = self.state['my_targets']
        
        # 简单检查：如果自己赢了
        remaining = [b for b in my_targets if balls[b].state.s != 4]
        if not remaining and balls['8'].state.s == 4:
            return True
        return False

class MCTSAgent(Agent):
    def __init__(self, simulation_budget=20, max_depth=3, c_puct=1.414):
        """
        初始化 MCTS Agent
        
        参数:
            simulation_budget: 每次决策的最大模拟次数 (Iterations)
            max_depth: 搜索最大深度
            c_puct: UCB 探索常数
        """
        super().__init__()
        self.simulation_budget = simulation_budget
        self.max_depth = max_depth
        self.c_puct = c_puct
        
        # 复用 NewAgent 的启发式逻辑来生成候选动作和评估
        self.heuristic_agent = NewAgent()
        
    def decision(self, balls, my_targets, table):
        """MCTS 决策入口"""
        root_state = {
            'balls': copy.deepcopy(balls),
            'table': copy.deepcopy(table),
            'my_targets': list(my_targets)
        }
        
        root = MCTSNode(state=root_state)
        
        # MCTS 主循环
        start_time = time.time()
        for i in range(self.simulation_budget):
            node = self._select(root)
            reward = self._simulate(node)
            self._backpropagate(node, reward)
            
            # 简单的超时控制 (防止单步思考太久)
            if time.time() - start_time > 10.0: # 最多思考 10 秒
                break
                
        # 选择最佳动作 (访问次数最多)
        if not root.children:
            print("[MCTS] 搜索未生成任何子节点，使用随机动作")
            return self._random_action()
            
        best_child = max(root.children, key=lambda c: c.visits)
        print(f"[MCTS] 搜索完成 (Iter: {root.visits}, Best Val: {best_child.value:.2f})")
        return best_child.action

    def _select(self, node):
        """选择阶段：应用 UCB 策略向下遍历直到找到未完全扩展的节点"""
        while not node.is_terminal():
            if not node.is_fully_expanded():
                return self._expand(node)
            else:
                if not node.children:
                    return node # 无法扩展且无子节点
                node = self._best_child(node)
        return node

    def _expand(self, node):
        """扩展阶段：生成一个新子节点"""
        if node.untried_actions is None:
            # 使用 heuristic agent 生成候选动作
            # 这里我们利用 NewAgent 的逻辑来筛选高概率动作，而不是完全随机采样
            # 这样可以大幅缩减搜索空间
            candidates = self._generate_candidates(node.state)
            node.untried_actions = candidates
            
        if not node.untried_actions:
            return node # 无动作可扩展
            
        action = node.untried_actions.pop(0)
        
        # 执行动作，获取新状态
        next_state = self._step(node.state, action)
        
        new_node = MCTSNode(state=next_state, parent=node, action=action)
        node.children.append(new_node)
        return new_node

    def _simulate(self, node):
        """模拟阶段 (Rollout)：快速推演到最大深度或终局"""
        current_state = copy.deepcopy(node.state)
        cumulative_reward = 0
        depth = 0
        
        # 计算当前节点的即时奖励 (作为 Value 的一部分)
        # 注意：这里我们简化处理，MCTS 的 Value 通常是最终胜负
        # 但在台球这种长程游戏中，使用中间奖励 (Heuristic Value) 效果更好
        
        # 这里我们直接评估当前状态的好坏
        # 如果是叶子节点，评估其静态价值
        val = self._evaluate_state(current_state)
        return val

    def _backpropagate(self, node, reward):
        """回溯阶段：更新路径上节点的统计信息"""
        while node is not None:
            node.visits += 1
            node.value_sum += reward
            node = node.parent

    def _best_child(self, node):
        """使用 UCB1 公式选择最佳子节点"""
        best_score = -float('inf')
        best_children = []
        
        for child in node.children:
            exploit = child.value_sum / child.visits
            explore = math.sqrt(2 * math.log(node.visits) / child.visits)
            score = exploit + self.c_puct * explore
            
            if score > best_score:
                best_score = score
                best_children = [child]
            elif score == best_score:
                best_children.append(child)
                
        return random.choice(best_children)

    def _generate_candidates(self, state):
        """使用 NewAgent 的启发式逻辑生成 Top N 候选动作"""
        # 临时借用 NewAgent 的内部方法
        # 注意：NewAgent.decision 内部会做模拟，我们只需要它的候选生成逻辑
        # 为了复用，我们稍微 hack 一下，或者重构代码。
        # 这里为了简单，我们手动调用 NewAgent 中类似的逻辑
        
        balls = state['balls']
        table = state['table']
        my_targets = state['my_targets']
        cue_ball = balls.get('cue')
        
        if not cue_ball or cue_ball.state.s == 4:
            return []
            
        # 重新利用 NewAgent 的逻辑
        # 为了避免大量重复代码，我们实例化一个 NewAgent 并调用其 helper
        # 但 NewAgent 没有直接公开 generate_candidates。
        # 我们这里简化实现一个基于 NewAgent 逻辑的生成器
        
        candidates = []
        
        # 1. 获取 NewAgent 的候选 (不进行物理模拟，只进行几何筛选)
        # 我们需要访问 NewAgent 的私有方法，或者复制逻辑。
        # 鉴于 NewAgent 代码在 agent.py，我们可以考虑在 agent.py 中重构
        # 但为了不破坏现有结构，我在这里简化复制几何计算部分
        
        remaining_targets = [bid for bid in my_targets if balls[bid].state.s != 4]
        if not remaining_targets:
            remaining_targets = ['8']
            
        cue_pos = cue_ball.state.rvw[0]
        ball_radius = cue_ball.params.R
        
        for target_id in remaining_targets:
            target_ball = balls[target_id]
            target_pos = target_ball.state.rvw[0]
            
            for pocket in table.pockets.values():
                pocket_pos = pocket.center
                
                phi, cut_angle, dist = self.heuristic_agent._calculate_cut_angle(
                    cue_pos, target_pos, pocket_pos, ball_radius
                )
                
                if cut_angle > 80: continue
                
                # 简单路径检查
                if not self.heuristic_agent._is_path_clear(target_pos, pocket_pos, balls, [target_id, 'cue'], ball_radius):
                    continue
                    
                # 估算力度
                base_v = 1.5 + dist * 1.5 + (cut_angle / 90) * 2.0
                base_v = min(base_v, 6.0)
                
                candidates.append({
                    'phi': phi,
                    'V0': base_v,
                    'theta': 0, 'a': 0, 'b': 0,
                    'score': 100 - cut_angle  # 简单排序依据
                })
        
        # 排序并取前 5 个作为 MCTS 的动作分支
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[:5]

    def _step(self, state, action):
        """在模拟器中执行动作，返回新状态"""
        # 深拷贝环境
        sim_balls = {bid: copy.deepcopy(ball) for bid, ball in state['balls'].items()}
        sim_table = copy.deepcopy(state['table'])
        
        cue = pt.Cue(cue_ball_id="cue")
        cue.set_state(V0=action['V0'], phi=action['phi'], theta=action['theta'], a=action['a'], b=action['b'])
        
        shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
        simulate_with_timeout(shot)
        
        # 更新目标球 (简单的逻辑：如果打进了，目标不变；如果没打进，可能换手)
        # 但 MCTS 通常只规划自己的回合。如果换手了，我们可以评估为“对手回合”的价值
        # 这里简化：只返回物理状态，逻辑状态由 Value Function 评估
        
        return {
            'balls': shot.balls,
            'table': sim_table, # table 状态通常不变，除非有损坏逻辑
            'my_targets': state['my_targets']
        }

    def _evaluate_state(self, state):
        """
        评估状态价值 (Heuristic Value)
        
        状态越好，分数越高。
        考虑因素：
        1. 己方剩余球数量 (越少越好)
        2. 白球位置 (是否安全，是否好打下一个)
        3. 是否犯规 (模拟中能检测到的)
        """
        balls = state['balls']
        my_targets = state['my_targets']
        
        remaining = [bid for bid in my_targets if balls[bid].state.s != 4]
        
        # 基础分：剩余球越少分越高
        score = (8 - len(remaining)) * 100
        
        # 如果黑8进了且合法
        if not remaining and balls['8'].state.s == 4:
            score += 1000
            
        # 如果白球进了 (犯规)
        if balls['cue'].state.s == 4:
            score -= 500
            
        return score
