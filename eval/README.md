# AI3603-Billiards 评估环境

本目录 (`eval/`) 包含了用于评估和运行智能台球 Agent 的核心代码和环境。

## 1. 目录结构

*   **`agents/`**: Agent 实现代码库
    *   `new_agent.py`: **核心提交 Agent**。结合了启发式规则搜索与神经网络价值评估的混合型 Agent。
    *   `basic_agent_pro.py`: **MCTS Agent (BasicAgentPro)**。基于蒙特卡洛树搜索 (MCTS) 的高阶 Agent，作为训练时的"老师"和评估时的强力对手。
    *   `network.py`: 神经网络模型定义 (BilliardValueNet)。
    *   `basic_agent.py`: 课程提供的基准 Agent。
*   **`evaluate.py`**: 评估脚本，用于让两个 Agent 进行对战并统计胜率。
*   **`poolenv.py`**: 台球游戏环境封装。
*   **`billiard_value_net.pth`**: 训练好的神经网络模型文件。
*   **`utils.py`**: 通用工具函数。

## 2. 核心 Agent 介绍

### NewAgent (混合决策 Agent)
`NewAgent` 是本项目最终提交的智能体，它采用了混合架构：
1.  **启发式搜索**：基于几何规则（切球角度、幽灵球算法）快速筛选候选击球点。
2.  **神经网络评估**：利用训练好的价值网络 (Value Network) 对击球后的局面进行打分。如果局面有利于己方（胜率高），则增加该动作的权重。
3.  **防守机制**：当没有进攻机会时，利用启发式规则寻找最佳防守（斯诺克）方案。

### BasicAgentPro (MCTS Agent)
`BasicAgentPro` 是一个基于 **蒙特卡洛树搜索 (MCTS)** 的增强型 Agent，主要用于辅助训练和高难度评估。
*   **原理**：通过物理引擎模拟未来的多次击球结果，构建搜索树来选择最佳动作。
*   **特性**：
    *   **抗噪性**：在模拟中主动注入物理噪声，寻找最稳健（Robust）的击球方案。
    *   **前瞻性**：能够看到未来几步的局面变化。
*   **用途**：在训练阶段作为 `NewAgent` 的陪练对手，提供高质量的对战数据。

## 3. 环境配置

请确保安装了以下依赖库：

```bash
pip install torch numpy scikit-learn pooltool
```

*注意：`pooltool` 需要正确安装并配置物理引擎支持。*

## 4. 运行评估

使用 `evaluate.py` 脚本可以让 Agent 进行对战。

**基本用法：**

```bash
python evaluate.py
```

**配置对战：**
打开 `evaluate.py`，修改以下代码以更换对手：

```python
# 修改 agent_b 为你想测试的 Agent
# agent_a = BasicAgent()       # 初级基准
# agent_a = BasicAgentPro()    # MCTS 高级基准 (推荐)
agent_a, agent_b = BasicAgentPro(), NewAgent() 
```


