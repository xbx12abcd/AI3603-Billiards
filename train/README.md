# 训练指南

本目录包含了用于优化 Agent 参数的训练脚本。

## 目录结构

*   `train.py`: 用于优化 `NewAgent` (基于启发式) 的权重参数。
*   `train_mcts.py`: 用于优化 `MCTSAgent` (基于蒙特卡洛树搜索) 的超参数。

## 环境配置

确保你已经安装了以下依赖：
*   `pooltool`
*   `numpy`
*   `bayesian-optimization`
*   `scikit-learn`

如果尚未安装，请在项目根目录下运行：
```bash
# Ubuntu / Linux / macOS
conda activate poolenv
pip install bayesian-optimization scikit-learn numpy

# Windows
.\.venv\Scripts\activate
pip install bayesian-optimization scikit-learn numpy
```

---

## 1. 优化 NewAgent (train.py)

我们使用 **贝叶斯优化** 来自动搜索 `NewAgent` 的最佳启发式权重。优化目标是 Agent 在 N 局对战中对阵 `BasicAgent` 的平均胜率。

### 待优化参数

| 参数名              | 含义                                                  | 搜索范围      | 默认值 |
| ------------------- | ----------------------------------------------------- | ------------- | ------ |
| `w_cut_angle`       | 切球角度的惩罚权重                                    | [0.5, 5.0]    | 1.5    |
| `w_distance`        | 白球到目标距离的惩罚权重                              | [5.0, 20.0]   | 10.0   |
| `w_safety_penalty`  | 防守/解球方案的基础负分（越低越倾向于只在绝境时防守） | [10.0, 100.0] | 30.0   |
| `w_cushion_penalty` | 白球停在贴库位置的惩罚分数                            | [0.0, 20.0]   | 5.0    |
| `w_position`        | 走位权重：惩罚下一杆目标球距离                        | [0.0, 10.0]   | 5.0    |
| `w_safety_quality`  | 防守质量权重：奖励防守后白球远离对手球                | [0.0, 5.0]    | 2.0    |
| `w_lookahead`       | 前瞻权重：奖励能创造下一杆好机会的击球                | [0.0, 5.0]    | 2.0    |

### 运行训练

在项目根目录下运行：
```bash
python train/train.py
```

### 应用结果

脚本运行结束后，会输出最佳参数组合。请将这些值更新到 `agent.py` 中 `NewAgent` 类的 `__init__` 方法里：

```python
# agent.py
class NewAgent(Agent):
    def __init__(self, weights=None):
        # ...
        self.weights = {
            'w_cut_angle': 1.12,       # 替换为训练结果
            'w_distance': 12.45,       # 替换为训练结果
            'w_safety_penalty': 45.67, # 替换为训练结果
            'w_cushion_penalty': 5.23, # 替换为训练结果
            'w_position': 3.5,         # 替换为训练结果
            'w_safety_quality': 1.8,   # 替换为训练结果
            'w_lookahead': 2.5         # 替换为训练结果
        }
        # ...
```

---

## 2. 优化 MCTSAgent (train_mcts.py)

如果你启用了 `MCTSAgent`，可以使用此脚本优化其核心参数（如 UCB 探索常数）。

### 待优化参数

| 参数名   | 含义                                        | 搜索范围   |
| -------- | ------------------------------------------- | ---------- |
| `c_puct` | UCB 公式中的探索常数 (控制探索与利用的平衡) | [0.5, 5.0] |

### 运行训练

```bash
python train/train_mcts.py
```

### 应用结果

将得到的最佳 `c_puct` 值更新到 `mcts_agent.py` 或在实例化 `MCTSAgent` 时传入：

```python
# mcts_agent.py 或调用处
agent = MCTSAgent(c_puct=2.5)  # 替换为训练结果
```
