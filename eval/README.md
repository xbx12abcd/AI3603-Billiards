# AI3603-Billiards 测试复现指南

本目录 (`eval/`) 包含了复现 `NewAgent` 测试结果所需的所有代码和文件。

## 1. 目录结构

*   `agent.py`: 核心代码，包含 `NewAgent` 的实现逻辑（修改自发布版本）。
*   `evaluate.py`: 测试脚本，用于运行 `NewAgent` 与 `BasicAgent` 的对战。
*   `poolenv.py`: 台球环境模拟代码（未修改）。
*   `utils.py`: 工具函数（未修改）。

## 2. 环境配置

测试环境依赖与项目主环境一致。请确保已安装以下库：

*   `pooltool` (基于源码安装，需包含 billiards 扩展)
*   `numpy`
*   `scikit-learn` (BasicAgent 依赖)
*   `bayesian-optimization` (BasicAgent 依赖)

**快速配置命令 (Windows/Linux/macOS):**

```bash
# 如果使用 Conda
conda activate poolenv
pip install numpy scikit-learn bayesian-optimization

# 如果使用 venv
# Windows:
# .\.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate
pip install numpy scikit-learn bayesian-optimization
```

## 3. 复现测试结果

我们提供了一个一键运行的测试脚本 `evaluate.py`，默认设置为进行 100 局对战。

**运行命令：**

在 `eval` 目录下运行：

```bash
python evaluate.py
```

**预期结果：**

脚本运行结束后，将输出胜率统计。`NewAgent` (Agent B) 的预期胜率应 **>= 70%** (通常在 80% - 90% 之间)。

输出示例：
```text
最终结果： {'AGENT_A_WIN': 10, 'AGENT_B_WIN': 90, 'SAME': 0, 'AGENT_A_SCORE': 10.0, 'AGENT_B_SCORE': 90.0}
```

## 4. 算法与超参数说明

### 核心算法
`NewAgent` 基于 **两步前瞻 (2-Step Lookahead)** 和 **启发式评分** 机制：
1.  **进攻筛选**：优先筛选出进球概率高（切角小、路径无阻挡）的候选球。
2.  **走位规划 (Lookahead)**：对于每个候选击球，模拟进球后的白球位置，并计算该位置对下一杆进攻的有利程度（"好形"得分）。
3.  **动态防守**：当没有好的进攻机会（得分低于阈值）时，自动切换到防守模式，选择能将白球停在对手难打区域的方案。

### 关键超参数 (在 `agent.py` 中设置)

这些参数经过贝叶斯优化微调，硬编码在 `NewAgent.__init__` 中：

```python
self.weights = {
    'w_cut_angle': 1.5,       # 切角惩罚：越大越倾向于打直球
    'w_distance': 10.0,       # 距离惩罚：控制长台进攻风险
    'w_safety_penalty': 30.0, # 防守门槛：降低此值可增加防守频率 (当前值鼓励在劣势时果断防守)
    'w_cushion_penalty': 5.0, # 贴库惩罚
    'w_position': 5.0,        # 基础走位权重
    'w_safety_quality': 2.0,  # 防守质量权重
    'w_lookahead': 2.0        # 前瞻权重：大幅提升此值以强化连续进攻能力
}
```

### Checkpoint / 权重文件
本算法为**非学习型启发式算法**，所有策略逻辑和参数均直接包含在 `agent.py` 代码中，因此**无需**加载额外的 checkpoint 模型文件。
