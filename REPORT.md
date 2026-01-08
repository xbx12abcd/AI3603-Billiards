# AI Billiards Agent Project Report

## 1. Methodology and Approach

### 1.1 Evolution of the Solution
Our approach to building a high-performance billiards agent evolved through several stages, each addressing limitations of the previous one:

1.  **Phase 1: Geometric Heuristics (Greedy Strategy)**
    *   **Approach**: We initially implemented a `NewAgent` that selected shots purely based on geometric properties: distance to pocket and cut angle.
    *   **Outcome**: The agent could pot easy balls but lacked "position play" (shape). It often left the cue ball in difficult positions (e.g., stuck against a cushion or snookered) after a successful pot, leading to a loss of turn.

2.  **Phase 2: Monte Carlo Tree Search (Exploration)**
    *   **Approach**: We implemented an `MCTSAgent` (in `mcts_agent.py`) using the Upper Confidence Bound for Trees (UCT) algorithm. This treated the game as a tree search problem.
    *   **Outcome**: While theoretically robust, MCTS struggled with the high computational cost of the `pooltool` physics simulation. With a limited time budget per move, the search tree remained too shallow to effectively plan long sequences.

3.  **Phase 3: Heuristic Search with 2-Step Lookahead (Final Strategy)**
    *   **Approach**: We combined the speed of heuristics with the foresight of search. We enhanced `NewAgent` to simulate top candidate shots and, crucially, evaluate the *resulting* state for the *next* shot possibilities.
    *   **Outcome**: This "Lookahead" mechanism proved to be the game-changer. By explicitly optimizing for the next shot's quality (Position Play), the agent began to string together multiple pots (runs), significantly increasing its win rate.

### 1.2 Lessons Learned
*   **Simulation Cost vs. Depth**: In continuous physics environments like billiards, deep search (like MCTS) is expensive. A broader, shallower search that evaluates "state quality" (heuristics) is often more effective than a deep but narrow search.
*   **The Importance of "Shape"**: In 8-ball, potting the current ball is only half the task. The win rate is determined by where the cue ball stops. Adding the `w_lookahead` weight was the single most impactful change.
*   **Defense is Mandatory**: A purely offensive agent will foul when snookered. Implementing a fallback "Safety" strategy (kicking or gentle touches) when no offensive shot is viable prevents giving the opponent "ball-in-hand".

---

## 2. Implementation Details

### 2.1 The `NewAgent` Architecture
The final agent (`agent.py`) operates in a three-stage pipeline:

#### Step 1: Candidate Generation
The agent generates a list of potential actions:
*   **Offensive Shots**: For every legal target ball, it calculates the necessary cut angle (`phi`) and velocity (`V0`) to pot it into every pocket. It filters out physically impossible shots (cut angle > 80°) or blocked paths using a ghost-ball collision check.
*   **Safety Shots**: It generates defensive candidates (kick shots or simple safety touches) to be used if no offensive shot is good enough.

#### Step 2: Heuristic Filtering & Optimization
Candidates are scored and sorted based on geometric difficulty. We used **Bayesian Optimization** (via `train/train.py`) to tune the weights for these heuristics.
*   **Key Weights**:
    *   `w_cut_angle`: Penalizes thin cuts (harder to execute).
    *   `w_distance`: Penalizes long-distance shots.

#### Step 3: Simulation and Lookahead Evaluation
The top candidates (e.g., top 15) are simulated using `pooltool`. The resulting state is evaluated:
1.  **Immediate Reward**: Did the target ball go in? Did we foul?
2.  **Lookahead Score (`_evaluate_position_quality`)**:
    *   If the shot was successful, the agent examines the new cue ball position.
    *   It calculates the "Opportunity Score" for all remaining target balls from this new position.
    *   **Logic**: `LookaheadScore = max(100 - NextCutAngle * w - NextDistance * w)`
    *   This rewards shots that leave the cue ball with a straight, close shot on the next ball.
3.  **Final Decision**: The shot with the highest combined score (`Immediate + w_lookahead * LookaheadScore`) is chosen.

### 2.2 Final Hyperparameters
The optimized parameters used in `NewAgent`:
*   **`w_lookahead = 2.0`**: High weight on position play. This drives the agent to plan ahead.
*   **`w_cut_angle = 1.5`**: Moderate penalty for thin cuts.
*   **`w_safety_penalty = 30.0`**: The threshold for switching to defense. If the best offensive shot scores below -30 (risky), the agent plays safe.

---

## 3. Experimental Results

### 3.1 Experimental Setup
*   **Test Script**: `evaluate.py`
*   **Opponent**: `BasicAgent` (Baseline provided in the course).
*   **Metric**: Win rate over a series of games.

### 3.2 Performance
After tuning the lookahead weights, we conducted validation runs.

*   **Win Rate**: **90%** (9 wins out of 10 games in validation run).
*   **Score Breakdown**:
    *   **Offensive Consistency**: The agent frequently clears 3-4 balls in a row due to the lookahead logic.
    *   **Defensive Stability**: In the 1 loss, the agent successfully played safety shots but was eventually outplayed in a complex endgame.
*   **Comparison**: The `NewAgent` significantly outperforms the `BasicAgent`, which lacks the lookahead capability and often misses difficult cut shots or scratches.

### 3.3 Conclusion
The implementation successfully meets the project goal (Win Rate ≥ 0.7). The combination of geometric heuristics for candidate selection and a 2-step lookahead for evaluation provides a robust and efficient solution for the 8-ball billiards task.
