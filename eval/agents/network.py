import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BilliardValueNet(nn.Module):
    """
    台球局面价值网络
    输入: 局面特征 (球的位置、归属等)
    输出: 当前局面的胜率估计 (Value)
    """
    def __init__(self, input_dim=68, hidden_dim=256):
        super(BilliardValueNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x)) # 输出范围 [-1, 1], 1代表必胜, -1代表必败
        return x

def state_to_tensor(balls, my_targets, table_w, table_l):
    """
    将球的字典状态转换为神经网络输入张量
    
    Feature Engineering:
    - Cue Ball: [x, y]
    - 8 Ball: [x, y, is_pocketed]
    - My Balls (7 slots): [x, y, is_pocketed] * 7 (sorted or fixed order)
    - Enemy Balls (7 slots): [x, y, is_pocketed] * 7
    
    Normalization:
    - x / table_w
    - y / table_l
    """
    features = []
    
    # 1. Cue Ball
    if 'cue' in balls:
        cue = balls['cue']
        features.extend([cue.state.rvw[0][0] / table_w, cue.state.rvw[0][1] / table_l])
    else:
        features.extend([0.0, 0.0]) # Should not happen in active play usually
        
    # 2. 8 Ball
    if '8' in balls:
        b8 = balls['8']
        is_pocketed = 1.0 if b8.state.s == 4 else 0.0
        features.extend([b8.state.rvw[0][0] / table_w, b8.state.rvw[0][1] / table_l, is_pocketed])
    else:
        features.extend([0.0, 0.0, 1.0])

    # Helper to process group
    def process_group(target_ids):
        group_feats = []
        # Sort ids to maintain consistency? Or just 1-7, 9-15?
        # Better to sort by ID to keep position in vector consistent for specific balls,
        # OR sort by position? 
        # Given balls are identical physics-wise, sorting by status (on-table vs pocketed) 
        # or spatial position might be better for CNN, but for MLP fixed ID order is okay.
        # Let's use sorted IDs.
        
        # We need to know which specific balls are in "my_targets".
        # balls dict has all balls.
        
        # Identify all potential balls in this group (Solid: 1-7, Stripe: 9-15)
        # We need to know if the current player is Solid or Stripe.
        # my_targets is a list of strings.
        
        # Assuming standard 8-ball: 1-7 and 9-15.
        # We need to fill 7 slots.
        
        # Let's find all balls that belong to "my side".
        # Note: my_targets only contains *remaining* targets. 
        # We need to reconstruct the full set to keep vector size constant.
        
        # Heuristic: 
        # If my_targets contains any of '1'..'7', I am Solid.
        # If my_targets contains any of '9'..'15', I am Stripe.
        # If my_targets is only ['8'], we need to infer from context or pass "my_suit".
        # For simplicity, we'll iterate 1-7 and 9-15.
        
        pass
    
    # Identify suits
    # This is a bit tricky if we only have `my_targets` which shrinks.
    # We will simply dump ALL balls 1-7 and 9-15 into the feature vector, 
    # but add a flag "is_mine" for each.
    
    # Revised Features:
    # Loop 1 to 15:
    # [x, y, is_pocketed, is_mine, is_target]
    
    for i in range(1, 16):
        bid = str(i)
        if bid == '8': continue # Handled separately
        
        if bid in balls:
            b = balls[bid]
            is_pocketed = 1.0 if b.state.s == 4 else 0.0
            x = b.state.rvw[0][0] / table_w
            y = b.state.rvw[0][1] / table_l
        else:
            # Should not happen if passed full balls dict, but if filtered:
            is_pocketed = 1.0
            x, y = 0.0, 0.0
            
        is_mine = 1.0 if bid in my_targets else 0.0 
        # Wait, if I already pocketed '1', it's not in my_targets anymore.
        # But it was mine. 
        # Ideally we want: "Is this ball belonging to my suit?"
        # We'll use a simplified check: 
        # If my_targets has solids, then 1-7 are mine.
        
        is_target = 1.0 if bid in my_targets else 0.0
        
        features.extend([x, y, is_pocketed, is_target])
        
    # Total dims:
    # Cue: 2
    # 8-ball: 3
    # 1-7, 9-15 (14 balls): 14 * 4 = 56
    # Total: 2 + 3 + 56 = 61
    
    return torch.FloatTensor(features)

