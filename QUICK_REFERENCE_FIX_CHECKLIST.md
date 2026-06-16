# ⚡ Quick Reference: Fix Checklist (Copy-Paste Ready)

**Purpose:** Fast implementation guide with code snippets  
**Audience:** Developer implementing fixes  
**Time to read:** 5 minutes

---

## 📋 CHECKLIST: Fix #1 (Enable Selector - 1 LINE)

### Status: 🟢 IMMEDIATE (no retraining needed)

**File:** `Model/flow_matching_model.py`  
**Line:** 6210

**Change:**
```python
# BEFORE:
def sample(self, batch_list, use_selector=False, ...):

# AFTER:
def sample(self, batch_list, use_selector=True, ...):
```

**Expected gain:** +10-15 km  
**Risk:** ZERO  

**Verification:**
```bash
python -c "
from Model.flow_matching_model import TCFlowMatching
model = TCFlowMatching(...)
import inspect
sig = inspect.signature(model.sample)
selector_default = sig.parameters['use_selector'].default
print(f'use_selector default: {selector_default}')
assert selector_default == True, 'Fix not applied!'
print('✅ Fix #1 verified')
"
```

---

## 📋 CHECKLIST: Fix #2 (Environment Loss - 5 HOURS)

### Status: 🟡 THIS WEEK

**Files to modify:**
1. `Model/flow_matching_model.py` — Add env loss computation
2. `scripts/train_flowmatching.py` — Update GradNorm terms

### Step 1: Add to `flow_matching_model.py` __init__ (line ~500)

```python
# Add this in __init__ method:
self.env_weight_fc = nn.Sequential(
    nn.Linear(256 + 256, 128),  # steering_feat + env_kine_feat
    nn.ReLU(),
    nn.Linear(128, 1)
)
self.register_parameter('env_loss_weight', 
                       nn.Parameter(torch.tensor(0.1)))
```

### Step 2: Add helper methods to `flow_matching_model.py` (after forward_with_ctx)

```python
def _compute_steering_weight(self, env_kine_feat, steering_feat):
    if steering_feat is None or env_kine_feat is None:
        return torch.ones((env_kine_feat.shape[0], 1), 
                         device=env_kine_feat.device)
    
    combined = torch.cat([steering_feat, env_kine_feat], dim=-1)
    weight = torch.sigmoid(self.env_weight_fc(combined))
    return weight

def _extract_env_velocity_component(self, obs_traj, env_data, target_vel):
    batch_size = obs_traj.shape[0]
    
    steering_dir = env_data.get('steering_direction', None)
    if steering_dir is None:
        steering_dir = target_vel / (torch.norm(target_vel, dim=-1, keepdim=True) + 1e-6)
    
    env_vel_magnitude = torch.sum(target_vel * steering_dir, dim=-1, keepdim=True)
    env_vel_component = env_vel_magnitude * steering_dir
    
    return env_vel_component
```

### Step 3: Update loss in `get_loss_breakdown()` (line ~5997)

```python
# BEFORE:
l_fm = F.mse_loss(pred_vel, u_target)

# AFTER:
l_fm = F.mse_loss(pred_vel, u_target)

if steering_feat is not None and env_kine_feat is not None:
    env_vel_gt = self._extract_env_velocity_component(
        obs_traj, env_data, u_target)
    steer_weight = self._compute_steering_weight(
        env_kine_feat, steering_feat)
    l_steering = F.mse_loss(
        pred_vel * steer_weight,
        env_vel_gt * steer_weight
    )
    lambda_steering = torch.sigmoid(self.env_loss_weight)
else:
    l_steering = torch.tensor(0.0, device=l_fm.device)
    lambda_steering = 0.0

l_base = l_fm + lambda_steering * l_steering
```

### Step 4: Update return dict in `get_loss_breakdown()`

```python
# Add to returned loss_dict:
"l_steering": l_steering.item(),
```

### Step 5: Update `scripts/train_flowmatching.py` (line ~164)

```python
# BEFORE:
GRADNORM_TERMS = ["l_dpe", "l_vel_reg", "l_heading", "l_speed", "l_accel"]

# AFTER:
GRADNORM_TERMS = ["l_dpe", "l_vel_reg", "l_heading", "l_speed", "l_accel", "l_steering"]
```

### Step 6: Update DEFAULT_LAMBDA (line ~170)

```python
# Add to dict:
"l_steering": 0.10,
```

### Step 7: Update AUX_TERMS (line ~169)

```python
# BEFORE:
AUX_TERMS = ["l_vel_reg", "l_heading", "l_speed", "l_accel"]

# AFTER:
AUX_TERMS = ["l_vel_reg", "l_heading", "l_speed", "l_accel", "l_steering"]
```

**Expected gain:** +20-30 km  
**Timeline:** 5h coding + 24h retraining from ep0  
**Risk:** LOW (can reduce weight if breaks easy_ADE)

**Test:**
```bash
python scripts/train_flowmatching.py \
  --num_epochs 10 \
  --batch_size 4 \
  --device cuda:0
# Should see l_steering decrease over 10 epochs
```

---

## 📋 CHECKLIST: Fix #3 (Trajectory Loss - 8 HOURS)

### Status: 🟡 WEEK 2

**Files:**
1. `Model/flow_matching_model.py` — Add trajectory loss computation
2. `scripts/train_flowmatching.py` — Integrate into training loop

### Step 1: Add to `flow_matching_model.py`

```python
def compute_trajectory_loss(self, generated_candidates, gt_trajectory, 
                            lambda_dict=None):
    """Trajectory loss: full 72h optimization"""
    batch_size, num_candidates, num_steps, _ = generated_candidates.shape
    device = generated_candidates.device
    
    # Compute ADE for each candidate
    distances = torch.norm(
        generated_candidates - gt_trajectory.unsqueeze(1),
        dim=-1
    )
    ades = torch.mean(distances, dim=-1)
    
    # Find best candidate
    best_idx = torch.argmin(ades, dim=1)
    batch_idx = torch.arange(batch_size, device=device)
    best_trajectory = generated_candidates[batch_idx, best_idx]
    
    # Loss 1: Pull best to ground truth
    loss_pull = F.mse_loss(best_trajectory, gt_trajectory)
    
    # Loss 2: Contrastive on top-10
    K = min(10, num_candidates)
    sorted_idx = torch.argsort(ades, dim=1)
    
    loss_push = 0.0
    for i in range(K):
        for j in range(i + 1, K):
            idx_i = sorted_idx[:, i]
            idx_j = sorted_idx[:, j]
            
            cand_i = generated_candidates[batch_idx, idx_i]
            cand_j = generated_candidates[batch_idx, idx_j]
            
            margin = 0.1
            loss_push += F.relu(F.mse_loss(cand_i, gt_trajectory) - 
                               F.mse_loss(cand_j, gt_trajectory) + margin)
    
    loss_push = loss_push / max((K * (K - 1) // 2), 1)
    
    # Loss 3: Diversity
    mean_traj = torch.mean(generated_candidates, dim=1, keepdim=True)
    diversity = torch.mean(torch.norm(
        generated_candidates - mean_traj, dim=-1))
    
    diversity_threshold = 50.0
    loss_diversity = F.relu(diversity_threshold - diversity)
    
    loss_traj = loss_pull + 0.5 * loss_push + 0.2 * loss_diversity
    
    return loss_traj, best_idx
```

### Step 2: Add to __init__ (line ~500)

```python
self.trajectory_loss_weight = nn.Parameter(torch.tensor(0.1))
```

### Step 3: Modify sample() to return candidates

```python
# In sample() method, add return_candidates parameter
def sample(self, batch_list, ..., return_candidates=False):
    # ... existing code ...
    
    # After generating DDIM candidates:
    if return_candidates:
        return final_predictions, all_candidates
    else:
        return final_predictions
```

### Step 4: Update training loop in `scripts/train_flowmatching.py`

```python
# In training step (around line 8500)
with torch.no_grad():
    preds, candidates = model.sample(
        batch_list, 
        return_candidates=True
    )

loss_traj, best_idx = model.compute_trajectory_loss(
    candidates, 
    batch_list['gt_trajectory']
)

l_fm, loss_dict = model.get_loss_breakdown(batch_list, epoch)

lambda_traj = torch.sigmoid(model.trajectory_loss_weight)
loss_total = l_fm + lambda_traj * loss_traj
```

### Step 5: Update GradNorm (scripts/train_flowmatching.py)

```python
GRADNORM_TERMS = [..., "l_trajectory"]
DEFAULT_LAMBDA["l_trajectory"] = 0.30
AUX_TERMS = [..., "l_trajectory"]
```

**Expected gain:** +20-40 km (biggest!)  
**Timeline:** 8h coding + 48h retraining  
**Risk:** MEDIUM (complex, may need weight tuning)

---

## 📋 CHECKLIST: Fix #4 (Selector t-Schedule - 3 HOURS)

### Status: 🟡 WEEK 2

**File:** `Model/flow_matching_model.py`

Add robust selector training:

```python
def train_selector_robust(self, batch_list, epoch):
    """Train selector with random t."""
    batch_size = len(batch_list)
    device = batch_list[0]['obs'].device
    
    t_selector = torch.rand(batch_size, device=device)
    
    candidates = self._generate_candidates_at_random_t(batch_list, t_selector)
    ades = self._compute_ades(candidates, batch_list['gt_trajectory'])
    scores = self.selector(batch_list['context'], candidates)
    
    # Normalize ADE to target score
    ade_min = torch.min(ades, dim=1)[0].unsqueeze(1)
    ade_max = torch.max(ades, dim=1)[0].unsqueeze(1)
    target_scores = 1.0 - (ades - ade_min) / (ade_max - ade_min + 1e-6)
    
    loss = F.mse_loss(scores, target_scores)
    
    return loss
```

Update training loop:
```python
if phase == 3:
    loss_selector = model.train_selector_robust(batch_list, epoch)
    loss_fm, loss_dict = model.get_loss_breakdown(batch_list, epoch)
    loss_total = 0.8 * loss_fm + 0.2 * loss_selector
```

**Expected gain:** +5-8 km

---

## 📋 CHECKLIST: Fix #5 (Hard Threshold - 2 HOURS)

### Status: 🟡 WEEK 2

**File:** `Model/flow_matching_model.py`

Add canonical hard score function:

```python
def get_hard_score(self, obs_traj):
    """Canonical hard classification used everywhere."""
    speeds = torch.norm(obs_traj[:, 1:] - obs_traj[:, :-1], dim=-1)
    mean_speed = torch.mean(speeds)
    speed_variance = torch.var(speeds)
    
    headings = torch.atan2(obs_traj[:, 1:, 1], obs_traj[:, 1:, 0])
    heading_changes = torch.abs(headings[1:] - headings[:-1])
    mean_heading_change = torch.mean(heading_changes)
    
    variance_score = torch.tanh(speed_variance / 2.0)
    heading_score = torch.tanh(mean_heading_change / 0.1)
    
    hard_score = 0.6 * variance_score + 0.4 * heading_score
    
    return hard_score
```

Use everywhere:
```python
HARD_THRESHOLD = 0.5

# Training
is_hard = model.get_hard_score(obs_traj) > HARD_THRESHOLD

# Inference
is_hard_inf = model.get_hard_score(obs_traj) > HARD_THRESHOLD
```

**Expected gain:** +3-5 km

---

## 🚀 EXECUTION PLAN

### Week 1
- [ ] **Day 1:** Fix #1 (1 line) — Retrain ep0 with FIX-LR-1/PHASE4-1
  - Expected: 325 → 310 km immediately, 280 km after training
  
- [ ] **Day 2-3:** Fix #2 (Environment loss) — 5h coding
  - Retrain ep0
  - Expected: 280 → 250 km
  
- [ ] **Day 4:** Fix #4 + Fix #5 — 5h coding
  - Quick retrain from previous checkpoint
  - Expected: 250 → 240 km

### Week 2
- [ ] **Day 8-12:** Fix #3 (Trajectory loss) — 8h coding
  - This is the big one
  - Retrain full ep0-100
  - Expected: 240 → 210-220 km

### Week 3
- [ ] Tuning + hard case focus
- [ ] Expected: 210-220 → 205-215 km

### Week 4
- [ ] Mode conditioning (optional bonus)
- [ ] Expected: 205 → 190-200 km

---

## 📊 FINAL METRICS TO TRACK

Each week, measure:

```python
# Monitor these metrics
print(f"val_ade: {val_ade:.1f} km")
print(f"easy_ade: {easy_ade:.1f} km")
print(f"hard_ade: {hard_ade:.1f} km")
print(f"diversity: {diversity:.1f} km")
print(f"selector_confidence: {selector_conf:.3f}")
```

**Success criteria:**
- [ ] Week 1: ADE 280-290 km
- [ ] Week 2: ADE 240-250 km  
- [ ] Week 3: ADE 220-230 km
- [ ] Week 4: ADE 195-210 km ✅ Beats ST-Trans!

---

## 🆘 QUICK TROUBLESHOOTING

| Problem | Check |
|---------|-------|
| Training diverges (NaN) | Loss weights too large, reduce λ_env to 0.05 |
| easy_ADE spikes | Steering loss too aggressive, reduce λ |
| hard_ADE doesn't improve | Insufficient hard samples in batch, check split |
| LR wrong | Verify FIX-LR-1 imported correctly, T_max_orig saved |
| Selector doesn't help | Check use_selector=True, verify confidence >0.5 |
| Trajectory loss breaks | May be computing wrong dimension, check tensor shapes |

---

## ✅ DONE CHECKLIST

- [x] FIX-LR-1: Learning rate schedule (merged commit a247241)
- [x] FIX-PHASE4-1: Phase 4 freeze flag (merged commit a247241)
- [ ] Fix #1: Enable selector (1 line, today)
- [ ] Fix #2: Environment loss (5h, this week)
- [ ] Fix #3: Trajectory loss (8h, week 2)
- [ ] Fix #4: Selector t-schedule (3h, week 2)
- [ ] Fix #5: Hard threshold (2h, week 2)
- [ ] Retrain from ep0 with all fixes
- [ ] Achieve 195-210 km ADE
- [ ] Beat ST-Trans! 🎉

