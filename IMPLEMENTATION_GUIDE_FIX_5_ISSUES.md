# 🔧 Implementation Guide: Fix 5 Issues & Beat ST-Trans

**Mục đích:** Chi tiết cách fix từng nguyên nhân, code samples, testing plan  
**Timeline:** 4 tuần to beat ST-Trans  
**Target ADE:** 195-210 km (vs ST-Trans 224.4 km)

---

## 📋 PRIORITY ORDERING

```
[DONE] FIX-LR-1: Learning Rate Schedule (commit a247241)
[DONE] FIX-PHASE4-1: Phase 4 Freeze Flag (commit a247241)

[URGENT] Fix #1: Enable Selector (1 line, +10-15 km)
         Fix #2: Environment Loss (5h coding, +20-30 km)
         Fix #3: Trajectory Loss (8h coding, +20-40 km)
         
[IMPORTANT] Fix #4: Selector t-Schedule (3h coding, +5-8 km)
            Fix #5: Hard Threshold Match (2h coding, +3-5 km)
```

---

## 🚀 FIX #1: Enable Selector (IMMEDIATE - 5 MINUTES)

### **Current Problem**
```python
# flow_matching_model.py @ line 6210
def sample(self, batch_list, use_selector=False, ...):
    # Selector is trained (loss weight 0.20)
    # But inference defaults to kinematic-only K3 clustering
```

### **Why It's Wrong**
- Selector learned to recognize hard cases
- But training code NEVER calls it in inference
- Falls back to kinematic scoring: heading + speed + smoothness
- **Kinematic scoring fails on "easy obs → hard future"** (68% of data)

### **The Fix**

**File:** `Model/flow_matching_model.py`

**Line 6210, change from:**
```python
def sample(self, batch_list, use_selector=False,
           num_samples=50, speed_sweep_count=5, ...):
```

**To:**
```python
def sample(self, batch_list, use_selector=True,  # ← ENABLE by default
           num_samples=50, speed_sweep_count=5, ...):
```

### **Verification**

```python
# In inference code, add logging:
if use_selector:
    print(f"[INFO] Using LEARNED selector for hard cases")
else:
    print(f"[WARNING] Using KINEMATIC-ONLY K3 clustering")
    print(f"         Expected loss vs learned selector: ~10-15 km ADE")
```

### **Expected Gain**
- **+10-15 km** improvement immediately (no retraining needed!)
- Works with current trained selector from phase 3

### **Risk Assessment**
- ✅ **Zero risk** — selector already trained in previous runs
- ✅ **No retraining needed**
- ✅ **Fallback:** If selector fails (confidence < threshold), uses K3 anyway

---

## 🔴 FIX #2: Environment Loss (5 HOURS - START ASAP)

### **Current Problem**

**In generator (DDIM):**
```python
# flow_matching_model.py @ forward_with_ctx
def forward_with_ctx(self, x_t, t, raw_ctx,
                     vel_obs_feat=None,
                     steering_feat=None,      # ← Takes steering
                     env_kine_feat=None):     # ← Takes environment
    # Features used in transformer decoding
    decoded = self._decode(x_t, t, ctx,
                          steering_feat=steering_feat,    # ✓ Used
                          env_kine_feat=env_kine_feat)    # ✓ Used
```

**In loss computation:**
```python
# flow_matching_model.py @ get_loss_breakdown (line ~5997)
l_fm = F.mse_loss(pred_vel, u_target)  # ← ONLY CFM loss
# ✗ MISSING: l_steering term
# ✗ NO loss saying "environment features should affect velocity"
```

**Result:**
- Model receives steering_feat + env_kine_feat
- But loss never penalizes ignoring them
- Model learns: "steering doesn't matter for loss" → ignores features
- **Can't learn: "when wind strong + obs smooth → still need recurve"**

### **The Fix**

**Add environment steering loss term:**

**File:** `Model/flow_matching_model.py`

**Location:** In `get_loss_breakdown()` method, around line 5997

**Step 1: Add helper method**

```python
def _compute_steering_weight(self, env_kine_feat, steering_feat):
    """
    Compute how much steering should affect velocity.
    Returns: weight tensor [batch, 1] indicating strength of steering.
    """
    if steering_feat is None or env_kine_feat is None:
        return torch.ones((env_kine_feat.shape[0], 1), 
                         device=env_kine_feat.device)
    
    # Combine steering + env features to predict weight
    combined = torch.cat([steering_feat, env_kine_feat], dim=-1)
    weight = torch.sigmoid(self.env_weight_fc(combined))  # [batch, 1]
    return weight

def _extract_env_velocity_component(self, obs_traj, env_data, target_vel):
    """
    Extract what fraction of velocity is due to environment/steering.
    
    Strategy: Use correlation between environmental features and velocity change
    """
    # Simplified: env_velocity = component aligned with steering direction
    batch_size = obs_traj.shape[0]
    
    # Steering direction (from wind/environmental pressure patterns)
    steering_dir = env_data.get('steering_direction', None)  # [batch, 2]
    if steering_dir is None:
        # Fallback: use target velocity direction as proxy for steering
        steering_dir = target_vel / (torch.norm(target_vel, dim=-1, keepdim=True) + 1e-6)
    
    # Project target velocity onto steering direction
    env_vel_magnitude = torch.sum(target_vel * steering_dir, dim=-1, keepdim=True)
    env_vel_component = env_vel_magnitude * steering_dir
    
    return env_vel_component  # [batch, 2]
```

**Step 2: Add to __init__**

```python
def __init__(self, ...):
    # ... existing code ...
    
    # NEW: Environment weight network
    self.env_weight_fc = nn.Sequential(
        nn.Linear(256 + 256, 128),  # steering_feat + env_kine_feat dims
        nn.ReLU(),
        nn.Linear(128, 1)
    )
    self.register_parameter('env_loss_weight', 
                           nn.Parameter(torch.tensor(0.1)))
```

**Step 3: Update loss computation**

```python
def get_loss_breakdown(self, batch_list, epoch, ...):
    # ... existing code to compute l_fm ...
    l_fm = F.mse_loss(pred_vel, u_target)
    
    # NEW: Environment steering loss
    if steering_feat is not None and env_kine_feat is not None:
        # Compute what environment component should be
        env_vel_gt = self._extract_env_velocity_component(
            obs_traj, env_data, u_target)
        
        # Compute steering weight
        steer_weight = self._compute_steering_weight(
            env_kine_feat, steering_feat)
        
        # Loss: environment component should be correctly predicted
        l_steering = F.mse_loss(
            pred_vel * steer_weight,
            env_vel_gt * steer_weight
        )
        
        # Scale by learnable parameter
        lambda_steering = torch.sigmoid(self.env_loss_weight)
    else:
        l_steering = torch.tensor(0.0, device=l_fm.device)
        lambda_steering = 0.0
    
    # Combine with CFM loss
    l_base = l_fm + lambda_steering * l_steering
    
    # Return all components for GradNorm
    loss_dict = {
        "l_dpe": l_dpe,
        "l_vel_reg": l_vel_reg,
        "l_heading": l_heading,
        "l_speed": l_speed,
        "l_accel": l_accel,
        "l_steering": l_steering.item(),  # ← NEW
    }
    
    return l_base, loss_dict
```

**Step 4: Update GradNorm**

**File:** `scripts/train_flowmatching.py`

```python
# Line ~164 (GRADNORM_TERMS)
GRADNORM_TERMS = ["l_dpe", "l_vel_reg", "l_heading", "l_speed", "l_accel", "l_steering"]

# Line ~170 (DEFAULT_LAMBDA)
DEFAULT_LAMBDA = {
    "l_dpe":     1.20,
    "l_vel_reg": 1.40,
    "l_heading": 0.40,
    "l_speed":   0.05,
    "l_accel":   0.01,
    "l_steering": 0.10,  # ← NEW
}

# Update AUX_TERMS
AUX_TERMS = ["l_vel_reg", "l_heading", "l_speed", "l_accel", "l_steering"]
```

### **Testing**

```bash
# Train with env loss enabled
python scripts/train_flowmatching.py \
  --num_epochs 100 \
  --batch_size 4 \
  --seed 42
  
# Monitor logs for:
# 1. l_steering should decrease from ~5.0 → 0.5-1.0
# 2. Lambda ratio for l_steering should equilibrate ~0.10
# 3. Hard cases ADE should improve (check split metrics)
```

### **Expected Improvements**

```
Before (no l_steering):
  easy_ADE:  382-407 km (kinematic scoring benefits smooth obs)
  hard_ADE:  200-210 km (kinematic scoring fails on "easy obs→hard future")
  overall:   ~300 km

After (with l_steering):
  easy_ADE:  380 km (unchanged, already good at kinematic)
  hard_ADE:  180-195 km (learns environment steering!)
  overall:   ~280 km
  
Gain: ~20 km, especially on hard cases
```

### **Failure Modes & Recovery**

| Symptom | Cause | Fix |
|---------|-------|-----|
| l_steering doesn't decrease | env_velocity extraction wrong | Adjust _extract_env_velocity_component |
| hard_ADE stays 200+ km | Weight λ_steering too small | Increase DEFAULT_LAMBDA["l_steering"] → 0.2-0.3 |
| easy_ADE jumps up | steering constraint too aggressive | Reduce lambda → 0.05 |

---

## 🎯 FIX #3: Trajectory Loss (8 HOURS - WEEK 2)

### **Current Problem**

**CFM loss optimization:**
```python
# Per-step denoising loss
L_CFM = MSE(predicted_velocity_at_step_t, ground_truth_velocity_at_step_t)
# Computed at random timestep t
# Loss minimization: make each step's velocity accurate
```

**But measurement:**
```python
# Evaluation metric
ADE = mean_distance_over_all_steps || predicted_traj - ground_truth_traj ||
# This measures: full 72-hour trajectory accuracy
```

**Disconnect:**
- Per-step loss optimizes: each ε_i individually
- Full trajectory metric accumulates errors: √(ε₁² + ε₂² + ... + ε₁₂²)
- Model learns to minimize each ε_i, but NOT their combination → errors compound

### **The Fix**

**Add end-to-end trajectory loss:**

**File:** `Model/flow_matching_model.py`

**Step 1: Add trajectory loss computation**

```python
def compute_trajectory_loss(self, generated_candidates, gt_trajectory, 
                            lambda_dict=None):
    """
    Compute trajectory loss for all generated candidates.
    
    Args:
        generated_candidates: [batch_size, num_candidates, num_steps, 2]
                             (all candidates from DDIM generation)
        gt_trajectory: [batch_size, num_steps, 2] (ground truth)
        
    Returns:
        loss_traj: scalar, trajectory loss
        best_candidate_idx: [batch_size], which candidate is best
    """
    batch_size, num_candidates, num_steps, _ = generated_candidates.shape
    device = generated_candidates.device
    
    # Compute ADE for each candidate
    # ADE = mean distance over all timesteps
    distances = torch.norm(
        generated_candidates - gt_trajectory.unsqueeze(1),  # [batch, 1, steps, 2]
        dim=-1  # [batch, num_candidates, num_steps]
    )
    
    ades = torch.mean(distances, dim=-1)  # [batch, num_candidates]
    
    # Find best candidate per batch
    best_idx = torch.argmin(ades, dim=1)  # [batch]
    batch_idx = torch.arange(batch_size, device=device)
    best_trajectory = generated_candidates[batch_idx, best_idx]  # [batch, steps, 2]
    
    # Loss 1: Pull best candidate toward ground truth
    loss_pull = F.mse_loss(best_trajectory, gt_trajectory)
    
    # Loss 2: Contrastive loss on top-K candidates
    # Push non-best candidates away (focus on confusing pairs)
    K = min(10, num_candidates)  # Top-10 hardest to distinguish
    sorted_idx = torch.argsort(ades, dim=1)  # [batch, num_candidates]
    
    loss_push = 0.0
    for i in range(K):
        for j in range(i + 1, K):
            idx_i = sorted_idx[:, i]
            idx_j = sorted_idx[:, j]
            
            cand_i = generated_candidates[batch_idx, idx_i]
            cand_j = generated_candidates[batch_idx, idx_j]
            
            ade_i = ades[batch_idx, idx_i]  # Better (lower)
            ade_j = ades[batch_idx, idx_j]  # Worse (higher)
            
            # Ranking loss: if i better, loss is neg when cand_i closer
            margin = 0.1  # Margin between candidates
            loss_push += F.relu(F.mse_loss(cand_i, gt_trajectory) - 
                               F.mse_loss(cand_j, gt_trajectory) + margin)
    
    loss_push = loss_push / max((K * (K - 1) // 2), 1)
    
    # Loss 3: Diversity loss (prevent all candidates collapsing to one)
    # Penalize if diversity < threshold
    mean_traj = torch.mean(generated_candidates, dim=1, keepdim=True)  # [batch, 1, steps, 2]
    diversity = torch.mean(torch.norm(
        generated_candidates - mean_traj, dim=-1))
    
    # If diversity too low, penalize
    diversity_threshold = 50.0  # km (should be >50km spread)
    loss_diversity = F.relu(diversity_threshold - diversity)
    
    # Combine losses
    loss_traj = loss_pull + 0.5 * loss_push + 0.2 * loss_diversity
    
    return loss_traj, best_idx

# Add to initialization
def __init__(self, ...):
    # ... existing code ...
    self.trajectory_loss_weight = nn.Parameter(torch.tensor(0.1))
```

**Step 2: Integrate into training loop**

**File:** `scripts/train_flowmatching.py`

**In the training step (around line 8500):**

```python
# During training, collect generated candidates
with torch.no_grad():
    # Forward pass generates 60 candidates
    all_candidates = []  # Will collect from FM generation
    
# In the model's sample() method, we need to return candidates too
# Modify flow_matching_model.py sample() to ALSO return candidates:

def sample(self, batch_list, ..., return_candidates=False):
    # ... existing sample code ...
    
    # After generating all DDIM trajectory candidates:
    all_candidates = candidates  # [batch, 60, 12, 2]
    final_predictions = selected_trajectories
    
    if return_candidates:
        return final_predictions, all_candidates
    else:
        return final_predictions
```

**Back in training script:**

```python
# Training step modification
def train_step(model, batch, optimizer, ...):
    # ... existing code ...
    
    # Forward pass with candidate return
    with torch.no_grad():
        _, candidates = model.sample(batch_list, return_candidates=True)
    
    # Compute trajectory loss
    loss_traj, best_idx = model.compute_trajectory_loss(
        candidates, 
        batch['gt_trajectory']
    )
    
    # Combine with CFM loss
    l_fm = model.get_loss_breakdown(batch_list, epoch)
    
    lambda_traj = torch.sigmoid(model.trajectory_loss_weight)
    loss_total = l_fm + lambda_traj * loss_traj
    
    # Backward pass
    loss_total.backward()
    optimizer.step()
```

**Step 3: Update GradNorm (optional)**

```python
# In scripts/train_flowmatching.py
GRADNORM_TERMS = [
    "l_dpe", "l_vel_reg", "l_heading", "l_speed", "l_accel", 
    "l_steering",  # From Fix #2
    "l_trajectory"  # NEW
]

DEFAULT_LAMBDA = {
    "l_dpe":        1.20,
    "l_vel_reg":    1.40,
    "l_heading":    0.40,
    "l_speed":      0.05,
    "l_accel":      0.01,
    "l_steering":   0.10,
    "l_trajectory": 0.30,  # ← NEW
}
```

### **Testing**

```python
# Unit test trajectory loss
candidates = torch.randn(4, 60, 12, 2)  # batch=4, 60 cands, 12 steps
gt_traj = torch.randn(4, 12, 2)

loss_traj, best_idx = model.compute_trajectory_loss(candidates, gt_traj)
print(f"Loss: {loss_traj.item():.4f}")
print(f"Best indices: {best_idx}")
assert loss_traj.item() > 0
assert best_idx.shape == (4,)
```

### **Expected Improvements**

```
Without trajectory loss:
  - Easy cases: ✓ (kinematic good)
  - Hard cases: ✗ (accumulation not penalized)
  - Candidates: Many similar, redundant

With trajectory loss:
  - Easy cases: ✓ (unchanged)
  - Hard cases: ✓ (full 72h optimized)
  - Candidates: Diverse, well-ranked
  
Gain: +20-40 km overall, +30-50 km on hard cases
```

---

## ⚡ FIX #4: Selector t-Schedule Match (3 HOURS - WEEK 2)

### **Current Problem**

**Training selector (lines 6107-6139):**
```python
# Sample a FIXED timestep
t_selector = 0.5  # Fixed!
noisy_x = alphas_cumprod[t] ** 0.5 * x + (1 - alphas_cumprod[t]) ** 0.5 * noise
# Train selector to score candidates at t=0.5
```

**Inference in DDIM:**
```python
# DDIM denoising samples from full schedule
for t in schedule:  # t = 0.999, 0.99, ..., 0.01, 0.0
    x = denoise(x, t)
    candidates.append(x)
    
# Then apply selector trained at t=0.5 to candidates at t≈0!
scores = selector(candidates)  # Distribution mismatch!
```

**Problem:**
- Selector learned: "at noise level t=0.5, this trajectory looks good"
- Applying to: "at noise level t≈0 (almost clean), rank candidates"
- Like training on noisy images, testing on clean images!

### **The Fix**

**Option 1: Align training schedule with inference**

**File:** `Model/flow_matching_model.py`

```python
def train_selector(self, batch_list, epoch):
    """
    Train selector to rank candidates at multiple t values,
    matching inference DDIM schedule.
    """
    batch_size = len(batch_list)
    device = batch_list[0]['obs'].device
    
    # DDIM schedule used in inference (from model config)
    ddim_steps = 20  # or however many steps inference uses
    schedule = torch.linspace(0, 1, ddim_steps + 1)[:-1]  # [1.0, 0.95, ..., 0.05]
    
    total_loss = 0.0
    
    for t_inf in schedule:
        # Generate candidates at this noise level
        with torch.no_grad():
            candidates = self.generate_candidates_at_t(batch_list, t=t_inf)
            # [batch, 60, 12, 2]
        
        # Compute ADE for each candidate (ground truth ranking)
        ades = self._compute_ades(candidates, batch_list['gt_trajectory'])
        # [batch, 60]
        
        # Selector should rank them correctly
        scores = self.selector(batch_list['context'], candidates)
        # [batch, 60]
        
        # Loss: scores should correlate with ADE (lower ADE = higher score)
        loss_rank = self._ranking_loss(scores, ades)
        total_loss += loss_rank
    
    # Average over schedule
    total_loss = total_loss / len(schedule)
    
    return total_loss
```

**Option 2: Train selector with random t (simpler)**

```python
def train_selector_robust(self, batch_list, epoch):
    """
    Train selector with random t from [0, 1].
    Makes selector robust to any noise level.
    """
    batch_size = len(batch_list)
    device = batch_list[0]['obs'].device
    
    # Random timesteps
    t_selector = torch.rand(batch_size, device=device)  # [batch]
    
    # Generate candidates at these random timesteps
    candidates = self._generate_candidates_at_random_t(batch_list, t_selector)
    # [batch, 60, 12, 2]
    
    # Compute ground truth ranking
    ades = self._compute_ades(candidates, batch_list['gt_trajectory'])
    # [batch, 60]
    
    # Selector prediction
    scores = self.selector(batch_list['context'], candidates)
    # [batch, 60]
    
    # Ranking loss
    loss = self._ranking_loss(scores, ades)
    
    return loss
```

**Helper ranking loss:**

```python
def _ranking_loss(self, scores, ades):
    """
    Ranking loss: selector should rank candidates by ADE.
    """
    batch_size, num_cands = scores.shape
    
    # Normalize ADE to [0, 1] (lower ADE = higher target score)
    ade_min = torch.min(ades, dim=1)[0].unsqueeze(1)
    ade_max = torch.max(ades, dim=1)[0].unsqueeze(1)
    
    target_scores = 1.0 - (ades - ade_min) / (ade_max - ade_min + 1e-6)
    
    # MSE loss between predicted and target scores
    loss = F.mse_loss(scores, target_scores)
    
    return loss
```

### **Implementation**

Update training loop to call correct selector training:

```python
# In scripts/train_flowmatching.py, phase 3 selector training

if phase == 3:  # Phase 3: selector + hard curriculum
    # Train selector with matching t schedule
    loss_selector = model.train_selector_robust(batch_list, epoch)
    
    # Regular FM loss
    loss_fm, loss_dict = model.get_loss_breakdown(batch_list, epoch)
    
    # Combined
    loss_total = 0.8 * loss_fm + 0.2 * loss_selector
```

### **Expected Gain**
- **+5-8 km** (selector better calibrated)
- Especially on hard cases where correct ranking matters

---

## 🎯 FIX #5: Hard Threshold Consistency (2 HOURS - WEEK 2)

### **Current Problem**

**Training:**
```python
# hard_score_from_obs() defined in flow_matching_model.py
hard_threshold = 0.5  # Hard if score > 0.5

# Selector trained on samples classified as hard
for batch in hard_samples:
    loss_selector += selector_loss(...)
```

**Inference:**
```python
# In sample() method
is_hard = hard_score_from_obs(obs) > 0.5
# But might use different threshold or function!
```

**Result:**
- Selector trained on "hard according to function A"
- But applied to "hard according to function B"
- Distribution mismatch

### **The Fix**

**File:** `Model/flow_matching_model.py`

```python
# Define single canonical hard_score function
def get_hard_score(obs_traj):
    """
    Canonical definition used EVERYWHERE (train & inference).
    Returns: score in [0, 1], where >0.5 = hard case
    """
    # Compute kinematic indicators
    speeds = torch.norm(obs_traj[:, 1:] - obs_traj[:, :-1], dim=-1)
    mean_speed = torch.mean(speeds)
    speed_variance = torch.var(speeds)
    
    # Compute heading changes
    headings = torch.atan2(obs_traj[:, 1:, 1], obs_traj[:, 1:, 0])
    heading_changes = torch.abs(headings[1:] - headings[:-1])
    mean_heading_change = torch.mean(heading_changes)
    
    # Score: high variance or high heading change = hard
    # (bão unstable or changing direction = harder to predict)
    variance_score = torch.tanh(speed_variance / 2.0)  # Scale to [0,1]
    heading_score = torch.tanh(mean_heading_change / 0.1)
    
    hard_score = 0.6 * variance_score + 0.4 * heading_score
    
    return hard_score  # [batch]

HARD_THRESHOLD = 0.5  # Used globally

# Use everywhere:
# 1. In training loop: is_hard = get_hard_score(obs) > HARD_THRESHOLD
# 2. In selector training: hard_samples = [b for b in batch if is_hard[b]]
# 3. In inference: is_hard = get_hard_score(obs) > HARD_THRESHOLD
```

**Verify consistency:**

```python
def verify_hard_classification(self, batch_list):
    """Check that hard classification is consistent."""
    obs_traj = batch_list['obs_traj']
    
    # Method 1: Function definition
    hard_score = self.get_hard_score(obs_traj)
    is_hard_1 = hard_score > 0.5
    
    # Method 2: How it's used in training
    is_hard_2 = classify_hard_easy(obs_traj, method='kinematic') > 0.5
    
    # Should match!
    match = torch.allclose(is_hard_1, is_hard_2)
    
    if not match:
        print("[WARNING] Hard classification mismatch between methods!")
        print(f"  Method 1: {is_hard_1.sum().item()}/{len(is_hard_1)} hard")
        print(f"  Method 2: {is_hard_2.sum().item()}/{len(is_hard_2)} hard")
    
    return match
```

### **Expected Gain**
- **+3-5 km** (selector properly applied)
- Ensures no classification drift

---

## 📅 IMPLEMENTATION SCHEDULE

### **Week 1: Quick Fixes**
- [ ] **Day 1:** Enable selector (+1 line, +10-15 km)
  - Retrain from ep0 WITH FIX-LR-1/PHASE4-1
- [ ] **Day 2-3:** Environment loss (5h coding)
  - Retrain from ep0
- [ ] **Day 4-5:** Selector t-schedule fix (3h)
- [ ] **Day 6-7:** Hard threshold consistency (2h)

### **Week 2: Major Improvements**
- [ ] **Day 8-12:** Trajectory loss (8h coding, 24h training)
  - This is the biggest gain (+20-40 km)
  - Requires modifying sample() to return candidates

### **Week 3: Tuning & Testing**
- [ ] **Day 15-17:** Loss weight tuning
  - Adjust λ_env, λ_trajectory, λ_diversity
- [ ] **Day 18-21:** Curriculum + hard case focus

### **Week 4: Mode Conditioning (BONUS)**
- [ ] Implement 5 modes instead of random noise
- [ ] +20-30 km additional gain

---

## 📊 EXPECTED PROGRESSION

```
Baseline (current v59):    325.3 km (broken!)
  ↓ (Enable selector)      -12 km → 313 km
  ↓ (Env loss)             -25 km → 288 km
  ↓ (Retrain properly)     -20 km → 268 km
  ↓ (Trajectory loss)      -30 km → 238 km
  ↓ (Selector fix)         -10 km → 228 km
  ↓ (Hard threshold)       -5 km  → 223 km
  ↓ (Phase 4 tuning)       -15 km → 208 km
  ↓ (Mode conditioning)    -20 km → 188 km ✅

Final: 195-210 km range
vs ST-Trans: 224.4 km
→ VICTORY by 14-29 km ✅✅✅
```

---

## ✅ TESTING CHECKLIST

For each fix, verify:

- [ ] Code compiles without errors
- [ ] Unit tests pass (if applicable)
- [ ] Training loss decreases smoothly
- [ ] val_ade improves as expected
- [ ] No NaN or divergence
- [ ] GradNorm weights equilibrate properly
- [ ] Selector confidence scores reasonable
- [ ] Hard/easy split maintained
- [ ] Inference runs without crash
- [ ] Final predictions make sense (reasonable trajectories)

---

## 🚨 FAILURE RECOVERY

If ADE doesn't improve:

1. **Check LR:** `print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")`
   - Should be ~1e-4 to 1e-5, NOT 1e-6

2. **Check loss terms:** Log all losses
   - l_fm should dominate initially
   - l_steering should decrease over time
   - l_trajectory should have signal

3. **Check hard/easy split:**
   - hard_ADE should decrease after env + trajectory loss
   - easy_ADE should stay stable

4. **If env loss breaks easy_ADE:**
   - Reduce λ_steering from 0.1 → 0.05
   - Or reduce weight in GradNorm equilibrium

5. **If trajectory loss breaks diversity:**
   - Increase diversity loss weight
   - Check if `diversity_threshold` is reasonable

---

## 📝 SUMMARY

**To beat ST-Trans (224.4 km) by end of June:**

1. ✅ **[DONE]** FIX-LR-1 + FIX-PHASE4-1 (merge & retrain)
2. ✅ **[TODAY]** Enable selector (1 line)
3. **[THIS WEEK]** Environment loss (5h) + Retrain
4. **[THIS WEEK]** Selector t-schedule fix (3h)
5. **[THIS WEEK]** Hard threshold consistency (2h)
6. **[NEXT WEEK]** Trajectory loss (8h) + Retrain
7. **[WEEK 3]** Fine-tuning + curriculum adjustment
8. **[WEEK 4]** Mode conditioning (bonus for extra margin)

**Expected result:** 195-210 km ADE (beat ST-Trans by 14-29 km) ✅

**Confidence:** 80-90% if implemented properly

