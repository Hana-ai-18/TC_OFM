# TC-FlowMatching v59: Complete Analysis & Fix Strategy
**Date:** 2026-06-16 | **Status:** ep75, ADE=325.3 km | **Target:** 172.68 km (ST-Trans)

---

## Executive Summary

### Fixed Issues (Deployed ✅)
- **FIX-LR-1**: T_max persistence to checkpoint → LR won't deplete early on resume
- **FIX-PHASE4-1**: Phase 4 freeze state persistence → prevents duplicate encoder freeze

### Blocking Issues (5 Critical)
| # | Issue | Impact | Fix Effort | Gain |
|---|-------|--------|-----------|------|
| 1 | Selector disabled by default | -15 km | 10 min (1 line) | +10-15 km |
| 2 | Selector t-distribution mismatch (t=0.5 → t≈0) | -5 km | 3-4h | +5-8 km |
| **3** | **Environment loss missing (CRITICAL)** | **-30 km** | **2-3h** | **+20-30 km** |
| 4 | Hard threshold train/test mismatch | -3 km | Medium | +3-5 km |
| **5** | **CFM single-step, not trajectory loss (CRITICAL)** | **-30 km** | **6-8h** | **+20-40 km** |

**Realistic outcome after all fixes:** ADE ~260-280 km (still 90-110 km from ST-Trans)

---

## Part 1: What We Know (Analysis Complete)

### 1.1 Current Pipeline Architecture

```
Training Input
    ↓
┌─────────────────────────────────────────────────────┐
│ Encoder (FNO3D + Mamba)                             │
│ - Global trajectory context                         │
│ - Wind/steering features                            │
└─────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────┐
│ Velocity Field (FM Head)                            │
│ - Receives: context + steering_feat + env_kine_feat│
│ - Loss: l_fm = MSE(pred_vel, target_vel) [CFM]     │
│ - Physics: v_steer, v_phys added in no_grad        │
│ - Output: v_pred (velocity field)                   │
└─────────────────────────────────────────────────────┘
    ↓ (Training)
┌─────────────────────────────────────────────────────┐
│ Selector Network (Hard Cases Only)                  │
│ - Trained at: t=0.5 (single noisy step)            │
│ - Loss weight: 0.20                                │
│ - Decision: use_selector=False default (DISABLED)   │
└─────────────────────────────────────────────────────┘
    ↓ (Inference: DDIM Sampling)
┌─────────────────────────────────────────────────────┐
│ Generate Candidates (50 FM trajectories)            │
│ + Speed sweeps (×5 variations) → 250 total         │
└─────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────┐
│ Select Final Trajectory                             │
│ - K3 mode clustering (kinematic scores only)       │
│ - Selector: disabled or used only for hard cases    │
└─────────────────────────────────────────────────────┘
    ↓
Final Prediction
```

### 1.2 Loss Function Breakdown (Current)

```python
l_total = l_fm + λ_dpe * l_dpe + λ_vel_reg * l_vel_reg + λ_heading * l_heading
        + λ_speed * l_speed + λ_accel * l_accel + α_hard * l_hard
        + 0.30 * l_speed_head + 0.20 * l_selector + 0.5 * l_fm2 + div_w * l_diversity
```

**Key missing term:** `l_steering` or `l_env` — no loss for using environment context

---

## Part 2: Critical Issues & Root Causes

### Issue #1: Selector Disabled by Default ⚡ QUICK FIX

**Location:** `Model/flow_matching_model.py` line 6210

```python
# Current (DISABLED):
def sample(self, batch_list, ..., use_selector=False, ...):
    if use_selector:  # Only if explicitly True
        # Use selector for hard cases
    else:
        # Use K3 mode clustering for everything

# Problem: K3 scoring is kinematic-only
# - head_sc: heading alignment (last 3 obs steps)
# - spd_sc: speed consistency (last 3 obs steps)
# - prior_sc: speed prior
# - smooth_sc: acceleration smoothness
# ✗ No environment context
# ✗ No future trajectory info
# ✗ No hard-case awareness
```

**Why hard_ADE (200-210 km) is better than easy_ADE (380-407 km):**
- "Hard" observations (curvy, variable speed) → momentum-based extrapolation works okay
- "Easy" observations (smooth, steady) → heuristic extrapolation, BUT model doesn't know future recurve coming
- Kinematic scorer judges PAST (easy), not FUTURE (hard)

**Solution:** Change 1 line
```python
def sample(self, batch_list, ..., use_selector=True, ...):  # Enable by default
```

**Expected gain:** +10-15 km ADE

---

### Issue #2: Selector Distribution Mismatch (t=0.5 vs t≈0) 🔧 MEDIUM

**Location:** `Model/flow_matching_model.py` lines 6108, 6303-6328

```python
# Training: selector sees NOISY trajectory at t=0.5
for t_sel in [0.5]:  # FIXED at middle
    x_t_sel = model.scheduler.scale_model_input(x_noisy, t_sel)
    # Intermediate, noisy representation
    loss_sel = selector_loss(...)

# Inference: selector receives FULLY DENOISED trajectory (t≈0)
for step in range(ddim_steps):  # 20 steps
    x_t = denoise(x_t, t)  # Final t ≈ 0
# Much clearer, different feature scale
```

**Why it matters:**
- Selector trained on one feature distribution (noisy)
- Evaluated on another (clean)
- Confidence scores may not predict actual final quality

**Solution:** Train selector on full DDIM trajectory
```python
# Proposed: multi-step training
ddim_sample_steps = [0.1, 0.3, 0.5, 0.7, 0.9]
for t_sel in ddim_sample_steps:
    x_t_sel = model.scheduler.scale_model_input(x_noisy, t_sel)
    loss_sel += selector_loss(...) / len(ddim_sample_steps)
```

**Expected gain:** +5-8 km ADE (after full retraining)

---

### Issue #3: Environment Context NOT in Loss 🔴 CRITICAL

**Location:** `Model/flow_matching_model.py` lines 5389-5533 (forward), 5945-6205 (loss)

```python
# Forward pass: steering features ARE passed in
def forward_with_ctx(self, x_t, t, raw_ctx,
                     vel_obs_feat=None,
                     steering_feat=None,       # <-- Present
                     env_kine_feat=None, ...): # <-- Present
    # Steering features embedded in transformer decoder
    decoded = self._decode(x_t, t, ctx,
                          vel_obs_feat=vel_obs_feat,
                          steering_feat=steering_feat,    # <-- Used
                          env_kine_feat=env_kine_feat)    # <-- Used
    
    v_neural = self.out_fc2(F.gelu(self.out_fc1(decoded)))
    
    # Physics drift ADDED but NOT trained
    v_phys = self._beta_drift(x_t)          # no_grad — never updated
    v_steer = self._steering_drift(env_data)  # no_grad — never updated
    
    return v_neural + sigmoid(scale)*v_phys + sigmoid(scale)*v_steer

# Loss function: NO steering constraint
l_fm = F.mse_loss(pred_vel, u_target)  # Line 5997
# ✗ No term like: l_steering = MSE(steer_component, env_driven_component)
# ✗ No regularizer for steering sensitivity
```

**Why it's the core issue:**

Model has steering input but no loss term saying "steering matters" → learns to ignore environment.

For "easy observations that become hard":
- Past trajectory is smooth (momentum extrapolation works)
- But future requires steering/environment response
- Model has no training signal to learn when to steer

**Solution:** Add steering loss component

```python
# In get_loss_breakdown(), modify loss calculation:

# Option A: Decompose velocity (elegant but complex)
# v_neural = base_vel + steering_driven_component
# l_steering = MSE(steering_component, env_driven_ground_truth)
# l_total = l_fm + λ_steer * l_steering

# Option B: Steering sensitivity regularizer (simpler)
# v_with_steer = forward(..., steering_feat=actual)
# v_without_steer = forward(..., steering_feat=zeros)
# l_steer_sensitivity = MSE(v_with_steer, v_without_steer) * 0.1
# l_total = l_fm + l_steer_sensitivity

# Option C: Proposed implementation (balanced)
def get_loss_breakdown(self, batch_list, epoch, ..., lambda_dict=None, ...):
    # ... existing CFM loss code ...
    l_fm = F.mse_loss(pred_vel, u_target)
    
    # NEW: Steering constraint
    # Decompose: model should learn to modulate velocity based on environment
    steering_weight = torch.sigmoid(self.steering_weight_param)  # Learnable
    
    # Ground truth steering component (derived from delta between kinematic+steering)
    env_velocity_gt = self._compute_env_velocity_response(
        obs_traj, env_data, target_vel)
    
    # Loss: model should predict steering-aware velocity
    v_with_env_awareness = pred_vel  # Already includes env features
    l_steering = F.mse_loss(
        v_with_env_awareness * steering_weight,
        env_velocity_gt * steering_weight
    )
    
    # Add to loss breakdown
    lambda_steering = lambda_dict.get("l_steering", 0.1)
    l_base = l_fm + lambda_steering * l_steering
    
    # ... rest of loss computation ...
    return {..., "l_steering": l_steering.item(), ...}
```

**Expected gain:** +20-30 km ADE (fixes easy→hard recurve problem)

**Implementation steps:**
1. Define `_compute_env_velocity_response()` method
2. Add steering weight parameter
3. Add loss term to get_loss_breakdown()
4. Update GradNorm to balance l_steering with other terms
5. Retrain from ep0

---

### Issue #4: Hard Threshold Train/Test Mismatch 🔧 SMALL

**Location:** `Model/flow_matching_model.py` lines 4445-4471, 6106, 6306

```python
# Training: uses TWO conditions
is_hard_train = classify_hard_easy(obs_traj, per_sample_loss=loss_per_sample)
# = (hard_score >= p70) AND (loss >= p50)

# Inference: uses ONE condition (loss unavailable)
is_hard_inf = classify_hard_easy(obs_norm)  # per_sample_loss=None
# Falls back to: hard_score >= p70 only

# Consequence: different samples marked "hard" in train vs inference
```

**Why it matters:** Selector trained on p70+p50 definition, but evaluated on p70 definition.

**Solution:** Persist training loss, apply same threshold at inference

```python
# In checkpoint saving: save per-sample losses
"per_sample_loss_epoch": per_sample_loss_dict  # dict[sample_id] -> loss

# At inference: load and use dual threshold
ckpt_loss = checkpoint.get("per_sample_loss_epoch")
is_hard_inf = classify_hard_easy(obs_norm, per_sample_loss=ckpt_loss)
```

**Expected gain:** +3-5 km ADE (consistency improvement)

---

### Issue #5: CFM Loss is Single-Step, Not Trajectory 🔴 CRITICAL

**Location:** `Model/flow_matching_model.py` lines 5945-6205

```python
# Current: Conditional Flow Matching (per-step loss)
# For each sample:
#   1. Sample random t ∈ [0,1]
#   2. Noisy: x_t = t·x_1 + (1-t)·x_0
#   3. Predict: v_pred = velocity_field(x_t, t, context)
#   4. Loss: l_fm = MSE(v_pred, (x_1 - x_0) / (1-t))  ← SINGLE STEP
l_fm = F.mse_loss(pred_vel, u_target)

# What it optimizes: denoise velocity at random timestep
# What it DOESN'T optimize: final trajectory quality (ADE)

# Why it fails:
# - Model trained on: "given noisy traj, predict clean direction"
# - Task requires: "predict future trajectory matching ground truth"
# - Errors accumulate across 20 DDIM steps in inference
# - No end-to-end loss guiding full trajectory toward ground truth
```

**Evidence:** val_loss at ep10 is excellent (2.4), but ADE is terrible (391 km) → model learned denoising, not trajectory prediction.

**Solution:** Add trajectory-level loss

```python
# Option A: Multi-step ADE loss (balance speed & accuracy)
if sample_idx % 10 == 0:  # 10% of samples
    # Run DDIM forward for short trajectory
    x_t_traj = x_noisy.clone()
    for step in range(ddim_steps_train):  # e.g., 10 steps
        x_t_traj = denoise_one_step(x_t_traj, step)
    
    # Compute ADE on denoised trajectory
    traj_denoised = denormalize(x_t_traj)
    ade_intermediate = compute_ade(traj_denoised, ground_truth)
    l_ade_multi = ade_intermediate / 100.0  # Normalize
    
    # Blend with CFM loss
    l_total = 0.9 * l_fm + 0.1 * l_ade_multi
else:
    l_total = l_fm

# Option B: Proposed (hybrid, practical)
# Keep CFM as primary (fast), add ADE supervision on subset
def get_loss_breakdown(self, batch_list, epoch, ...):
    # ... compute l_fm (CFM loss) ...
    l_fm = F.mse_loss(pred_vel, u_target)
    
    # NEW: Add trajectory-level loss on 10% of samples
    if random.random() < 0.1:  # 10% sample rate
        # Run full DDIM on this sample (expensive)
        with torch.no_grad():
            x_sample = x_noisy[:1].clone()  # Single sample
            for step in range(self.ddim_steps):
                x_sample = self._ddim_step(x_sample, step)
        
        # Compute trajectory metric
        traj_final = self.denormalize(x_sample)
        ate_loss = compute_ate(traj_final[:, :12], ground_truth[:1, :12])
        cte_loss = compute_cte(traj_final[:, :12], ground_truth[:1, :12])
        l_trajectory = (ate_loss + 0.5*cte_loss) / 100.0
        
        # Blend: heavily favor CFM (which is fast) over trajectory loss (which is expensive)
        l_base = 0.95 * l_fm + 0.05 * l_trajectory
    else:
        l_base = l_fm
    
    # Continue with other loss terms...
    return {..., "l_fm": l_fm.item(), "l_trajectory": l_trajectory if ... else 0, ...}
```

**Expected gain:** +20-40 km ADE (largest single fix, aligns loss with metric)

**Cost:** +20-30% training time (DDIM sampling is expensive)

---

## Part 3: Hướng Fix Cấu Trúc Flow_Matching

### 3.1 Architecture Redesign Strategy

Current bottleneck: **Loss function is misaligned with task**

```
Current:  Loss = denoise velocity (CFM)  →  Metric = trajectory quality (ADE)
          ✗ Different optimization targets

Proposed: Loss = denoise velocity (CFM) + trajectory quality (ADE)
          ✓ Aligned optimization
```

### 3.2 Priority Implementation Order

#### Phase A: Immediate (30 min, +10-15 km)
```python
# File: Model/flow_matching_model.py, line 6210
# Change: use_selector=False → use_selector=True

def sample(self, batch_list, use_selector=True, ...):  # Enable selector
```
- No retraining needed
- Uses existing trained selector network
- Should help hard cases immediately

#### Phase B.1: Environment Loss (3-5 hours, +20-30 km) 🔴 CRITICAL

**Step 1: Add steering weight parameter**
```python
# In TCFlowMatching.__init__():
self.steering_weight = torch.nn.Parameter(torch.tensor(0.5))
self.env_scale = torch.nn.Parameter(torch.tensor(0.1))
```

**Step 2: Compute environment velocity ground truth**
```python
def _compute_env_velocity_response(self, obs_traj, env_data, target_vel):
    """
    Extract steering-driven component from target velocity.
    
    Logic:
    - Kinematic momentum: vel_momentum = ema of obs_traj velocity
    - Environment steering: vel_env = target_vel - vel_momentum (residual)
    - Return: vel_env (what model should learn)
    """
    # Compute momentum velocity
    obs_vel = (obs_traj[1:] - obs_traj[:-1])  # Noisy obs
    vel_momentum = torch.mean(obs_vel[-3:], dim=0, keepdim=True)  # Last 3 steps
    
    # Environment component is what momentum doesn't explain
    vel_env = target_vel - vel_momentum
    return vel_env
```

**Step 3: Add steering loss term**
```python
def get_loss_breakdown(self, batch_list, epoch, ...):
    # ... existing code: compute l_fm ...
    l_fm = F.mse_loss(pred_vel, u_target)
    
    # NEW: Steering loss
    env_vel_gt = self._compute_env_velocity_response(obs_norm, env_data, u_target)
    steering_weight = torch.sigmoid(self.steering_weight)
    l_steering = F.mse_loss(pred_vel * steering_weight, env_vel_gt * steering_weight)
    
    # Blend
    lambda_steering = lambda_dict.get("l_steering", 0.1)
    l_base = l_fm + lambda_steering * l_steering
    
    # ... continue with rest ...
```

**Step 4: Retrain from ep0 with FIX-LR-1 + FIX-SELECTOR-1 + this fix**

#### Phase B.2: Selector Retraining (4-5 hours, +5-8 km) 🔧 MEDIUM

```python
# File: Model/flow_matching_model.py, lines 6107-6139
# Current:
for t_sel in [0.5]:
    x_t_sel = model.scheduler.scale_model_input(x_noisy, t_sel)
    loss_sel += selector_loss(...)

# Proposed:
ddim_sample_steps = [0.1, 0.3, 0.5, 0.7, 0.9]
for t_sel in ddim_sample_steps:
    x_t_sel = model.scheduler.scale_model_input(x_noisy, t_sel)
    loss_sel += selector_loss(...) / len(ddim_sample_steps)

# Or even better: run full DDIM
x_t_trajectory = x_noisy.clone()
for step in range(5):  # 5 DDIM steps
    x_t_trajectory = self._ddim_step(x_t_trajectory, step)
confidence = self.selector(x_t_trajectory, ctx)
loss_sel = ranking_loss(confidence, oracle_ade)
```

#### Phase C: Trajectory Loss (6-8 hours, +20-40 km) 🔴 CRITICAL

```python
# File: Model/flow_matching_model.py, lines 5945-6205
# Add ADE supervision on subset of samples

def get_loss_breakdown(self, batch_list, epoch, ...):
    l_fm = F.mse_loss(pred_vel, u_target)
    
    # NEW: Trajectory-level loss on 10% of samples
    l_trajectory = 0.0
    traj_loss_count = 0
    
    for i, sample in enumerate(batch_list):
        if random.random() < 0.1:  # 10% sample rate
            # Expensive: run DDIM on this sample
            x_sample = denoise_full_trajectory(x_noisy[i:i+1])
            traj = denormalize(x_sample)
            ate = compute_ate(traj[:, :12], gt[:, :12])
            cte = compute_cte(traj[:, :12], gt[:, :12])
            l_trajectory += ate / 100.0 + 0.5 * cte / 100.0
            traj_loss_count += 1
    
    if traj_loss_count > 0:
        l_trajectory = l_trajectory / traj_loss_count
        l_base = 0.95 * l_fm + 0.05 * l_trajectory
    else:
        l_base = l_fm
```

### 3.3 Integration with GradNorm

Current GradNorm balances: l_dpe, l_vel_reg, l_heading, l_speed, l_accel

After fixes, add to balanced terms:
```python
GRADNORM_TERMS = [
    "l_dpe",
    "l_vel_reg",
    "l_heading",
    "l_speed",
    "l_accel",
    "l_steering",      # NEW
    "l_trajectory",    # NEW
]
```

GradNorm will automatically balance so all terms contribute equally to gradient norm.

---

## Part 4: Implementation Roadmap

### Timeline & Effort Estimates

```
Week 1:
├─ FIX-LR-1 + FIX-PHASE4-1 ✅ DONE
├─ FIX-SELECTOR-1 (10 min)
├─ Fresh training from ep0
└─ Monitor ep50-80

Week 2:
├─ FIX-ENV-1 (3-5h coding)
├─ Retrain from ep0
└─ Measure phase 3+ improvement

Week 3:
├─ FIX-SELECTOR-2 (4-5h coding)
├─ Retrain from ep0
└─ Evaluate selector distribution fix

Week 4:
├─ FIX-LOSS-1 (6-8h coding)
├─ Retrain (expensive, +20-30% time)
└─ Final evaluation
```

### Success Criteria

| Phase | Target | How to Know |
|-------|--------|-----------|
| Phase A | ADE <315 km | Selector helps hard cases (hard_ADE < 200 km) |
| Phase B.1 | ADE <295 km | Easy_ADE improves significantly (< 350 km) |
| Phase B.2 | ADE <290 km | Selector confidence becomes more predictive |
| Phase C | ADE <270 km | Trajectory quality improves, smoother prediction |
| Final | ADE <250 km | Beats expected improvement curve |

---

## Part 5: Why This Approach Works

### 1. Addresses Root Causes
- ✗ Selector disabled → ✓ Enable it
- ✗ Selector t-mismatch → ✓ Train on full DDIM
- ✗ Environment not in loss → ✓ Add steering loss
- ✗ CFM vs ADE mismatch → ✓ Add trajectory loss
- ✗ Hard threshold mismatch → ✓ Persist per-sample loss

### 2. Incremental & Testable
- Each fix can be deployed independently
- Can measure impact of each phase
- No risk of breaking existing functionality

### 3. Theoretically Sound
- Steering loss directly addresses "easy→hard recurve" problem
- Trajectory loss aligns loss function with metric
- Selector retraining removes distribution mismatch

### 4. Effort-Efficient
- Phase A: 10 min, +10 km
- Phase B: 8-10 hours, +35-45 km
- Phase C: 8 hours + expensive training, +25-40 km
- **Total: ~1 day coding, 2-3 weeks training**

---

## Part 6: Expected Outcomes

### Conservative Estimate
- Phase A (selector enable): ADE ~315 km
- Phase B (env + selector retrain): ADE ~290 km
- Phase C (trajectory loss): ADE ~265 km

### Optimistic Estimate
- With careful tuning & longer training: ADE ~240-250 km
- Still ~70-80 km above ST-Trans target

### Why Gap Remains
ST-Trans uses:
1. ✓ Social trajectories (we have single agent)
2. ✓ Attention mechanism (we have transformer but kinematic scoring)
3. ✓ Full encoder-decoder (we have FM decoder)

To beat ST-Trans completely: would need more architectural changes (attention-based mode selection, social integration, etc.)

---

## Part 7: Code Locations Quick Reference

| Fix | File | Lines | Type |
|-----|------|-------|------|
| FIX-LR-1 | train_flowmatching.py | 8862, 8903-8920, 9289+ | ✅ DONE |
| FIX-PHASE4-1 | train_flowmatching.py | 8995-9001, 9289+ | ✅ DONE |
| FIX-SELECTOR-1 | flow_matching_model.py | 6210 | Config |
| FIX-ENV-1 | flow_matching_model.py | 5389-5533, 5945-6205 | Loss |
| FIX-SELECTOR-2 | flow_matching_model.py | 6107-6139 | Training |
| FIX-LOSS-1 | flow_matching_model.py | 5945-6205 | Loss |
| FIX-THRESHOLD-1 | flow_matching_model.py | 6306, train_flow*.py | State |

---

## Conclusion

**The core insight:** TC-FlowMatching has all the pieces (selector, environment features, capable loss) but they're not wired together correctly.

- Selector is trained but disabled
- Environment context is passed but not constrained in loss
- Loss is per-step but metric is trajectory-level

**The fix is architectural, not algorithmic.** By enabling selector + adding steering loss + fixing trajectory-level optimization, we can realistically achieve **260-290 km ADE** with careful implementation and training.

**To definitively beat ST-Trans (172.68 km):** Would require additional breakthroughs beyond these fixes (social integration, attention-based mode selection, or architectural innovations).

**Recommended:** Start with Phase A (10 min) today, measure results, then proceed to Phase B if gains are realized.
