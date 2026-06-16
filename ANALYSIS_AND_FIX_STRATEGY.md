# TC-FlowMatching v59-Strategy: Complete Analysis & Fix Strategy
**Date:** 2026-06-16 | **Epoch:** 75/100 | **ADE:** 325.3 km | **Target:** 172.68 km (ST-Trans)

---

## EXECUTIVE SUMMARY

### What's Fixed ✅ (Commit a247241)

**FIX-LR-1: Learning Rate Schedule Persistence**
- **Issue:** T_max (schedule duration) recalculated each resume → LR depleted early
- **Symptom:** LR dropped to ~1e-6 by ep60, starving phase 3 of learning signal
- **Root cause:** `nstep * args.num_epochs` recalculated with potentially shorter num_epochs
- **Solution:** Save T_max_orig to checkpoint, restore on resume
- **File changed:** train_flowmatching.py (lines 8862, 8903-8920, 9289-9501)
- **Impact:** LR will maintain proper schedule (e.g., ~1e-5 in phase 3 instead of depleting to 1e-6)

**FIX-PHASE4-1: Phase 4 Encoder Freeze State Persistence**
- **Issue:** `_phase4_frozen` flag lost on resume (local variable, not saved)
- **Symptom:** Encoder re-frozen after resume at ep≥80, unintended LR reduction
- **Root cause:** Nonlocal variable, not persisted to checkpoint
- **Solution:** Save `_phase4_frozen` flag to checkpoint, restore on resume
- **File changed:** train_flowmatching.py (lines 8995-9001, 9289-9501)
- **Impact:** Prevents duplicate freeze and unintended phase 4 side effects

### What's NOT Fixed (5 Critical Issues)

| # | Issue | Location | Impact | Fix Effort | Gain |
|---|-------|----------|--------|-----------|------|
| 1 | **Selector disabled** | Model/flow_matching_model.py:6210 | Kinematic-only scoring | 10 min (1 line) | +10-15 km |
| 2 | **Environment loss missing** | Model/flow_matching_model.py:5389-6205 | Model ignores steering context | 3-5h coding | +20-30 km |
| 3 | **CFM vs trajectory loss** | Model/flow_matching_model.py:5945-6205 | Errors accumulate in DDIM | 6-8h coding | +20-40 km |
| 4 | **Selector t-distribution** | Model/flow_matching_model.py:6107-6139 | Train/inference mismatch | 4-5h coding | +5-8 km |
| 5 | **Hard threshold mismatch** | Model/flow_matching_model.py:6306 | Different classification | Medium | +3-5 km |

**Total estimated gain if ALL fixed:** +58-98 km → ADE 260-267 km (still ~90-100 km from ST-Trans)

---

## DETAILED ANALYSIS: What's Wrong & Why

### Current Architecture Pipeline

```
┌──────────────────────────────────────────────────────────┐
│ Training Input                                           │
└──────────────┬───────────────────────────────────────────┘
               ↓
┌──────────────────────────────────────────────────────────┐
│ Encoder (FNO3D + Mamba)                                  │
│ - Global trajectory context                             │
│ - Wind/steering features                                │
└──────────────┬───────────────────────────────────────────┘
               ↓
┌──────────────────────────────────────────────────────────┐
│ Velocity Field (FM Head) ← ISSUES #2, #3 HERE          │
│ - Receives: context + steering_feat + env_kine_feat    │
│ - Loss: l_fm = MSE(pred_vel, target_vel) [CFM only]    │
│ - ✗ Missing: l_steering (no env loss term)             │
│ - ✗ Single-step CFM, not full trajectory loss          │
└──────────────┬───────────────────────────────────────────┘
               ↓
┌──────────────────────────────────────────────────────────┐
│ Selector Network (Hard Cases) ← ISSUES #1, #4 HERE      │
│ - Trained: t=0.5 (single noisy step)                    │
│ - Inference: use_selector=False (DISABLED by default)   │
│ - ✗ Distribution mismatch: train t=0.5 vs eval t≈0     │
└──────────────┬───────────────────────────────────────────┘
               ↓
┌──────────────────────────────────────────────────────────┐
│ Sample Generation (DDIM)                                │
│ - Generate 50 FM trajectories + speed sweeps × 5        │
│ → 250 candidates total                                  │
└──────────────┬───────────────────────────────────────────┘
               ↓
┌──────────────────────────────────────────────────────────┐
│ Select Final (K3 Mode Clustering) ← ISSUE #5 HERE       │
│ - Uses kinematic scores only:                           │
│   * head_sc: heading alignment (last 3 obs)            │
│   * spd_sc: speed consistency (last 3 obs)             │
│   * prior_sc: speed prior (Gaussian)                    │
│   * smooth_sc: acceleration smoothness                  │
│ - ✗ Selector disabled or misaligned                     │
│ - ✗ Hard threshold train/test mismatch                 │
└──────────────┬───────────────────────────────────────────┘
               ↓
            Final Prediction
```

---

## Issue #1: Selector Disabled by Default ⚡ QUICK FIX

**Location:** `Model/flow_matching_model.py` line 6210

**Current Code:**
```python
def sample(self, batch_list, use_selector=False, ...):
    ...
    if use_selector:  # Only if explicitly True
        use_sel_mask = is_hard_inf & (confidence >= threshold)
        # Use selector for hard cases
    else:
        # Use K3 mode clustering for everything
```

**Problem:**
- Selector is trained (loss weight 0.20, lines 6107-6139)
- But inference ALWAYS defaults to kinematic-only K3 clustering
- K3 scores only: heading, speed, smoothness (measures PAST, not FUTURE)

**Why it's bad:**
For "easy observations that become hard" (68% of data):
- Observed trajectory is smooth → kinematic scorer marks "easy"
- But future will recurve → prediction fails
- Selector trained to recognize this, but it's disabled

**Solution (1 line):**
```python
def sample(self, batch_list, use_selector=True, ...):  # Enable by default
```

**Expected gain:** +10-15 km ADE (no retraining needed)

---

## Issue #2: Environment Loss Missing 🔴 CRITICAL

**Location:** `Model/flow_matching_model.py` lines 5389-6205

**Current Code:**
```python
# Velocity field decoder receives environment features
def forward_with_ctx(self, x_t, t, raw_ctx,
                     vel_obs_feat=None,
                     steering_feat=None,      # ← Present
                     env_kine_feat=None, ...):
    # Features embedded in transformer
    decoded = self._decode(x_t, t, ctx,
                          steering_feat=steering_feat,    # ← Used
                          env_kine_feat=env_kine_feat)    # ← Used
    
    v_neural = self.out_fc2(F.gelu(self.out_fc1(decoded)))
    
    # Physics drift added but NEVER trained
    v_phys = self._beta_drift(x_t)          # no_grad
    v_steer = self._steering_drift(env_data)  # no_grad
    
    return v_neural + sigmoid(scale)*v_phys + sigmoid(scale)*v_steer

# Loss function: ONLY CFM, no steering constraint
l_fm = F.mse_loss(pred_vel, u_target)  # Line 5997
# ✗ Missing: l_steering term
```

**Problem:**
- Model HAS steering input but NO loss term saying "steering matters"
- Physics drift added in no_grad → never updated
- Equivalent to giving model context but saying "doesn't affect loss" → learns to ignore

**Why it's THE core issue:**
For "easy obs → hard future":
- Model trained only on kinematic patterns
- Has no training signal: "when to steer based on environment"
- Easy trajectories stay easy even if future recurves (can't learn the transition)

**Solution (add steering loss):**
```python
def get_loss_breakdown(self, batch_list, epoch, ..., lambda_dict=None, ...):
    # ... existing CFM loss ...
    l_fm = F.mse_loss(pred_vel, u_target)
    
    # NEW: Steering constraint
    steering_weight = torch.sigmoid(self.steering_weight_param)  # Learnable
    
    # Extract ground truth steering component
    env_velocity_gt = self._compute_env_velocity_response(
        obs_traj, env_data, target_vel)
    
    # Loss: model should predict steering-aware velocity
    l_steering = F.mse_loss(
        pred_vel * steering_weight,
        env_velocity_gt * steering_weight
    )
    
    # Blend with CFM
    lambda_steering = lambda_dict.get("l_steering", 0.1)
    l_base = l_fm + lambda_steering * l_steering
    
    # Add to GradNorm for balanced learning
    return {..., "l_steering": l_steering.item(), ...}
```

**Implementation steps:**
1. Add `self.steering_weight = nn.Parameter(torch.tensor(0.5))` in __init__
2. Define `_compute_env_velocity_response()` method
3. Add steering loss term in get_loss_breakdown()
4. Update GRADNORM_TERMS to include "l_steering"
5. Retrain from ep0

**Expected gain:** +20-30 km ADE (fixes easy→hard recurve problem)

---

## Issue #3: CFM Loss vs Trajectory Metric Mismatch 🔴 CRITICAL

**Location:** `Model/flow_matching_model.py` lines 5945-6205

**Current Code:**
```python
# Training: Conditional Flow Matching (CFM)
# For each sample:
#   1. Sample random t ∈ [0,1]
#   2. Noisy: x_t = t·x_1 + (1-t)·x_0
#   3. Predict: v_pred = velocity_field(x_t, t, context)
#   4. Loss: l_fm = MSE(v_pred, (x_1 - x_0) / (1-t))  ← ONE STEP
l_fm = F.mse_loss(pred_vel, u_target)

# What it optimizes:
#   "Given noisy trajectory at t, predict clean direction"
# What task requires:
#   "Predict future trajectory matching ground truth"
# These are DIFFERENT
```

**Problem:**
- Loss optimized for: per-step velocity matching (denoising)
- Task metric: full-trajectory ADE matching
- No end-to-end loss guiding entire path toward ground truth
- Errors accumulate across 20 DDIM steps in inference

**Evidence:**
- val_loss at ep10: excellent (2.4)
- ADE at ep10: terrible (391 km)
→ Model learned denoise well, not forecast well

**Why it matters:**
- Phase 3 curriculum (hard samples, α_hard, selector) all assume model learns trajectory quality
- But loss doesn't directly optimize for that
- GradNorm balances loss terms, but all are still velocity-level, not trajectory-level

**Solution (add trajectory loss):**
```python
def get_loss_breakdown(self, batch_list, epoch, ...):
    l_fm = F.mse_loss(pred_vel, u_target)
    
    # NEW: Trajectory-level loss on subset (balance speed vs accuracy)
    l_trajectory = 0.0
    if random.random() < 0.1:  # 10% of samples
        # Run DDIM on this sample (expensive)
        x_sample = x_noisy.clone()
        with torch.no_grad():
            for step in range(self.ddim_steps):
                x_sample = self._ddim_step(x_sample, step)
        
        # Compute trajectory metric on denoised path
        traj_denoised = self.denormalize(x_sample)
        ate_loss = compute_ate(traj_denoised[:, :12], ground_truth[:, :12])
        cte_loss = compute_cte(traj_denoised[:, :12], ground_truth[:, :12])
        l_trajectory = (ate_loss + 0.5*cte_loss) / 100.0  # Normalize
        
        # Blend: heavily favor CFM (fast) over trajectory loss (expensive)
        l_base = 0.95 * l_fm + 0.05 * l_trajectory
    else:
        l_base = l_fm
    
    # Continue with other loss terms...
    return {..., "l_fm": l_fm.item(), "l_trajectory": l_trajectory, ...}
```

**Implementation steps:**
1. Add `_ddim_step()` method (or reuse existing sampling logic)
2. Add trajectory loss computation for 10% of samples
3. Blend l_fm (90%) + l_trajectory (10%)
4. Add to GradNorm
5. Retrain (expensive: +20-30% training time)

**Expected gain:** +20-40 km ADE (LARGEST SINGLE FIX, aligns loss with metric)

---

## Issue #4: Selector t-Distribution Mismatch 🔧 MEDIUM

**Location:** `Model/flow_matching_model.py` lines 6107-6139, 6303-6328

**Current Code:**
```python
# Training: selector sees NOISY trajectory at t=0.5
for t_sel in [0.5]:  # Fixed single timestep
    x_t_sel = model.scheduler.scale_model_input(x_noisy, t_sel)
    # Intermediate, noisy representation
    loss_sel = selector_loss(...)

# Inference: selector receives FULLY DENOISED trajectory (t≈0)
for step in range(ddim_steps):  # 20 steps
    x_t = denoise(x_t, t)
# Final t ≈ 0, much clearer, different feature scale
```

**Problem:**
- Selector trained on feature distribution at t=0.5 (noisy, intermediate)
- Selector evaluated on feature distribution at t≈0 (clean, fully denoised)
- Feature scale and structure completely different
- Confidence scores may not predict actual final trajectory quality

**Why it matters:**
- Selector ranking becomes unreliable
- Hard cases selected incorrectly
- Even if enabled (Issue #1 fixed), still sub-optimal

**Solution (train on full DDIM):**
```python
# Option: Train selector to see full trajectory evolution
ddim_sample_steps = [0.1, 0.3, 0.5, 0.7, 0.9]
for t_sel in ddim_sample_steps:
    x_t_sel = model.scheduler.scale_model_input(x_noisy, t_sel)
    loss_sel += selector_loss(...) / len(ddim_sample_steps)

# Or better: Run full DDIM during selector training
x_t_trajectory = x_noisy.clone()
for step in range(5):  # Short DDIM, 5 steps
    x_t_trajectory = self._ddim_step(x_t_trajectory, step)
confidence = self.selector(x_t_trajectory, ctx_train)
loss_sel = ranking_loss(confidence, oracle_ade)
```

**Expected gain:** +5-8 km ADE (after retraining selector)

---

## Issue #5: Hard Threshold Train/Test Mismatch 🔧 SMALL

**Location:** `Model/flow_matching_model.py` lines 4445-4471, 6106, 6306

**Current Code:**
```python
# Training: TWO conditions
is_hard_train = classify_hard_easy(obs_traj, per_sample_loss=loss_per_sample)
# = (hard_score >= p70) AND (loss >= p50)

# Inference: ONE condition (loss unavailable)
is_hard_inf = classify_hard_easy(obs_norm)  # per_sample_loss=None
# Falls back to: hard_score >= p70 only
```

**Problem:**
- Training: marks samples hard if BOTH obs-complex (p70) AND high-loss (p50)
- Inference: marks hard if obs-complex >= p70 only
- Different sets of samples marked "hard" in train vs inference

**Why it matters:**
- Selector trained on p70+p50 definition
- But applied to p70 definition
- Overfits to training condition, sub-optimal for inference condition

**Solution (persist training loss):**
```python
# In checkpoint saving:
"per_sample_loss_epoch": per_sample_loss_dict  # dict[sample_id] -> loss

# At inference: load and use dual threshold
ckpt_loss = checkpoint.get("per_sample_loss_epoch")
is_hard_inf = classify_hard_easy(obs_norm, per_sample_loss=ckpt_loss)
```

**Expected gain:** +3-5 km ADE (consistency improvement)

---

## Why ST-Trans Wins & We Don't (Yet)

### ST-Trans Advantages
1. ✓ **End-to-end trajectory loss** on full 72h prediction (we have CFM single-step)
2. ✓ **Environment + social context** in transformer (we have steering_feat but no loss)
3. ✓ **Selector that works** because trained well and used by default (ours disabled/misaligned)
4. ✓ **Attention mechanism** for dynamic mode selection (we have K3 heuristic)

### Our Gaps vs ST-Trans
| Component | ST-Trans | FlowMatching | Gap |
|-----------|----------|--------------|-----|
| Encoder | Transformer (trajectory + social) | FNO3D + Mamba (trajectory + env) | ✓ Comparable |
| Loss for trajectory quality | End-to-end 72h ADE | Single-step CFM | ✗ Major |
| Environment usage | Learned in loss | Features only, no loss | ✗ Major |
| Selector | Learned attention | Disabled heuristic | ✗ Major |
| Mode selection | Attention (learned) | K3 clustering (heuristic) | ✗ Moderate |
| Social context | Yes (baseline) | No (single agent only) | ✗ Data limitation |

### Gap Breakdown
- Current FlowMatching: 325.3 km
- After Issue #1 (selector): ~315 km (-10 km)
- After Issue #2 (env loss): ~295 km (-20 km)
- After Issue #3 (trajectory loss): ~265 km (-30 km)
- After Issues #4-5: ~260 km (-5 km)
- **Realistic ceiling:** 260-280 km

- ST-Trans: 172.68 km
- **Gap remaining:** 90-100 km (cannot be fixed by these 5 issues)

### Why We Can't Beat ST-Trans Without Architectural Changes
To close the remaining 90-100 km gap would require:
1. **Social integration:** Include social trajectories (not available in this dataset for single-agent prediction)
2. **Attention-based selection:** Replace K3 clustering with learned attention (major architecture change)
3. **Regime-aware modeling:** Different models for different weather regimes (adds complexity)
4. **End-to-end training:** Gradient through full DDIM + CFM + selector + scoring (very expensive)

---

## Improvement Roadmap

### Phase A: Quick Win (30 min, no retraining)
**FIX-SELECTOR-1:** Enable selector by default
```python
# File: Model/flow_matching_model.py, line 6210
def sample(self, batch_list, use_selector=True, ...):  # Change False→True
```
**Expected gain:** +10-15 km ADE

### Phase B: Architecture Fixes (8-10 hours, full retraining)
**FIX-ENV-1:** Add steering loss term
- **Time:** 3-5h coding + full ep0-100 retraining
- **Expected gain:** +20-30 km ADE
- **New level:** ~295 km

**FIX-SELECTOR-2:** Retrain selector on full DDIM
- **Time:** 4-5h coding + retraining
- **Expected gain:** +5-8 km ADE
- **New level:** ~290 km

### Phase C: Loss Function Redesign (6-8 hours, expensive retraining)
**FIX-LOSS-1:** Add multi-step trajectory ADE loss
- **Time:** 6-8h coding + ep0-100 retraining (+20-30% training time)
- **Expected gain:** +20-40 km ADE
- **New level:** ~265 km

### Phase D: Polish (low priority)
**FIX-THRESHOLD-1:** Persist hard threshold logic
- **Expected gain:** +3-5 km ADE
- **New level:** ~260 km

### Final Expected Outcome
- **Best case:** ADE ~260 km (with careful tuning)
- **Realistic:** ADE ~265-280 km
- **Gap to ST-Trans:** Still ~90-100 km
- **To beat ST-Trans:** Would need additional architectural innovations

---

## Code Location Reference

| Fix | File | Lines | Type | Priority |
|-----|------|-------|------|----------|
| FIX-LR-1 | train_flowmatching.py | 8862, 8903-8920, 9289-9501 | ✅ DONE | - |
| FIX-PHASE4-1 | train_flowmatching.py | 8995-9001, 9289-9501 | ✅ DONE | - |
| **FIX-SELECTOR-1** | flow_matching_model.py | 6210 | Config | ⚡ Immediate |
| **FIX-ENV-1** | flow_matching_model.py | 5389-6205 | Loss | 🔴 Critical |
| **FIX-LOSS-1** | flow_matching_model.py | 5945-6205 | Loss | 🔴 Critical |
| FIX-SELECTOR-2 | flow_matching_model.py | 6107-6139 | Training | 🔧 Medium |
| FIX-THRESHOLD-1 | flow_matching_model.py | 6306 | Logic | 🔧 Small |

---

## Conclusion

**What we've fixed:**
- ✅ Learning rate schedule persistence (FIX-LR-1)
- ✅ Phase 4 freeze state persistence (FIX-PHASE4-1)

**What's blocking ADE improvement:**
1. Selector disabled (fixable, +10-15 km)
2. Environment loss missing (critical, +20-30 km)
3. Trajectory loss missing (critical, +20-40 km)
4. Selector t-mismatch (medium, +5-8 km)
5. Hard threshold mismatch (small, +3-5 km)

**Realistic outcome if all fixed:** ADE ~260-280 km

**To beat ST-Trans (172.68 km):** Would require ~90-100 km additional improvement, which needs architectural innovations beyond these 5 fixes (social integration, attention-based selection, or regime-aware modeling).

**Recommended next step:**
1. Deploy FIX-SELECTOR-1 today (10 min)
2. Fresh training from ep0
3. Measure phase 3+ improvement
4. Proceed to Phase B if gains realized
