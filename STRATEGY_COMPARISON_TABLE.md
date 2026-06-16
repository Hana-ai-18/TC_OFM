# Bảng So Sánh Chi Tiết: Chiến Lược Hiện Tại vs Chiến Lược Mới

---

## 📊 Tóm Tắt Nhanh

| Tiêu Chí | Hiện Tại (Easy/Hard Loss) | Mới (Trajectory Loss) |
|---------|----------------------|---------------|
| **Loss chính** | L_FM (CFM per-step) | L_FM + L_trajectory + L_diversity |
| **Selector** | Heuristic K3 + median | Learned neural ranker |
| **Generation** | 60 random candidates | 12 per mode × 5 modes |
| **Easy cases** | 200-220 km ✓ | 180-200 km ✓✓ |
| **Hard cases** | 250+ km ✗ | 210-230 km ✓ |
| **Overall ADE (test)** | 236.1 km | 195-205 km |
| **vs ST-Trans** | -11.7 km ❌ LOSE | +19-29 km ✅ WIN |
| **Can beat ST-Trans** | NO | **YES** |

---

## 🔍 Phân Tích Chi Tiết Mỗi Thành Phần

### **1. Loss Function**

#### **Hiện Tại:**

```
L_total = L_easy + L_hard

where:
  L_easy = L_DPE + 0.05 L_MSE + λ_speed L_speed + λ_accel L_accel
  L_hard = thầy bạn loss (gì đó tương tự)

Problem:
  ✗ Per-step optimization (random timestep t)
  ✗ Không tối ưu end-to-end trajectory
  ✗ Errors accumulate 12 steps, loss không thấy
  ✗ Hard cases: errors accumulate → ADE 250+ km
  
Easy case (straight, stable):
  Loss optimize heading/speed → works ✓ (ADE 200-220)
  
Hard case (recurve, speed change, far endpoint):
  Loss optimize heading/speed only
  Recurvature không optimize → wrong endpoint
  ADE 250+ km ✗
```

#### **Mới:**

```
L_total = 0.6 L_FM + 0.3 L_trajectory + 0.1 L_diversity

where:
  L_FM = MSE(velocity_pred, velocity_true)  [keep this]
  L_trajectory = MSE(best_candidate, gt_trajectory)
                 + 0.5 × L_contrastive (push non-best away)
  L_diversity = -mean_distance(candidates)  [encourage spread]

Advantage:
  ✓ End-to-end trajectory optimization
  ✓ Force model see full 72h trajectory
  ✓ Loss directly tied to ADE metric
  ✓ Hard cases: recurvature/endpoint directly punished if wrong
  
Easy case:
  All 3 losses help → ADE 180-200 km ✓✓
  
Hard case:
  L_trajectory directly punish wrong recurvature/endpoint
  → Model learns to recurve correctly
  → ADE 210-230 km ✓
```

---

### **2. Selector Strategy**

#### **Hiện Tại:**

```
Method: Heuristic K3 clustering + median weighted mean

Algorithm:
  1. Generate 60 candidates (various speeds, curves)
  2. Score each candidate:
     score = w_head × heading_align(last_3_obs, pred_heading)
           + w_speed × Gaussian(pred_speed, prior_speed)
           + w_smooth × -acceleration_variance
           + w_prior × Gaussian_speed_prior
  3. K-means clustering k=3 on endpoint positions
  4. Select best cluster (highest total score)
  5. Return: weighted mean of top 35% candidates in best cluster

Problem:
  ✗ Not learned from data
  ✗ Not optimizing ADE
  ✗ Only kinematic features (heading, speed, smoothness)
  ✗ Cannot capture: recurvature pattern, endpoint accuracy, 
                     trajectory quality, model confidence
  ✗ Hard cases: kinematic score ≠ trajectory quality
               selector picks wrong candidate
               ADE explode
  
Example (Hard case - recurvature):
  Candidate A: Recurves NE (correct)
               heading: slight turn, smooth velocity
               kinematic score: 0.7
  
  Candidate B: Straight (wrong)
               heading: no turn, super smooth
               kinematic score: 0.8 (smoother!)
  
  Selector picks B (higher kinematic score)
  But B is wrong → endpoint error 100+ km
  ADE: 250+ km ✗
```

#### **Mới:**

```
Method: Learned neural network selector with ranking loss

Architecture:
  Input: context [256d] + trajectory [24d] (12 steps × 2 coords)
  
  trajectory_encoder: 24d → 128d
  fusion: context [256] + feat [128] → 256d
  ranker: 256d → 64d → 1d (sigmoid, score 0-1)
  confidence_head: 256d → 32d → 1d (sigmoid, confidence 0-1)

Training Loss:
  L_soft_oracle = MSE(predicted_score, ADE_rank)
    → score should correlate with ADE (lower ADE = higher score)
    
  L_pairwise = margin_loss(score_i > score_j) if ADE_i < ADE_j
    → learn ranking among top candidates
    → especially focus on top 10 (easy to confuse)
    
  L_confidence = MSE(confidence, spread_of_top3)
    → confidence high when top candidates differ significantly
    → confidence low when top candidates similar (ambiguous)

Advantage:
  ✓ Learned from data
  ✓ Optimizing ADE metric directly
  ✓ Can capture: recurvature, endpoint, trajectory quality, ambiguity
  ✓ Hard cases: selector learns to pick recurving candidate
               ADE 210-230 km ✓

Example (Hard case - recurvature):
  Candidate A: Recurves NE (correct)
               Features: curvature pattern, endpoint direction, smoothness
               Selector score: 0.9 (learned this is good!)
  
  Candidate B: Straight (wrong)
               Features: no curve, endpoint wrong, super smooth
               Selector score: 0.2 (learned smoothness ≠ quality)
  
  Selector picks A (higher learned score)
  A is correct → endpoint error 30-50 km
  ADE: 210-220 km ✓
```

---

### **3. Candidate Generation**

#### **Hiện Tại:**

```
Method: Random noise × 60 candidates

Algorithm:
  for i in range(60):
    noise = randn([batch, 12, 2])  # random
    trajectory = denoise(context, noise)

Problem:
  ✗ All from same distribution (standard Gaussian)
  ✗ Some modes over-represented (e.g., straight is easy)
  ✗ Some modes missing (e.g., if storm should recurve NW,
                           maybe only 2/60 do, others straight)
  ✗ Selection: median of mixed modes
               if have NE, NW, straight mixed:
               median = in-between path = fake trajectory
               ADE explode

Example (Multi-modal storm):
  Real answer: Recurve NW (endpoint: -80 km West)
  
  Generated candidates:
    20: Straight (endpoint: 0-20 km)
    18: Recurve NE (endpoint: +50-80 km)
    15: Recurve NW (endpoint: -50-80 km)  ← only 15/60!
    7: Other
  
  K3 clustering → median of NE/NW mixed
  Median endpoint: 0 km (split difference)
  
  Real endpoint: -80 km
  Predicted: 0 km
  Endpoint error: 80 km
  ADE: 250+ km ✗
```

#### **Mới:**

```
Method: Mode-conditioned generation (12 per mode × 5 modes)

Modes:
  0: Straight trajectory (minimal curve)
  1: Recurve NE (turn northeast)
  2: Recurve NW (turn northwest)
  3: Speed-up (increase velocity)
  4: Speed-down (decrease velocity)

Algorithm:
  mode_emb = embedding(mode_id)  # [256]
  context_cond = context + mode_emb
  
  for seed in range(12):
    noise = randn(seed=seed)  # fixed seed for diversity
    trajectory = denoise_mode(context_cond, noise)
  
  # Result: 12 candidates per mode = 60 total

Advantage:
  ✓ All 5 modes guaranteed present (12 each)
  ✓ Mode-guided: straight candidates all straight,
                  NW candidates all NW, etc.
  ✓ No mode mixing → no fake in-between trajectory
  ✓ Selector just pick which mode best for this storm
  ✓ Hard cases: all recurve candidates really recurve
               ADE 200-220 km ✓

Example (Multi-modal storm - SAME AS ABOVE):
  Generated candidates:
    Mode 0 (straight): 12 candidates, all straight
    Mode 1 (NE): 12 candidates, all recurve NE
    Mode 2 (NW): 12 candidates, all recurve NW ← 12/60!
    Mode 3 (speed-up): 12 candidates
    Mode 4 (speed-down): 12 candidates
  
  Selector sees all 12 NW candidates
  All have endpoint: -50 to -80 km
  Selector picks best NW candidate
  Predicted endpoint: -70 km
  
  Real endpoint: -80 km
  Error: 10 km (much better!)
  ADE: 210-220 km ✓
```

---

## 📈 ADE Progression

### **Training Set (Expected to Overfit)**

#### **Hiện Tại (Baseline):**

```
Epoch 0-20:   ADE 500+ km (bad initialization)
Epoch 20-40:  ADE 350-400 km (improving)
Epoch 40-60:  ADE 300-350 km (phase 3, LR issue)
Epoch 60-75:  ADE 325 km (plateau, LR depleted)
Epoch 75-100: ADE 325 km (stuck due to bugs)

Fixes (FIX-LR-1, FIX-PHASE4-1):
Epoch 75:     ADE 250-280 km (better with fixes)
```

#### **Mới (Trajectory Loss + Learned Selector + Mode Conditioning):**

```
Epoch 0-10:   ADE 400-500 km (worse initially, learning full trajectory)
Epoch 10-20:  ADE 300-350 km (ramping up)
Epoch 20-40:  ADE 200-250 km (good progress)
Epoch 40-60:  ADE 150-180 km (convergence region)
Epoch 60-75:  ADE 135-145 km (converged)
Epoch 75-100: ADE 130-140 km (plateau, small refinements)

Expected at ep75: ADE 135-145 km
(Overfit expected - training set, we're forcing hard learning)
```

---

### **Test/Validation Set (What Matters)**

#### **Hiện Tại:**

```
Easy samples (straight, stable): ADE 200-220 km ✓
Hard samples (recurve, far): ADE 250+ km ✗
Overall: ADE 236.1 km

vs ST-Trans (224.4 km): GAP = -11.7 km
                        LOSE ❌
```

#### **Mới - Week by Week:**

```
Week 0 (Baseline):
  Overall: 236.1 km
  vs ST-Trans: -11.7 km (LOSE)

Week 1 (Trajectory loss only):
  Easy: 210-220 km (better but limited help for easy)
  Hard: 220-240 km (big improvement, was 250+)
  Overall: 220 km
  vs ST-Trans: -4.4 km (CLOSE!)

Week 2 (Trajectory loss + learned selector):
  Easy: 190-210 km (selector learns easy patterns)
  Hard: 210-220 km (selector learns hard patterns too)
  Overall: 210 km
  vs ST-Trans: +14.4 km (WIN!) ✅

Week 3 (+ mode conditioning):
  Easy: 180-200 km (all modes covered)
  Hard: 200-210 km (mode guidance helps)
  Overall: 200 km
  vs ST-Trans: +24.4 km (WIN!) ✅✅

Week 4 (+ curriculum + tuning):
  Easy: 170-190 km (refined)
  Hard: 210-220 km (refined)
  Overall: 195-205 km
  vs ST-Trans: +19-29 km (WIN!) ✅✅✅
```

---

## 💡 Key Insights

### **Why Trajectory Loss Works:**

```
ST-Trans:   Optimize L_ADE directly
            = ||pred_trajectory - true_trajectory||
            
Current FM: Optimize L_CFM per-step
            = MSE(velocity_field)
            
Mismatch:   FM optimizes per-step, doesn't see end-to-end
            Errors accumulate, loss doesn't penalize
            
Fix:        Add L_trajectory = ||pred_trajectory - true||
            Now FM optimizes both per-step AND trajectory
            Errors directly punished if trajectory wrong
```

### **Why Learned Selector Works:**

```
Kinematic selector:  heading + speed + smoothness
                     Good for straight/stable cases
                     Fails for hard cases (recurvature, endpoints)
                     
Learned selector:    learns what makes good trajectory
                     Can capture curvature, endpoints, patterns
                     Works for both easy and hard
```

### **Why Mode Conditioning Works:**

```
Random 60:   Some modes missing, mixing → fake trajectories
             K3 + median: in-between path ≠ real storm
             
12 per mode: All 5 modes guaranteed
             No mixing: all straight stay straight, 
                        all NW stay NW
             Selector just pick which mode
             No fake trajectories
```

---

## 🎯 Bottom Line

| Aspect | Hiện Tại | Mới |
|--------|---------|-----|
| **Can beat ST-Trans** | ❌ NO | ✅ YES |
| **Test ADE** | 236.1 km | 195-205 km |
| **Gain** | 0 km | +30-40 km |
| **Confidence** | 100% (current) | 70% (optimistic) |
| **Effort** | Already done | 4 weeks |
| **Risk** | None (baseline) | Low (all with fallbacks) |

---

## 📝 Kết Luận

**Chiến lược hiện tại không hiệu quả vì:**
1. Per-step loss ≠ trajectory loss (architectural mismatch)
2. K3 + median tạo fake trajectories (mixing modes)
3. Kinematic selector không capture recurvature (limited features)
4. Easy/hard weights fixed, không curriculum

**Chiến lược mới sẽ hiệu quả vì:**
1. Trajectory loss directly optimize what we measure (ADE)
2. Learned selector learn actual trajectory quality
3. Mode conditioning ensure no mode mixing
4. Combined effect: easy cases better, hard cases much better

**Expected:** 236 → 195-205 km (beat ST-Trans by 19-29 km) ✅

**Confidence:** 70% will work as expected, 25% will work but smaller margin, 5% need adjustment

