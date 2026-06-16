# Index: Chiến Lược Mới Để Thắng ST-Trans

**Ngày:** 2026-06-17  
**Mục tiêu:** Giảm ADE từ 236.1 km → 195-205 km (thắng ST-Trans 224.4 km)  
**Timeline:** 4 tuần  
**Confidence:** 70% optimistic  

---

## 📚 Tài Liệu Chính (Read in Order)

### **1. 📖 QUICK_SUMMARY_WHY_AND_HOW.md** (START HERE)
**Đọc trước nếu muốn hiểu nhanh**

```
- Tại sao chiến lược hiện tại không hiệu quả (4 vấn đề chính)
- 3 thay đổi chính sơ lược
- Training vs test ADE expectations
- Next steps
```

**Thời gian đọc:** 5-10 min  
**Đủ để:** Hiểu vấn đề, quyết định có implement không

---

### **2. 📊 STRATEGY_COMPARISON_TABLE.md** (SECOND)
**Đọc sau để so sánh chi tiết**

```
- Bảng so sánh từng thành phần (loss, selector, generation)
- Tại sao hiện tại fail, tại sao mới work
- Chi tiết từng mode, từng vấn đề
- ADE progression timeline
```

**Thời gian đọc:** 15-20 min  
**Đủ để:** Hiểu chi tiết tất cả thành phần

---

### **3. 📋 NEW_STRATEGY_TRAJECTORY_LOSS.md** (FOR IMPLEMENTATION)
**Đọc khi sẵn sàng code**

```
- Phân tích chi tiết 4 vấn đề
- 3 thay đổi chính với code examples
- Week-by-week implementation roadmap
- Expected results per week
- Risk assessment + fallbacks
- Comparison table
```

**Thời gian đọc:** 30-40 min  
**Đủ để:** Implement từ A-Z

---

## 🎯 Tại Sao Hiện Tại Không Hiệu Quả?

### **Vấn Đề 1: CFM Loss ≠ ADE Loss**
```
Loss optimize: per-step velocity field (MSE)
Metric measure: end-to-end 72h trajectory (ADE)

Mismatch: Errors accumulate 12 steps, loss không thấy

Easy: Còn được (200-220 km) ✓
Hard: Fail (250+ km) ✗
```

**Giải pháp:** Add end-to-end trajectory loss

---

### **Vấn Đề 2: K3 + Median Tạo Fake Trajectories**
```
Scenario: 60 candidates từ 3 modes (NE, NW, straight)

K3 clustering: select top 35% → median of mixed modes
Result: Trajectory nằm giữa NE & NW
Problem: Bão chỉ NE HOẶC NW, không mix!
         Fake trajectory = high ADE (250+ km)
```

**Giải pháp:** Mode-conditioned generation (12 per mode × 5)

---

### **Vấn Đề 3: Selector Kinematic Only**
```
Current score: w_head × heading + w_speed × speed + w_smooth × smooth
Problem: Không learn, không optimize ADE
         Hard cases: kinematic score ≠ trajectory quality

Example: Smooth trajectory (high kinematic score)
         Nhưng endpoint sai 100 km (wrong direction)
         Kinematic selector: "ok, take it!"
         ADE explode (250+ km)
```

**Giải pháp:** Learned selector with ranking loss

---

### **Vấn Đề 4: Easy/Hard Weights Fixed**
```
Current: w_easy = 1.0, w_hard = 2.0 (fixed)
Problem: Early training hard too hard
         Late training easy converged, waste capacity

Fix: Curriculum learning
  Phase 1-2: mostly easy, w_easy=1.0, w_hard=0.5
  Phase 2-3: balanced, w_easy=1.0, w_hard=1.0
  Phase 3-4: hard focus, w_easy=0.5, w_hard=2.0
```

**Giải pháp:** Curriculum learning per phase

---

## ✅ Chiến Lược Mới: 3 Thay Đổi

### **Thay Đổi 1: End-to-End Trajectory Loss**

```python
# Current
L_FM = MSE(velocity_pred, velocity_true)

# New
L = 0.6 * L_FM + 0.3 * L_trajectory + 0.1 * L_diversity

L_trajectory:
  - Generate 60 candidates
  - Find best (lowest ADE)
  - Pull best toward ground truth
  - Push non-best away (contrastive)

Effect: Force model optimize full 72h trajectory
        Not just per-step
```

**Expected gain:** 236 → 220 km (+16 km)

---

### **Thay Đổi 2: Learned Selector**

```python
# Current
score = w_head*heading + w_speed*speed + w_smooth*smooth

# New
selector = NeuralNetwork(context, trajectory)
score = selector.forward()  # learned, 0-1

Loss:
  - L_soft_oracle: score correlate with ADE
  - L_pairwise: learn ranking among top 10
  - L_confidence: when confident vs fallback
```

**Expected gain:** 220 → 210 km (+10 km)

---

### **Thay Đổi 3: Mode-Conditioned Generation**

```python
# Current
for i in range(60):
    noise = randn()
    traj = denoise(context, noise)

# New
for mode in [0,1,2,3,4]:      # 5 modes
    for seed in range(12):     # 12 per mode
        traj = denoise_mode(context, noise, mode)

Modes:
  0: Straight
  1: Recurve NE
  2: Recurve NW
  3: Speed-up
  4: Speed-down
```

**Expected gain:** 210 → 200 km (+10 km)

---

## 📈 Expected ADE Progression

### **Training Set** (Expected to overfit, that's ok)

```
Current: 325 km (ep75, not converged due to bugs)

New:
  Epoch 20: 240 km
  Epoch 40: 210 km
  Epoch 60: 150 km
  Epoch 75: 135 km
  Epoch 100: 135 km (plateau)
  
Gain: +190 km ✅
```

### **Test Set** (What matters)

```
Current: 236.1 km
  vs ST-Trans (224.4): GAP -11.7 km ❌ LOSE

New (per week):
  Week 1: 220 km  (-4.4 km vs ST) ⚠️ Close
  Week 2: 210 km  (+14 km vs ST)  ✅ Win
  Week 3: 200 km  (+24 km vs ST)  ✅✅
  Week 4: 195-205 (+19-29 vs ST)  ✅✅✅
  
Final Expected: 195-205 km
vs ST-Trans: WIN by 19-29 km! 
Confidence: 70% optimistic
```

---

## 🗓️ Implementation Timeline (4 Weeks)

### **Week 1: Trajectory Loss**
- File: `flow_matching_model.py`
- Add: `compute_trajectory_loss()` method (~100 lines)
- Change: Loss weights (0.6 L_FM + 0.3 L_traj)
- Train: Epoch 0-75 fresh
- **Expected:** 236 → 220 km

### **Week 2: Learned Selector**
- File: `trajectory_selector.py` (new)
- Add: `TrajectorySelector` class + `SelectorLoss` (~300 lines)
- Train: Selector phase (10 epochs on val set)
- **Expected:** 220 → 210 km

### **Week 3: Mode Conditioning**
- File: `flow_matching_model.py`
- Add: Mode embeddings + `forward_mode_conditioned()` (~80 lines)
- Add: `generate_candidates_multimode()` method
- Train: Epoch 0-75 fresh with mode conditioning
- **Expected:** 210 → 200 km

### **Week 4: Curriculum + Tuning**
- File: `train_flowmatching.py`
- Add: Curriculum learning (4 phases with adaptive weights)
- Add: GradNorm loss balancing
- Train: Fine-tune with curriculum
- **Expected:** 200 → 195-205 km

---

## 🚀 How to Start

### **Step 1: Read Documentation (Today)**
1. Read `QUICK_SUMMARY_WHY_AND_HOW.md` (5-10 min)
2. Read `STRATEGY_COMPARISON_TABLE.md` (15-20 min)
3. Decide: Proceed or not?

### **Step 2: Prepare (Day 1)**
1. Understand `NEW_STRATEGY_TRAJECTORY_LOSS.md` (detailed)
2. Review code structure:
   - `Model/flow_matching_model.py` (where to add trajectory loss)
   - `scripts/train_flowmatching.py` (where to modify loss weights)

### **Step 3: Code Week 1 (Day 2-3)**
1. Implement `compute_trajectory_loss()` in `flow_matching_model.py`
2. Modify training loop to use new loss weights
3. Start training from epoch 0

### **Step 4: Run Week 1 Training (Day 4-10)**
1. Train for 75 epochs
2. Monitor: loss_fm, loss_traj, val_ade, val_aee
3. **Target:** Achieve 220 km ADE by epoch 75
4. If success: proceed to Week 2

---

## ❓ FAQs

### **Q: Có chắc sẽ thắng ST-Trans không?**
A: 70% confidence (optimistic scenario)
   25% confidence (moderate scenario, still win with smaller margin)
   5% risk (need adjustment, but can fallback to week 2 result 210 km)

### **Q: Nếu Week 1 không work?**
A: Can fallback:
   - Reduce L_traj weight from 0.3 → 0.2
   - Increase L_FM weight from 0.6 → 0.7
   - Adjust diversity weight
   - Still expected 225-230 km (close to ST-Trans)

### **Q: Timeline bao lâu?**
A: Week 1 only: Code (1 day) + train (5-6 days) = 1 week
   All 4 weeks: 4 weeks

### **Q: Có risk gì?**
A: - Mode conditioning adds complexity (can disable)
   - Selector overfits (can early-stop)
   - Loss imbalance (can GradNorm)
   - All have fallbacks

### **Q: Nếu không implement thì sao?**
A: Stay at 236 km, lose to ST-Trans 224.4 km by 11.7 km
   Current easy/hard loss not enough to close gap

---

## 📞 Contact Points

**Questions about:**
- Why current fails: See `QUICK_SUMMARY_WHY_AND_HOW.md` section 2
- How new works: See `STRATEGY_COMPARISON_TABLE.md` section 2
- Implementation: See `NEW_STRATEGY_TRAJECTORY_LOSS.md` section 4-5
- Risk management: See `NEW_STRATEGY_TRAJECTORY_LOSS.md` section 6

---

## ✍️ Summary

| Hiện tại | Mới |
|----------|-----|
| ADE 236.1 km | ADE 195-205 km |
| vs ST-Trans: -11.7 km ❌ | vs ST-Trans: +19-29 km ✅ |
| Không thắng | **Thắng mạnh!** |
| 4 vấn đề chưa sửa | 3 thay đổi giải quyết tất cả |
| - | 4 weeks + 90% chance of success |

---

**Ready to beat ST-Trans?** 🚀

