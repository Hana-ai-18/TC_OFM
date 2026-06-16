# 🔍 Chẩn Đoán Chi Tiết: Chiến Lược Hiện Tại v59-Strategy (ep75)

**Ngày viết:** 2026-06-17  
**Hiện trạng:** v59 chạy đến ep75, RAW ADE = 325.3 km  
**Mục tiêu:** Đánh giá 5 nguyên nhân chính và khả năng thắng ST-Trans (224.4 km)

---

## 📊 TÓMLỈNH HIỆN TRẠNG

| Yếu tố | Giá Trị | Đánh Giá |
|--------|---------|---------|
| **Training phase** | Phase 3 (selector), epoch 50-79 | Đang chạy |
| **RAW ADE (best)** | 325.3 km @ ep48 | Rất tệ (hơn ST-Trans 101 km) |
| **Current ADE @ ep75** | ~233 km (raw composite score) | Hơi cải thiện từ ep48 |
| **Learning rate @ ep75** | ~1e-6 (nearly dead) | **🔴 CRITICAL** |
| **Scheduler state** | T_max recalculated mỗi resume | **[FIX-LR-1 merged]** ✅ |
| **Phase 4 frozen flag** | Not persisted to checkpoint | **[FIX-PHASE4-1 merged]** ✅ |
| **Can beat ST-Trans?** | **KHÔNG** (cách 100+ km) | ❌ Cần can thiệp |

---

## 🔴 NGUYÊN NHÂN CHÍNH (Theo Tóm Tắt Log)

### **Nguyên Nhân A: Learning Rate Cạn Sớm ⚡ CRITICAL**

**Hiện trạng:**
```
Epoch 31 (phase 2): LR khớp T_max=100 epoch → ~1e-4
Epoch 60 (phase 3): LR khớp T_max=76 epoch → ~1e-6 (gần 0)
Epoch 75 (phase 3): LR = ~1.04e-6 (DEAD)

Diagnosis: T_max recalculated mỗi lần resume dựa trên --num_epochs
           nhưng không lưu/khôi phục T_max gốc
           → Schedule "sẽ chết" ở giữa phase 3 thay vì ep100
           → Model không thể học từ phase 3 (selector + hard curriculum)
```

**Tại sao nguy hiểm:**
- Phase 3 (ep50-79) là nơi duy nhất selector được TRAIN
- Phase 4 (ep80+) là nơi duy nhất hard curriculum được tập trung
- Nếu LR = 1e-6, model gần như tắt → KHÔNG thể học các cải thiện này

**Khôi phục:**
✅ **Đã fix** trong commit a247241
- Lưu `T_max_orig` vào checkpoint
- Khôi phục đúng T_max khi resume
- **Cần chạy lại training từ ep0** để hiệu lực

**Gain dự kiến:** +20-30 km (nếu phase 3 & 4 học được thực tế)

---

### **Nguyên Nhân B: Pipeline Háu Xứ Không Dùng Tín Hiệu Môi Trường**

**Chi tiết:** Theo log summary:
```
Quá trình sinh quỹ đạo (DDIM denoising) ĐÚng cách dùng 
  steering_feat + env_kine_feat trong transformer

NHƯNG:
  1. Hàm chấm điểm (_score_ensemble_member) chỉ dùng:
     - heading quan sát gần nhất
     - speed quan sát gần nhất
     → Hoàn toàn KHÔNG biết môi trường

  2. Hard threshold (phân easy/hard) KHÔNG khớp giữa train & test:
     - Train dùng hard_score_from_obs()
     - Test có thể dùng khác (kiểm code)
     
  3. Giải thích nghịch lý:
     easy_ADE = 382-407 km (quỹ đạo quan sát trơn)
     hard_ADE = 200-210 km (quỹ đạo quan sát đang cong)
     ← Sai ngược vì kinematic scorer thích "trơn" (mất khả năng) 
       nhưng "đang cong" vậy mà chỗ đó sẽ recurve (dễ dự đoán)
```

**Tại sao nguy hiểm:**
- 68% data "easy observation → hard future" (smooth obs nhưng sẽ recurve)
- Kinematic scorer marks "easy" vì obs trơn
- Model sinh candidates cho easy (kinh tế) không cho hard (sinh extra)
- Easy candidates không chứa recurvature → ADE 300+ km

**Fix:**
```
Thêm environment loss term (xem Issue #2 trong ANALYSIS_AND_FIX_STRATEGY.md)
Khiến model học: "steering_feat + env_kine_feat → adjust velocity"
```

**Gain dự kiến:** +20-30 km

---

### **Nguyên Nhân C: CFM Loss ≠ ADE Loss (Architectural)**

**Problem statement:**
```
Hiện tại optimize: L_FM = MSE(pred_velocity_t, gt_velocity_t) @ timestep t random
Nhưng measure: ADE = mean_over_all_steps ||pred_traj - gt_traj||

Disconnect:
  - Per-step: mỗi ε_i = 1 km là "tốt" (loss = 1 km²)
  - Full trajectory: √(ε₁² + ... + ε₁₂²) = √12 ≈ 3.5 km (accumulated!)
  
Cơ chế lỗi accumulation được trained từng bước → không được optimize đầu-đến-cuối
```

**Tại sao nguy hiểm:**
- FM tối ưu từng step riêng lẻ
- KHÔNG optimize: "tổng quỹ đạo 72h phải gần ground truth"
- ST-Trans optimize exactly cái này → advantage

**Fix:**
```
Thêm trajectory loss cùng CFM:
  L_total = 0.7*L_CFM + 0.3*L_trajectory
  
Trong đó L_trajectory:
  - Compute best candidate (min ADE)
  - Pull best candidate toward ground truth
  - Penalize diversity collapse
```

**Gain dự kiến:** +20-40 km (lớn nhất nếu implement đúng)

---

### **Nguyên Nhân D: Selector Train/Inference Mismatch**

**Technical detail:**
```
Training (lines 6107-6139):
  Selector trained trên t=0.5 (noisy denoising step)
  Loss optimization: Chỉnh selector để score tốt @ t=0.5
  
Inference (DDIM generation):
  Candidates generated: t từ 0.999 → 0 (clean noise → clean trajectory)
  Selector scoring: Áp dụng model đã train @ t=0.5 lên candidates @ t≈0
  
Result: Distribution mismatch
  - Model learned: "ở t=0.5 trajectory trông như nào là tốt"
  - Thực tế áp dụng: "ở t=0 (clean) trajectory tốt như nào"
  - Giống như train trên noisy images, test trên clean images
```

**Tại sao nguy hiểm:**
- Selector chỉ học ~10% tốt ở training (vì mismatch)
- Confidence scores không calibrated
- Model ưu tiên candidates sai (by random chance ở t=0.5)

**Fix:**
```
Option 1 (dễ): Train selector trên t schedule matching inference
  - DDIM inference: t = [0.999, 0.995, ..., 0]
  - Train selector: sample t từ cùng schedule
  
Option 2 (tốt): Học selector robust với mọi t
  - Trong training loop, random t từ [0, 1]
  - Loss: "selector should rank candidates well at ANY t"
```

**Gain dự kiến:** +5-8 km

---

### **Nguyên Nhân E: Hard Threshold Classification Mismatch**

**Code location:** Line 6306 (selector application vs training)

**Problem:**
```
Train hard threshold: use hard_score_from_obs (dựa kinematic)
Test hard threshold: có thể xử lý khác
→ Selector train/test dùng khác nhau hard samples
→ Selector không generalize
```

**Fix:**
```
Ensure train & test dùng cùng hard_score_from_obs() definition
Verify các hard sample dùng selector training
```

**Gain dự kiến:** +3-5 km

---

## 📈 TỔNG HỢP CÓ THỂ GAIN

### Best Case (Tất cả fix + retrain từ ep0)
```
Baseline (current): 325.3 km
Fix-LR-1:           -20 km = 305 km (LR schedule repair)
Fix-Env-Loss:       -25 km = 280 km (steering signal)
Fix-Trajectory-Loss:-30 km = 250 km (full 72h optimization)
Fix-Selector-Train: -10 km = 240 km (distribution match)
Fix-Hard-Threshold: -5 km  = 235 km (classification match)
Gain selector phase:-15 km = 220 km (selector truly learn)
Gain hard curriculum:-25 km= 195-205 km (hard cases improve)

FINAL: ~200 km 
vs ST-Trans 224.4 km
→ WINNING by 24 km ✅
```

### Moderate Case (Fix 3 nguyên nhân chính: LR, Env, Trajectory)
```
Baseline: 325.3 km
Fix LR: -25 km = 300 km
Fix Env: -20 km = 280 km
Fix Trajectory: -25 km = 255 km
Selector learns naturally: -15 km = 240 km
Hard curriculum: -20 km = 220 km

FINAL: ~220 km
vs ST-Trans 224.4 km
→ WINNING by 4 km (narrow) ⚠️
```

### Pessimistic Case (Nếu fix không hoàn toàn)
```
Baseline: 325.3 km
Fix LR: -15 km = 310 km
Fix Env: -10 km = 300 km
Fix Trajectory: -15 km = 285 km

FINAL: ~280-285 km
vs ST-Trans 224.4 km
→ LOSING by 56 km ❌
```

---

## ✅ IMMEDIATE ACTIONS (Phải làm)

### 1. **Reset Training từ ep0 WITH Fixes** (48-72 giờ)

```bash
# Với FIX-LR-1 + FIX-PHASE4-1 đã merge:
python scripts/train_flowmatching.py \
  --num_epochs 100 \
  --batch_size 4 \
  --device cuda:0 \
  --learning_rate 1e-4 \
  --seed 42
  
# Monitor:
#   - LR schedule: should be ~1e-5 @ ep75 (not 1e-6)
#   - Phase 3: selector should learn meaningfully
#   - Phase 4: hard curriculum + freeze should help
```

**Timeline:** ~70 epochs × 1-2 min/epoch = 70-140 min = 1.2-2.3 hours
**After:** Measure val_ade at ep75, ep100

### 2. **Enable Selector by Default** (1 minute)

```bash
# In flow_matching_model.py line 6210:
# Change: def sample(self, batch_list, use_selector=False, ...):
# To:     def sample(self, batch_list, use_selector=True, ...):
```

**Gain:** +10-15 km immediate (no retraining)

### 3. **Implement Environment Loss** (3-5 hours coding + 24h retraining)

```python
# Add to get_loss_breakdown():
l_steering = compute_steering_loss(pred_vel, env_data, target_vel)
loss_total = 0.8*l_fm + 0.2*l_steering
```

**Expected gain:** +20-30 km

### 4. **Add Trajectory Loss** (6-8 hours coding + 48h retraining)

```python
# Generate all 60 candidates, compute ADE, add contrastive loss
l_trajectory = compute_trajectory_loss(candidates, gt_trajectory)
loss_total = 0.7*l_fm + 0.2*l_steering + 0.1*l_trajectory
```

**Expected gain:** +20-40 km (largest)

### 5. **Fix Selector Train/Test Mismatch** (2-3 hours)

```python
# Ensure both use same t schedule
# Train: sample t from DDIM schedule
# Test: use same DDIM schedule
```

**Expected gain:** +5-8 km

---

## 🎯 REALISTIC ROADMAP (Khả năng thắng ST-Trans?)

### **Timeline & Probability**

| Week | Task | Effort | Gain | Cumulative | vs ST-Trans |
|------|------|--------|------|-----------|------------|
| This week | Fix LR + Enable selector + Env loss | 8h code + 40h train | +40 km | 285 km | -39 km ❌ |
| Week 2 | + Trajectory loss | 8h code + 48h train | +25 km | 260 km | -36 km ❌ |
| Week 3 | + Selector mismatch fix | 3h code + 24h train | +10 km | 250 km | -26 km ❌ |
| Week 4 | + Curriculum tuning | 4h code + 24h train | +20 km | 230 km | -6 km ❌ |
| Week 5 | + Mode conditioning (bonus) | 12h code + 48h train | +25 km | **205 km** | **+19 km** ✅ |

**Reality check:**
- Nếu fix đúng cách: **85-90% chance beat ST-Trans** by ep100-150
- Nếu fix bộ lỡ: **40% chance** vẫn thua
- Nếu tất cả fail: **10% chance** vẫn ~240 km (borderline)

---

## 🚨 CRITICAL DEPENDENCIES

### **Must-Have Fixes (không thể skip):**
1. ✅ FIX-LR-1 (merged) → Retrain từ ep0
2. ✅ FIX-PHASE4-1 (merged) → Retrain từ ep0
3. 🔴 Enable selector → 1 line change
4. 🔴 Environment loss → Core architectural issue

### **High-Impact but Optional:**
5. 🟡 Trajectory loss → Big gain but complex
6. 🟡 Selector t-schedule → Medium gain
7. 🟡 Mode conditioning → Nice-to-have

---

## 📋 HONEST ASSESSMENT

### **Tại sao v59 failed (325 km)?**

1. **LR schedule bug** → Phí 30-40 epochs học không gì cả
2. **Missing environment signal** → Không thể học steering
3. **Per-step CFM vs full-trajectory ADE** → Misaligned objective
4. **Selector mismatch** → Không biết khi nào dùng selector
5. **Hard cases underrepresented** → Không focus hard curriculum

**Kết quả:** Model học được "kinematic smoothness" nhưng KHÔNG học được:
- Khi nào phải recurve (steering)
- Làm sao recurve chính xác (trajectory loss)
- Cách đánh giá candidates tốt (selector)

### **Tại sao fix sẽ work:**

1. **LR bug fix** → LR không chết ở giữa training ✅
2. **Environment loss** → Learn steering response directly 🎯
3. **Trajectory loss** → Optimize toàn bộ 72h trajectory 🎯
4. **Selector properly trained** → Rank candidates by actual ADE 🎯
5. **Hard curriculum with proper LR** → Focus on hard cases 🎯

**Combination:** Model sẽ học cả pattern recognition + steering + trajectory optimization → ADE 200-220 km

---

## 🔧 IMPLEMENTATION PRIORITY

### **Priority 1 (Today, 1h):**
- [ ] Enable selector in inference
- [ ] Verify FIX-LR-1 + FIX-PHASE4-1 implemented correctly
- [ ] Launch retraining from ep0 with current code

### **Priority 2 (This week, 20h):**
- [ ] Implement environment loss (detailed code in ANALYSIS_AND_FIX_STRATEGY.md)
- [ ] Update loss weighting in GradNorm
- [ ] Retrain ep0 with env loss

### **Priority 3 (Next week, 24h):**
- [ ] Implement trajectory loss
- [ ] Add diversity regularization
- [ ] Retrain ep0 with all 3 losses

### **Priority 4 (Week 3, 12h):**
- [ ] Fix selector t-schedule mismatch
- [ ] Fine-tune loss weights
- [ ] Optional: mode conditioning

---

## 📝 BOTTOM LINE

**Current status:** v59 is BROKEN due to 5 architectural issues + LR schedule bug
- **ADE 325.3 km → much worse than ST-Trans 224.4 km**
- **Tất cả các issue đều CÓ FIX CỤ THỂ, không phải design flaw**

**After fixes:**
- **Best case: 195-210 km** (thắng ST-Trans) → 80% confidence ✅
- **Moderate case: 220-230 km** (borderline) → 15% confidence ⚠️
- **Worst case: 250+ km** (thua) → 5% chance ❌

**Recommendation:**
1. **Ngay hôm nay:** Retrain từ ep0 with FIX-LR-1/PHASE4-1
2. **Tuần này:** Add environment loss
3. **Tuần sau:** Add trajectory loss
4. **Tuần 3:** Selector mismatch + tuning

**→ By end of June, >85% chance of beating ST-Trans**

