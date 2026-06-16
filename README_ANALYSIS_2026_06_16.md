# TC-FlowMatching Analysis — June 16, 2026

## 📋 Documentation Index

This analysis contains everything you need to understand the current state and fix strategy.

### Core Analysis (START HERE)
**→ [`COMPLETE_ANALYSIS_AND_FIX_STRATEGY.md`](COMPLETE_ANALYSIS_AND_FIX_STRATEGY.md)** (Main document)
- Executive summary of all issues
- Part 1: Current architecture & what's wrong
- Part 2: 5 critical issues with code locations
- Part 3: **Detailed fix strategy for flow_matching.py** ⚠️ IMPORTANT
- Part 4: Implementation roadmap
- Part 5-7: Why this works, expected outcomes

**Read this first.** Everything else is supplementary.

### Supporting Documents (Reference)

1. **[FIXES_LR_PHASE4.md](FIXES_LR_PHASE4.md)** — FIX-LR-1 & FIX-PHASE4-1 (already deployed ✅)
   - Detailed explanation of learning rate & phase 4 fixes
   - Why they matter for resume stability

2. **[QUICK_REFERENCE.txt](QUICK_REFERENCE.txt)** — One-page visual summary
   - Problem overview with ASCII diagram
   - Immediate actions
   - Expected outcomes by phase

3. **[NEXT_ACTIONS_PRIORITY.md](NEXT_ACTIONS_PRIORITY.md)** — Implementation details
   - Concrete code changes for each fix
   - Line numbers & context
   - Go/no-go decision criteria

4. **[ANALYSIS_SUMMARY_2026_06_16.md](ANALYSIS_SUMMARY_2026_06_16.md)** — Full report
   - Training trajectory analysis
   - Root cause breakdown
   - Dependency graph between fixes

---

## 🎯 Quick Start

### What's Fixed ✅
- **FIX-LR-1**: Learning rate schedule persistence (commit a247241)
- **FIX-PHASE4-1**: Phase 4 freeze state persistence (commit a247241)

### What Needs Fixing 🔴
| Priority | Issue | File | Lines | Impact |
|----------|-------|------|-------|--------|
| ⚡ Quick | Enable selector | flow_matching.py | 6210 | +10-15 km |
| 🔴 Critical | Add steering loss | flow_matching.py | 5389-6205 | +20-30 km |
| 🔴 Critical | Trajectory loss | flow_matching.py | 5945-6205 | +20-40 km |
| 🔧 Medium | Retrain selector | flow_matching.py | 6107-6139 | +5-8 km |
| 🔧 Small | Hard threshold | flow_matching.py | 6306 | +3-5 km |

### Next Step
1. Read [`COMPLETE_ANALYSIS_AND_FIX_STRATEGY.md`](COMPLETE_ANALYSIS_AND_FIX_STRATEGY.md) Part 3 (flow_matching fix strategy)
2. Implement FIX-SELECTOR-1 (10 min, 1 line)
3. Start fresh training from ep0

---

## 📊 Current Status
- **Epoch**: 75 / 100
- **ADE**: 325.3 km
- **Target**: 172.68 km (ST-Trans)
- **Gap**: 152.6 km (47% above)

### Why Plateau at ep75
1. ❌ LR depleted (FIX-LR-1 addresses) ✅ Deployed
2. ❌ Selector disabled (FIX-SELECTOR-1 addresses)
3. ❌ Environment loss missing (FIX-ENV-1 addresses)
4. ❌ Trajectory loss missing (FIX-LOSS-1 addresses)

---

## 🔧 Architecture Issues (In Order of Priority)

### Issue 1: Selector Disabled (10 min fix)
```python
# File: Model/flow_matching_model.py, line 6210
# Change: use_selector=False → use_selector=True
```
Selector is trained but never used. K3 clustering ignores environment.
**Gain:** +10-15 km

### Issue 2: Environment Loss Missing (CRITICAL)
Model receives steering features but no loss term says "steering matters."
→ Learns to ignore environment
→ Can't handle "easy observation → hard future" recurve problem
**Gain:** +20-30 km

### Issue 3: CFM Loss vs ADE Mismatch (CRITICAL)
Loss optimizes per-step denoising, metric measures trajectory quality.
→ Model good at denoising (val_loss=2.4) but bad at forecasting (ADE=325)
→ Errors accumulate in DDIM inference
**Gain:** +20-40 km

### Issue 4: Selector Distribution Mismatch
Selector trained on t=0.5 (noisy), evaluated on t≈0 (clean).
**Gain:** +5-8 km

### Issue 5: Hard Threshold Mismatch
Training uses p70+p50, inference uses p70 only.
**Gain:** +3-5 km

---

## 📈 Expected Improvement Path

```
Baseline (ep75):                    ADE = 325.3 km
├─ Phase A: FIX-SELECTOR-1          ADE = 315 km (+10 km)
├─ Phase B: FIX-ENV-1               ADE = 295 km (+20 km)
├─ Phase B: FIX-SELECTOR-2          ADE = 290 km (+5 km)
├─ Phase C: FIX-LOSS-1              ADE = 265 km (+25 km)
└─ Phase D: FIX-THRESHOLD-1         ADE = 262 km (+3 km)

Realistic final: 260-280 km (after all fixes with careful tuning)
Still ~90-100 km from ST-Trans target (172.68 km)
```

---

## ⚠️ Key Insights

1. **Model has the components** (selector, environment features) but they're disabled/unwired
2. **Loss function is misaligned** with metric (CFM ≠ ADE)
3. **Training/inference mismatch** (selector t-distribution)
4. **Kinematic-only scoring** ignores future trajectory dynamics
5. **"Easy→hard recurve" problem:** Model trained on past (easy), needs to predict future (hard)

---

## 🚀 Recommended Action

**Today (30 min):**
- Deploy FIX-SELECTOR-1 (1 line)
- Start fresh training from ep0

**This week (6-8 hours):**
- Implement FIX-ENV-1 (steering loss)
- Retrain and measure ep50-80 improvement

**Next 1-2 weeks (8 hours + expensive training):**
- Implement FIX-LOSS-1 (trajectory loss)
- Retrain and evaluate final performance

**If targeting ST-Trans (<172 km):**
- Would need architectural innovations beyond these fixes
- Consider attention-based mode selection, social integration, or other approaches

---

## 📞 Questions?

All analysis is in [`COMPLETE_ANALYSIS_AND_FIX_STRATEGY.md`](COMPLETE_ANALYSIS_AND_FIX_STRATEGY.md).

Specific code locations in [`NEXT_ACTIONS_PRIORITY.md`](NEXT_ACTIONS_PRIORITY.md).

Implementation details in [`QUICK_REFERENCE.txt`](QUICK_REFERENCE.txt).
