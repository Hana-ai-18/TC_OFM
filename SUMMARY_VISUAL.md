# 📊 SUMMARY - Visual Quick Guide

## HIỆN TRẠNG vs TARGET

```
325 km ─────────────────────────────────── ST-Trans 224 km
 ↑                                              ↑
Current FM v59                           Goal: Beat this
(tệ, cách 101 km)                       (195-205 km)
```

## HAI CÁCH ĐỂ THẮNG

```
┌─────────────────────────────────────────────────────────────┐
│ CÁCH 1: QUICK WIN (4 tuần, 240 km)                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ Fix 5 bugs:                                                 │
│  ✓ Enable selector           +10-15 km                     │
│  ✓ Environment loss          +20-30 km                     │
│  ✓ Trajectory loss           +20-40 km                     │
│  ✓ Selector scheduling       +5-8 km                       │
│  ✓ Hard threshold            +3-5 km                       │
│                                                             │
│ Result: 325 → 240 km                                        │
│ vs ST-Trans 224 km: LOSE -16 km ❌                         │
│                                                             │
│ Pros: Easy, nhanh, low risk                               │
│ Cons: Not enough to beat ST-Trans                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ CÁCH 2: WINNING PATH (12 tuần, 195-205 km)                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ Architecture change + engineering:                          │
│                                                             │
│ Week 1:    Fix bugs + lightweight DDIM                      │
│            → 325 → 280 km  (+45)                            │
│                                                             │
│ Week 2-3:  Add Trajectory Decoder                           │
│            → 280 → 245 km  (+35)                            │
│                                                             │
│ Week 4-6:  Regime-Aware (straight vs turning)              │
│            → 245 → 210 km  (+35)                            │
│                                                             │
│ Week 7:    Physics Ensemble                                 │
│            → 210 → 200 km  (+10)                            │
│                                                             │
│ Week 8-10: Fine-tune                                        │
│            → 200 → 195 km  (+5)                             │
│                                                             │
│ Result: 325 → 195 km                                        │
│ vs ST-Trans 224 km: WIN +29 km ✅✅                        │
│                                                             │
│ Pros: Beat ST-Trans, innovative, research-worthy           │
│ Cons: 12 weeks, 600h GPU, medium complexity                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## CORE IDEA OF CÁCH 2

```
┌──────────────────────────────────────────────────────────┐
│                  CURRENT FM (BAD)                        │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Iterative DDIM:                                         │
│  ┌────────┐  ┌──────┐  ┌──────┐         ┌──────┐       │
│  │ Noise  │→ │Step 1│→ │Step 2│→ ... → │Step12│       │
│  └────────┘  └──────┘  └──────┘         └──────┘       │
│                                                          │
│  Problem:                                                │
│  - Mỗi step chỉ nhìn vị trí hiện tại                    │
│  - Không thấy toàn bộ 72h trajectory                    │
│  - Errors accumulate                                     │
│  - Result: 325 km                                        │
│                                                          │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│              NEW FM + DECODER (GOOD)                     │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Lightweight DDIM + Parallel Decoder:                    │
│  ┌────────┐  ┌──────┐  ┌──────────────────────┐        │
│  │ Noise  │→ │ 5    │→ │  Transformer Decoder │        │
│  └────────┘  │Steps │  │                      │        │
│              └──────┘  └──────┬───────────────┘        │
│                               │                        │
│         Output: [step1 step2 ... step12] (parallel)   │
│                                                          │
│  Advantage:                                              │
│  + Sees full 72h context                                │
│  + Optimized end-to-end                                 │
│  + Regime-aware (straight vs turning)                   │
│  + Physics baseline (stability)                         │
│  Result: 195 km                                          │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

## COMPONENTS TO ADD

```
┌──────────────┐
│  TrajectoryDecoder (200 lines)
├──────────────┤
│ - Learned queries (12 per step)
│ - Cross-attention: queries ← context
│ - Self-attention: steps ← steps  
│ - Output: refined trajectory
└──────────────┘

┌──────────────┐
│  RegimeClassifier (100 lines)
├──────────────┤
│ - Detect: straight? turning? complex?
│ - 3 separate decoders per regime
│ - Blend outputs based on regime prob
└──────────────┘

┌──────────────┐
│  Physics (30 lines)
├──────────────┤
│ - Momentum + damping
│ - Ensemble: 0.8*learned + 0.2*physics
└──────────────┘
```

## EFFORT BREAKDOWN

```
Code:       80 hours
  - Decoder:          40h
  - Regime class:     20h
  - Physics:          10h
  - Training mods:    10h

Training:   600 hours
  - Week 1:           40h
  - Week 2-3:         100h
  - Week 4-6:         200h
  - Week 7:           100h
  - Week 8-10:        160h

Total:      680 hours = ~3-4 weeks of 24/7 GPU
            or ~8-10 weeks of normal 8h/day training
```

## EXPECTED METRICS

```
ADE Progression:
325 ──────┐
          │ Fix bugs   
280 ──────┤
          │ Decoder
245 ──────┤
          │ Regime
210 ──────┤
          │ Physics
200 ──────┤
          │ Fine-tune
195 ──────┴─→ BEAT ST-Trans (224 km) by 29 km ✅
```

## DECISION TREE

```
                    BEAT ST-TRANS?
                         │
            ┌────────────┴────────────┐
            │                         │
        MUST WIN              DON'T NEED TO
            │                         │
       (Path 2)               (Path 1 ok)
            │                         │
      12 weeks              4 weeks
      195 km                240 km
      70-80% prob           Easier
      Innovative            Engineering
```

## WHAT YOU NEED TO DECIDE

```
☐ Have 12 weeks?
  YES → Path 2 is doable
  NO  → Path 1 only

☐ Beat ST-Trans is critical?
  YES → Path 2 required
  NO  → Path 1 acceptable

☐ Have 600h GPU?
  YES → Path 2 can do
  NO  → Path 1 use 150h

☐ Research or engineering?
  RESEARCH → Path 2 (publication)
  ENGINEERING → Path 1 (improvement)
```

## IF YOU CHOOSE PATH 2

```
Week 1:  FIX BUGS
         ├─ Enable selector
         ├─ Environment loss
         ├─ Trajectory loss
         ├─ Selector fixes
         └─ Hard threshold
         Result: 280 km

Week 2-3: ADD DECODER
         ├─ TrajectoryDecoder
         ├─ Cross-attention
         └─ Self-attention
         Result: 245 km

Week 4-6: REGIME AWARE
         ├─ Regime classifier
         ├─ 3 decoders
         └─ Gating network
         Result: 210 km

Week 7:  PHYSICS
         ├─ Momentum model
         └─ Ensemble weights
         Result: 200 km

Week 8-10: FINE-TUNE
         ├─ Curriculum learning
         ├─ Loss tuning
         └─ Validation
         Result: 195 km ✅
```

## BOTTOM LINE

```
Question: Có cách nào để đánh thắng?

Answer:   CÓ.

Path 1: 240 km (easy, no beat)
Path 2: 195 km (hard, beat +29km)

Choose Path 2 if:
  ✓ Beat ST-Trans is must-have
  ✓ Have 12 weeks
  ✓ Have 600h GPU
  ✓ Want innovation

Choose Path 1 if:
  ✓ Time is tight
  ✓ 240 km improvement is good enough
  ✓ Want quick confidence

I recommend: PATH 2
Confidence: 70-80% to win
```

