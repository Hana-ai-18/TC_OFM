# 🎯 CÁCH 2: NGẮN GỌN

## 🏗️ ARCHITECTURE CHANGE

**Từ:**
```
noise → DDIM 50 bước → trajectory
```

**Thành:**
```
noise → DDIM 5 bước → Decoder → 12 bước cùng lúc
```

## 4 COMPONENT MỚI

### 1. **Lightweight DDIM (Week 1)**
- Giảm 50 bước → 5 bước
- Nhanh hơn, giữ cái chính

### 2. **Trajectory Decoder (Week 2-3)**
```python
class Decoder:
  Input: [context 256D, rough_traj 12×2]
  Output: refined_traj 12×2
  
  Cơ chế:
    - Learned queries (12 cái, 1 per step)
    - Cross-attention: queries ← context
    - Self-attention: steps ← steps
    - Output: refined trajectory
```

### 3. **Regime Classifier (Week 4-6)**
```python
Detect: Is storm straight? turning? accelerating?

Nếu straight: dùng decoder_straight
Nếu turning:  dùng decoder_turning
Nếu complex:  dùng decoder_complex

Blend 3 decoders output dựa regime probability
```

### 4. **Physics Baseline (Week 7)**
```python
Physics: momentum + damping
  pos[t+1] = pos[t] + vel[t]
  vel[t+1] = vel[t] * 0.95

Ensemble: 0.8*learned + 0.2*physics
```

## 📈 PROGRESSION

```
Start:           325 km
Week 1:  +45 km → 280 km (fixes + lightweight DDIM)
Week 2-3: +35 km → 245 km (add decoder)
Week 4-6: +35 km → 210 km (regime aware)
Week 7:   +10 km → 200 km (physics ensemble)
Week 8-10:+5 km → 195 km (fine-tune)

BEAT ST-TRANS (224 km) by 29 km ✅
```

## 💻 CODE CHANGES (3 files only)

### File 1: Model/flow_matching_model.py

```python
# Add 200 lines:

class TrajectoryDecoder(nn.Module):
    def __init__(self):
        self.queries = nn.Parameter(randn(12, 512))
        self.cross_attn = MultiheadAttention(512, 8)
        self.self_attn = MultiheadAttention(512, 8)
        self.refine = Linear(512) → Linear(2)
    
    def forward(self, context, rough_traj):
        queries = self.queries  # [12, 512]
        attended = cross_attn(queries, context, context)
        refined = self_attn(attended, attended, attended)
        delta = self.refine(refined)
        return rough_traj + 0.1*delta

class RegimeAwareDecoder(nn.Module):
    def __init__(self):
        self.classify_regime = small_mlp(obs) → regime_scores
        self.decoder_straight = TrajectoryDecoder()
        self.decoder_turning = TrajectoryDecoder()
        self.decoder_complex = TrajectoryDecoder()
    
    def forward(self, context, rough_traj, obs):
        regime = self.classify_regime(obs)  # [B, 3]
        
        out_straight = decoder_straight(context, rough_traj)
        out_turning = decoder_turning(context, rough_traj)
        out_complex = decoder_complex(context, rough_traj)
        
        return (regime[:, 0:1] * out_straight + 
                regime[:, 1:2] * out_turning +
                regime[:, 2:3] * out_complex)

# Modify TCFlowMatching.__init__():
self.decoder = RegimeAwareDecoder()
self.physics_weight = Parameter(0.2)

# Add new forward:
def forward_with_decoder(self, batch):
    context = self.net._context(batch)
    x_rough = self._lightweight_ddim(batch, steps=5)  # NEW: only 5 steps
    x_refined = self.decoder(context, x_rough, batch[0])
    return x_refined
```

### File 2: scripts/train_flowmatching.py

```python
# Add training loop:

def train_decoder_phase(model, batch, epoch):
    # Forward with new decoder
    pred = model.forward_with_decoder(batch)
    
    # Loss
    gt = batch[1]
    loss_traj = mse_loss(pred, gt)
    loss_curve = curvature_loss(pred)  # Penalize sharp turns
    
    total_loss = loss_traj + 0.1*loss_curve
    
    return total_loss

# Modify main training:
if epoch < 50:
    # Phase 1-3: fix bugs + lightweight DDIM
    loss = train_standard(...)
else:
    # Phase 4+: decoder training
    loss = train_decoder_phase(...)
```

### File 3: Model/physics.py (new file)

```python
def physics_forecast(obs_traj, pred_len=12):
    """Simple momentum-based forecast"""
    if len(obs_traj) < 2:
        return obs_traj[-1].repeat(pred_len, 1)
    
    vel = obs_traj[-1] - obs_traj[-2]
    damping = 0.95
    
    forecast = []
    pos = obs_traj[-1]
    for _ in range(pred_len):
        pos = pos + vel
        vel = vel * damping
        forecast.append(pos)
    
    return stack(forecast)

# In training:
phys_pred = physics_forecast(obs)
learned_pred = model(obs)
final_pred = 0.8 * learned_pred + 0.2 * phys_pred
```

## ⏱️ TIMELINE

| Week | What | Expected ADE |
|------|------|-------------|
| 1 | Fix bugs + DDIM 5 | 280 km |
| 2-3 | Add decoder | 245 km |
| 4-6 | Regime classification | 210 km |
| 7 | Physics ensemble | 200 km |
| 8-10 | Fine-tune | 195 km ✅ |

**Total: 10 weeks, ~600h GPU**

## 🎯 WHY IT WORKS

1. **Parallel generation** → Sees full trajectory context
2. **Regime-aware** → Different strategy per storm type
3. **Physics** → Stability + constraint
4. **FM backbone** → Keep learned velocity field

**Result:** Combines best parts of both worlds

## ✅ READY TO START?

**If YES:**
- Week 1: Start fixing bugs + implement decoder
- Week 2: Retrain with decoder
- Week 3-10: Add regime + physics + fine-tune

**If you have questions:** Ask now, I'll clarify

