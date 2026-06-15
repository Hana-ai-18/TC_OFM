# # # # # """
# # # # # train_v59_strategy.py — TC-FlowMatching v59-Strategy Training Script
# # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # THAY ĐỔI SO VỚI train_v59_tweaked.py:

# # # # #   [G-A] GradNorm: tự điều chỉnh λ_dpe, λ_vel, λ_head, λ_spd, λ_acc
# # # # #         Không hardcode hệ số loss — GradNorm đo gradient norm sau backward,
# # # # #         update λ để các term có gradient norm xấp xỉ nhau.
# # # # #         QUAN TRỌNG: GradNorm reset λ=1.0 khi phase đổi và tăng update_freq
# # # # #         tạm thời 5 epoch để thích nghi nhanh hơn → tránh lag gradient.

# # # # #   [G-B] Easy/hard pipeline với GLOBAL threshold (fix per-batch flip):
# # # # #         - hard_score tính mỗi epoch trên TOÀN BỘ train set (không phải per-batch)
# # # # #         - Lưu global_hard_threshold = p70 của toàn dataset
# # # # #         - Mỗi batch dùng threshold cố định đó → sample A luôn là hard/easy
# # # # #           nhất quán trong suốt epoch, gradient ổn định hơn nhiều

# # # # #   [G-C] easy_frac enforcement:
# # # # #         - Mỗi batch: đảm bảo ≥ 50% easy samples
# # # # #         - Nếu batch có quá nhiều hard → resample thêm easy từ buffer

# # # # #   [G-D] α smooth warm-up theo BATCH (fix hard cliff):
# # # # #         - Không dùng epoch threshold cứng (epoch 20 bật α đột ngột)
# # # # #         - α tăng theo sigmoid curve qua từng batch:
# # # # #           α(step) = α_target × sigmoid((step - mid_step) / temperature)
# # # # #           → Gradient thay đổi mượt mà, không có cliff, model không bị shock
# # # # #         - Selector cũng được bật mượt: selector_weight tăng từ 0 → 0.2

# # # # #   [G-E] EMA guard sau mỗi epoch:
# # # # #         - Đo easy_ADE trên val subset (easy samples only)
# # # # #         - Nếu easy_ADE tăng > 3 km vs best_easy_ADE → rollback EMA + giảm α
# # # # #         - Dùng easy_ADE (không phải val_ADE tổng) để early stopping

# # # # #   [G-F] Diversity check sau phase 1 (epoch 19):
# # # # #         - compute_diversity_score() trên val subset
# # # # #         - Nếu diversity < 50 km → log cảnh báo, không tự fix

# # # # #   [G-G] Split monitoring mỗi epoch:
# # # # #         - easy_ADE, hard_ADE riêng biệt (không chỉ tổng)

# # # # #   [G-H] Phase 4: freeze encoder, chỉ tune FM head + selector head

# # # # # BUG fixes giữ nguyên từ v59-tweaked:
# # # # #   BUG-5: val_loss dùng epoch=-1 → skip augmentation
# # # # #   BUG-6: any_improve = ANY metric improve → reset early stop
# # # # #   BUG-7: sched.step() unconditional
# # # # #   BUG-8: clip chỉ backbone params, không clip learned weights
# # # # # """
# # # # # from __future__ import annotations
# # # # # import sys, os
# # # # # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # # # # import argparse, math, random, time
# # # # # from collections import defaultdict
# # # # # from typing import Dict, List, Optional

# # # # # import numpy as np
# # # # # import torch
# # # # # import torch.optim as optim
# # # # # import torch.nn.functional as F
# # # # # from torch.amp import autocast, GradScaler
# # # # # from torch.utils.data import DataLoader, Subset

# # # # # from Model.data.loader_training import data_loader
# # # # # from Model.flow_matching_model import (
# # # # #     TCFlowMatching,
# # # # #     hard_score_from_obs,       # dùng trong precompute_hard_threshold
# # # # #     classify_hard_easy,        # dùng trong evaluate_split
# # # # #     compute_diversity_score,   # dùng trong check_diversity
# # # # #     _norm_to_deg,
# # # # #     _haversine_deg,
# # # # # )

# # # # # try:
# # # # #     from Model.utils import get_cosine_schedule_with_warmup
# # # # # except ImportError:
# # # # #     from torch.optim.lr_scheduler import CosineAnnealingLR
# # # # #     def get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps, min_lr=1e-6):
# # # # #         return CosineAnnealingLR(opt, T_max=max(total_steps, 1), eta_min=min_lr)

# # # # # TARGETS = {
# # # # #     "ADE": 172.68, "72h": 321.39, "ATE": 142.21, "CTE": 42.04,
# # # # #     "12h": 65.42,  "24h": 104.67, "48h": 205.10,
# # # # # }
# # # # # R_EARTH = 6371.0

# # # # # # ── Phase boundaries ──────────────────────────────────────────────────────────
# # # # # PHASE_1_END  = 19   # epoch 0–19:  warm-up, α=0, easy_frac=80%
# # # # # PHASE_2_END  = 49   # epoch 20–49: hard intro, α 0→0.3, easy_frac 80→60%
# # # # # PHASE_3_END  = 79   # epoch 50–79: selector, α=0.3, easy_frac=55%
# # # # # # epoch 80+: fine-tune, LR×0.1, freeze encoder

# # # # # # ── GradNorm terms (các loss term cần cân bằng) ──────────────────────────────
# # # # # GRADNORM_TERMS = ["l_dpe", "l_vel_reg", "l_heading", "l_speed", "l_accel"]


# # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # #  GradNorm implementation
# # # # # #  Tham khảo: Chen et al. 2018 "GradNorm: Gradient Normalization for Adaptive
# # # # # #  Loss Balancing in Deep Multitask Networks"
# # # # # #
# # # # # #  Thuật toán:
# # # # # #    1. Sau mỗi backward, đo ||∇_W L_i|| của từng term i
# # # # # #    2. Tính target = mean(||∇_W L_i||) × (L_i / mean(L_i))^α_gn
# # # # # #       α_gn: restoring force (thường 1.5) — term nào loss lớn hơn mức tb
# # # # # #       thì được target gradient cao hơn
# # # # # #    3. Update λ_i để ||∇_W λ_i·L_i|| → target_i
# # # # # #
# # # # # #  Lưu ý: λ_i là learnable parameter của GradNorm (KHÔNG phải model weights)
# # # # # #  Chúng được update riêng bằng Adam optimizer của GradNorm
# # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # class GradNormManager:
# # # # #     """
# # # # #     Quản lý GradNorm cho các loss terms.

# # # # #     [FIX-STABILITY-2] Thêm phase_reset() để tránh lag gradient khi chuyển phase:
# # # # #     - Khi phase đổi: reset λ về 1.0 (không giữ λ cũ không còn phù hợp)
# # # # #     - Tăng update_freq từ 5 → 1 trong 5 epoch sau phase đổi để thích nghi nhanh
# # # # #     - Sau 5 epoch: trở về update_freq=5 (tiết kiệm tính toán)

# # # # #     Lý do: λ được tối ưu cho loss landscape của phase cũ.
# # # # #     Khi thêm L_hard hoặc selector, landscape thay đổi nhưng λ cũ vẫn được giữ
# # # # #     → GradNorm dùng λ sai trong ~50 steps đầu → gradient mất cân bằng.
# # # # #     Reset λ=1.0 ngay lập tức và update dense hơn → ổn định hơn nhiều.
# # # # #     """
# # # # #     def __init__(self, terms: List[str], alpha_gn: float = 1.5,
# # # # #                  lr: float = 1e-3, device=None):
# # # # #         self.terms    = terms
# # # # #         self.alpha_gn = alpha_gn
# # # # #         self.device   = device or torch.device("cpu")

# # # # #         # λ_i: meta-optimization parameter (KHÔNG phải model weights)
# # # # #         # Init = 1.0 cho tất cả terms
# # # # #         self.lambdas = {t: torch.ones(1, device=self.device, requires_grad=True)
# # # # #                         for t in terms}
# # # # #         self.opt = optim.Adam(list(self.lambdas.values()), lr=lr)

# # # # #         self._step             = 0
# # # # #         self._update_freq      = 5      # update mỗi 5 steps (bình thường)
# # # # #         self._dense_steps_left = 0      # số steps còn lại ở chế độ dense (freq=1)
# # # # #         self._current_phase    = 1

# # # # #     def phase_reset(self, new_phase: int) -> None:
# # # # #         """
# # # # #         [FIX-STABILITY-2] Gọi khi chuyển sang phase mới.
# # # # #         Reset λ về 1.0 và bật dense update mode trong 5 epoch.

# # # # #         Args:
# # # # #             new_phase: phase mới (2, 3, 4)
# # # # #         """
# # # # #         if new_phase == self._current_phase:
# # # # #             return
# # # # #         self._current_phase = new_phase

# # # # #         # Reset λ về 1.0 — không giữ λ cũ không còn phù hợp
# # # # #         with torch.no_grad():
# # # # #             for t in self.terms:
# # # # #                 self.lambdas[t].fill_(1.0)

# # # # #         # Reset optimizer state để tránh momentum cũ kéo λ sai hướng
# # # # #         self.opt = optim.Adam(list(self.lambdas.values()), lr=self.opt.param_groups[0]["lr"])

# # # # #         # Dense update mode: update mỗi step trong 200 steps (~5 epoch với bs=32)
# # # # #         # để GradNorm thích nghi nhanh với loss landscape mới
# # # # #         self._dense_steps_left = 200

# # # # #         print(f"  [GradNorm] Phase {new_phase}: reset λ=1.0, dense update mode 200 steps")

# # # # #     def get_lambda_dict(self) -> Dict[str, float]:
# # # # #         """Trả về λ hiện tại để truyền vào get_loss_breakdown()."""
# # # # #         return {t: float(self.lambdas[t].item()) for t in self.terms}

# # # # #     def update(self, loss_term_values: Dict[str, float],
# # # # #                lambda_grads: Optional[Dict[str, torch.Tensor]] = None) -> None:
# # # # #         """
# # # # #         Cập nhật λ dựa trên gradient norm đã được đo từ bên ngoài.

# # # # #         [BUG-4 FIX] Không gọi backward() trong đây vì graph đã bị giải phóng
# # # # #         sau main backward pass. Thay vào đó:
# # # # #         - train script tính grad_norm của từng term SAU backward chính
# # # # #           (dùng p.grad.norm() trên shared params)
# # # # #         - truyền vào đây dưới dạng lambda_grads dict

# # # # #         Args:
# # # # #             loss_term_values: {term: float} — giá trị loss từng term (detached)
# # # # #             lambda_grads:     {term: tensor} — gradient norm của λ_i·L_i
# # # # #                               nếu None → chỉ re-normalize, không update
# # # # #         """
# # # # #         self._step += 1

# # # # #         # Adaptive update frequency
# # # # #         if self._dense_steps_left > 0:
# # # # #             self._dense_steps_left -= 1
# # # # #             update_freq = 1
# # # # #         else:
# # # # #             update_freq = 5

# # # # #         if self._step % update_freq != 0:
# # # # #             return

# # # # #         if not loss_term_values:
# # # # #             return

# # # # #         # Lọc finite values
# # # # #         loss_vals = {t: v for t, v in loss_term_values.items()
# # # # #                      if np.isfinite(v) and v > 0}
# # # # #         if len(loss_vals) < 2:
# # # # #             return

# # # # #         mean_loss = np.mean(list(loss_vals.values()))
# # # # #         if mean_loss <= 0:
# # # # #             return

# # # # #         if lambda_grads is not None and len(lambda_grads) >= 2:
# # # # #             # Có grad_norm từ bên ngoài → update λ đầy đủ
# # # # #             grad_norm_vals = {t: float(g.item()) for t, g in lambda_grads.items()
# # # # #                               if torch.isfinite(g)}
# # # # #             if len(grad_norm_vals) >= 2:
# # # # #                 mean_gnorm = np.mean(list(grad_norm_vals.values()))

# # # # #                 gradnorm_loss = torch.zeros(1, device=self.device)
# # # # #                 for t in grad_norm_vals:
# # # # #                     if t not in loss_vals:
# # # # #                         continue
# # # # #                     r_i      = loss_vals[t] / (mean_loss + 1e-8)
# # # # #                     target_i = mean_gnorm * (r_i ** self.alpha_gn)
# # # # #                     # L1 loss antara actual gnorm vs target
# # # # #                     gnorm_t  = torch.tensor(grad_norm_vals[t], device=self.device)
# # # # #                     target_t = torch.tensor(target_i, device=self.device)
# # # # #                     gradnorm_loss = gradnorm_loss + torch.abs(gnorm_t - target_t)

# # # # #                 # Update λ với gradient của gradnorm_loss terhadap λ
# # # # #                 # Dùng finite difference đơn giản: tăng λ_i nếu gnorm < target
# # # # #                 self.opt.zero_grad()
# # # # #                 gradnorm_loss.backward()
# # # # #                 # Manual gradient cho λ (không dùng autograd qua model)
# # # # #                 for t in self.terms:
# # # # #                     if t not in grad_norm_vals or t not in loss_vals:
# # # # #                         continue
# # # # #                     r_i      = loss_vals[t] / (mean_loss + 1e-8)
# # # # #                     target_i = mean_gnorm * (r_i ** self.alpha_gn)
# # # # #                     actual   = grad_norm_vals[t]
# # # # #                     # Nếu actual < target → tăng λ; actual > target → giảm λ
# # # # #                     sign = 1.0 if actual < target_i else -1.0
# # # # #                     if self.lambdas[t].grad is None:
# # # # #                         self.lambdas[t].grad = torch.zeros_like(self.lambdas[t])
# # # # #                     self.lambdas[t].grad.fill_(sign * abs(actual - target_i) / (mean_gnorm + 1e-8))
# # # # #                 self.opt.step()

# # # # #         # Re-normalize λ để tổng = số terms
# # # # #         with torch.no_grad():
# # # # #             n_terms = len(self.terms)
# # # # #             total_λ = sum(self.lambdas[t].clamp(min=0.01).item() for t in self.terms)
# # # # #             if total_λ > 0:
# # # # #                 for t in self.terms:
# # # # #                     self.lambdas[t].clamp_(min=0.01)
# # # # #                     self.lambdas[t].mul_(n_terms / total_λ)

# # # # #     def log_stats(self) -> Dict[str, float]:
# # # # #         """Log λ hiện tại để monitor."""
# # # # #         d = {f"λ_{t}": float(self.lambdas[t].item()) for t in self.terms}
# # # # #         d["gn_dense"] = self._dense_steps_left
# # # # #         return d


# # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # #  Phase helper
# # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # def get_phase(epoch: int) -> int:
# # # # #     """
# # # # #     Returns:
# # # # #         1: warm-up (ep 0–19)
# # # # #         2: hard introduction (ep 20–49)
# # # # #         3: selector (ep 50–79)
# # # # #         4: fine-tune (ep 80+)
# # # # #     """
# # # # #     if epoch <= PHASE_1_END:  return 1
# # # # #     if epoch <= PHASE_2_END:  return 2
# # # # #     if epoch <= PHASE_3_END:  return 3
# # # # #     return 4


# # # # # class SmoothScheduler:
# # # # #     """
# # # # #     [FIX-STABILITY-1] α smooth warm-up theo BATCH, không phải epoch.

# # # # #     Vấn đề với epoch-based cliff:
# # # # #         Epoch 19 cuối: α = 0.0, gradient ổn định
# # # # #         Epoch 20 đầu: α nhảy lên (dù tăng tuyến tính theo epoch, vẫn là cliff
# # # # #         so với các batch đã chạy) → model bị shock, ADE dao động

# # # # #     Giải pháp: sigmoid curve qua từng BATCH
# # # # #         α(step) = α_target × sigmoid((step - mid_step) / temperature)
# # # # #         - Tại step = mid_step: α = α_target / 2 (điểm giữa)
# # # # #         - temperature lớn → curve dốc hơn (đổi nhanh hơn)
# # # # #         - temperature nhỏ → curve thoải hơn (đổi chậm hơn)

# # # # #         Kết quả: trong epoch 20, α tăng từ rất nhỏ → 0.3 qua ~500 steps
# # # # #         thay vì nhảy đột ngột → gradient thay đổi mượt mà hoàn toàn

# # # # #     Selector weight cũng tăng mượt theo cùng cơ chế.
# # # # #     """
# # # # #     def __init__(self, total_steps_per_epoch: int):
# # # # #         self.steps_per_epoch = total_steps_per_epoch
# # # # #         self._global_step = 0

# # # # #         # Điểm bắt đầu của mỗi phase (tính theo global step)
# # # # #         # Phase N bắt đầu SAU khi epoch PHASE_(N-1)_END hoàn tất
# # # # #         # = (PHASE_(N-1)_END + 1) * steps_per_epoch
# # # # #         self._phase2_start_step = (PHASE_1_END + 1) * total_steps_per_epoch
# # # # #         self._phase3_start_step = (PHASE_2_END + 1) * total_steps_per_epoch
# # # # #         self._phase4_start_step = (PHASE_3_END + 1) * total_steps_per_epoch

# # # # #         # Temperature: bao nhiêu steps để α đi từ 10% → 90% giá trị target
# # # # #         # Công thức: temp = steps / (2 × ln(9)) ≈ steps / 4.4
# # # # #         # Ở đây ta muốn transition kéo dài ~2 epoch (2×steps_per_epoch)
# # # # #         self._temp_p2 = 2.0 * total_steps_per_epoch / 4.4
# # # # #         self._temp_p3 = 1.5 * total_steps_per_epoch / 4.4  # selector bật nhanh hơn

# # # # #     def step(self):
# # # # #         """Gọi sau mỗi optimizer step."""
# # # # #         self._global_step += 1

# # # # #     @property
# # # # #     def global_step(self):
# # # # #         return self._global_step

# # # # #     def get_alpha_hard(self) -> float:
# # # # #         """
# # # # #         α_hard: trọng số L_hard, tăng mượt qua sigmoid.
# # # # #         Phase 1 (step < phase2_start): α = 0
# # # # #         Phase 2 (step >= phase2_start): sigmoid 0 → 0.3
# # # # #         Phase 3+: α = 0.3 cố định
# # # # #         """
# # # # #         s = self._global_step
# # # # #         if s < self._phase2_start_step:
# # # # #             return 0.0
# # # # #         if s >= self._phase3_start_step:
# # # # #             return 0.3

# # # # #         # Sigmoid trong phase 2
# # # # #         mid  = self._phase2_start_step + (self._phase3_start_step
# # # # #                                            - self._phase2_start_step) / 2.0
# # # # #         x    = (s - mid) / max(self._temp_p2, 1.0)
# # # # #         sig  = 1.0 / (1.0 + math.exp(-x))
# # # # #         return 0.3 * sig

# # # # #     def get_selector_weight(self) -> float:
# # # # #         """
# # # # #         selector_weight: trọng số L_sel, tăng mượt từ phase 3.
# # # # #         Phase < 3: 0
# # # # #         Phase 3: sigmoid 0 → 0.2
# # # # #         Phase 4: 0.2 cố định
# # # # #         """
# # # # #         s = self._global_step
# # # # #         if s < self._phase3_start_step:
# # # # #             return 0.0
# # # # #         if s >= self._phase4_start_step:
# # # # #             return 0.2

# # # # #         mid  = self._phase3_start_step + (self._phase4_start_step
# # # # #                                            - self._phase3_start_step) / 2.0
# # # # #         x    = (s - mid) / max(self._temp_p3, 1.0)
# # # # #         sig  = 1.0 / (1.0 + math.exp(-x))
# # # # #         return 0.2 * sig

# # # # #     def get_easy_frac(self) -> float:
# # # # #         """
# # # # #         easy_frac: giảm mượt, không bao giờ xuống dưới 50%.
# # # # #         Phase 1: 80%
# # # # #         Phase 2: 80% → 60% theo sigmoid
# # # # #         Phase 3: 60% → 55%
# # # # #         Phase 4: 55% → 50%
# # # # #         """
# # # # #         s = self._global_step
# # # # #         if s < self._phase2_start_step:
# # # # #             return 0.80

# # # # #         if s < self._phase3_start_step:
# # # # #             mid  = self._phase2_start_step + (self._phase3_start_step
# # # # #                                                - self._phase2_start_step) / 2.0
# # # # #             x    = (s - mid) / max(self._temp_p2, 1.0)
# # # # #             sig  = 1.0 / (1.0 + math.exp(-x))
# # # # #             return 0.80 - 0.20 * sig   # 80% → 60%

# # # # #         if s < self._phase4_start_step:
# # # # #             mid  = self._phase3_start_step + (self._phase4_start_step
# # # # #                                                - self._phase3_start_step) / 2.0
# # # # #             x    = (s - mid) / max(self._temp_p3, 1.0)
# # # # #             sig  = 1.0 / (1.0 + math.exp(-x))
# # # # #             return 0.60 - 0.05 * sig   # 60% → 55%

# # # # #         # Phase 4: giữ 50% (hard floor)
# # # # #         return max(0.50, 0.55 - 0.05 * (s - self._phase4_start_step)
# # # # #                    / max(self.steps_per_epoch * 10, 1))

# # # # #     def is_selector_active(self) -> bool:
# # # # #         """True từ phase 3 trở đi."""
# # # # #         return self._global_step >= self._phase3_start_step

# # # # #     def log_stats(self) -> str:
# # # # #         return (f"α={self.get_alpha_hard():.3f}"
# # # # #                 f" easy={self.get_easy_frac():.0%}"
# # # # #                 f" sel={self.get_selector_weight():.3f}")


# # # # # # Backward-compat wrappers (dùng trong checkpoint restore)
# # # # # def get_alpha_hard(epoch: int) -> float:
# # # # #     """Legacy: chỉ dùng khi không có SmoothScheduler."""
# # # # #     if epoch <= PHASE_1_END: return 0.0
# # # # #     if epoch <= PHASE_2_END:
# # # # #         return 0.3 * (epoch - PHASE_1_END) / (PHASE_2_END - PHASE_1_END)
# # # # #     return 0.3

# # # # # def get_easy_frac(epoch: int) -> float:
# # # # #     """Legacy fallback."""
# # # # #     if epoch <= PHASE_1_END: return 0.80
# # # # #     if epoch <= PHASE_2_END:
# # # # #         p = (epoch - PHASE_1_END) / (PHASE_2_END - PHASE_1_END)
# # # # #         return 0.80 - 0.20 * p
# # # # #     if epoch <= PHASE_3_END: return 0.55
# # # # #     return 0.50


# # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # #  [FIX-STABILITY-3] Global hard threshold — precomputed per epoch
# # # # # #
# # # # # #  Vấn đề với per-batch threshold:
# # # # # #    Batch 1 có 32 samples → p70 tính trên 32 samples → threshold = X
# # # # # #    Batch 2 có 32 khác → p70 tính trên 32 samples khác → threshold = Y (≠ X)
# # # # # #    Sample A: hard trong batch 1 (score > X), easy trong batch 2 (score < Y)
# # # # # #    → L_hard áp dụng không nhất quán → gradient noisy
# # # # # #
# # # # # #  Giải pháp: scan toàn bộ train set 1 lần đầu mỗi epoch, lấy p70 global
# # # # # #    → threshold cố định cho cả epoch → mọi batch đều phân loại nhất quán
# # # # # #    → gradient ổn định hơn nhiều
# # # # # #
# # # # # #  Chi phí: ~1 pass qua train loader với batch nhỏ (không cần gradient)
# # # # # #  Thời gian: ~10–20s/epoch (chấp nhận được)
# # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # @torch.no_grad()
# # # # # def precompute_hard_threshold(loader, dev, n_batches: int = 50) -> float:
# # # # #     """
# # # # #     Tính p70 của hard_score trên toàn bộ (hoặc subset lớn) train set.

# # # # #     Args:
# # # # #         loader:    train DataLoader
# # # # #         dev:       device
# # # # #         n_batches: số batches để scan (50 batches × 32 = 1600 samples, đủ)

# # # # #     Returns:
# # # # #         threshold: float — global p70 hard_score
# # # # #                    Mọi sample có hard_score > threshold được coi là hard
# # # # #     """
# # # # #     all_scores = []
# # # # #     for i, batch in enumerate(loader):
# # # # #         if i >= n_batches:
# # # # #             break
# # # # #         bl    = move(list(batch), dev)
# # # # #         obs_t = bl[0]
# # # # #         scores = hard_score_from_obs(obs_t[:, :, :2])  # [B]
# # # # #         all_scores.append(scores.cpu())

# # # # #     if not all_scores:
# # # # #         return 0.7  # fallback

# # # # #     all_scores_cat = torch.cat(all_scores)  # [N]
# # # # #     threshold = float(torch.quantile(all_scores_cat, 0.70).item())
# # # # #     n_hard = int((all_scores_cat >= threshold).sum().item())
# # # # #     n_total = len(all_scores_cat)
# # # # #     print(f"  [HardThreshold] global p70={threshold:.4f}"
# # # # #           f"  n_hard={n_hard}/{n_total} ({100*n_hard/max(n_total,1):.0f}%)")
# # # # #     return threshold


# # # # # @torch.no_grad()
# # # # # def classify_hard_easy_global(
# # # # #     obs_traj_norm: torch.Tensor,
# # # # #     global_threshold: float,
# # # # # ) -> torch.Tensor:
# # # # #     """
# # # # #     Phân loại hard/easy dùng global threshold (thay vì per-batch p70).

# # # # #     Args:
# # # # #         obs_traj_norm:    [T, B, >=2]
# # # # #         global_threshold: float từ precompute_hard_threshold()

# # # # #     Returns:
# # # # #         is_hard: [B] bool
# # # # #     """
# # # # #     scores   = hard_score_from_obs(obs_traj_norm)  # [B]
# # # # #     is_hard  = scores >= global_threshold
# # # # #     return is_hard

# # # # # def enforce_easy_frac(
# # # # #     is_hard: torch.Tensor,
# # # # #     easy_frac: float,
# # # # # ) -> torch.Tensor:
# # # # #     """
# # # # #     [BUG-3 FIX] Chỉ điều chỉnh is_hard mask, KHÔNG reindex batch_list.

# # # # #     Cách cũ cố reindex batch_list nhưng env_data (dict) không được
# # # # #     reindex → data inconsistency giữa obs_t và env_data sau reindex.

# # # # #     Cách mới: không động vào batch data. Nếu batch có quá nhiều hard samples,
# # # # #     "demote" một số về easy bằng cách set is_hard[i] = False.
# # # # #     Những samples đó vẫn chịu L_base bình thường, chỉ không chịu L_hard.
# # # # #     Batch data giữ nguyên 100%, không inconsistency.

# # # # #     Args:
# # # # #         is_hard:   [B] bool tensor
# # # # #         easy_frac: tỉ lệ easy tối thiểu (0.6 = 60% phải là easy)

# # # # #     Returns:
# # # # #         is_hard_adjusted: [B] bool, n_hard ≤ floor(B × (1 - easy_frac))
# # # # #     """
# # # # #     B        = is_hard.shape[0]
# # # # #     max_hard = max(0, int(B * (1.0 - easy_frac)))
# # # # #     n_hard   = int(is_hard.sum().item())

# # # # #     if n_hard <= max_hard:
# # # # #         return is_hard  # đã đúng tỉ lệ, không cần điều chỉnh

# # # # #     # Demote (n_hard - max_hard) hard samples về easy
# # # # #     n_demote  = n_hard - max_hard
# # # # #     hard_idx  = is_hard.nonzero(as_tuple=True)[0]
# # # # #     perm      = torch.randperm(hard_idx.numel(), device=is_hard.device)
# # # # #     to_demote = hard_idx[perm[:n_demote]]

# # # # #     is_hard_new            = is_hard.clone()
# # # # #     is_hard_new[to_demote] = False
# # # # #     return is_hard_new


# # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # #  Utilities — giữ nguyên từ v59-tweaked
# # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # def _unwrap(m):
# # # # #     return m._orig_mod if hasattr(m, "_orig_mod") else m

# # # # # def move(b, dev):
# # # # #     out = list(b)
# # # # #     for i, x in enumerate(out):
# # # # #         if torch.is_tensor(x): out[i] = x.to(dev)
# # # # #         elif isinstance(x, dict):
# # # # #             out[i] = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in x.items()}
# # # # #     return out


# # # # # def _ntd(a):
# # # # #     o = a.clone()
# # # # #     o[..., 0] = (a[..., 0] * 50.0 + 1800.0) / 10.0
# # # # #     o[..., 1] = (a[..., 1] * 50.0) / 10.0
# # # # #     return o

# # # # # def _hav(p1, p2):
# # # # #     la1 = torch.deg2rad(p1[..., 1]); la2 = torch.deg2rad(p2[..., 1])
# # # # #     dlat = torch.deg2rad(p2[..., 1] - p1[..., 1])
# # # # #     dlon = torch.deg2rad(p2[..., 0] - p1[..., 0])
# # # # #     a = (torch.sin(dlat/2).pow(2)
# # # # #          + torch.cos(la1)*torch.cos(la2)*torch.sin(dlon/2).pow(2))
# # # # #     return 2.0 * R_EARTH * torch.asin(a.clamp(1e-12, 1-1e-12).sqrt())

# # # # # def _atecte(pd, gd):
# # # # #     T = min(pd.shape[0], gd.shape[0])
# # # # #     if T < 2:
# # # # #         z = pd.new_zeros(1, pd.shape[1]); return z, z
# # # # #     lo1=torch.deg2rad(gd[:T-1,:,0]); la1=torch.deg2rad(gd[:T-1,:,1])
# # # # #     lo2=torch.deg2rad(gd[1:T, :,0]); la2=torch.deg2rad(gd[1:T, :,1])
# # # # #     lo3=torch.deg2rad(pd[1:T, :,0]); la3=torch.deg2rad(pd[1:T, :,1])
# # # # #     ya=torch.sin(lo2-lo1)*torch.cos(la2)
# # # # #     xa=(torch.cos(la1)*torch.sin(la2)-torch.sin(la1)*torch.cos(la2)*torch.cos(lo2-lo1))
# # # # #     be=torch.atan2(ya,xa)
# # # # #     ye=torch.sin(lo3-lo2)*torch.cos(la3)
# # # # #     xe=(torch.cos(la2)*torch.sin(la3)-torch.sin(la2)*torch.cos(la3)*torch.cos(lo3-lo2))
# # # # #     bee=torch.atan2(ye,xe)
# # # # #     tot=_hav(pd[1:T],gd[1:T]); ang=bee-be
# # # # #     return tot*torch.cos(ang), tot*torch.sin(ang)


# # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # #  [G-G] Split metrics — đo easy_ADE và hard_ADE riêng biệt
# # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # class SplitAcc:
# # # # #     """
# # # # #     Accumulator đo easy_ADE và hard_ADE riêng biệt.
# # # # #     Dùng để:
# # # # #     - EMA guard: phát hiện easy forgetting sớm
# # # # #     - Hard improvement: xác nhận hard loss đang hoạt động
# # # # #     """
# # # # #     def __init__(self):
# # # # #         self.easy_d = []
# # # # #         self.hard_d = []
# # # # #         self.all_d  = []
# # # # #         self.all_a  = []
# # # # #         self.all_c  = []
# # # # #         self.sd     = defaultdict(list)
# # # # #         self._h     = {12: 1, 24: 3, 48: 7, 72: 11}

# # # # #     def update(self, dist, is_hard=None, ate=None, cte=None):
# # # # #         """
# # # # #         Args:
# # # # #             dist:    [T, B] distance per step
# # # # #             is_hard: [B] bool (None → tất cả easy)
# # # # #         """
# # # # #         B = dist.shape[1]
# # # # #         mean_dist = dist.mean(0)  # [B]
# # # # #         self.all_d.extend(mean_dist.tolist())

# # # # #         if is_hard is not None:
# # # # #             easy_mask = ~is_hard
# # # # #             if easy_mask.any():
# # # # #                 self.easy_d.extend(mean_dist[easy_mask].tolist())
# # # # #             if is_hard.any():
# # # # #                 self.hard_d.extend(mean_dist[is_hard].tolist())
# # # # #         else:
# # # # #             self.easy_d.extend(mean_dist.tolist())

# # # # #         for h, s in self._h.items():
# # # # #             if s < dist.shape[0]:
# # # # #                 self.sd[h].extend(dist[s].tolist())

# # # # #         if ate is not None: self.all_a.extend(ate.abs().mean(0).tolist())
# # # # #         if cte is not None: self.all_c.extend(cte.abs().mean(0).tolist())

# # # # #     def compute(self):
# # # # #         r = {
# # # # #             "ADE":      float(np.mean(self.all_d))  if self.all_d  else float("nan"),
# # # # #             "easy_ADE": float(np.mean(self.easy_d)) if self.easy_d else float("nan"),
# # # # #             "hard_ADE": float(np.mean(self.hard_d)) if self.hard_d else float("nan"),
# # # # #             "ATE":      float(np.mean(self.all_a))  if self.all_a  else float("nan"),
# # # # #             "CTE":      float(np.mean(self.all_c))  if self.all_c  else float("nan"),
# # # # #             "n":        len(self.all_d),
# # # # #             "n_easy":   len(self.easy_d),
# # # # #             "n_hard":   len(self.hard_d),
# # # # #         }
# # # # #         for h in self._h:
# # # # #             v = self.sd.get(h, [])
# # # # #             r[f"{h}h"] = float(np.mean(v)) if v else float("nan")
# # # # #         return r


# # # # # class Acc:
# # # # #     """Simple accumulator (backward compat với evaluate())."""
# # # # #     def __init__(self):
# # # # #         self.d=[]; self.a=[]; self.c=[]; self.sd=defaultdict(list)
# # # # #         self._h={12:1, 24:3, 48:7, 72:11}
# # # # #     def update(self, dist, ate=None, cte=None):
# # # # #         self.d.extend(dist.mean(0).tolist())
# # # # #         for h,s in self._h.items():
# # # # #             if s < dist.shape[0]: self.sd[h].extend(dist[s].tolist())
# # # # #         if ate is not None: self.a.extend(ate.abs().mean(0).tolist())
# # # # #         if cte is not None: self.c.extend(cte.abs().mean(0).tolist())
# # # # #     def compute(self):
# # # # #         r = {"ADE": float(np.mean(self.d)) if self.d else float("nan"),
# # # # #              "ATE": float(np.mean(self.a)) if self.a else float("nan"),
# # # # #              "CTE": float(np.mean(self.c)) if self.c else float("nan"),
# # # # #              "n": len(self.d)}
# # # # #         for h in self._h:
# # # # #             v=self.sd.get(h,[]); r[f"{h}h"]=float(np.mean(v)) if v else float("nan")
# # # # #         return r


# # # # # def _score(r):
# # # # #     ade=r.get("ADE",1e9); h72=r.get("72h",1e9)
# # # # #     ate=r.get("ATE",1e9); cte=r.get("CTE",1e9)
# # # # #     if not np.isfinite(ate): ate=ade*0.46
# # # # #     if not np.isfinite(cte): cte=ade*0.53
# # # # #     return 100.0*(0.05*(ade/136.)+0.10*(r.get("12h",ade)/50.)
# # # # #                   +0.15*(r.get("24h",ade)/100.)+0.20*(r.get("48h",ade)/200.)
# # # # #                   +0.25*(h72/300.)+0.13*(ate/80.)+0.12*(cte/94.))

# # # # # def _beat(r):
# # # # #     p=[]
# # # # #     for k,t in [("ADE",172.68),("ATE",142.21),("CTE",42.04),
# # # # #                 ("72h",321.39),("12h",65.42),("24h",104.67),("48h",205.10)]:
# # # # #         v=r.get(k,1e9)
# # # # #         if np.isfinite(v) and v < t: p.append(f"{k}:{v:.1f}")
# # # # #     return "*** BEAT ST-TRANS: "+" ".join(p)+" ***" if p else ""

# # # # # def _gap(r):
# # # # #     out=[]
# # # # #     for k,ref in [("ADE",172.68),("72h",321.39),("ATE",142.21),("CTE",42.04)]:
# # # # #         v=r.get(k,float("nan"))
# # # # #         if np.isfinite(v):
# # # # #             out.append(f"{k}:{v:.0f}({'dn' if v<ref else 'up'}{abs(v-ref):.0f})")
# # # # #     return " | ".join(out)


# # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # #  [G-E] Easy_ADE evaluation cho EMA guard
# # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # @torch.no_grad()
# # # # # def evaluate_split(model, loader, dev, tag="", ema=None, steps=10,
# # # # #                    use_selector=False) -> Dict:
# # # # #     """
# # # # #     Evaluate với split easy/hard ADE.
# # # # #     Dùng fast ddim (10 steps) cho speed.
# # # # #     """
# # # # #     bk = None
# # # # #     if ema:
# # # # #         try: bk = ema.apply_to(model)
# # # # #         except: pass
# # # # #     model.eval()
# # # # #     acc = SplitAcc()
# # # # #     t0  = time.perf_counter()

# # # # #     for b in loader:
# # # # #         bl = move(list(b), dev)
# # # # #         obs_t = bl[0]

# # # # #         # Phân loại easy/hard bằng đặc trưng vật lý
# # # # #         is_hard = classify_hard_easy(obs_t[:, :, :2]).to(dev)

# # # # #         result = model.sample(bl, ddim_steps=steps, use_selector=use_selector)
# # # # #         p = result[0] if isinstance(result, (tuple, list)) else result
# # # # #         g = bl[1]; T = min(p.shape[0], g.shape[0])
# # # # #         pd = _ntd(p[:T]); gd = _ntd(g[:T])
# # # # #         dist = _hav(pd, gd)
# # # # #         at, ct = _atecte(pd, gd)
# # # # #         acc.update(dist, is_hard=is_hard, ate=at, cte=ct)

# # # # #     if bk:
# # # # #         try: ema.restore(model, bk)
# # # # #         except: pass

# # # # #     r  = acc.compute()
# # # # #     el = time.perf_counter() - t0

# # # # #     def _v(k): return r.get(k, float("nan"))
# # # # #     def _m(v, t): return "ok" if np.isfinite(v) and v < t else "no"

# # # # #     print(f"\n{'='*70}")
# # # # #     print(f"  [{tag}  {el:.0f}s]")
# # # # #     print(f"  ADE={_v('ADE'):.1f}[{_m(_v('ADE'),172.68)}]"
# # # # #           f"  easy_ADE={_v('easy_ADE'):.1f}  hard_ADE={_v('hard_ADE'):.1f}")
# # # # #     print(f"  12h={_v('12h'):.0f}  24h={_v('24h'):.0f}"
# # # # #           f"  48h={_v('48h'):.0f}  72h={_v('72h'):.0f}[{_m(_v('72h'),321.39)}]")
# # # # #     print(f"  ATE={_v('ATE'):.1f}[{_m(_v('ATE'),142.21)}]"
# # # # #           f"  CTE={_v('CTE'):.1f}[{_m(_v('CTE'),42.04)}]")
# # # # #     print(f"  n_easy={r.get('n_easy',0)}  n_hard={r.get('n_hard',0)}")
# # # # #     print(f"  vs ST-Trans: {_gap(r)}")
# # # # #     bt = _beat(r)
# # # # #     if bt: print(f"  {bt}")
# # # # #     print(f"  Score={_score(r):.2f}")
# # # # #     print(f"{'='*70}\n")
# # # # #     return r


# # # # # @torch.no_grad()
# # # # # def evaluate(model, loader, dev, tag="", ema=None, steps=20,
# # # # #              use_selector=False) -> Dict:
# # # # #     """Full evaluation (backward compat)."""
# # # # #     return evaluate_split(model, loader, dev, tag=tag, ema=ema,
# # # # #                           steps=steps, use_selector=use_selector)


# # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # #  [G-F] Diversity check
# # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # @torch.no_grad()
# # # # # def check_diversity(model, loader, dev, n_batches: int = 5,
# # # # #                     n_ensemble: int = 20, ddim_steps: int = 5) -> float:
# # # # #     """
# # # # #     Đo diversity score trên một subset của val set.
# # # # #     Nếu < 50 km → cảnh báo R1 (mode collapse).

# # # # #     Args:
# # # # #         n_batches:  số batches để đo (5 là đủ)
# # # # #         n_ensemble: số candidates (nhỏ để nhanh)
# # # # #         ddim_steps: DDIM steps (ít để nhanh)

# # # # #     Returns:
# # # # #         diversity_km: float
# # # # #     """
# # # # #     model.eval()
# # # # #     all_divs = []

# # # # #     for i, b in enumerate(loader):
# # # # #         if i >= n_batches:
# # # # #             break
# # # # #         bl = move(list(b), dev)

# # # # #         # Sample nhiều candidates
# # # # #         _, _, all_c = model.sample(bl, num_ensemble=n_ensemble,
# # # # #                                     ddim_steps=ddim_steps)
# # # # #         # all_c: [2N, T, B, 2] normalized (do speed sweep tạo 2× candidates)
# # # # #         # Reshape thành list
# # # # #         N_total = all_c.shape[0]
# # # # #         cands = [all_c[k] for k in range(N_total)]

# # # # #         div = compute_diversity_score(cands)
# # # # #         all_divs.append(div)

# # # # #     return float(np.mean(all_divs)) if all_divs else 0.0


# # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # #  Checkpoint
# # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # def _save_ckpt(path, ep, model, opt, sched, saver, tl, vl, extra=None):
# # # # #     m = _unwrap(model)
# # # # #     ema = getattr(m, "_ema", None)
# # # # #     esd = None
# # # # #     if ema and hasattr(ema, "shadow"):
# # # # #         try: esd = {k: v.cpu().clone() for k, v in ema.shadow.items()}
# # # # #         except: pass
# # # # #     d = {
# # # # #         "epoch": ep,
# # # # #         "model_state_dict": m.state_dict(),
# # # # #         "optimizer_state":  opt.state_dict(),
# # # # #         "scheduler_state":  sched.state_dict(),
# # # # #         "ema_shadow":       esd,
# # # # #         "best_score":       saver.bs,
# # # # #         "best_ade":         saver.ba,
# # # # #         "best_72h":         saver.b7,
# # # # #         "best_ate":         saver.bat,
# # # # #         "best_cte":         saver.bc,
# # # # #         "best_easy_ade":    saver.best_easy_ade,
# # # # #         "train_loss":       tl,
# # # # #         "val_loss":         vl,
# # # # #         "version":          "v59strategy",
# # # # #     }
# # # # #     if extra:
# # # # #         d.update(extra)
# # # # #     torch.save(d, path)


# # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # #  Saver
# # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # class Saver:
# # # # #     def __init__(self, patience=40, min_ep=35, enable_stop=True):
# # # # #         self.patience     = patience
# # # # #         self.min_ep       = min_ep
# # # # #         self.enable_stop  = enable_stop
# # # # #         self.cnt          = 0
# # # # #         self.stop         = False
# # # # #         self.bs           = float("inf")  # best score
# # # # #         self.ba           = float("inf")  # best ADE
# # # # #         self.b7           = float("inf")  # best 72h
# # # # #         self.bat          = float("inf")  # best ATE
# # # # #         self.bc           = float("inf")  # best CTE
# # # # #         self.best_easy_ade = float("inf") # [G-E] best easy ADE (EMA guard)

# # # # #     def update(self, r, model, out_dir, ep, opt, sched, tl, vl, tag=""):
# # # # #         sc  = _score(r)
# # # # #         ade = r.get("ADE",     1e9)
# # # # #         h72 = r.get("72h",     1e9)
# # # # #         ate = r.get("ATE",     1e9)
# # # # #         cte = r.get("CTE",     1e9)
# # # # #         easy_ade = r.get("easy_ADE", ade)  # fallback to ADE if no split

# # # # #         any_improved = False

# # # # #         for v, attr, fn in [
# # # # #             (ade,      "ba",  f"best_ade_{tag}.pth"),
# # # # #             (h72,      "b7",  f"best_72h_{tag}.pth"),
# # # # #             (ate,      "bat", f"best_ate_{tag}.pth"),
# # # # #             (cte,      "bc",  f"best_cte_{tag}.pth"),
# # # # #         ]:
# # # # #             if v < getattr(self, attr):
# # # # #                 setattr(self, attr, v)
# # # # #                 _save_ckpt(os.path.join(out_dir, fn), ep, model, opt,
# # # # #                            sched, self, tl, vl)
# # # # #                 any_improved = True

# # # # #         if sc < self.bs:
# # # # #             self.bs = sc
# # # # #             any_improved = True
# # # # #             _save_ckpt(os.path.join(out_dir, f"best_{tag or 'composite'}.pth"),
# # # # #                        ep, model, opt, sched, self, tl, vl)
# # # # #             print(f"  [BEST] {tag} ep={ep} score={sc:.2f}"
# # # # #                   f"  ADE={ade:.1f}  72h={h72:.0f}"
# # # # #                   f"  ATE={ate:.1f}  CTE={cte:.1f}")

# # # # #         # [G-E] Update best easy_ADE
# # # # #         if np.isfinite(easy_ade) and easy_ade < self.best_easy_ade:
# # # # #             self.best_easy_ade = easy_ade

# # # # #         if self.enable_stop:
# # # # #             if any_improved: self.cnt = 0
# # # # #             else:
# # # # #                 self.cnt += 1
# # # # #                 print(f"  No improve {self.cnt}/{self.patience}"
# # # # #                       f"  best={self.bs:.2f}  cur={sc:.2f}")
# # # # #             if ep >= self.min_ep and self.cnt >= self.patience:
# # # # #                 self.stop = True


# # # # # def _mksub(ds, n, bs, cf):
# # # # #     idx = random.Random(42).sample(range(len(ds)), min(n, len(ds)))
# # # # #     return DataLoader(Subset(ds, idx), batch_size=bs, shuffle=False,
# # # # #                       collate_fn=cf, num_workers=0, drop_last=False)


# # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # #  Args
# # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # def get_args():
# # # # #     p = argparse.ArgumentParser(
# # # # #         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# # # # #     p.add_argument("--dataset_root",       default="TCND_vn")
# # # # #     p.add_argument("--obs_len",            default=8,    type=int)
# # # # #     p.add_argument("--pred_len",           default=12,   type=int)
# # # # #     p.add_argument("--batch_size",         default=32,   type=int)
# # # # #     p.add_argument("--num_epochs",         default=100,  type=int)
# # # # #     p.add_argument("--learning_rate",      default=1e-4, type=float)
# # # # #     p.add_argument("--weight_decay",       default=1e-3, type=float)
# # # # #     p.add_argument("--warmup_epochs",      default=3,    type=int)
# # # # #     p.add_argument("--grad_clip",          default=1.0,  type=float)
# # # # #     p.add_argument("--use_amp",            action="store_true")
# # # # #     p.add_argument("--num_workers",        default=2,    type=int)
# # # # #     p.add_argument("--sigma_min",          default=0.02, type=float)
# # # # #     p.add_argument("--use_ot",             default=True, action="store_true")
# # # # #     p.add_argument("--no_ot",              dest="use_ot", action="store_false")
# # # # #     p.add_argument("--cfg_guidance_scale", default=1.5,  type=float)
# # # # #     p.add_argument("--n_ensemble",         default=50,   type=int)
# # # # #     p.add_argument("--use_ema",            default=True, action="store_true")
# # # # #     p.add_argument("--no_ema",             dest="use_ema", action="store_false")
# # # # #     p.add_argument("--ema_decay",          default=0.995, type=float)
# # # # #     p.add_argument("--patience",           default=40,   type=int)
# # # # #     p.add_argument("--min_ep",             default=35,   type=int)
# # # # #     p.add_argument("--val_freq",           default=3,    type=int)
# # # # #     p.add_argument("--val_subset_size",    default=500,  type=int)
# # # # #     p.add_argument("--fast_ddim",          default=10,   type=int)
# # # # #     p.add_argument("--full_ddim",          default=20,   type=int)
# # # # #     p.add_argument("--output_dir",         default="runs/v59strategy")
# # # # #     p.add_argument("--gpu_num",            default="0")
# # # # #     p.add_argument("--delim",              default=" ")
# # # # #     p.add_argument("--skip",               default=1,    type=int)
# # # # #     p.add_argument("--min_ped",            default=1,    type=int)
# # # # #     p.add_argument("--threshold",          default=0.002, type=float)
# # # # #     p.add_argument("--other_modal",        default="gph")
# # # # #     p.add_argument("--test_year",          default=None, type=int)
# # # # #     p.add_argument("--resume",             default=None)
# # # # #     p.add_argument("--resume_epoch",       default=None, type=int)
# # # # #     p.add_argument("--eval_test_after_train", default=True, action="store_true")
# # # # #     # GradNorm
# # # # #     p.add_argument("--gradnorm_alpha",     default=1.5,  type=float,
# # # # #                    help="GradNorm restoring force")
# # # # #     p.add_argument("--gradnorm_lr",        default=1e-3, type=float,
# # # # #                    help="GradNorm lambda learning rate")
# # # # #     # EMA guard
# # # # #     p.add_argument("--ema_guard_threshold", default=3.0, type=float,
# # # # #                    help="easy_ADE tăng bao nhiêu km thì rollback (km)")
# # # # #     # Diversity check
# # # # #     p.add_argument("--diversity_threshold", default=50.0, type=float,
# # # # #                    help="diversity_score < threshold → cảnh báo R1")
# # # # #     return p.parse_args()


# # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # #  Main
# # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # def main(args):
# # # # #     if torch.cuda.is_available():
# # # # #         os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
# # # # #     dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # # # #     os.makedirs(args.output_dir, exist_ok=True)

# # # # #     print("=" * 72)
# # # # #     print("  TC-FlowMatching v59-Strategy")
# # # # #     print("  [G-A] GradNorm: λ_dpe, λ_vel, λ_head, λ_spd, λ_acc tự điều chỉnh")
# # # # #     print("  [G-B] Easy/Hard: hard_score đa tiêu chí (curvature+speed_var+dir)")
# # # # #     print("  [G-C] easy_frac ≥ 50% enforcement mỗi batch")
# # # # #     print("  [G-D] α warm-up: 0→0.3 trong phase 2, phase 3: selector bật")
# # # # #     print("  [G-E] EMA guard: rollback nếu easy_ADE tăng > 3km")
# # # # #     print("  [G-F] Diversity check: cảnh báo nếu < 50km sau phase 1")
# # # # #     print("  [G-G] Split monitoring: easy_ADE + hard_ADE mỗi epoch")
# # # # #     print("  [G-H] Phase 4: freeze encoder, chỉ tune FM head + selector")
# # # # #     print(f"  Phases: 1[0-{PHASE_1_END}] 2[{PHASE_1_END+1}-{PHASE_2_END}]"
# # # # #           f" 3[{PHASE_2_END+1}-{PHASE_3_END}] 4[{PHASE_3_END+1}+]")
# # # # #     print(f"  Target: ADE<{TARGETS['ADE']} 72h<{TARGETS['72h']}"
# # # # #           f" ATE<{TARGETS['ATE']} CTE<{TARGETS['CTE']}")
# # # # #     print("=" * 72)

# # # # #     # Data
# # # # #     trd, trl = data_loader(args, {"root": args.dataset_root, "type": "train"}, test=False)
# # # # #     vd,  vl  = data_loader(args, {"root": args.dataset_root, "type": "val"},   test=True)
# # # # #     from Model.data.trajectoriesWithMe_unet_training import seq_collate
# # # # #     vsub = _mksub(vd, args.val_subset_size, args.batch_size, seq_collate)
# # # # #     print(f"  train:{len(trd)} seqs  val:{len(vd)} seqs")

# # # # #     # Model
# # # # #     model = TCFlowMatching(
# # # # #         pred_len=args.pred_len, obs_len=args.obs_len,
# # # # #         sigma_min=args.sigma_min, use_ema=args.use_ema,
# # # # #         ema_decay=args.ema_decay, use_ate_ot=args.use_ot,
# # # # #         cfg_guidance_scale=args.cfg_guidance_scale,
# # # # #     ).to(dev)
# # # # #     model.init_ema()

# # # # #     n_total = sum(p.numel() for p in model.parameters() if p.requires_grad)

# # # # #     # BUG-8: backbone vs learned weights (step_weights, selector small params)
# # # # #     # Backbone: numel > 100
# # # # #     # Selector weights cũng là model weights, clip bình thường
# # # # #     # Chỉ KHÔNG clip step_weights (numel = 12)
# # # # #     backbone_params = [p for p in model.parameters()
# # # # #                        if p.requires_grad and p.numel() > 100]
# # # # #     step_w_params   = [p for p in model.parameters()
# # # # #                        if p.requires_grad and p.numel() <= 12]

# # # # #     print(f"  params total: {n_total:,}")
# # # # #     print(f"  backbone (clipped at {args.grad_clip}): "
# # # # #           f"{sum(p.numel() for p in backbone_params):,}")
# # # # #     print(f"  step_weights (NOT clipped): "
# # # # #           f"{sum(p.numel() for p in step_w_params):,}")

# # # # #     # Optimizer & scheduler
# # # # #     opt    = optim.AdamW(model.parameters(),
# # # # #                           lr=args.learning_rate,
# # # # #                           weight_decay=args.weight_decay)
# # # # #     scaler = GradScaler("cuda", enabled=args.use_amp)
# # # # #     nstep  = len(trl)
# # # # #     total  = nstep * args.num_epochs
# # # # #     wstp   = nstep * args.warmup_epochs
# # # # #     sched  = get_cosine_schedule_with_warmup(opt, wstp, total, min_lr=1e-6)

# # # # #     # [G-A] GradNorm manager
# # # # #     gradnorm = GradNormManager(
# # # # #         terms=GRADNORM_TERMS,
# # # # #         alpha_gn=args.gradnorm_alpha,
# # # # #         lr=args.gradnorm_lr,
# # # # #         device=dev,
# # # # #     )

# # # # #     # [FIX-STABILITY-1] SmoothScheduler — α theo batch, không phải epoch
# # # # #     smooth_sched = SmoothScheduler(total_steps_per_epoch=nstep)

# # # # #     # Savers
# # # # #     fast_saver = Saver(patience=args.patience, min_ep=args.min_ep,
# # # # #                         enable_stop=False)
# # # # #     full_saver = Saver(patience=args.patience, min_ep=args.min_ep,
# # # # #                         enable_stop=True)

# # # # #     # Resume
# # # # #     start = 0
# # # # #     if args.resume and os.path.exists(args.resume):
# # # # #         print(f"  Loading: {args.resume}")
# # # # #         ck = torch.load(args.resume, map_location=dev)
# # # # #         m  = _unwrap(model)
# # # # #         ms, _ = m.load_state_dict(ck["model_state_dict"], strict=False)
# # # # #         if ms: print(f"  Missing: {len(ms)}")
# # # # #         ema = getattr(m, "_ema", None)
# # # # #         if ema and ck.get("ema_shadow"):
# # # # #             for k, v in ck["ema_shadow"].items():
# # # # #                 if k in ema.shadow: ema.shadow[k].copy_(v.to(dev))
# # # # #         try: opt.load_state_dict(ck["optimizer_state"])
# # # # #         except Exception as e: print(f"  Opt: {e}")
# # # # #         try: sched.load_state_dict(ck["scheduler_state"])
# # # # #         except:
# # # # #             for _ in range(ck.get("epoch", 0) * nstep): sched.step()
# # # # #         for a, attr in [("best_score","bs"), ("best_ade","ba"), ("best_72h","b7"),
# # # # #                         ("best_ate","bat"), ("best_cte","bc"),
# # # # #                         ("best_easy_ade","best_easy_ade")]:
# # # # #             if a in ck:
# # # # #                 setattr(full_saver, attr, ck[a])
# # # # #                 setattr(fast_saver, attr, ck[a])
# # # # #         start = args.resume_epoch or ck.get("epoch", 0) + 1
# # # # #         print(f"  Resume ep={start}  best_score={full_saver.bs:.2f}"
# # # # #               f"  best_easy_ADE={full_saver.best_easy_ade:.1f}")
# # # # #         # Restore smooth_sched global step
# # # # #         smooth_sched._global_step = start * nstep

# # # # #     try:
# # # # #         model = torch.compile(model, mode="reduce-overhead")
# # # # #         print("  torch.compile: ok")
# # # # #     except:
# # # # #         pass

# # # # #     # ── Phase 4 encoder list (dùng khi freeze) ────────────────────────────
# # # # #     def _get_encoder_params(m):
# # # # #         """Trả về params của encoder để freeze ở phase 4."""
# # # # #         enc = _unwrap(m)
# # # # #         params = []
# # # # #         # Freeze: spatial_enc, enc_1d, env_enc (các encoder đầu vào)
# # # # #         for mod_name in ["net.spatial_enc", "net.enc_1d", "net.env_enc",
# # # # #                           "net.bottleneck_pool", "net.bottleneck_proj",
# # # # #                           "net.decoder_proj"]:
# # # # #             parts = mod_name.split(".")
# # # # #             mod = enc
# # # # #             for p in parts:
# # # # #                 if hasattr(mod, p):
# # # # #                     mod = getattr(mod, p)
# # # # #                 else:
# # # # #                     mod = None
# # # # #                     break
# # # # #             if mod is not None:
# # # # #                 params.extend(list(mod.parameters()))
# # # # #         return params

# # # # #     _phase4_frozen = False

# # # # #     def _maybe_freeze_encoder(ep, m):
# # # # #         nonlocal _phase4_frozen
# # # # #         if get_phase(ep) == 4 and not _phase4_frozen:
# # # # #             enc_params = _get_encoder_params(m)
# # # # #             for p in enc_params:
# # # # #                 p.requires_grad_(False)
# # # # #             _phase4_frozen = True
# # # # #             print(f"  [Phase 4] Froze encoder params"
# # # # #                   f" (n={sum(1 for p in enc_params)})")
# # # # #             # Giảm LR ×0.1 cho phase 4
# # # # #             for g in opt.param_groups:
# # # # #                 g["lr"] = g["lr"] * 0.1
# # # # #             print(f"  [Phase 4] LR reduced to {opt.param_groups[0]['lr']:.2e}")

# # # # #     # ── EMA guard state ────────────────────────────────────────────────────
# # # # #     best_easy_ade_ema  = float("inf")
# # # # #     ema_rollback_count = 0
# # # # #     _prev_phase        = 1
# # # # #     # [FIX-STABILITY-3] Global hard threshold — precomputed mỗi epoch
# # # # #     global_hard_threshold = 0.5  # default, sẽ được tính lại ngay

# # # # #     print(f"  Training: {nstep} steps/ep, start ep={start}")
# # # # #     print("=" * 72)
# # # # #     ts = time.perf_counter()

# # # # #     for ep in range(start, args.num_epochs):
# # # # #         phase = get_phase(ep)

# # # # #         # [FIX-STABILITY-2] GradNorm reset khi chuyển phase
# # # # #         if phase != _prev_phase:
# # # # #             gradnorm.phase_reset(phase)
# # # # #             _prev_phase = phase
# # # # #             print(f"  [Phase {phase}] Chuyển phase tại epoch {ep}")

# # # # #         # [FIX-STABILITY-3] Precompute global hard threshold mỗi epoch
# # # # #         # Chạy trước training loop để threshold ổn định trong suốt epoch
# # # # #         # Chỉ chạy từ phase 2 trở đi (phase 1 không dùng hard loss)
# # # # #         if phase >= 2:
# # # # #             global_hard_threshold = precompute_hard_threshold(trl, dev, n_batches=50)

# # # # #         # [G-H] Phase 4: freeze encoder
# # # # #         _maybe_freeze_encoder(ep, model)

# # # # #         model.train()
# # # # #         sl = 0.0
# # # # #         t0 = time.perf_counter()

# # # # #         # Lấy lambda hiện tại từ GradNorm
# # # # #         lambda_dict = gradnorm.get_lambda_dict()

# # # # #         for i, batch in enumerate(trl):
# # # # #             bl    = move(list(batch), dev)
# # # # #             obs_t = bl[0]

# # # # #             # [FIX-STABILITY-1] α, easy_frac từ SmoothScheduler (per-batch)
# # # # #             alpha_hard    = smooth_sched.get_alpha_hard()
# # # # #             easy_frac     = smooth_sched.get_easy_frac()
# # # # #             sel_weight    = smooth_sched.get_selector_weight()
# # # # #             train_selector = smooth_sched.is_selector_active() and (sel_weight > 0.01)

# # # # #             # [FIX-STABILITY-3] Dùng global threshold thay vì per-batch p70
# # # # #             with torch.no_grad():
# # # # #                 is_hard = classify_hard_easy_global(
# # # # #                     obs_t[:, :, :2], global_hard_threshold).to(dev)

# # # # #             # [G-C] Enforce easy_frac — chỉ điều chỉnh mask, không reindex batch
# # # # #             is_hard = enforce_easy_frac(is_hard, easy_frac)

# # # # #             with autocast(device_type="cuda", enabled=args.use_amp):
# # # # #                 bd = model.get_loss_breakdown(
# # # # #                     bl,
# # # # #                     epoch=ep,
# # # # #                     alpha_hard=alpha_hard,
# # # # #                     is_hard=is_hard,
# # # # #                     train_selector=train_selector,
# # # # #                     lambda_dict=lambda_dict,
# # # # #                 )

# # # # #             opt.zero_grad()
# # # # #             scaler.scale(bd["total"]).backward()
# # # # #             scaler.unscale_(opt)

# # # # #             # [G-A] GradNorm: đo grad_norm từ p.grad SAU backward, TRƯỚC clip
# # # # #             # Đây là cách đúng — không gọi backward() lần 2 (graph đã giải phóng)
# # # # #             # Dùng grad norm của ctx_fc2.weight (shared param cuối context fusion)
# # # # #             try:
# # # # #                 shared_p_name = "net.ctx_fc2.weight"
# # # # #                 shared_p = None
# # # # #                 m_unwrapped = _unwrap(model)
# # # # #                 for name, p in m_unwrapped.named_parameters():
# # # # #                     if name == shared_p_name and p.grad is not None:
# # # # #                         shared_p = p
# # # # #                         break

# # # # #                 if shared_p is not None and shared_p.grad is not None:
# # # # #                     # Tính grad_norm của từng term: ước lượng bằng cách nhìn
# # # # #                     # vào current grad norm của shared_p × λ_i / λ_mean
# # # # #                     base_gnorm = shared_p.grad.norm().item()
# # # # #                     lambda_dict_cur = gradnorm.get_lambda_dict()
# # # # #                     lambda_mean = np.mean(list(lambda_dict_cur.values()))

# # # # #                     lambda_grads_approx = {}
# # # # #                     for t in GRADNORM_TERMS:
# # # # #                         λ_i = lambda_dict_cur.get(t, 1.0)
# # # # #                         # Ước lượng: gnorm_i ≈ base_gnorm × λ_i / λ_mean
# # # # #                         # (gần đúng, đủ để GradNorm cân bằng)
# # # # #                         lambda_grads_approx[t] = torch.tensor(
# # # # #                             base_gnorm * λ_i / max(lambda_mean, 1e-6),
# # # # #                             device=dev)

# # # # #                     loss_vals_float = {
# # # # #                         t: float(bd[t].item()) if torch.is_tensor(bd.get(t)) else float(bd.get(t, 0.0))
# # # # #                         for t in GRADNORM_TERMS
# # # # #                         if t in bd
# # # # #                     }
# # # # #                     gradnorm.update(loss_vals_float, lambda_grads_approx)
# # # # #                     lambda_dict = gradnorm.get_lambda_dict()
# # # # #             except Exception:
# # # # #                 pass  # GradNorm không critical

# # # # #             # BUG-8: clip chỉ backbone
# # # # #             torch.nn.utils.clip_grad_norm_(backbone_params, args.grad_clip)

# # # # #             scaler.step(opt)
# # # # #             scaler.update()
# # # # #             # BUG-7: unconditional
# # # # #             try: sched.step()
# # # # #             except: pass
# # # # #             model.ema_update()

# # # # #             # [FIX-STABILITY-1] Advance smooth scheduler mỗi step
# # # # #             smooth_sched.step()

# # # # #             sl += bd["total"].item()

# # # # #             if i % 20 == 0:
# # # # #                 lr     = opt.param_groups[0]["lr"]
# # # # #                 tot    = bd["total"].item()
# # # # #                 warn   = " [!>10]" if tot > 10.0 else ""
# # # # #                 sw_st  = _unwrap(model).step_weights.stats()
# # # # #                 gn_log = gradnorm.log_stats()
# # # # #                 n_hard_batch = int(is_hard.sum().item())

# # # # #                 print(
# # # # #                     f"  [{ep:>3}|Ph{phase}][{i:>3}/{nstep}]"
# # # # #                     f"  tot={tot:.3f}{warn}"
# # # # #                     f"  fm={bd.get('l_fm',0):.4f}"
# # # # #                     f"  dpe={bd.get('dpe',0):.4f}"
# # # # #                     f"  vel={bd.get('vel_reg',0):.4f}"
# # # # #                     f"  hard={bd.get('l_hard_total',0):.4f}"
# # # # #                     f"  {smooth_sched.log_stats()}"
# # # # #                     f"  n_h={n_hard_batch}/{obs_t.shape[1]}"
# # # # #                     f"  sw72={sw_st['sw_72h']:.2f}"
# # # # #                     f"  gn_dense={gn_log.get('gn_dense',0)}"
# # # # #                     f"  lr={lr:.2e}"
# # # # #                 )

# # # # #         avt  = sl / nstep
# # # # #         t_ep = time.perf_counter() - t0

# # # # #         # BUG-5: val với epoch=-1 → skip augmentation
# # # # #         model.eval()
# # # # #         vls = 0.0
# # # # #         with torch.no_grad():
# # # # #             for batch in vl:
# # # # #                 bv = move(list(batch), dev)
# # # # #                 with autocast(device_type="cuda", enabled=args.use_amp):
# # # # #                     vls += model.get_loss(bv, epoch=-1).item()
# # # # #         avv = vls / len(vl)

# # # # #         lr_cur   = opt.param_groups[0]["lr"]
# # # # #         gn_stats = gradnorm.log_stats()
# # # # #         alpha_hard_ep = smooth_sched.get_alpha_hard()
# # # # #         easy_frac_ep  = smooth_sched.get_easy_frac()
# # # # #         use_sel_eval  = smooth_sched.is_selector_active()

# # # # #         print(f"  Epoch {ep:>3}|Ph{phase}"
# # # # #               f" | train={avt:.4f} val={avv:.4f}"
# # # # #               f" | {smooth_sched.log_stats()}"
# # # # #               f" | λ_dpe={gn_stats.get('λ_l_dpe',1.0):.2f}"
# # # # #               f" λ_vel={gn_stats.get('λ_l_vel_reg',1.0):.2f}"
# # # # #               f" | lr={lr_cur:.2e} | {t_ep:.0f}s")

# # # # #         # [G-G] Fast eval với split monitoring (dùng fast ddim)
# # # # #         rf = evaluate_split(model, vsub, dev,
# # # # #                              tag=f"FAST ep{ep}",
# # # # #                              steps=args.fast_ddim,
# # # # #                              use_selector=use_sel_eval)
# # # # #         fast_saver.update(rf, model, args.output_dir, ep, opt, sched, avt, avv, tag="fast")

# # # # #         # [G-E] EMA guard: kiểm tra easy_ADE sau mỗi epoch
# # # # #         easy_ade_curr = rf.get("easy_ADE", rf.get("ADE", float("inf")))
# # # # #         if np.isfinite(easy_ade_curr):
# # # # #             if easy_ade_curr < best_easy_ade_ema:
# # # # #                 best_easy_ade_ema = easy_ade_curr

# # # # #             elif (easy_ade_curr - best_easy_ade_ema) > args.ema_guard_threshold:
# # # # #                 ema_rollback_count += 1
# # # # #                 print(f"\n  [EMA GUARD] easy_ADE tăng "
# # # # #                       f"{easy_ade_curr - best_easy_ade_ema:.1f} km"
# # # # #                       f" (best={best_easy_ade_ema:.1f} curr={easy_ade_curr:.1f})")
# # # # #                 print(f"  [EMA GUARD] Rollback #{ema_rollback_count}: restore EMA weights")

# # # # #                 # Restore từ EMA
# # # # #                 m   = _unwrap(model)
# # # # #                 ema = getattr(m, "_ema", None)
# # # # #                 if ema:
# # # # #                     try: ema.apply_to(model)
# # # # #                     except: pass

# # # # #                 # Đẩy smooth_sched lùi lại để α giảm xuống
# # # # #                 # Hiệu quả: α sẽ nhỏ hơn ở các batch tiếp theo
# # # # #                 pullback_steps = nstep * 3  # tương đương 3 epoch
# # # # #                 smooth_sched._global_step = max(
# # # # #                     smooth_sched._phase2_start_step,
# # # # #                     smooth_sched._global_step - pullback_steps
# # # # #                 )
# # # # #                 print(f"  [EMA GUARD] α giảm về {smooth_sched.get_alpha_hard():.3f}"
# # # # #                       f" (step pulled back {pullback_steps} steps)")

# # # # #                 # Lưu rollback checkpoint
# # # # #                 _save_ckpt(
# # # # #                     os.path.join(args.output_dir,
# # # # #                                  f"ema_guard_rollback_ep{ep}.pth"),
# # # # #                     ep, model, opt, sched, full_saver, avt, avv,
# # # # #                     extra={"ema_rollback": True,
# # # # #                            "easy_ade_trigger": easy_ade_curr},
# # # # #                 )

# # # # #                 if ema_rollback_count >= 3:
# # # # #                     print(f"  [EMA GUARD] 3 lần rollback — dừng tăng hard loss")
# # # # #                     if phase == 3:
# # # # #                         print(f"  [EMA GUARD] Tắt selector training")
# # # # #                         # Đưa smooth_sched về trước phase 3
# # # # #                         smooth_sched._global_step = min(
# # # # #                             smooth_sched._global_step,
# # # # #                             smooth_sched._phase3_start_step - 1
# # # # #                         )

# # # # #         # [G-F] Diversity check cuối phase 1
# # # # #         div_score = None
# # # # #         if ep == PHASE_1_END:
# # # # #             print(f"\n  {'='*50}")
# # # # #             print(f"  [CHECKPOINT] Cuối phase 1 — kiểm tra diversity...")
# # # # #             div_score = check_diversity(model, vsub, dev)
# # # # #             print(f"  Diversity score: {div_score:.1f} km")
# # # # #             if div_score < args.diversity_threshold:
# # # # #                 print(f"  [CẢNH BÁO R1] diversity={div_score:.1f} km"
# # # # #                       f" < {args.diversity_threshold} km")
# # # # #                 print(f"  → FM candidates không đủ đa dạng → selector sẽ không hiệu quả")
# # # # #                 print(f"  → Hành động: kiểm tra noise_std hoặc thêm L_diversity")
# # # # #             else:
# # # # #                 print(f"  [OK] diversity={div_score:.1f} km >= "
# # # # #                       f"{args.diversity_threshold} km → tiến hành phase 2")
# # # # #             print(f"  {'='*50}\n")

# # # # #         # Đánh giá đầy đủ mỗi val_freq epoch
# # # # #         if ep % args.val_freq == 0:
# # # # #             em = getattr(_unwrap(model), "_ema", None)
# # # # #             rr = evaluate_split(model, vl, dev,
# # # # #                                   tag=f"RAW ep{ep}",
# # # # #                                   steps=args.full_ddim,
# # # # #                                   use_selector=use_sel_eval)
# # # # #             full_saver.update(rr, model, args.output_dir, ep, opt, sched,
# # # # #                               avt, avv, tag="raw")

# # # # #             if em and ep >= 3:
# # # # #                 re = evaluate_split(model, vl, dev,
# # # # #                                      tag=f"EMA ep{ep}",
# # # # #                                      ema=em,
# # # # #                                      steps=args.full_ddim,
# # # # #                                      use_selector=use_sel_eval)
# # # # #                 full_saver.update(re, model, args.output_dir, ep, opt, sched,
# # # # #                                   avt, avv, tag="ema")

# # # # #         # Periodic checkpoint
# # # # #         if ep % 10 == 0 or ep == args.num_epochs - 1:
# # # # #             _save_ckpt(
# # # # #                 os.path.join(args.output_dir, f"ckpt_ep{ep:03d}.pth"),
# # # # #                 ep, model, opt, sched, full_saver, avt, avv,
# # # # #                 extra={"phase": phase,
# # # # #                        "alpha_hard": smooth_sched.get_alpha_hard(),
# # # # #                        "smooth_sched_step": smooth_sched.global_step,
# # # # #                        "global_hard_threshold": global_hard_threshold,
# # # # #                        "diversity": div_score},
# # # # #             )

# # # # #         if full_saver.stop:
# # # # #             print(f"  Early stop ep={ep}")
# # # # #             break

# # # # #     th = (time.perf_counter() - ts) / 3600.0
# # # # #     print(f"\n  Best: ADE={full_saver.ba:.1f}  72h={full_saver.b7:.0f}"
# # # # #           f"  ATE={full_saver.bat:.1f}  CTE={full_saver.bc:.1f}"
# # # # #           f"  easy_ADE={full_saver.best_easy_ade:.1f}"
# # # # #           f"  ({th:.2f}h)")

# # # # #     # Post-training test
# # # # #     if args.eval_test_after_train:
# # # # #         print("\n" + "="*72 + "\n  POST-TRAINING TEST\n" + "="*72)
# # # # #         try: _, tl2 = data_loader(args, {"root": args.dataset_root, "type": "test"}, test=True)
# # # # #         except: print("  No test → using val"); tl2 = vl

# # # # #         for fn, lb in [("best_raw.pth", "RAW"), ("best_ema.pth", "EMA"),
# # # # #                         ("best_ade_raw.pth", "BEST_ADE"),
# # # # #                         ("best_cte_raw.pth", "BEST_CTE")]:
# # # # #             pp = os.path.join(args.output_dir, fn)
# # # # #             if not os.path.exists(pp): continue
# # # # #             ck = torch.load(pp, map_location=dev)
# # # # #             _unwrap(model).load_state_dict(ck["model_state_dict"], strict=False)
# # # # #             em = getattr(_unwrap(model), "_ema", None)
# # # # #             if em and ck.get("ema_shadow"):
# # # # #                 for k, v in ck["ema_shadow"].items():
# # # # #                     if k in em.shadow: em.shadow[k].copy_(v.to(dev))
# # # # #             r = evaluate_split(model, tl2, dev, tag=f"TEST/{lb}",
# # # # #                                 steps=args.full_ddim,
# # # # #                                 use_selector=(get_phase(ck.get("epoch", 80)) >= 3))
# # # # #             print(f"\n  --- {lb} (ep={ck.get('epoch','?')}) ---")
# # # # #             for key, ref in [("ADE", 172.68), ("72h", 321.39),
# # # # #                               ("ATE", 142.21), ("CTE", 42.04)]:
# # # # #                 v = r.get(key, float("nan"))
# # # # #                 mk = "BEAT!" if np.isfinite(v) and v < ref else f"need {ref:.0f}"
# # # # #                 print(f"    {key:<8}: {v:>8.1f} km  [{mk}  gap:{v-ref:+.1f}]")
# # # # #             ea = r.get("easy_ADE", float("nan"))
# # # # #             ha = r.get("hard_ADE", float("nan"))
# # # # #             print(f"    easy_ADE={ea:.1f}  hard_ADE={ha:.1f}")

# # # # #     print("=" * 72)


# # # # # if __name__ == "__main__":
# # # # #     args = get_args()
# # # # #     np.random.seed(42)
# # # # #     torch.manual_seed(42)
# # # # #     if torch.cuda.is_available():
# # # # #         torch.cuda.manual_seed_all(42)
# # # # #     main(args)

# # # # """
# # # # train_v59_strategy.py — TC-FlowMatching v59-Strategy Training Script
# # # # ══════════════════════════════════════════════════════════════════════════════

# # # # THAY ĐỔI SO VỚI train_v59_tweaked.py:

# # # #   [G-A] GradNorm: tự điều chỉnh λ_dpe, λ_vel, λ_head, λ_spd, λ_acc
# # # #         Không hardcode hệ số loss — GradNorm đo loss ratio sau backward,
# # # #         update λ để các term có gradient norm xấp xỉ nhau.
# # # #         GradNorm reset λ=1.0 khi phase đổi và tăng update_freq
# # # #         tạm thời 200 steps để thích nghi nhanh hơn → tránh lag gradient.

# # # #   [G-B] Easy/hard pipeline với GLOBAL threshold (fix per-batch flip):
# # # #         - hard_score tính mỗi epoch trên TOÀN BỘ train set (không phải per-batch)
# # # #         - Lưu global_hard_threshold = p70 của toàn dataset
# # # #         - Mỗi batch dùng threshold cố định đó → sample A luôn là hard/easy
# # # #           nhất quán trong suốt epoch, gradient ổn định hơn nhiều
# # # #         - Phase 1: is_hard = zeros (alpha_hard=0, không cần phân loại)

# # # #   [G-C] easy_frac enforcement:
# # # #         - Mỗi batch: đảm bảo ≥ easy_frac easy samples
# # # #         - Chỉ điều chỉnh is_hard mask, KHÔNG reindex batch data

# # # #   [G-D] α smooth warm-up theo BATCH (fix hard cliff):
# # # #         - Sigmoid curve qua từng batch
# # # #         - Selector weight tăng từ 0 → 0.2 tương tự

# # # #   [G-E] EMA guard sau mỗi epoch:
# # # #         - Đo easy_ADE trên val subset (easy samples only, dùng global threshold)
# # # #         - Nếu easy_ADE tăng > 3 km vs best_easy_ADE → rollback EMA + giảm α
# # # #         - FIXED: sau rollback restore weights đúng rồi tiếp tục train với
# # # #           current model (không để model bị kẹt ở EMA weights mãi)

# # # #   [G-F] Diversity check sau phase 1 (epoch 19)

# # # #   [G-G] Split monitoring mỗi epoch (easy_ADE, hard_ADE riêng biệt)
# # # #         dùng global_hard_threshold để consistent với training

# # # #   [G-H] Phase 4: freeze encoder, chỉ tune FM head + selector head

# # # # BUG fixes giữ nguyên từ v59-tweaked:
# # # #   BUG-5: val_loss dùng epoch=-1 → skip augmentation
# # # #   BUG-6: any_improve = ANY metric improve → reset early stop
# # # #   BUG-7: sched.step() unconditional
# # # #   BUG-8: clip chỉ backbone params, không clip learned weights

# # # # BUG fixes MỚI trong v59-strategy (so với phiên bản trước):
# # # #   FIX-1:  Phase 1 không tính is_hard (threshold chưa được compute)
# # # #   FIX-2:  GradNorm dùng loss-ratio weighting thực sự thay vì approximation sai
# # # #   FIX-3:  GradNorm.update() xóa dead backward() call, sửa opt.zero_grad() order
# # # #   FIX-4:  evaluate_split() nhận global_hard_threshold để consistent split
# # # #   FIX-5:  EMA guard rollback: restore EMA weights đúng cách rồi tiếp tục
# # # #           train với current model (không bị kẹt EMA weights vĩnh viễn)
# # # #   FIX-6:  phase_reset() reset _step=0 để dense mode đếm đúng
# # # #   FIX-7:  get_easy_frac() phase 4 dùng sigmoid giống các phase khác
# # # #   FIX-8:  _save_ckpt lưu smooth_sched_step cho mọi loại checkpoint
# # # # """
# # # # from __future__ import annotations
# # # # import sys, os
# # # # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # # # import argparse, math, random, time
# # # # from collections import defaultdict
# # # # from typing import Dict, List, Optional

# # # # import numpy as np
# # # # import torch
# # # # import torch.optim as optim
# # # # import torch.nn.functional as F
# # # # from torch.amp import autocast, GradScaler
# # # # from torch.utils.data import DataLoader, Subset

# # # # from Model.data.loader_training import data_loader
# # # # from Model.flow_matching_model import (
# # # #     TCFlowMatching,
# # # #     hard_score_from_obs,
# # # #     classify_hard_easy,
# # # #     compute_diversity_score,
# # # #     _norm_to_deg,
# # # #     _haversine_deg,
# # # # )

# # # # try:
# # # #     from Model.utils import get_cosine_schedule_with_warmup
# # # # except ImportError:
# # # #     from torch.optim.lr_scheduler import CosineAnnealingLR
# # # #     def get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps, min_lr=1e-6):
# # # #         return CosineAnnealingLR(opt, T_max=max(total_steps, 1), eta_min=min_lr)

# # # # TARGETS = {
# # # #     "ADE": 172.68, "72h": 321.39, "ATE": 142.21, "CTE": 42.04,
# # # #     "12h": 65.42,  "24h": 104.67, "48h": 205.10,
# # # # }
# # # # R_EARTH = 6371.0

# # # # # ── Phase boundaries ──────────────────────────────────────────────────────────
# # # # PHASE_1_END  = 19   # epoch 0–19:  warm-up, α=0, easy_frac=80%
# # # # PHASE_2_END  = 49   # epoch 20–49: hard intro, α 0→0.3, easy_frac 80→60%
# # # # PHASE_3_END  = 79   # epoch 50–79: selector, α=0.3, easy_frac=55%
# # # # # epoch 80+: fine-tune, LR×0.1, freeze encoder

# # # # # ── GradNorm terms ────────────────────────────────────────────────────────────
# # # # GRADNORM_TERMS = ["l_dpe", "l_vel_reg", "l_heading", "l_speed", "l_accel"]


# # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # #  GradNorm implementation
# # # # #
# # # # #  Thuật toán loss-ratio weighting:
# # # # #    1. Sau mỗi backward, lấy giá trị loss của từng term (đã detach)
# # # # #    2. Tính mean_loss = mean({l_i})
# # # # #    3. r_i = l_i / mean_loss  (relative magnitude của term i)
# # # # #    4. Nếu r_i > 1 → term này lớn hơn TB → tăng λ_i (cần học nhiều hơn)
# # # # #       Nếu r_i < 1 → term này nhỏ hơn TB → giảm λ_i
# # # # #    5. Update λ_i qua Adam với grad được set thủ công
# # # # #    6. Re-normalize: sum(λ_i) = n_terms sau mỗi update
# # # # #
# # # # #  Lưu ý quan trọng:
# # # # #    - λ_i là meta-parameters (KHÔNG phải model weights)
# # # # #    - Gradient KHÔNG chạy qua model, chỉ update λ ngoài graph
# # # # #    - Không cần backward() vì ta set grad thủ công dựa trên loss ratio
# # # # #
# # # # #  [FIX-2] Thay thế approximation sai (base_gnorm × λ_i / λ_mean):
# # # # #    Cách cũ dùng cùng base_gnorm cho mọi term → ratio giữa approx
# # # # #    luôn bằng ratio giữa λ → GradNorm chỉ re-normalize về 1.0 mãi mãi.
# # # # #    Cách mới: dùng loss ratio thực sự để drive λ update.
# # # # #
# # # # #  [FIX-3] Xóa dead backward() call trong update():
# # # # #    gradnorm_loss.backward() vô nghĩa vì gradnorm_loss không connected
# # # # #    với self.lambdas (là torch.tensor constants, không có grad_fn).
# # # # #    Đồng thời sửa thứ tự: zero_grad() PHẢI gọi trước khi set grad,
# # # # #    không phải trước backward() của loss không liên quan.
# # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # class GradNormManager:
# # # #     """
# # # #     Quản lý GradNorm cho các loss terms.

# # # #     [FIX-6] phase_reset() reset _step=0 để dense mode đếm đúng:
# # # #     - Khi phase đổi: reset λ=1.0, reset optimizer state, reset _step=0
# # # #     - Bật dense update mode trong 200 steps để thích nghi nhanh

# # # #     [FIX-2+3] update() dùng loss-ratio weighting thực sự:
# # # #     - Không gọi backward() (dead code, tensor không connected với lambdas)
# # # #     - Set grad thủ công dựa trên loss ratio r_i = l_i / mean_loss
# # # #     - zero_grad() đúng thứ tự: TRƯỚC khi set grad
# # # #     """
# # # #     def __init__(self, terms: List[str], alpha_gn: float = 1.5,
# # # #                  lr: float = 1e-3, device=None):
# # # #         self.terms    = terms
# # # #         self.alpha_gn = alpha_gn  # restoring force (giữ cho backward compat)
# # # #         self.device   = device or torch.device("cpu")

# # # #         # λ_i: meta-optimization parameter (KHÔNG phải model weights)
# # # #         self.lambdas = {t: torch.ones(1, device=self.device, requires_grad=True)
# # # #                         for t in terms}
# # # #         self.opt = optim.Adam(list(self.lambdas.values()), lr=lr)

# # # #         self._step             = 0
# # # #         self._update_freq      = 5
# # # #         self._dense_steps_left = 0
# # # #         self._current_phase    = 1

# # # #     def phase_reset(self, new_phase: int) -> None:
# # # #         """
# # # #         [FIX-6] Gọi khi chuyển sang phase mới.
# # # #         Reset λ=1.0, reset optimizer state, reset _step=0.
# # # #         Bật dense update mode trong 200 steps.
# # # #         """
# # # #         if new_phase == self._current_phase:
# # # #             return
# # # #         self._current_phase = new_phase

# # # #         with torch.no_grad():
# # # #             for t in self.terms:
# # # #                 self.lambdas[t].fill_(1.0)

# # # #         # Reset optimizer để tránh momentum cũ kéo λ sai hướng
# # # #         self.opt = optim.Adam(list(self.lambdas.values()),
# # # #                                lr=self.opt.param_groups[0]["lr"])

# # # #         # [FIX-6] Reset _step=0 để dense mode đếm đúng từ đầu
# # # #         self._step = 0
# # # #         self._dense_steps_left = 200

# # # #         print(f"  [GradNorm] Phase {new_phase}: reset λ=1.0, _step=0, "
# # # #               f"dense update 200 steps")

# # # #     def get_lambda_dict(self) -> Dict[str, float]:
# # # #         """Trả về λ hiện tại để truyền vào get_loss_breakdown()."""
# # # #         return {t: float(self.lambdas[t].item()) for t in self.terms}

# # # #     def update(self, loss_term_values: Dict[str, float]) -> None:
# # # #         """
# # # #         [FIX-2+3] Cập nhật λ dựa trên loss ratio thực sự.

# # # #         Thuật toán:
# # # #           r_i = l_i / mean_loss
# # # #           grad_i = -(r_i - 1.0) * scale
# # # #             → r_i > 1: term lớn hơn TB, cần tăng λ → grad âm (Adam minimize)
# # # #             → r_i < 1: term nhỏ hơn TB, cần giảm λ → grad dương

# # # #         KHÔNG cần backward() vì:
# # # #           - gradnorm_loss cũ là torch.tensor constants, không có grad_fn
# # # #           - lambdas không trong computation graph của model loss
# # # #           - Ta set grad thủ công dựa trên loss ratio

# # # #         Args:
# # # #             loss_term_values: {term_name: float_value} — loss của từng term
# # # #         """
# # # #         self._step += 1

# # # #         # Adaptive update frequency
# # # #         if self._dense_steps_left > 0:
# # # #             self._dense_steps_left -= 1
# # # #             update_freq = 1
# # # #         else:
# # # #             update_freq = self._update_freq

# # # #         if self._step % update_freq != 0:
# # # #             return

# # # #         # Lọc finite và positive values
# # # #         loss_vals = {t: v for t, v in loss_term_values.items()
# # # #                      if t in self.terms and np.isfinite(v) and v > 0}
# # # #         if len(loss_vals) < 2:
# # # #             # Chưa đủ terms để cân bằng, chỉ re-normalize
# # # #             self._renormalize()
# # # #             return

# # # #         mean_loss = np.mean(list(loss_vals.values()))
# # # #         if mean_loss <= 0:
# # # #             return

# # # #         # [FIX-3] zero_grad() TRƯỚC khi set grad (không phải trước backward)
# # # #         self.opt.zero_grad()

# # # #         for t in self.terms:
# # # #             if t not in loss_vals:
# # # #                 continue
# # # #             r_i = loss_vals[t] / (mean_loss + 1e-8)
# # # #             # r_i > 1 → term này lớn hơn TB → cần tăng λ_i
# # # #             # Để Adam tăng λ_i, ta set grad âm (Adam: param -= lr * grad)
# # # #             # → grad = -(r_i - 1.0) * scale
# # # #             # Scale 0.1 để update không quá mạnh trong 1 step
# # # #             grad_val = -(r_i - 1.0) * 0.1
# # # #             if self.lambdas[t].grad is None:
# # # #                 self.lambdas[t].grad = torch.zeros_like(self.lambdas[t])
# # # #             self.lambdas[t].grad.fill_(grad_val)

# # # #         self.opt.step()
# # # #         self._renormalize()

# # # #     def _renormalize(self) -> None:
# # # #         """
# # # #         Re-normalize λ để sum(λ) = n_terms sau mỗi update.

# # # #         2-pass algorithm để tránh bug:
# # # #           Pass 1: clamp_ tất cả về [0.01, ∞) IN-PLACE
# # # #           Pass 2: tính tổng trên giá trị ĐÃ clamped, rồi normalize
# # # #         Tách 2 bước đảm bảo total_λ tính từ giá trị thực tế sau clamp,
# # # #         không phải giá trị trước clamp (tránh discrepancy khi λ < 0.01).
# # # #         """
# # # #         with torch.no_grad():
# # # #             n_terms = len(self.terms)
# # # #             # Pass 1: clamp in-place tất cả trước
# # # #             for t in self.terms:
# # # #                 self.lambdas[t].clamp_(min=0.01)
# # # #             # Pass 2: tính tổng từ giá trị đã clamped rồi normalize
# # # #             total_λ = sum(float(self.lambdas[t].item()) for t in self.terms)
# # # #             if total_λ > 0:
# # # #                 scale = n_terms / total_λ
# # # #                 for t in self.terms:
# # # #                     self.lambdas[t].mul_(scale)

# # # #     def log_stats(self) -> Dict[str, float]:
# # # #         """Log λ hiện tại để monitor."""
# # # #         d = {f"λ_{t}": float(self.lambdas[t].item()) for t in self.terms}
# # # #         d["gn_dense"] = self._dense_steps_left
# # # #         d["gn_step"]  = self._step
# # # #         return d


# # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # #  Phase helper
# # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # def get_phase(epoch: int) -> int:
# # # #     if epoch <= PHASE_1_END:  return 1
# # # #     if epoch <= PHASE_2_END:  return 2
# # # #     if epoch <= PHASE_3_END:  return 3
# # # #     return 4


# # # # class SmoothScheduler:
# # # #     """
# # # #     α smooth warm-up theo BATCH, không phải epoch.

# # # #     Sigmoid curve qua từng BATCH:
# # # #         α(step) = α_target × sigmoid((step - mid_step) / temperature)

# # # #     [FIX-7] get_easy_frac() phase 4 dùng sigmoid thay vì linear:
# # # #         Consistent với phương pháp của các phase khác.
# # # #         Đảm bảo không bao giờ xuống dưới 0.50.
# # # #     """
# # # #     def __init__(self, total_steps_per_epoch: int):
# # # #         self.steps_per_epoch = total_steps_per_epoch
# # # #         self._global_step = 0

# # # #         self._phase2_start_step = (PHASE_1_END + 1) * total_steps_per_epoch
# # # #         self._phase3_start_step = (PHASE_2_END + 1) * total_steps_per_epoch
# # # #         self._phase4_start_step = (PHASE_3_END + 1) * total_steps_per_epoch

# # # #         # Temperature: transition kéo dài ~2 epoch
# # # #         self._temp_p2 = 2.0 * total_steps_per_epoch / 4.4
# # # #         self._temp_p3 = 1.5 * total_steps_per_epoch / 4.4
# # # #         # [FIX-7] Temperature cho phase 4 (giảm từ 55%→50% qua 5 epoch)
# # # #         self._temp_p4 = 2.5 * total_steps_per_epoch / 4.4

# # # #     def step(self):
# # # #         self._global_step += 1

# # # #     @property
# # # #     def global_step(self):
# # # #         return self._global_step

# # # #     def get_alpha_hard(self) -> float:
# # # #         s = self._global_step
# # # #         if s < self._phase2_start_step:
# # # #             return 0.0
# # # #         if s >= self._phase3_start_step:
# # # #             return 0.3

# # # #         mid = self._phase2_start_step + (
# # # #             self._phase3_start_step - self._phase2_start_step) / 2.0
# # # #         x   = (s - mid) / max(self._temp_p2, 1.0)
# # # #         sig = 1.0 / (1.0 + math.exp(-x))
# # # #         return 0.3 * sig

# # # #     def get_selector_weight(self) -> float:
# # # #         s = self._global_step
# # # #         if s < self._phase3_start_step:
# # # #             return 0.0
# # # #         if s >= self._phase4_start_step:
# # # #             return 0.2

# # # #         mid = self._phase3_start_step + (
# # # #             self._phase4_start_step - self._phase3_start_step) / 2.0
# # # #         x   = (s - mid) / max(self._temp_p3, 1.0)
# # # #         sig = 1.0 / (1.0 + math.exp(-x))
# # # #         return 0.2 * sig

# # # #     def get_easy_frac(self) -> float:
# # # #         """
# # # #         [FIX-7] Phase 4 dùng sigmoid thay vì linear.
# # # #         Tất cả phases đều smooth qua sigmoid, không có linear segment.
# # # #         Floor cứng ở 0.50.
# # # #         """
# # # #         s = self._global_step

# # # #         # Phase 1: 80% cố định
# # # #         if s < self._phase2_start_step:
# # # #             return 0.80

# # # #         # Phase 2: 80% → 60% qua sigmoid
# # # #         if s < self._phase3_start_step:
# # # #             mid = self._phase2_start_step + (
# # # #                 self._phase3_start_step - self._phase2_start_step) / 2.0
# # # #             x   = (s - mid) / max(self._temp_p2, 1.0)
# # # #             sig = 1.0 / (1.0 + math.exp(-x))
# # # #             return 0.80 - 0.20 * sig  # 80% → 60%

# # # #         # Phase 3: 60% → 55% qua sigmoid
# # # #         if s < self._phase4_start_step:
# # # #             mid = self._phase3_start_step + (
# # # #                 self._phase4_start_step - self._phase3_start_step) / 2.0
# # # #             x   = (s - mid) / max(self._temp_p3, 1.0)
# # # #             sig = 1.0 / (1.0 + math.exp(-x))
# # # #             return 0.60 - 0.05 * sig  # 60% → 55%

# # # #         # [FIX-7] Phase 4: 55% → 50% qua sigmoid (không phải linear)
# # # #         # Midpoint: 5 epoch sau khi vào phase 4
# # # #         phase4_mid = self._phase4_start_step + 5 * self.steps_per_epoch
# # # #         x   = (s - phase4_mid) / max(self._temp_p4, 1.0)
# # # #         sig = 1.0 / (1.0 + math.exp(-x))
# # # #         # Sigmoid đi từ gần 0 → gần 1 khi s tăng
# # # #         # Kết quả: 0.55 - 0.05*sig → giảm từ ~0.55 về ~0.50
# # # #         return max(0.50, 0.55 - 0.05 * sig)

# # # #     def is_selector_active(self) -> bool:
# # # #         return self._global_step >= self._phase3_start_step

# # # #     def log_stats(self) -> str:
# # # #         return (f"α={self.get_alpha_hard():.3f}"
# # # #                 f" easy={self.get_easy_frac():.0%}"
# # # #                 f" sel={self.get_selector_weight():.3f}")


# # # # # Backward-compat wrappers
# # # # def get_alpha_hard(epoch: int) -> float:
# # # #     if epoch <= PHASE_1_END: return 0.0
# # # #     if epoch <= PHASE_2_END:
# # # #         return 0.3 * (epoch - PHASE_1_END) / (PHASE_2_END - PHASE_1_END)
# # # #     return 0.3

# # # # def get_easy_frac(epoch: int) -> float:
# # # #     if epoch <= PHASE_1_END: return 0.80
# # # #     if epoch <= PHASE_2_END:
# # # #         p = (epoch - PHASE_1_END) / (PHASE_2_END - PHASE_1_END)
# # # #         return 0.80 - 0.20 * p
# # # #     if epoch <= PHASE_3_END: return 0.55
# # # #     return 0.50


# # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # #  Global hard threshold
# # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # @torch.no_grad()
# # # # def precompute_hard_threshold(loader, dev, n_batches: int = 50) -> float:
# # # #     """
# # # #     Tính p70 của hard_score trên subset của train set.
# # # #     Chỉ gọi từ phase 2 trở đi (phase 1 không cần).
# # # #     """
# # # #     all_scores = []
# # # #     for i, batch in enumerate(loader):
# # # #         if i >= n_batches:
# # # #             break
# # # #         bl    = move(list(batch), dev)
# # # #         obs_t = bl[0]
# # # #         scores = hard_score_from_obs(obs_t[:, :, :2])  # [B]
# # # #         all_scores.append(scores.cpu())

# # # #     if not all_scores:
# # # #         return 0.7  # fallback

# # # #     all_scores_cat = torch.cat(all_scores)
# # # #     threshold = float(torch.quantile(all_scores_cat, 0.70).item())
# # # #     n_hard  = int((all_scores_cat >= threshold).sum().item())
# # # #     n_total = len(all_scores_cat)
# # # #     print(f"  [HardThreshold] global p70={threshold:.4f}"
# # # #           f"  n_hard={n_hard}/{n_total} ({100*n_hard/max(n_total,1):.0f}%)")
# # # #     return threshold


# # # # @torch.no_grad()
# # # # def classify_hard_easy_global(
# # # #     obs_traj_norm: torch.Tensor,
# # # #     global_threshold: float,
# # # # ) -> torch.Tensor:
# # # #     """
# # # #     Phân loại hard/easy dùng global threshold (thay vì per-batch p70).
# # # #     """
# # # #     scores  = hard_score_from_obs(obs_traj_norm)  # [B]
# # # #     is_hard = scores >= global_threshold
# # # #     return is_hard


# # # # def enforce_easy_frac(
# # # #     is_hard: torch.Tensor,
# # # #     easy_frac: float,
# # # # ) -> torch.Tensor:
# # # #     """
# # # #     Chỉ điều chỉnh is_hard mask, KHÔNG reindex batch data.
# # # #     Nếu batch có quá nhiều hard → demote một số về easy bằng cách
# # # #     set is_hard[i] = False (batch data giữ nguyên 100%).
# # # #     """
# # # #     B        = is_hard.shape[0]
# # # #     max_hard = max(0, int(B * (1.0 - easy_frac)))
# # # #     n_hard   = int(is_hard.sum().item())

# # # #     if n_hard <= max_hard:
# # # #         return is_hard

# # # #     n_demote  = n_hard - max_hard
# # # #     hard_idx  = is_hard.nonzero(as_tuple=True)[0]
# # # #     perm      = torch.randperm(hard_idx.numel(), device=is_hard.device)
# # # #     to_demote = hard_idx[perm[:n_demote]]

# # # #     is_hard_new            = is_hard.clone()
# # # #     is_hard_new[to_demote] = False
# # # #     return is_hard_new


# # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # #  Utilities
# # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # def _unwrap(m):
# # # #     return m._orig_mod if hasattr(m, "_orig_mod") else m

# # # # def move(b, dev):
# # # #     out = list(b)
# # # #     for i, x in enumerate(out):
# # # #         if torch.is_tensor(x): out[i] = x.to(dev)
# # # #         elif isinstance(x, dict):
# # # #             out[i] = {k: v.to(dev) if torch.is_tensor(v) else v
# # # #                       for k, v in x.items()}
# # # #     return out

# # # # def _ntd(a):
# # # #     o = a.clone()
# # # #     o[..., 0] = (a[..., 0] * 50.0 + 1800.0) / 10.0
# # # #     o[..., 1] = (a[..., 1] * 50.0) / 10.0
# # # #     return o

# # # # def _hav(p1, p2):
# # # #     la1 = torch.deg2rad(p1[..., 1]); la2 = torch.deg2rad(p2[..., 1])
# # # #     dlat = torch.deg2rad(p2[..., 1] - p1[..., 1])
# # # #     dlon = torch.deg2rad(p2[..., 0] - p1[..., 0])
# # # #     a = (torch.sin(dlat/2).pow(2)
# # # #          + torch.cos(la1)*torch.cos(la2)*torch.sin(dlon/2).pow(2))
# # # #     return 2.0 * R_EARTH * torch.asin(a.clamp(1e-12, 1-1e-12).sqrt())

# # # # def _atecte(pd, gd):
# # # #     T = min(pd.shape[0], gd.shape[0])
# # # #     if T < 2:
# # # #         z = pd.new_zeros(1, pd.shape[1]); return z, z
# # # #     lo1=torch.deg2rad(gd[:T-1,:,0]); la1=torch.deg2rad(gd[:T-1,:,1])
# # # #     lo2=torch.deg2rad(gd[1:T, :,0]); la2=torch.deg2rad(gd[1:T, :,1])
# # # #     lo3=torch.deg2rad(pd[1:T, :,0]); la3=torch.deg2rad(pd[1:T, :,1])
# # # #     ya=torch.sin(lo2-lo1)*torch.cos(la2)
# # # #     xa=(torch.cos(la1)*torch.sin(la2)
# # # #         -torch.sin(la1)*torch.cos(la2)*torch.cos(lo2-lo1))
# # # #     be=torch.atan2(ya,xa)
# # # #     ye=torch.sin(lo3-lo2)*torch.cos(la3)
# # # #     xe=(torch.cos(la2)*torch.sin(la3)
# # # #         -torch.sin(la2)*torch.cos(la3)*torch.cos(lo3-lo2))
# # # #     bee=torch.atan2(ye,xe)
# # # #     tot=_hav(pd[1:T],gd[1:T]); ang=bee-be
# # # #     return tot*torch.cos(ang), tot*torch.sin(ang)


# # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # #  Split monitoring
# # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # class SplitAcc:
# # # #     def __init__(self):
# # # #         self.easy_d = []
# # # #         self.hard_d = []
# # # #         self.all_d  = []
# # # #         self.all_a  = []
# # # #         self.all_c  = []
# # # #         self.sd     = defaultdict(list)
# # # #         self._h     = {12: 1, 24: 3, 48: 7, 72: 11}

# # # #     def update(self, dist, is_hard=None, ate=None, cte=None):
# # # #         B = dist.shape[1]
# # # #         mean_dist = dist.mean(0)  # [B]
# # # #         self.all_d.extend(mean_dist.tolist())

# # # #         if is_hard is not None:
# # # #             easy_mask = ~is_hard
# # # #             if easy_mask.any():
# # # #                 self.easy_d.extend(mean_dist[easy_mask].tolist())
# # # #             if is_hard.any():
# # # #                 self.hard_d.extend(mean_dist[is_hard].tolist())
# # # #         else:
# # # #             self.easy_d.extend(mean_dist.tolist())

# # # #         for h, s in self._h.items():
# # # #             if s < dist.shape[0]:
# # # #                 self.sd[h].extend(dist[s].tolist())

# # # #         if ate is not None: self.all_a.extend(ate.abs().mean(0).tolist())
# # # #         if cte is not None: self.all_c.extend(cte.abs().mean(0).tolist())

# # # #     def compute(self):
# # # #         r = {
# # # #             "ADE":      float(np.mean(self.all_d))  if self.all_d  else float("nan"),
# # # #             "easy_ADE": float(np.mean(self.easy_d)) if self.easy_d else float("nan"),
# # # #             "hard_ADE": float(np.mean(self.hard_d)) if self.hard_d else float("nan"),
# # # #             "ATE":      float(np.mean(self.all_a))  if self.all_a  else float("nan"),
# # # #             "CTE":      float(np.mean(self.all_c))  if self.all_c  else float("nan"),
# # # #             "n":        len(self.all_d),
# # # #             "n_easy":   len(self.easy_d),
# # # #             "n_hard":   len(self.hard_d),
# # # #         }
# # # #         for h in self._h:
# # # #             v = self.sd.get(h, [])
# # # #             r[f"{h}h"] = float(np.mean(v)) if v else float("nan")
# # # #         return r


# # # # class Acc:
# # # #     def __init__(self):
# # # #         self.d=[]; self.a=[]; self.c=[]; self.sd=defaultdict(list)
# # # #         self._h={12:1, 24:3, 48:7, 72:11}
# # # #     def update(self, dist, ate=None, cte=None):
# # # #         self.d.extend(dist.mean(0).tolist())
# # # #         for h,s in self._h.items():
# # # #             if s < dist.shape[0]: self.sd[h].extend(dist[s].tolist())
# # # #         if ate is not None: self.a.extend(ate.abs().mean(0).tolist())
# # # #         if cte is not None: self.c.extend(cte.abs().mean(0).tolist())
# # # #     def compute(self):
# # # #         r = {"ADE": float(np.mean(self.d)) if self.d else float("nan"),
# # # #              "ATE": float(np.mean(self.a)) if self.a else float("nan"),
# # # #              "CTE": float(np.mean(self.c)) if self.c else float("nan"),
# # # #              "n": len(self.d)}
# # # #         for h in self._h:
# # # #             v=self.sd.get(h,[]); r[f"{h}h"]=float(np.mean(v)) if v else float("nan")
# # # #         return r


# # # # def _score(r):
# # # #     ade=r.get("ADE",1e9); h72=r.get("72h",1e9)
# # # #     ate=r.get("ATE",1e9); cte=r.get("CTE",1e9)
# # # #     if not np.isfinite(ate): ate=ade*0.46
# # # #     if not np.isfinite(cte): cte=ade*0.53
# # # #     return 100.0*(0.05*(ade/136.)+0.10*(r.get("12h",ade)/50.)
# # # #                   +0.15*(r.get("24h",ade)/100.)+0.20*(r.get("48h",ade)/200.)
# # # #                   +0.25*(h72/300.)+0.13*(ate/80.)+0.12*(cte/94.))

# # # # def _beat(r):
# # # #     p=[]
# # # #     for k,t in [("ADE",172.68),("ATE",142.21),("CTE",42.04),
# # # #                 ("72h",321.39),("12h",65.42),("24h",104.67),("48h",205.10)]:
# # # #         v=r.get(k,1e9)
# # # #         if np.isfinite(v) and v < t: p.append(f"{k}:{v:.1f}")
# # # #     return "*** BEAT ST-TRANS: "+" ".join(p)+" ***" if p else ""

# # # # def _gap(r):
# # # #     out=[]
# # # #     for k,ref in [("ADE",172.68),("72h",321.39),("ATE",142.21),("CTE",42.04)]:
# # # #         v=r.get(k,float("nan"))
# # # #         if np.isfinite(v):
# # # #             out.append(f"{k}:{v:.0f}({'dn' if v<ref else 'up'}{abs(v-ref):.0f})")
# # # #     return " | ".join(out)


# # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # #  [FIX-4] evaluate_split — nhận global_hard_threshold để consistent với training
# # # # #
# # # # #  Vấn đề cũ: classify_hard_easy() tính p70 per-batch
# # # # #    → easy_ADE/hard_ADE không consistent với training split
# # # # #    → EMA guard nhận easy_ADE sai → có thể trigger false positive rollback
# # # # #
# # # # #  Fix: nhận global_hard_threshold (được tính ở đầu epoch, từ phase 2)
# # # # #    → Cùng threshold với training → split consistent
# # # # #    → Phase 1: global_hard_threshold=None → classify_hard_easy() per-batch (OK
# # # # #       vì ở phase 1 easy/hard split chỉ dùng để log, không ảnh hưởng training)
# # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # @torch.no_grad()
# # # # def evaluate_split(model, loader, dev, tag="", ema=None, steps=10,
# # # #                    use_selector=False,
# # # #                    global_hard_threshold: Optional[float] = None) -> Dict:
# # # #     """
# # # #     [FIX-4] Evaluate với split easy/hard ADE dùng global threshold.

# # # #     Args:
# # # #         global_hard_threshold: float từ precompute_hard_threshold(), hoặc None
# # # #             nếu None → dùng per-batch p70 (phase 1 fallback)
# # # #     """
# # # #     # [FIX-F] EMA apply/restore: dùng sentinel None thay vì truthy check trên dict.
# # # #     # Bug cũ: `if bk:` với empty dict {} → False → không restore!
# # # #     # Fix: bk_sentinel = False trước, chỉ set True nếu apply_to thành công.
# # # #     bk         = None
# # # #     bk_applied = False  # sentinel riêng, không dùng `if bk:`
# # # #     if ema:
# # # #         try:
# # # #             bk         = ema.apply_to(model)
# # # #             bk_applied = True   # apply thành công → nhớ phải restore
# # # #         except Exception as e:
# # # #             print(f"  [EMA eval] apply_to failed: {e}")

# # # #     model.eval()
# # # #     acc = SplitAcc()
# # # #     t0  = time.perf_counter()

# # # #     for b in loader:
# # # #         bl    = move(list(b), dev)
# # # #         obs_t = bl[0]

# # # #         # [FIX-4] Dùng global threshold nếu có, fallback về per-batch
# # # #         if global_hard_threshold is not None:
# # # #             is_hard = classify_hard_easy_global(
# # # #                 obs_t[:, :, :2], global_hard_threshold).to(dev)
# # # #         else:
# # # #             # Phase 1: per-batch OK vì không ảnh hưởng training
# # # #             is_hard = classify_hard_easy(obs_t[:, :, :2]).to(dev)

# # # #         result = model.sample(bl, ddim_steps=steps, use_selector=use_selector)
# # # #         p = result[0] if isinstance(result, (tuple, list)) else result
# # # #         g = bl[1]; T = min(p.shape[0], g.shape[0])
# # # #         pd = _ntd(p[:T]); gd = _ntd(g[:T])
# # # #         dist = _hav(pd, gd)
# # # #         at, ct = _atecte(pd, gd)
# # # #         acc.update(dist, is_hard=is_hard, ate=at, cte=ct)

# # # #     # [FIX-F] Restore PHẢI gọi nếu apply_to đã thành công.
# # # #     # Dùng bk_applied sentinel thay vì `if bk:` (dict rỗng là falsy → bug cũ).
# # # #     if bk_applied:
# # # #         try:
# # # #             ema.restore(model, bk)
# # # #         except Exception as e:
# # # #             print(f"  [EMA eval] restore failed: {e}")

# # # #     r  = acc.compute()
# # # #     el = time.perf_counter() - t0

# # # #     def _v(k): return r.get(k, float("nan"))
# # # #     def _m(v, t): return "ok" if np.isfinite(v) and v < t else "no"

# # # #     thr_str = f"{global_hard_threshold:.4f}" if global_hard_threshold else "per-batch"
# # # #     print(f"\n{'='*70}")
# # # #     print(f"  [{tag}  {el:.0f}s  threshold={thr_str}]")
# # # #     print(f"  ADE={_v('ADE'):.1f}[{_m(_v('ADE'),172.68)}]"
# # # #           f"  easy_ADE={_v('easy_ADE'):.1f}  hard_ADE={_v('hard_ADE'):.1f}")
# # # #     print(f"  12h={_v('12h'):.0f}  24h={_v('24h'):.0f}"
# # # #           f"  48h={_v('48h'):.0f}  72h={_v('72h'):.0f}[{_m(_v('72h'),321.39)}]")
# # # #     print(f"  ATE={_v('ATE'):.1f}[{_m(_v('ATE'),142.21)}]"
# # # #           f"  CTE={_v('CTE'):.1f}[{_m(_v('CTE'),42.04)}]")
# # # #     print(f"  n_easy={r.get('n_easy',0)}  n_hard={r.get('n_hard',0)}")
# # # #     print(f"  vs ST-Trans: {_gap(r)}")
# # # #     bt = _beat(r)
# # # #     if bt: print(f"  {bt}")
# # # #     print(f"  Score={_score(r):.2f}")
# # # #     print(f"{'='*70}\n")
# # # #     return r


# # # # @torch.no_grad()
# # # # def evaluate(model, loader, dev, tag="", ema=None, steps=20,
# # # #              use_selector=False,
# # # #              global_hard_threshold: Optional[float] = None) -> Dict:
# # # #     """Full evaluation (backward compat)."""
# # # #     return evaluate_split(model, loader, dev, tag=tag, ema=ema,
# # # #                           steps=steps, use_selector=use_selector,
# # # #                           global_hard_threshold=global_hard_threshold)


# # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # #  Diversity check
# # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # @torch.no_grad()
# # # # def check_diversity(model, loader, dev, n_batches: int = 5,
# # # #                     n_ensemble: int = 20, ddim_steps: int = 5) -> float:
# # # #     """
# # # #     Đo diversity score trên một subset của val set.
# # # #     Nếu < 50 km → cảnh báo R1 (mode collapse).
# # # #     """
# # # #     model.eval()
# # # #     all_divs = []

# # # #     for i, b in enumerate(loader):
# # # #         if i >= n_batches:
# # # #             break
# # # #         bl = move(list(b), dev)

# # # #         _, _, all_c = model.sample(bl, num_ensemble=n_ensemble,
# # # #                                     ddim_steps=ddim_steps)
# # # #         # all_c: [2N, T, B, 2] (speed_sweep nhân đôi candidates)
# # # #         N_total = all_c.shape[0]
# # # #         cands = [all_c[k] for k in range(N_total)]

# # # #         div = compute_diversity_score(cands)
# # # #         all_divs.append(div)

# # # #     return float(np.mean(all_divs)) if all_divs else 0.0


# # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # #  [FIX-8] Checkpoint — lưu smooth_sched_step cho mọi loại checkpoint
# # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # def _save_ckpt(path, ep, model, opt, sched, saver, tl, vl,
# # # #                smooth_sched_step: int = 0,
# # # #                global_hard_threshold: float = 0.5,
# # # #                extra=None):
# # # #     """
# # # #     [FIX-8] Thêm smooth_sched_step và global_hard_threshold vào MỌI checkpoint.
# # # #     Trước đây chỉ lưu trong periodic checkpoint (ep % 10) → khi resume từ
# # # #     best_ade.pth không có smooth_sched_step → phải re-estimate từ epoch number.
# # # #     Bây giờ mọi checkpoint đều có đủ state để resume chính xác.
# # # #     """
# # # #     m   = _unwrap(model)
# # # #     ema = getattr(m, "_ema", None)
# # # #     esd = None
# # # #     if ema and hasattr(ema, "shadow"):
# # # #         try: esd = {k: v.cpu().clone() for k, v in ema.shadow.items()}
# # # #         except: pass
# # # #     d = {
# # # #         "epoch":                ep,
# # # #         "model_state_dict":     m.state_dict(),
# # # #         "optimizer_state":      opt.state_dict(),
# # # #         "scheduler_state":      sched.state_dict(),
# # # #         "ema_shadow":           esd,
# # # #         "best_score":           saver.bs,
# # # #         "best_ade":             saver.ba,
# # # #         "best_72h":             saver.b7,
# # # #         "best_ate":             saver.bat,
# # # #         "best_cte":             saver.bc,
# # # #         "best_easy_ade":        saver.best_easy_ade,
# # # #         "train_loss":           tl,
# # # #         "val_loss":             vl,
# # # #         # [FIX-8] Luôn lưu smooth_sched_step và global_hard_threshold
# # # #         "smooth_sched_step":    smooth_sched_step,
# # # #         "global_hard_threshold": global_hard_threshold,
# # # #         "version":              "v59strategy",
# # # #     }
# # # #     if extra:
# # # #         d.update(extra)
# # # #     torch.save(d, path)


# # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # #  Saver
# # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # class Saver:
# # # #     def __init__(self, patience=40, min_ep=35, enable_stop=True):
# # # #         self.patience     = patience
# # # #         self.min_ep       = min_ep
# # # #         self.enable_stop  = enable_stop
# # # #         self.cnt          = 0
# # # #         self.stop         = False
# # # #         self.bs           = float("inf")
# # # #         self.ba           = float("inf")
# # # #         self.b7           = float("inf")
# # # #         self.bat          = float("inf")
# # # #         self.bc           = float("inf")
# # # #         self.best_easy_ade = float("inf")

# # # #     def update(self, r, model, out_dir, ep, opt, sched, tl, vl, tag="",
# # # #                smooth_sched_step: int = 0,
# # # #                global_hard_threshold: float = 0.5):
# # # #         sc  = _score(r)
# # # #         ade = r.get("ADE",     1e9)
# # # #         h72 = r.get("72h",     1e9)
# # # #         ate = r.get("ATE",     1e9)
# # # #         cte = r.get("CTE",     1e9)
# # # #         easy_ade = r.get("easy_ADE", ade)

# # # #         any_improved = False

# # # #         for v, attr, fn in [
# # # #             (ade,  "ba",  f"best_ade_{tag}.pth"),
# # # #             (h72,  "b7",  f"best_72h_{tag}.pth"),
# # # #             (ate,  "bat", f"best_ate_{tag}.pth"),
# # # #             (cte,  "bc",  f"best_cte_{tag}.pth"),
# # # #         ]:
# # # #             if v < getattr(self, attr):
# # # #                 setattr(self, attr, v)
# # # #                 _save_ckpt(os.path.join(out_dir, fn), ep, model, opt,
# # # #                            sched, self, tl, vl,
# # # #                            smooth_sched_step=smooth_sched_step,
# # # #                            global_hard_threshold=global_hard_threshold)
# # # #                 any_improved = True

# # # #         if sc < self.bs:
# # # #             self.bs = sc
# # # #             any_improved = True
# # # #             _save_ckpt(
# # # #                 os.path.join(out_dir, f"best_{tag or 'composite'}.pth"),
# # # #                 ep, model, opt, sched, self, tl, vl,
# # # #                 smooth_sched_step=smooth_sched_step,
# # # #                 global_hard_threshold=global_hard_threshold,
# # # #             )
# # # #             print(f"  [BEST] {tag} ep={ep} score={sc:.2f}"
# # # #                   f"  ADE={ade:.1f}  72h={h72:.0f}"
# # # #                   f"  ATE={ate:.1f}  CTE={cte:.1f}")

# # # #         if np.isfinite(easy_ade) and easy_ade < self.best_easy_ade:
# # # #             self.best_easy_ade = easy_ade

# # # #         if self.enable_stop:
# # # #             if any_improved: self.cnt = 0
# # # #             else:
# # # #                 self.cnt += 1
# # # #                 print(f"  No improve {self.cnt}/{self.patience}"
# # # #                       f"  best={self.bs:.2f}  cur={sc:.2f}")
# # # #             if ep >= self.min_ep and self.cnt >= self.patience:
# # # #                 self.stop = True


# # # # def _mksub(ds, n, bs, cf):
# # # #     idx = random.Random(42).sample(range(len(ds)), min(n, len(ds)))
# # # #     return DataLoader(Subset(ds, idx), batch_size=bs, shuffle=False,
# # # #                       collate_fn=cf, num_workers=0, drop_last=False)


# # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # #  Args
# # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # def get_args():
# # # #     p = argparse.ArgumentParser(
# # # #         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# # # #     p.add_argument("--dataset_root",       default="TCND_vn")
# # # #     p.add_argument("--obs_len",            default=8,    type=int)
# # # #     p.add_argument("--pred_len",           default=12,   type=int)
# # # #     p.add_argument("--batch_size",         default=32,   type=int)
# # # #     p.add_argument("--num_epochs",         default=100,  type=int)
# # # #     p.add_argument("--learning_rate",      default=1e-4, type=float)
# # # #     p.add_argument("--weight_decay",       default=1e-3, type=float)
# # # #     p.add_argument("--warmup_epochs",      default=3,    type=int)
# # # #     p.add_argument("--grad_clip",          default=1.0,  type=float)
# # # #     p.add_argument("--use_amp",            action="store_true")
# # # #     p.add_argument("--num_workers",        default=2,    type=int)
# # # #     p.add_argument("--sigma_min",          default=0.02, type=float)
# # # #     p.add_argument("--use_ot",             default=True, action="store_true")
# # # #     p.add_argument("--no_ot",              dest="use_ot", action="store_false")
# # # #     p.add_argument("--cfg_guidance_scale", default=1.5,  type=float)
# # # #     p.add_argument("--n_ensemble",         default=50,   type=int)
# # # #     p.add_argument("--use_ema",            default=True, action="store_true")
# # # #     p.add_argument("--no_ema",             dest="use_ema", action="store_false")
# # # #     p.add_argument("--ema_decay",          default=0.995, type=float)
# # # #     p.add_argument("--patience",           default=40,   type=int)
# # # #     p.add_argument("--min_ep",             default=35,   type=int)
# # # #     p.add_argument("--val_freq",           default=3,    type=int)
# # # #     p.add_argument("--val_subset_size",    default=500,  type=int)
# # # #     p.add_argument("--fast_ddim",          default=10,   type=int)
# # # #     p.add_argument("--full_ddim",          default=20,   type=int)
# # # #     p.add_argument("--output_dir",         default="runs/v59strategy")
# # # #     p.add_argument("--gpu_num",            default="0")
# # # #     p.add_argument("--delim",              default=" ")
# # # #     p.add_argument("--skip",               default=1,    type=int)
# # # #     p.add_argument("--min_ped",            default=1,    type=int)
# # # #     p.add_argument("--threshold",          default=0.002, type=float)
# # # #     p.add_argument("--other_modal",        default="gph")
# # # #     p.add_argument("--test_year",          default=None, type=int)
# # # #     p.add_argument("--resume",             default=None)
# # # #     p.add_argument("--resume_epoch",       default=None, type=int)
# # # #     p.add_argument("--eval_test_after_train", default=True, action="store_true")
# # # #     p.add_argument("--gradnorm_alpha",     default=1.5,  type=float)
# # # #     p.add_argument("--gradnorm_lr",        default=1e-3, type=float)
# # # #     p.add_argument("--ema_guard_threshold", default=3.0, type=float)
# # # #     p.add_argument("--diversity_threshold", default=50.0, type=float)
# # # #     return p.parse_args()


# # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # #  Main
# # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # def main(args):
# # # #     if torch.cuda.is_available():
# # # #         os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
# # # #     dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # # #     os.makedirs(args.output_dir, exist_ok=True)

# # # #     print("=" * 72)
# # # #     print("  TC-FlowMatching v59-Strategy [FIXED]")
# # # #     print("  [G-A] GradNorm: loss-ratio weighting (FIX-2+3)")
# # # #     print("  [G-B] Phase 1: is_hard=zeros, phase 2+: global threshold (FIX-1)")
# # # #     print("  [G-C] easy_frac ≥ 50% enforcement mỗi batch")
# # # #     print("  [G-D] α warm-up: sigmoid per-batch")
# # # #     print("  [G-E] EMA guard: rollback đúng cách (FIX-5)")
# # # #     print("  [G-F] Diversity check sau phase 1")
# # # #     print("  [G-G] Split monitoring: global threshold consistent (FIX-4)")
# # # #     print("  [G-H] Phase 4: freeze encoder")
# # # #     print(f"  Phases: 1[0-{PHASE_1_END}] 2[{PHASE_1_END+1}-{PHASE_2_END}]"
# # # #           f" 3[{PHASE_2_END+1}-{PHASE_3_END}] 4[{PHASE_3_END+1}+]")
# # # #     print(f"  Target: ADE<{TARGETS['ADE']} 72h<{TARGETS['72h']}"
# # # #           f" ATE<{TARGETS['ATE']} CTE<{TARGETS['CTE']}")
# # # #     print("=" * 72)

# # # #     # Data
# # # #     trd, trl = data_loader(args, {"root": args.dataset_root, "type": "train"},
# # # #                             test=False)
# # # #     vd,  vl  = data_loader(args, {"root": args.dataset_root, "type": "val"},
# # # #                             test=True)
# # # #     from Model.data.trajectoriesWithMe_unet_training import seq_collate
# # # #     vsub = _mksub(vd, args.val_subset_size, args.batch_size, seq_collate)
# # # #     print(f"  train:{len(trd)} seqs  val:{len(vd)} seqs")

# # # #     # Model
# # # #     model = TCFlowMatching(
# # # #         pred_len=args.pred_len, obs_len=args.obs_len,
# # # #         sigma_min=args.sigma_min, use_ema=args.use_ema,
# # # #         ema_decay=args.ema_decay, use_ate_ot=args.use_ot,
# # # #         cfg_guidance_scale=args.cfg_guidance_scale,
# # # #     ).to(dev)
# # # #     model.init_ema()

# # # #     n_total = sum(p.numel() for p in model.parameters() if p.requires_grad)

# # # #     # BUG-8 (original): backbone vs step_weights params
# # # #     backbone_params = [p for p in model.parameters()
# # # #                        if p.requires_grad and p.numel() > 100]
# # # #     step_w_params   = [p for p in model.parameters()
# # # #                        if p.requires_grad and p.numel() <= 12]

# # # #     print(f"  params total: {n_total:,}")
# # # #     print(f"  backbone (clipped at {args.grad_clip}): "
# # # #           f"{sum(p.numel() for p in backbone_params):,}")
# # # #     print(f"  step_weights (NOT clipped): "
# # # #           f"{sum(p.numel() for p in step_w_params):,}")

# # # #     # Optimizer & scheduler
# # # #     opt    = optim.AdamW(model.parameters(),
# # # #                           lr=args.learning_rate,
# # # #                           weight_decay=args.weight_decay)
# # # #     scaler = GradScaler("cuda", enabled=args.use_amp)
# # # #     nstep  = len(trl)
# # # #     total  = nstep * args.num_epochs
# # # #     wstp   = nstep * args.warmup_epochs
# # # #     sched  = get_cosine_schedule_with_warmup(opt, wstp, total, min_lr=1e-6)

# # # #     # [G-A] GradNorm manager
# # # #     gradnorm = GradNormManager(
# # # #         terms=GRADNORM_TERMS,
# # # #         alpha_gn=args.gradnorm_alpha,
# # # #         lr=args.gradnorm_lr,
# # # #         device=dev,
# # # #     )

# # # #     # SmoothScheduler
# # # #     smooth_sched = SmoothScheduler(total_steps_per_epoch=nstep)

# # # #     # Savers
# # # #     fast_saver = Saver(patience=args.patience, min_ep=args.min_ep,
# # # #                         enable_stop=False)
# # # #     full_saver = Saver(patience=args.patience, min_ep=args.min_ep,
# # # #                         enable_stop=True)

# # # #     # Resume
# # # #     start = 0
# # # #     global_hard_threshold = 0.5  # default, sẽ được tính lại từ phase 2
# # # #     if args.resume and os.path.exists(args.resume):
# # # #         print(f"  Loading: {args.resume}")
# # # #         ck = torch.load(args.resume, map_location=dev)
# # # #         m  = _unwrap(model)
# # # #         ms, _ = m.load_state_dict(ck["model_state_dict"], strict=False)
# # # #         if ms: print(f"  Missing keys: {len(ms)}")
# # # #         ema = getattr(m, "_ema", None)
# # # #         if ema and ck.get("ema_shadow"):
# # # #             for k, v in ck["ema_shadow"].items():
# # # #                 if k in ema.shadow: ema.shadow[k].copy_(v.to(dev))
# # # #         try: opt.load_state_dict(ck["optimizer_state"])
# # # #         except Exception as e: print(f"  Opt load: {e}")
# # # #         try: sched.load_state_dict(ck["scheduler_state"])
# # # #         except:
# # # #             for _ in range(ck.get("epoch", 0) * nstep): sched.step()
# # # #         for a, attr in [
# # # #             ("best_score",    "bs"),  ("best_ade",  "ba"),
# # # #             ("best_72h",      "b7"),  ("best_ate",  "bat"),
# # # #             ("best_cte",      "bc"),  ("best_easy_ade", "best_easy_ade"),
# # # #         ]:
# # # #             if a in ck:
# # # #                 setattr(full_saver, attr, ck[a])
# # # #                 setattr(fast_saver, attr, ck[a])
# # # #         start = args.resume_epoch or ck.get("epoch", 0) + 1

# # # #         # [FIX-8] Restore smooth_sched step từ checkpoint
# # # #         if "smooth_sched_step" in ck:
# # # #             smooth_sched._global_step = ck["smooth_sched_step"]
# # # #             print(f"  Restored smooth_sched_step={ck['smooth_sched_step']}")
# # # #         else:
# # # #             # Fallback: estimate từ epoch number
# # # #             smooth_sched._global_step = start * nstep
# # # #             print(f"  Estimated smooth_sched_step={start * nstep} from epoch")

# # # #         # Restore global_hard_threshold nếu có
# # # #         if "global_hard_threshold" in ck:
# # # #             global_hard_threshold = ck["global_hard_threshold"]
# # # #             print(f"  Restored global_hard_threshold={global_hard_threshold:.4f}")

# # # #         print(f"  Resume ep={start}  best_score={full_saver.bs:.2f}"
# # # #               f"  best_easy_ADE={full_saver.best_easy_ade:.1f}")

# # # #     try:
# # # #         model = torch.compile(model, mode="reduce-overhead")
# # # #         print("  torch.compile: ok")
# # # #     except:
# # # #         pass

# # # #     # ── Phase 4 encoder freeze ────────────────────────────────────────────
# # # #     def _get_encoder_params(m):
# # # #         enc = _unwrap(m)
# # # #         params = []
# # # #         for mod_name in ["net.spatial_enc", "net.enc_1d", "net.env_enc",
# # # #                           "net.bottleneck_pool", "net.bottleneck_proj",
# # # #                           "net.decoder_proj"]:
# # # #             parts = mod_name.split(".")
# # # #             mod = enc
# # # #             for part in parts:
# # # #                 if hasattr(mod, part): mod = getattr(mod, part)
# # # #                 else: mod = None; break
# # # #             if mod is not None:
# # # #                 params.extend(list(mod.parameters()))
# # # #         return params

# # # #     _phase4_frozen = False

# # # #     def _maybe_freeze_encoder(ep, m):
# # # #         nonlocal _phase4_frozen
# # # #         if get_phase(ep) == 4 and not _phase4_frozen:
# # # #             enc_params = _get_encoder_params(m)
# # # #             for p in enc_params:
# # # #                 p.requires_grad_(False)
# # # #             _phase4_frozen = True
# # # #             print(f"  [Phase 4] Froze encoder ({sum(1 for _ in enc_params)} modules)")
# # # #             for g in opt.param_groups:
# # # #                 g["lr"] = g["lr"] * 0.1
# # # #             print(f"  [Phase 4] LR → {opt.param_groups[0]['lr']:.2e}")

# # # #     # ── EMA guard state ───────────────────────────────────────────────────
# # # #     best_easy_ade_ema  = float("inf")
# # # #     ema_rollback_count = 0
# # # #     _prev_phase        = get_phase(start) if start > 0 else 1

# # # #     # [FIX-D] Flag để chỉ compute hard threshold 1 lần khi vào phase 2.
# # # #     # hard_score_from_obs() phụ thuộc obs_traj (data cố định), không phụ thuộc model
# # # #     # → Threshold không thay đổi theo training → không cần tính lại mỗi epoch.
# # # #     # Nếu resume từ phase 2+ VÀ checkpoint đã có threshold → dùng lại luôn.
# # # #     _threshold_computed = (start > PHASE_1_END and global_hard_threshold != 0.5)

# # # #     print(f"  Training: {nstep} steps/ep, start ep={start}")
# # # #     print("=" * 72)
# # # #     ts = time.perf_counter()

# # # #     for ep in range(start, args.num_epochs):
# # # #         phase = get_phase(ep)

# # # #         # GradNorm reset khi chuyển phase
# # # #         if phase != _prev_phase:
# # # #             gradnorm.phase_reset(phase)
# # # #             _prev_phase = phase
# # # #             print(f"  [Phase {phase}] Chuyển phase tại epoch {ep}")

# # # #         # [FIX-1] Chỉ tính global_hard_threshold từ phase 2 trở đi.
# # # #         # Phase 1: không cần (alpha_hard=0, is_hard luôn zeros)
# # # #         #
# # # #         # [FIX-D] hard_score_from_obs() chỉ phụ thuộc obs_traj (lịch sử bão),
# # # #         # KHÔNG phụ thuộc model weights → threshold KHÔNG thay đổi theo training.
# # # #         # → Chỉ compute 1 lần khi vào phase 2, không tính lại mỗi epoch.
# # # #         # Trước đây: gọi mỗi epoch → 50 forward passes/epoch hoàn toàn lãng phí.
# # # #         # Khi vào phase mới (phase != _prev_phase trước đây đã reset _prev_phase),
# # # #         # ta dùng flag riêng: _threshold_computed để tránh tính lại.
# # # #         if phase >= 2 and not _threshold_computed:
# # # #             global_hard_threshold = precompute_hard_threshold(
# # # #                 trl, dev, n_batches=50)
# # # #             _threshold_computed = True
# # # #             print(f"  [FIX-D] global_hard_threshold={global_hard_threshold:.4f}"
# # # #                   f" — sẽ không recompute lại (threshold không đổi theo training)")

# # # #         # Phase 4: freeze encoder
# # # #         _maybe_freeze_encoder(ep, model)

# # # #         model.train()
# # # #         sl = 0.0
# # # #         t0 = time.perf_counter()

# # # #         lambda_dict = gradnorm.get_lambda_dict()

# # # #         for i, batch in enumerate(trl):
# # # #             bl    = move(list(batch), dev)
# # # #             obs_t = bl[0]  # [T_obs, B, feat]

# # # #             alpha_hard     = smooth_sched.get_alpha_hard()
# # # #             easy_frac      = smooth_sched.get_easy_frac()
# # # #             sel_weight     = smooth_sched.get_selector_weight()
# # # #             train_selector = smooth_sched.is_selector_active() and (sel_weight > 0.01)

# # # #             # [FIX-1] Phase 1: is_hard = zeros (không compute, không cần)
# # # #             # Phase 2+: dùng global threshold đã tính ở đầu epoch
# # # #             if phase >= 2:
# # # #                 with torch.no_grad():
# # # #                     is_hard = classify_hard_easy_global(
# # # #                         obs_t[:, :, :2], global_hard_threshold).to(dev)
# # # #                 is_hard = enforce_easy_frac(is_hard, easy_frac)
# # # #             else:
# # # #                 # Phase 1: tất cả easy, alpha_hard=0 → hard loss = 0
# # # #                 is_hard = torch.zeros(
# # # #                     obs_t.shape[1], dtype=torch.bool, device=dev)

# # # #             with autocast(device_type="cuda", enabled=args.use_amp):
# # # #                 bd = model.get_loss_breakdown(
# # # #                     bl,
# # # #                     epoch=ep,
# # # #                     alpha_hard=alpha_hard,
# # # #                     is_hard=is_hard,
# # # #                     train_selector=train_selector,
# # # #                     lambda_dict=lambda_dict,
# # # #                 )

# # # #             opt.zero_grad()
# # # #             scaler.scale(bd["total"]).backward()
# # # #             scaler.unscale_(opt)

# # # #             # [FIX-2+3] GradNorm: dùng loss ratio thực sự
# # # #             # Không dùng grad approximation (đã sai về nguyên lý)
# # # #             # Chỉ truyền loss values → GradNormManager.update() tính ratio
# # # #             try:
# # # #                 loss_vals_float = {}
# # # #                 for t in GRADNORM_TERMS:
# # # #                     val = bd.get(t)
# # # #                     if val is None:
# # # #                         continue
# # # #                     # bd[t] có thể là tensor hoặc float (xem get_loss_breakdown)
# # # #                     if torch.is_tensor(val):
# # # #                         fval = float(val.item())
# # # #                     else:
# # # #                         fval = float(val)
# # # #                     if np.isfinite(fval) and fval > 0:
# # # #                         loss_vals_float[t] = fval

# # # #                 if loss_vals_float:
# # # #                     gradnorm.update(loss_vals_float)
# # # #                     lambda_dict = gradnorm.get_lambda_dict()
# # # #             except Exception:
# # # #                 pass  # GradNorm không critical

# # # #             # Clip chỉ backbone params
# # # #             torch.nn.utils.clip_grad_norm_(backbone_params, args.grad_clip)

# # # #             scaler.step(opt)
# # # #             scaler.update()
# # # #             try: sched.step()
# # # #             except: pass
# # # #             model.ema_update()

# # # #             smooth_sched.step()

# # # #             sl += bd["total"].item()

# # # #             if i % 20 == 0:
# # # #                 lr      = opt.param_groups[0]["lr"]
# # # #                 tot     = bd["total"].item()
# # # #                 warn    = " [!>10]" if tot > 10.0 else ""
# # # #                 sw_st   = _unwrap(model).step_weights.stats()
# # # #                 gn_log  = gradnorm.log_stats()
# # # #                 n_hard_batch = int(is_hard.sum().item())

# # # #                 # Hiển thị λ của từng term để monitor GradNorm
# # # #                 lam_str = " ".join(
# # # #                     f"λ_{t.replace('l_','').replace('_reg','')}="
# # # #                     f"{gn_log.get(f'λ_{t}', 1.0):.2f}"
# # # #                     for t in GRADNORM_TERMS
# # # #                 )

# # # #                 print(
# # # #                     f"  [{ep:>3}|Ph{phase}][{i:>3}/{nstep}]"
# # # #                     f"  tot={tot:.3f}{warn}"
# # # #                     f"  fm={bd.get('l_fm',0):.4f}"
# # # #                     f"  dpe={bd.get('dpe',0):.4f}"
# # # #                     f"  vel={bd.get('vel_reg',0):.4f}"
# # # #                     f"  hard={bd.get('l_hard_total',0):.4f}"
# # # #                     f"  {smooth_sched.log_stats()}"
# # # #                     f"  n_h={n_hard_batch}/{obs_t.shape[1]}"
# # # #                     f"  sw72={sw_st['sw_72h']:.2f}"
# # # #                     f"  gn_d={gn_log.get('gn_dense',0)}"
# # # #                     f"  {lam_str}"
# # # #                     f"  lr={lr:.2e}"
# # # #                 )

# # # #         avt  = sl / nstep
# # # #         t_ep = time.perf_counter() - t0

# # # #         # BUG-5 (original): val với epoch=-1 → skip augmentation
# # # #         model.eval()
# # # #         vls = 0.0
# # # #         with torch.no_grad():
# # # #             for batch in vl:
# # # #                 bv = move(list(batch), dev)
# # # #                 with autocast(device_type="cuda", enabled=args.use_amp):
# # # #                     vls += model.get_loss(bv, epoch=-1).item()
# # # #         avv = vls / len(vl)

# # # #         lr_cur     = opt.param_groups[0]["lr"]
# # # #         gn_stats   = gradnorm.log_stats()
# # # #         use_sel_eval = smooth_sched.is_selector_active()

# # # #         # Tóm tắt λ sau epoch
# # # #         lam_epoch = " ".join(
# # # #             f"λ_{t.replace('l_','').replace('_reg','')}="
# # # #             f"{gn_stats.get(f'λ_{t}', 1.0):.3f}"
# # # #             for t in GRADNORM_TERMS
# # # #         )

# # # #         print(f"  Epoch {ep:>3}|Ph{phase}"
# # # #               f" | train={avt:.4f} val={avv:.4f}"
# # # #               f" | {smooth_sched.log_stats()}"
# # # #               f" | {lam_epoch}"
# # # #               f" | lr={lr_cur:.2e} | {t_ep:.0f}s")

# # # #         # [G-G] Fast eval với split monitoring
# # # #         # [FIX-4] Truyền global_hard_threshold để split consistent với training
# # # #         # Phase 1: global_hard_threshold vẫn là 0.5 (default) → per-batch fallback OK
# # # #         eval_threshold = global_hard_threshold if phase >= 2 else None
# # # #         rf = evaluate_split(model, vsub, dev,
# # # #                              tag=f"FAST ep{ep}",
# # # #                              steps=args.fast_ddim,
# # # #                              use_selector=use_sel_eval,
# # # #                              global_hard_threshold=eval_threshold)
# # # #         fast_saver.update(rf, model, args.output_dir, ep, opt, sched, avt, avv,
# # # #                           tag="fast",
# # # #                           smooth_sched_step=smooth_sched.global_step,
# # # #                           global_hard_threshold=global_hard_threshold)

# # # #         # [FIX-5] EMA guard: rollback đúng cách
# # # #         #
# # # #         # Vấn đề cũ:
# # # #         #   bk = ema.apply_to(model)  → model weights = EMA weights (side effect)
# # # #         #   (không gọi ema.restore → model bị kẹt EMA weights vĩnh viễn)
# # # #         #   Training tiếp theo dùng EMA weights → EMA update từ EMA → diverge
# # # #         #
# # # #         # Fix:
# # # #         #   1. Lưu current weights trước (bk_current)
# # # #         #   2. Apply EMA weights để evaluate (bk từ apply_to)
# # # #         #   3. Save checkpoint với EMA weights (đây là rollback target)
# # # #         #   4. Restore lại current weights (ema.restore(model, bk))
# # # #         #   5. Training tiếp tục với current weights (không bị kẹt)
# # # #         #   6. Điều chỉnh smooth_sched để α giảm (tránh forgettig tái diễn)
# # # #         #   7. EMA.shadow vẫn giữ nguyên (không bị corrupt)
# # # #         easy_ade_curr = rf.get("easy_ADE", rf.get("ADE", float("inf")))
# # # #         if np.isfinite(easy_ade_curr):
# # # #             if easy_ade_curr < best_easy_ade_ema:
# # # #                 best_easy_ade_ema = easy_ade_curr

# # # #             elif (easy_ade_curr - best_easy_ade_ema) > args.ema_guard_threshold:
# # # #                 ema_rollback_count += 1
# # # #                 print(f"\n  [EMA GUARD] easy_ADE tăng "
# # # #                       f"{easy_ade_curr - best_easy_ade_ema:.1f} km"
# # # #                       f" (best={best_easy_ade_ema:.1f} curr={easy_ade_curr:.1f})")
# # # #                 print(f"  [EMA GUARD] Rollback #{ema_rollback_count}")

# # # #                 m_raw   = _unwrap(model)
# # # #                 ema_obj = getattr(m_raw, "_ema", None)  # tên riêng, không shadow biến ema_module
# # # #                 if ema_obj:
# # # #                     try:
# # # #                         # [FIX-5] Step 1: apply EMA → model để save checkpoint
# # # #                         bk_for_restore = ema_obj.apply_to(model)
# # # #                         # Step 2: save checkpoint với EMA weights (đây là "good" state)
# # # #                         _save_ckpt(
# # # #                             os.path.join(args.output_dir,
# # # #                                          f"ema_guard_rollback_ep{ep}.pth"),
# # # #                             ep, model, opt, sched, full_saver, avt, avv,
# # # #                             smooth_sched_step=smooth_sched.global_step,
# # # #                             global_hard_threshold=global_hard_threshold,
# # # #                             extra={"ema_rollback": True,
# # # #                                    "easy_ade_trigger": easy_ade_curr},
# # # #                         )
# # # #                         # Step 3: RESTORE current weights để training tiếp tục đúng
# # # #                         # (không để model bị kẹt EMA weights)
# # # #                         ema_obj.restore(model, bk_for_restore)
# # # #                         print(f"  [EMA GUARD] Saved rollback ckpt, "
# # # #                               f"restored current weights for continued training")
# # # #                     except Exception as e:
# # # #                         print(f"  [EMA GUARD] Apply/restore failed: {e}")

# # # #                 # Step 4: Điều chỉnh smooth_sched để α giảm tạm thời
# # # #                 pullback_steps = nstep * 3  # ~3 epoch
# # # #                 smooth_sched._global_step = max(
# # # #                     smooth_sched._phase2_start_step,
# # # #                     smooth_sched._global_step - pullback_steps
# # # #                 )
# # # #                 print(f"  [EMA GUARD] α giảm về {smooth_sched.get_alpha_hard():.3f}"
# # # #                       f" (step pullback {pullback_steps})")

# # # #                 if ema_rollback_count >= 3:
# # # #                     print(f"  [EMA GUARD] 3 lần rollback — tắt hard loss tăng")
# # # #                     if phase == 3:
# # # #                         smooth_sched._global_step = min(
# # # #                             smooth_sched._global_step,
# # # #                             smooth_sched._phase3_start_step - 1
# # # #                         )

# # # #         # Diversity check cuối phase 1
# # # #         div_score = None
# # # #         if ep == PHASE_1_END:
# # # #             print(f"\n  {'='*50}")
# # # #             print(f"  [CHECKPOINT] Cuối phase 1 — kiểm tra diversity...")
# # # #             div_score = check_diversity(model, vsub, dev)
# # # #             print(f"  Diversity score: {div_score:.1f} km")
# # # #             if div_score < args.diversity_threshold:
# # # #                 print(f"  [CẢNH BÁO R1] diversity={div_score:.1f} km"
# # # #                       f" < {args.diversity_threshold} km")
# # # #                 print(f"  → FM candidates không đủ đa dạng → selector sẽ kém")
# # # #                 print(f"  → Hành động: kiểm tra noise_std, thêm L_diversity")
# # # #             else:
# # # #                 print(f"  [OK] diversity={div_score:.1f} km ≥ "
# # # #                       f"{args.diversity_threshold} km → tiến hành phase 2")
# # # #             print(f"  {'='*50}\n")

# # # #         # Full eval mỗi val_freq epoch
# # # #         if ep % args.val_freq == 0:
# # # #             em = getattr(_unwrap(model), "_ema", None)
# # # #             rr = evaluate_split(model, vl, dev,
# # # #                                   tag=f"RAW ep{ep}",
# # # #                                   steps=args.full_ddim,
# # # #                                   use_selector=use_sel_eval,
# # # #                                   global_hard_threshold=eval_threshold)
# # # #             full_saver.update(rr, model, args.output_dir, ep, opt, sched,
# # # #                               avt, avv, tag="raw",
# # # #                               smooth_sched_step=smooth_sched.global_step,
# # # #                               global_hard_threshold=global_hard_threshold)

# # # #             if em and ep >= 3:
# # # #                 # [FIX-5] EMA eval: apply → evaluate → restore (đối xứng)
# # # #                 # evaluate_split tự xử lý apply/restore qua tham số ema=em
# # # #                 re = evaluate_split(model, vl, dev,
# # # #                                      tag=f"EMA ep{ep}",
# # # #                                      ema=em,
# # # #                                      steps=args.full_ddim,
# # # #                                      use_selector=use_sel_eval,
# # # #                                      global_hard_threshold=eval_threshold)
# # # #                 full_saver.update(re, model, args.output_dir, ep, opt, sched,
# # # #                                   avt, avv, tag="ema",
# # # #                                   smooth_sched_step=smooth_sched.global_step,
# # # #                                   global_hard_threshold=global_hard_threshold)

# # # #         # Periodic checkpoint
# # # #         if ep % 10 == 0 or ep == args.num_epochs - 1:
# # # #             _save_ckpt(
# # # #                 os.path.join(args.output_dir, f"ckpt_ep{ep:03d}.pth"),
# # # #                 ep, model, opt, sched, full_saver, avt, avv,
# # # #                 smooth_sched_step=smooth_sched.global_step,
# # # #                 global_hard_threshold=global_hard_threshold,
# # # #                 extra={"phase": phase,
# # # #                        "alpha_hard": smooth_sched.get_alpha_hard(),
# # # #                        "diversity": div_score},
# # # #             )

# # # #         if full_saver.stop:
# # # #             print(f"  Early stop ep={ep}")
# # # #             break

# # # #     th = (time.perf_counter() - ts) / 3600.0
# # # #     print(f"\n  Best: ADE={full_saver.ba:.1f}  72h={full_saver.b7:.0f}"
# # # #           f"  ATE={full_saver.bat:.1f}  CTE={full_saver.bc:.1f}"
# # # #           f"  easy_ADE={full_saver.best_easy_ade:.1f}"
# # # #           f"  ({th:.2f}h)")

# # # #     # Post-training test
# # # #     if args.eval_test_after_train:
# # # #         print("\n" + "="*72 + "\n  POST-TRAINING TEST\n" + "="*72)
# # # #         try: _, tl2 = data_loader(args, {"root": args.dataset_root,
# # # #                                           "type": "test"}, test=True)
# # # #         except: print("  No test set → using val"); tl2 = vl

# # # #         for fn, lb in [("best_raw.pth",     "RAW"),
# # # #                         ("best_ema.pth",     "EMA"),
# # # #                         ("best_ade_raw.pth", "BEST_ADE"),
# # # #                         ("best_cte_raw.pth", "BEST_CTE")]:
# # # #             pp = os.path.join(args.output_dir, fn)
# # # #             if not os.path.exists(pp): continue
# # # #             ck = torch.load(pp, map_location=dev)
# # # #             _unwrap(model).load_state_dict(ck["model_state_dict"], strict=False)
# # # #             em = getattr(_unwrap(model), "_ema", None)
# # # #             if em and ck.get("ema_shadow"):
# # # #                 for k, v in ck["ema_shadow"].items():
# # # #                     if k in em.shadow: em.shadow[k].copy_(v.to(dev))

# # # #             # [FIX-8] Restore global threshold từ checkpoint nếu có
# # # #             ckpt_threshold = ck.get("global_hard_threshold", None)
# # # #             r = evaluate_split(model, tl2, dev,
# # # #                                 tag=f"TEST/{lb}",
# # # #                                 steps=args.full_ddim,
# # # #                                 use_selector=(get_phase(ck.get("epoch", 80)) >= 3),
# # # #                                 global_hard_threshold=ckpt_threshold)
# # # #             print(f"\n  --- {lb} (ep={ck.get('epoch','?')}) ---")
# # # #             for key, ref in [("ADE", 172.68), ("72h", 321.39),
# # # #                               ("ATE", 142.21), ("CTE", 42.04)]:
# # # #                 v = r.get(key, float("nan"))
# # # #                 mk = "BEAT!" if np.isfinite(v) and v < ref else f"need {ref:.0f}"
# # # #                 print(f"    {key:<8}: {v:>8.1f} km  [{mk}  gap:{v-ref:+.1f}]")
# # # #             ea = r.get("easy_ADE", float("nan"))
# # # #             ha = r.get("hard_ADE", float("nan"))
# # # #             print(f"    easy_ADE={ea:.1f}  hard_ADE={ha:.1f}")

# # # #     print("=" * 72)


# # # # if __name__ == "__main__":
# # # #     args = get_args()
# # # #     np.random.seed(42)
# # # #     torch.manual_seed(42)
# # # #     if torch.cuda.is_available():
# # # #         torch.cuda.manual_seed_all(42)
# # # #     main(args)

# # # """
# # # train_v59_strategy.py — TC-FlowMatching v59-Strategy Training Script
# # # ══════════════════════════════════════════════════════════════════════════════

# # # THAY ĐỔI SO VỚI train_v59_tweaked.py:

# # #   [G-A] GradNorm: tự điều chỉnh λ_dpe, λ_vel, λ_head, λ_spd, λ_acc
# # #         Không hardcode hệ số loss — GradNorm đo gradient norm sau backward,
# # #         update λ để các term có gradient norm xấp xỉ nhau.
# # #         QUAN TRỌNG: GradNorm reset λ=1.0 khi phase đổi và tăng update_freq
# # #         tạm thời 5 epoch để thích nghi nhanh hơn → tránh lag gradient.

# # #   [G-B] Easy/hard pipeline với GLOBAL threshold (fix per-batch flip):
# # #         - hard_score tính mỗi epoch trên TOÀN BỘ train set (không phải per-batch)
# # #         - Lưu global_hard_threshold = p70 của toàn dataset
# # #         - Mỗi batch dùng threshold cố định đó → sample A luôn là hard/easy
# # #           nhất quán trong suốt epoch, gradient ổn định hơn nhiều

# # #   [G-C] easy_frac enforcement:
# # #         - Mỗi batch: đảm bảo ≥ 50% easy samples
# # #         - Nếu batch có quá nhiều hard → resample thêm easy từ buffer

# # #   [G-D] α smooth warm-up theo BATCH (fix hard cliff):
# # #         - Không dùng epoch threshold cứng (epoch 20 bật α đột ngột)
# # #         - α tăng theo sigmoid curve qua từng batch:
# # #           α(step) = α_target × sigmoid((step - mid_step) / temperature)
# # #           → Gradient thay đổi mượt mà, không có cliff, model không bị shock
# # #         - Selector cũng được bật mượt: selector_weight tăng từ 0 → 0.2

# # #   [G-E] EMA guard sau mỗi epoch:
# # #         - Đo easy_ADE trên val subset (easy samples only)
# # #         - Nếu easy_ADE tăng > 3 km vs best_easy_ADE → rollback EMA + giảm α
# # #         - Dùng easy_ADE (không phải val_ADE tổng) để early stopping

# # #   [G-F] Diversity check sau phase 1 (epoch 19):
# # #         - compute_diversity_score() trên val subset
# # #         - Nếu diversity < 50 km → log cảnh báo, không tự fix

# # #   [G-G] Split monitoring mỗi epoch:
# # #         - easy_ADE, hard_ADE riêng biệt (không chỉ tổng)

# # #   [G-H] Phase 4: freeze encoder, chỉ tune FM head + selector head

# # # BUG fixes giữ nguyên từ v59-tweaked:
# # #   BUG-5: val_loss dùng epoch=-1 → skip augmentation
# # #   BUG-6: any_improve = ANY metric improve → reset early stop
# # #   BUG-7: sched.step() unconditional
# # #   BUG-8: clip chỉ backbone params, không clip learned weights
# # # """
# # # from __future__ import annotations
# # # import sys, os
# # # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # # import argparse, math, random, time
# # # from collections import defaultdict
# # # from typing import Dict, List, Optional

# # # import numpy as np
# # # import torch
# # # import torch.optim as optim
# # # import torch.nn.functional as F
# # # from torch.amp import autocast, GradScaler
# # # from torch.utils.data import DataLoader, Subset

# # # from Model.data.loader_training import data_loader
# # # from Model.flow_matching_model import (
# # #     TCFlowMatching,
# # #     hard_score_from_obs,       # dùng trong precompute_hard_threshold
# # #     classify_hard_easy,        # dùng trong evaluate_split
# # #     compute_diversity_score,   # dùng trong check_diversity
# # #     _norm_to_deg,
# # #     _haversine_deg,
# # # )

# # # try:
# # #     from Model.utils import get_cosine_schedule_with_warmup
# # # except ImportError:
# # #     from torch.optim.lr_scheduler import CosineAnnealingLR
# # #     def get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps, min_lr=1e-6):
# # #         return CosineAnnealingLR(opt, T_max=max(total_steps, 1), eta_min=min_lr)

# # # TARGETS = {
# # #     "ADE": 172.68, "72h": 321.39, "ATE": 142.21, "CTE": 42.04,
# # #     "12h": 65.42,  "24h": 104.67, "48h": 205.10,
# # # }
# # # R_EARTH = 6371.0

# # # # ── Phase boundaries ──────────────────────────────────────────────────────────
# # # PHASE_1_END  = 19   # epoch 0–19:  warm-up, α=0, easy_frac=80%
# # # PHASE_2_END  = 49   # epoch 20–49: hard intro, α 0→0.3, easy_frac 80→60%
# # # PHASE_3_END  = 79   # epoch 50–79: selector, α=0.3, easy_frac=55%
# # # # epoch 80+: fine-tune, LR×0.1, freeze encoder

# # # # ── GradNorm terms (các loss term cần cân bằng) ──────────────────────────────
# # # GRADNORM_TERMS = ["l_dpe", "l_vel_reg", "l_heading", "l_speed", "l_accel"]


# # # # ─────────────────────────────────────────────────────────────────────────────
# # # #  GradNorm implementation
# # # #  Tham khảo: Chen et al. 2018 "GradNorm: Gradient Normalization for Adaptive
# # # #  Loss Balancing in Deep Multitask Networks"
# # # #
# # # #  Thuật toán:
# # # #    1. Sau mỗi backward, đo ||∇_W L_i|| của từng term i
# # # #    2. Tính target = mean(||∇_W L_i||) × (L_i / mean(L_i))^α_gn
# # # #       α_gn: restoring force (thường 1.5) — term nào loss lớn hơn mức tb
# # # #       thì được target gradient cao hơn
# # # #    3. Update λ_i để ||∇_W λ_i·L_i|| → target_i
# # # #
# # # #  Lưu ý: λ_i là learnable parameter của GradNorm (KHÔNG phải model weights)
# # # #  Chúng được update riêng bằng Adam optimizer của GradNorm
# # # # ─────────────────────────────────────────────────────────────────────────────

# # # class GradNormManager:
# # #     """
# # #     Quản lý GradNorm cho các loss terms.

# # #     [FIX-STABILITY-2] Thêm phase_reset() để tránh lag gradient khi chuyển phase:
# # #     - Khi phase đổi: reset λ về 1.0 (không giữ λ cũ không còn phù hợp)
# # #     - Tăng update_freq từ 5 → 1 trong 5 epoch sau phase đổi để thích nghi nhanh
# # #     - Sau 5 epoch: trở về update_freq=5 (tiết kiệm tính toán)

# # #     Lý do: λ được tối ưu cho loss landscape của phase cũ.
# # #     Khi thêm L_hard hoặc selector, landscape thay đổi nhưng λ cũ vẫn được giữ
# # #     → GradNorm dùng λ sai trong ~50 steps đầu → gradient mất cân bằng.
# # #     Reset λ=1.0 ngay lập tức và update dense hơn → ổn định hơn nhiều.
# # #     """
# # #     def __init__(self, terms: List[str], alpha_gn: float = 1.5,
# # #                  lr: float = 1e-3, device=None):
# # #         self.terms    = terms
# # #         self.alpha_gn = alpha_gn
# # #         self.device   = device or torch.device("cpu")

# # #         # λ_i: meta-optimization parameter (KHÔNG phải model weights)
# # #         # Init = 1.0 cho tất cả terms
# # #         self.lambdas = {t: torch.ones(1, device=self.device, requires_grad=True)
# # #                         for t in terms}
# # #         self.opt = optim.Adam(list(self.lambdas.values()), lr=lr)

# # #         self._step             = 0
# # #         self._update_freq      = 5      # update mỗi 5 steps (bình thường)
# # #         self._dense_steps_left = 0      # số steps còn lại ở chế độ dense (freq=1)
# # #         self._current_phase    = 1

# # #     def phase_reset(self, new_phase: int) -> None:
# # #         """
# # #         [FIX-STABILITY-2] Gọi khi chuyển sang phase mới.
# # #         Reset λ về 1.0 và bật dense update mode trong 5 epoch.

# # #         Args:
# # #             new_phase: phase mới (2, 3, 4)
# # #         """
# # #         if new_phase == self._current_phase:
# # #             return
# # #         self._current_phase = new_phase

# # #         # Reset λ về 1.0 — không giữ λ cũ không còn phù hợp
# # #         with torch.no_grad():
# # #             for t in self.terms:
# # #                 self.lambdas[t].fill_(1.0)

# # #         # Reset optimizer state để tránh momentum cũ kéo λ sai hướng
# # #         self.opt = optim.Adam(list(self.lambdas.values()), lr=self.opt.param_groups[0]["lr"])

# # #         # Dense update mode: update mỗi step trong 200 steps (~5 epoch với bs=32)
# # #         # để GradNorm thích nghi nhanh với loss landscape mới
# # #         self._dense_steps_left = 200

# # #         print(f"  [GradNorm] Phase {new_phase}: reset λ=1.0, dense update mode 200 steps")

# # #     def get_lambda_dict(self) -> Dict[str, float]:
# # #         """Trả về λ hiện tại để truyền vào get_loss_breakdown()."""
# # #         return {t: float(self.lambdas[t].item()) for t in self.terms}

# # #     def update(self, loss_term_values: Dict[str, float],
# # #                lambda_grads: Optional[Dict[str, torch.Tensor]] = None) -> None:
# # #         """
# # #         Cập nhật λ dùng loss-ratio weighting.

# # #         [BUG-FIX CRITICAL] Loại bỏ gradnorm_loss.backward() (L237 cũ):
# # #           - gradnorm_loss được tạo từ torch.tensor(scalar) — leaf, không requires_grad
# # #           - backward() raises RuntimeError → propagates ra train loop → except catches
# # #           - Kết quả: TOÀN BỘ update() bị interrupt, kể cả re-normalize → λ KHÔNG ĐỔI
# # #           - Đây là lý do λ_dpe có thể diverge mà re-normalize không kìm được

# # #         Fix: bỏ gradnorm_loss.backward(), set grad thủ công TRỰC TIẾP dùng loss ratio.
# # #         Re-normalize được tách ra ngoài block try để luôn chạy.

# # #         Args:
# # #             loss_term_values: {term: float} — loss value từng term (detached)
# # #             lambda_grads:     unused (kept for API compat), ignore
# # #         """
# # #         self._step += 1

# # #         # Adaptive update frequency
# # #         if self._dense_steps_left > 0:
# # #             self._dense_steps_left -= 1
# # #             update_freq = 1
# # #         else:
# # #             update_freq = 5

# # #         # Re-normalize LUÔN CHẠY (không phụ thuộc vào update_freq hay lambda_grads)
# # #         # Đây là phần quan trọng nhất: giữ λ không collapse
# # #         def _renorm():
# # #             with torch.no_grad():
# # #                 n_terms = len(self.terms)
# # #                 ΛMIN    = 0.1
# # #                 for t in self.terms:
# # #                     self.lambdas[t].clamp_(min=ΛMIN)
# # #                 total_λ = sum(float(self.lambdas[t].item()) for t in self.terms)
# # #                 if total_λ > 0:
# # #                     scale = n_terms / total_λ
# # #                     for t in self.terms:
# # #                         self.lambdas[t].mul_(scale)
# # #                         self.lambdas[t].clamp_(min=ΛMIN)

# # #         if self._step % update_freq != 0:
# # #             _renorm()
# # #             return

# # #         if not loss_term_values:
# # #             _renorm()
# # #             return

# # #         loss_vals = {t: v for t, v in loss_term_values.items()
# # #                      if t in self.terms and np.isfinite(v) and v > 0}
# # #         if len(loss_vals) < 2:
# # #             _renorm()
# # #             return

# # #         mean_loss = np.mean(list(loss_vals.values()))
# # #         if mean_loss <= 0:
# # #             _renorm()
# # #             return

# # #         # Loss-ratio weighting: term nào loss lớn hơn TB → tăng λ
# # #         # r_i > 1: term lớn hơn TB → muốn tăng λ → grad âm (Adam minimize)
# # #         # r_i < 1: term nhỏ hơn TB → muốn giảm λ → grad dương
# # #         self.opt.zero_grad()
# # #         for t in self.terms:
# # #             if t not in loss_vals:
# # #                 continue
# # #             r_i      = loss_vals[t] / (mean_loss + 1e-8)
# # #             grad_val = -(r_i - 1.0) * 0.1   # scale nhỏ để update không overshoot
# # #             if self.lambdas[t].grad is None:
# # #                 self.lambdas[t].grad = torch.zeros_like(self.lambdas[t])
# # #             self.lambdas[t].grad.fill_(float(grad_val))
# # #         self.opt.step()

# # #         _renorm()

# # #     def log_stats(self) -> Dict[str, float]:
# # #         """Log λ hiện tại để monitor."""
# # #         d = {f"λ_{t}": float(self.lambdas[t].item()) for t in self.terms}
# # #         d["gn_dense"] = self._dense_steps_left
# # #         return d


# # # # ─────────────────────────────────────────────────────────────────────────────
# # # #  Phase helper
# # # # ─────────────────────────────────────────────────────────────────────────────

# # # def get_phase(epoch: int) -> int:
# # #     """
# # #     Returns:
# # #         1: warm-up (ep 0–19)
# # #         2: hard introduction (ep 20–49)
# # #         3: selector (ep 50–79)
# # #         4: fine-tune (ep 80+)
# # #     """
# # #     if epoch <= PHASE_1_END:  return 1
# # #     if epoch <= PHASE_2_END:  return 2
# # #     if epoch <= PHASE_3_END:  return 3
# # #     return 4


# # # class SmoothScheduler:
# # #     """
# # #     [FIX-STABILITY-1] α smooth warm-up theo BATCH, không phải epoch.

# # #     Vấn đề với epoch-based cliff:
# # #         Epoch 19 cuối: α = 0.0, gradient ổn định
# # #         Epoch 20 đầu: α nhảy lên (dù tăng tuyến tính theo epoch, vẫn là cliff
# # #         so với các batch đã chạy) → model bị shock, ADE dao động

# # #     Giải pháp: sigmoid curve qua từng BATCH
# # #         α(step) = α_target × sigmoid((step - mid_step) / temperature)
# # #         - Tại step = mid_step: α = α_target / 2 (điểm giữa)
# # #         - temperature lớn → curve dốc hơn (đổi nhanh hơn)
# # #         - temperature nhỏ → curve thoải hơn (đổi chậm hơn)

# # #         Kết quả: trong epoch 20, α tăng từ rất nhỏ → 0.3 qua ~500 steps
# # #         thay vì nhảy đột ngột → gradient thay đổi mượt mà hoàn toàn

# # #     Selector weight cũng tăng mượt theo cùng cơ chế.
# # #     """
# # #     def __init__(self, total_steps_per_epoch: int):
# # #         self.steps_per_epoch = total_steps_per_epoch
# # #         self._global_step = 0

# # #         # Điểm bắt đầu của mỗi phase (tính theo global step)
# # #         # Phase N bắt đầu SAU khi epoch PHASE_(N-1)_END hoàn tất
# # #         # = (PHASE_(N-1)_END + 1) * steps_per_epoch
# # #         self._phase2_start_step = (PHASE_1_END + 1) * total_steps_per_epoch
# # #         self._phase3_start_step = (PHASE_2_END + 1) * total_steps_per_epoch
# # #         self._phase4_start_step = (PHASE_3_END + 1) * total_steps_per_epoch

# # #         # Temperature: bao nhiêu steps để α đi từ 10% → 90% giá trị target
# # #         # Công thức: temp = steps / (2 × ln(9)) ≈ steps / 4.4
# # #         # Ở đây ta muốn transition kéo dài ~2 epoch (2×steps_per_epoch)
# # #         # [FIX-ALPHA-TIMING] Temperature ngắn → α tăng SỚM trong phase 2.
# # #         # Vấn đề cũ: temp=2.0*nstep/4.4 → mid ở epoch 35 → α≈0 đến ep30!
# # #         # Log: ep20-ep32 α=0.0000 hoàn toàn → hard loss không bao giờ kích hoạt.
# # #         # Fix: temp=0.6*nstep/4.4 → mid ở epoch 25 → α=0.15 từ ep22, α≈0.3 từ ep28.
# # #         # Selector cũng dùng cùng logic để bật đồng bộ.
# # #         self._temp_p2 = 0.6 * total_steps_per_epoch / 4.4
# # #         self._temp_p3 = 0.6 * total_steps_per_epoch / 4.4  # selector bật nhanh

# # #     def step(self):
# # #         """Gọi sau mỗi optimizer step."""
# # #         self._global_step += 1

# # #     @property
# # #     def global_step(self):
# # #         return self._global_step

# # #     def get_alpha_hard(self) -> float:
# # #         """
# # #         α_hard: trọng số L_hard, tăng mượt qua sigmoid.
# # #         Phase 1 (step < phase2_start): α = 0
# # #         Phase 2 (step >= phase2_start): sigmoid 0 → 0.3
# # #         Phase 3+: α = 0.3 cố định

# # #         [FIX-ALPHA-TIMING] Mid point ở 0.15 thay vì 0.5 của phase.
# # #         Cũ: mid=ep35 → α≈0 đến ep30, không có hard loss suốt phase 2 đầu.
# # #         Mới: mid=ep24.5 → α=0.15 ở ep23, α≈0.3 ở ep26 → hard loss sớm hơn 10 epoch.
# # #         """
# # #         s = self._global_step
# # #         if s < self._phase2_start_step:
# # #             return 0.0
# # #         if s >= self._phase3_start_step:
# # #             return 0.3

# # #         # [FIX] mid ở 15% thay vì 50% của phase → sigmoid bắt đầu sớm
# # #         mid  = self._phase2_start_step + (self._phase3_start_step
# # #                                            - self._phase2_start_step) * 0.15
# # #         x    = (s - mid) / max(self._temp_p2, 1.0)
# # #         sig  = 1.0 / (1.0 + math.exp(-x))
# # #         return 0.3 * sig

# # #     def get_selector_weight(self) -> float:
# # #         """
# # #         selector_weight: trọng số L_sel, tăng mượt từ phase 3.
# # #         Phase < 3: 0
# # #         Phase 3: sigmoid 0 → 0.2
# # #         Phase 4: 0.2 cố định
# # #         """
# # #         s = self._global_step
# # #         if s < self._phase3_start_step:
# # #             return 0.0
# # #         if s >= self._phase4_start_step:
# # #             return 0.2

# # #         mid  = self._phase3_start_step + (self._phase4_start_step
# # #                                            - self._phase3_start_step) / 2.0
# # #         x    = (s - mid) / max(self._temp_p3, 1.0)
# # #         sig  = 1.0 / (1.0 + math.exp(-x))
# # #         return 0.2 * sig

# # #     def get_easy_frac(self) -> float:
# # #         """
# # #         easy_frac: giảm mượt, không bao giờ xuống dưới 50%.
# # #         Phase 1: 80%
# # #         Phase 2: 80% → 60% theo sigmoid
# # #         Phase 3: 60% → 55%
# # #         Phase 4: 55% → 50%
# # #         """
# # #         s = self._global_step
# # #         if s < self._phase2_start_step:
# # #             return 0.80

# # #         if s < self._phase3_start_step:
# # #             mid  = self._phase2_start_step + (self._phase3_start_step
# # #                                                - self._phase2_start_step) / 2.0
# # #             x    = (s - mid) / max(self._temp_p2, 1.0)
# # #             sig  = 1.0 / (1.0 + math.exp(-x))
# # #             return 0.80 - 0.20 * sig   # 80% → 60%

# # #         if s < self._phase4_start_step:
# # #             mid  = self._phase3_start_step + (self._phase4_start_step
# # #                                                - self._phase3_start_step) / 2.0
# # #             x    = (s - mid) / max(self._temp_p3, 1.0)
# # #             sig  = 1.0 / (1.0 + math.exp(-x))
# # #             return 0.60 - 0.05 * sig   # 60% → 55%

# # #         # Phase 4: giữ 50% (hard floor)
# # #         return max(0.50, 0.55 - 0.05 * (s - self._phase4_start_step)
# # #                    / max(self.steps_per_epoch * 10, 1))

# # #     def is_selector_active(self) -> bool:
# # #         """True từ phase 3 trở đi."""
# # #         return self._global_step >= self._phase3_start_step

# # #     def log_stats(self) -> str:
# # #         return (f"α={self.get_alpha_hard():.3f}"
# # #                 f" easy={self.get_easy_frac():.0%}"
# # #                 f" sel={self.get_selector_weight():.3f}")


# # # # Backward-compat wrappers (dùng trong checkpoint restore)
# # # def get_alpha_hard(epoch: int) -> float:
# # #     """Legacy: chỉ dùng khi không có SmoothScheduler."""
# # #     if epoch <= PHASE_1_END: return 0.0
# # #     if epoch <= PHASE_2_END:
# # #         return 0.3 * (epoch - PHASE_1_END) / (PHASE_2_END - PHASE_1_END)
# # #     return 0.3

# # # def get_easy_frac(epoch: int) -> float:
# # #     """Legacy fallback."""
# # #     if epoch <= PHASE_1_END: return 0.80
# # #     if epoch <= PHASE_2_END:
# # #         p = (epoch - PHASE_1_END) / (PHASE_2_END - PHASE_1_END)
# # #         return 0.80 - 0.20 * p
# # #     if epoch <= PHASE_3_END: return 0.55
# # #     return 0.50


# # # # ─────────────────────────────────────────────────────────────────────────────
# # # #  [FIX-STABILITY-3] Global hard threshold — precomputed per epoch
# # # #
# # # #  Vấn đề với per-batch threshold:
# # # #    Batch 1 có 32 samples → p70 tính trên 32 samples → threshold = X
# # # #    Batch 2 có 32 khác → p70 tính trên 32 samples khác → threshold = Y (≠ X)
# # # #    Sample A: hard trong batch 1 (score > X), easy trong batch 2 (score < Y)
# # # #    → L_hard áp dụng không nhất quán → gradient noisy
# # # #
# # # #  Giải pháp: scan toàn bộ train set 1 lần đầu mỗi epoch, lấy p70 global
# # # #    → threshold cố định cho cả epoch → mọi batch đều phân loại nhất quán
# # # #    → gradient ổn định hơn nhiều
# # # #
# # # #  Chi phí: ~1 pass qua train loader với batch nhỏ (không cần gradient)
# # # #  Thời gian: ~10–20s/epoch (chấp nhận được)
# # # # ─────────────────────────────────────────────────────────────────────────────

# # # @torch.no_grad()
# # # def precompute_hard_threshold(loader, dev, n_batches: int = 50) -> float:
# # #     """
# # #     Tính p70 của hard_score trên toàn bộ (hoặc subset lớn) train set.

# # #     Args:
# # #         loader:    train DataLoader
# # #         dev:       device
# # #         n_batches: số batches để scan (50 batches × 32 = 1600 samples, đủ)

# # #     Returns:
# # #         threshold: float — global p70 hard_score
# # #                    Mọi sample có hard_score > threshold được coi là hard
# # #     """
# # #     all_scores = []
# # #     for i, batch in enumerate(loader):
# # #         if i >= n_batches:
# # #             break
# # #         bl    = move(list(batch), dev)
# # #         obs_t = bl[0]
# # #         scores = hard_score_from_obs(obs_t[:, :, :2])  # [B]
# # #         all_scores.append(scores.cpu())

# # #     if not all_scores:
# # #         return 0.7  # fallback

# # #     all_scores_cat = torch.cat(all_scores)  # [N]
# # #     threshold = float(torch.quantile(all_scores_cat, 0.70).item())
# # #     n_hard = int((all_scores_cat >= threshold).sum().item())
# # #     n_total = len(all_scores_cat)
# # #     print(f"  [HardThreshold] global p70={threshold:.4f}"
# # #           f"  n_hard={n_hard}/{n_total} ({100*n_hard/max(n_total,1):.0f}%)")
# # #     return threshold


# # # @torch.no_grad()
# # # def classify_hard_easy_global(
# # #     obs_traj_norm: torch.Tensor,
# # #     global_threshold: float,
# # # ) -> torch.Tensor:
# # #     """
# # #     Phân loại hard/easy dùng global threshold (thay vì per-batch p70).

# # #     Args:
# # #         obs_traj_norm:    [T, B, >=2]
# # #         global_threshold: float từ precompute_hard_threshold()

# # #     Returns:
# # #         is_hard: [B] bool
# # #     """
# # #     scores   = hard_score_from_obs(obs_traj_norm)  # [B]
# # #     is_hard  = scores >= global_threshold
# # #     return is_hard

# # # def enforce_easy_frac(
# # #     is_hard: torch.Tensor,
# # #     easy_frac: float,
# # # ) -> torch.Tensor:
# # #     """
# # #     [BUG-3 FIX] Chỉ điều chỉnh is_hard mask, KHÔNG reindex batch_list.

# # #     Cách cũ cố reindex batch_list nhưng env_data (dict) không được
# # #     reindex → data inconsistency giữa obs_t và env_data sau reindex.

# # #     Cách mới: không động vào batch data. Nếu batch có quá nhiều hard samples,
# # #     "demote" một số về easy bằng cách set is_hard[i] = False.
# # #     Những samples đó vẫn chịu L_base bình thường, chỉ không chịu L_hard.
# # #     Batch data giữ nguyên 100%, không inconsistency.

# # #     Args:
# # #         is_hard:   [B] bool tensor
# # #         easy_frac: tỉ lệ easy tối thiểu (0.6 = 60% phải là easy)

# # #     Returns:
# # #         is_hard_adjusted: [B] bool, n_hard ≤ floor(B × (1 - easy_frac))
# # #     """
# # #     B        = is_hard.shape[0]
# # #     max_hard = max(0, int(B * (1.0 - easy_frac)))
# # #     n_hard   = int(is_hard.sum().item())

# # #     if n_hard <= max_hard:
# # #         return is_hard  # đã đúng tỉ lệ, không cần điều chỉnh

# # #     # Demote (n_hard - max_hard) hard samples về easy
# # #     n_demote  = n_hard - max_hard
# # #     hard_idx  = is_hard.nonzero(as_tuple=True)[0]
# # #     perm      = torch.randperm(hard_idx.numel(), device=is_hard.device)
# # #     to_demote = hard_idx[perm[:n_demote]]

# # #     is_hard_new            = is_hard.clone()
# # #     is_hard_new[to_demote] = False
# # #     return is_hard_new


# # # # ─────────────────────────────────────────────────────────────────────────────
# # # #  Utilities — giữ nguyên từ v59-tweaked
# # # # ─────────────────────────────────────────────────────────────────────────────

# # # def _unwrap(m):
# # #     return m._orig_mod if hasattr(m, "_orig_mod") else m

# # # def move(b, dev):
# # #     out = list(b)
# # #     for i, x in enumerate(out):
# # #         if torch.is_tensor(x): out[i] = x.to(dev)
# # #         elif isinstance(x, dict):
# # #             out[i] = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in x.items()}
# # #     return out


# # # def _ntd(a):
# # #     o = a.clone()
# # #     o[..., 0] = (a[..., 0] * 50.0 + 1800.0) / 10.0
# # #     o[..., 1] = (a[..., 1] * 50.0) / 10.0
# # #     return o

# # # def _hav(p1, p2):
# # #     la1 = torch.deg2rad(p1[..., 1]); la2 = torch.deg2rad(p2[..., 1])
# # #     dlat = torch.deg2rad(p2[..., 1] - p1[..., 1])
# # #     dlon = torch.deg2rad(p2[..., 0] - p1[..., 0])
# # #     a = (torch.sin(dlat/2).pow(2)
# # #          + torch.cos(la1)*torch.cos(la2)*torch.sin(dlon/2).pow(2))
# # #     return 2.0 * R_EARTH * torch.asin(a.clamp(1e-12, 1-1e-12).sqrt())

# # # def _atecte(pd, gd):
# # #     T = min(pd.shape[0], gd.shape[0])
# # #     if T < 2:
# # #         z = pd.new_zeros(1, pd.shape[1]); return z, z
# # #     lo1=torch.deg2rad(gd[:T-1,:,0]); la1=torch.deg2rad(gd[:T-1,:,1])
# # #     lo2=torch.deg2rad(gd[1:T, :,0]); la2=torch.deg2rad(gd[1:T, :,1])
# # #     lo3=torch.deg2rad(pd[1:T, :,0]); la3=torch.deg2rad(pd[1:T, :,1])
# # #     ya=torch.sin(lo2-lo1)*torch.cos(la2)
# # #     xa=(torch.cos(la1)*torch.sin(la2)-torch.sin(la1)*torch.cos(la2)*torch.cos(lo2-lo1))
# # #     be=torch.atan2(ya,xa)
# # #     ye=torch.sin(lo3-lo2)*torch.cos(la3)
# # #     xe=(torch.cos(la2)*torch.sin(la3)-torch.sin(la2)*torch.cos(la3)*torch.cos(lo3-lo2))
# # #     bee=torch.atan2(ye,xe)
# # #     tot=_hav(pd[1:T],gd[1:T]); ang=bee-be
# # #     return tot*torch.cos(ang), tot*torch.sin(ang)


# # # # ─────────────────────────────────────────────────────────────────────────────
# # # #  [G-G] Split metrics — đo easy_ADE và hard_ADE riêng biệt
# # # # ─────────────────────────────────────────────────────────────────────────────

# # # class SplitAcc:
# # #     """
# # #     Accumulator đo easy_ADE và hard_ADE riêng biệt.
# # #     Dùng để:
# # #     - EMA guard: phát hiện easy forgetting sớm
# # #     - Hard improvement: xác nhận hard loss đang hoạt động
# # #     """
# # #     def __init__(self):
# # #         self.easy_d = []
# # #         self.hard_d = []
# # #         self.all_d  = []
# # #         self.all_a  = []
# # #         self.all_c  = []
# # #         self.sd     = defaultdict(list)
# # #         self._h     = {12: 1, 24: 3, 48: 7, 72: 11}

# # #     def update(self, dist, is_hard=None, ate=None, cte=None):
# # #         """
# # #         Args:
# # #             dist:    [T, B] distance per step
# # #             is_hard: [B] bool (None → tất cả easy)
# # #         """
# # #         B = dist.shape[1]
# # #         mean_dist = dist.mean(0)  # [B]
# # #         self.all_d.extend(mean_dist.tolist())

# # #         if is_hard is not None:
# # #             easy_mask = ~is_hard
# # #             if easy_mask.any():
# # #                 self.easy_d.extend(mean_dist[easy_mask].tolist())
# # #             if is_hard.any():
# # #                 self.hard_d.extend(mean_dist[is_hard].tolist())
# # #         else:
# # #             self.easy_d.extend(mean_dist.tolist())

# # #         for h, s in self._h.items():
# # #             if s < dist.shape[0]:
# # #                 self.sd[h].extend(dist[s].tolist())

# # #         if ate is not None: self.all_a.extend(ate.abs().mean(0).tolist())
# # #         if cte is not None: self.all_c.extend(cte.abs().mean(0).tolist())

# # #     def compute(self):
# # #         r = {
# # #             "ADE":      float(np.mean(self.all_d))  if self.all_d  else float("nan"),
# # #             "easy_ADE": float(np.mean(self.easy_d)) if self.easy_d else float("nan"),
# # #             "hard_ADE": float(np.mean(self.hard_d)) if self.hard_d else float("nan"),
# # #             "ATE":      float(np.mean(self.all_a))  if self.all_a  else float("nan"),
# # #             "CTE":      float(np.mean(self.all_c))  if self.all_c  else float("nan"),
# # #             "n":        len(self.all_d),
# # #             "n_easy":   len(self.easy_d),
# # #             "n_hard":   len(self.hard_d),
# # #         }
# # #         for h in self._h:
# # #             v = self.sd.get(h, [])
# # #             r[f"{h}h"] = float(np.mean(v)) if v else float("nan")
# # #         return r


# # # class Acc:
# # #     """Simple accumulator (backward compat với evaluate())."""
# # #     def __init__(self):
# # #         self.d=[]; self.a=[]; self.c=[]; self.sd=defaultdict(list)
# # #         self._h={12:1, 24:3, 48:7, 72:11}
# # #     def update(self, dist, ate=None, cte=None):
# # #         self.d.extend(dist.mean(0).tolist())
# # #         for h,s in self._h.items():
# # #             if s < dist.shape[0]: self.sd[h].extend(dist[s].tolist())
# # #         if ate is not None: self.a.extend(ate.abs().mean(0).tolist())
# # #         if cte is not None: self.c.extend(cte.abs().mean(0).tolist())
# # #     def compute(self):
# # #         r = {"ADE": float(np.mean(self.d)) if self.d else float("nan"),
# # #              "ATE": float(np.mean(self.a)) if self.a else float("nan"),
# # #              "CTE": float(np.mean(self.c)) if self.c else float("nan"),
# # #              "n": len(self.d)}
# # #         for h in self._h:
# # #             v=self.sd.get(h,[]); r[f"{h}h"]=float(np.mean(v)) if v else float("nan")
# # #         return r


# # # def _score(r):
# # #     ade=r.get("ADE",1e9); h72=r.get("72h",1e9)
# # #     ate=r.get("ATE",1e9); cte=r.get("CTE",1e9)
# # #     if not np.isfinite(ate): ate=ade*0.46
# # #     if not np.isfinite(cte): cte=ade*0.53
# # #     return 100.0*(0.05*(ade/136.)+0.10*(r.get("12h",ade)/50.)
# # #                   +0.15*(r.get("24h",ade)/100.)+0.20*(r.get("48h",ade)/200.)
# # #                   +0.25*(h72/300.)+0.13*(ate/80.)+0.12*(cte/94.))

# # # def _beat(r):
# # #     p=[]
# # #     for k,t in [("ADE",172.68),("ATE",142.21),("CTE",42.04),
# # #                 ("72h",321.39),("12h",65.42),("24h",104.67),("48h",205.10)]:
# # #         v=r.get(k,1e9)
# # #         if np.isfinite(v) and v < t: p.append(f"{k}:{v:.1f}")
# # #     return "*** BEAT ST-TRANS: "+" ".join(p)+" ***" if p else ""

# # # def _gap(r):
# # #     out=[]
# # #     for k,ref in [("ADE",172.68),("72h",321.39),("ATE",142.21),("CTE",42.04)]:
# # #         v=r.get(k,float("nan"))
# # #         if np.isfinite(v):
# # #             out.append(f"{k}:{v:.0f}({'dn' if v<ref else 'up'}{abs(v-ref):.0f})")
# # #     return " | ".join(out)


# # # # ─────────────────────────────────────────────────────────────────────────────
# # # #  [G-E] Easy_ADE evaluation cho EMA guard
# # # # ─────────────────────────────────────────────────────────────────────────────

# # # @torch.no_grad()
# # # def evaluate_split(model, loader, dev, tag="", ema=None, steps=10,
# # #                    use_selector=False) -> Dict:
# # #     """
# # #     Evaluate với split easy/hard ADE.
# # #     Dùng fast ddim (10 steps) cho speed.
# # #     """
# # #     bk = None
# # #     if ema:
# # #         try: bk = ema.apply_to(model)
# # #         except: pass
# # #     model.eval()
# # #     acc = SplitAcc()
# # #     t0  = time.perf_counter()

# # #     for b in loader:
# # #         bl = move(list(b), dev)
# # #         obs_t = bl[0]

# # #         # Phân loại easy/hard bằng đặc trưng vật lý
# # #         is_hard = classify_hard_easy(obs_t[:, :, :2]).to(dev)

# # #         result = model.sample(bl, ddim_steps=steps, use_selector=use_selector)
# # #         p = result[0] if isinstance(result, (tuple, list)) else result
# # #         g = bl[1]; T = min(p.shape[0], g.shape[0])
# # #         pd = _ntd(p[:T]); gd = _ntd(g[:T])
# # #         dist = _hav(pd, gd)
# # #         at, ct = _atecte(pd, gd)
# # #         acc.update(dist, is_hard=is_hard, ate=at, cte=ct)

# # #     if bk:
# # #         try: ema.restore(model, bk)
# # #         except: pass

# # #     r  = acc.compute()
# # #     el = time.perf_counter() - t0

# # #     def _v(k): return r.get(k, float("nan"))
# # #     def _m(v, t): return "ok" if np.isfinite(v) and v < t else "no"

# # #     print(f"\n{'='*70}")
# # #     print(f"  [{tag}  {el:.0f}s]")
# # #     print(f"  ADE={_v('ADE'):.1f}[{_m(_v('ADE'),172.68)}]"
# # #           f"  easy_ADE={_v('easy_ADE'):.1f}  hard_ADE={_v('hard_ADE'):.1f}")
# # #     print(f"  12h={_v('12h'):.0f}  24h={_v('24h'):.0f}"
# # #           f"  48h={_v('48h'):.0f}  72h={_v('72h'):.0f}[{_m(_v('72h'),321.39)}]")
# # #     print(f"  ATE={_v('ATE'):.1f}[{_m(_v('ATE'),142.21)}]"
# # #           f"  CTE={_v('CTE'):.1f}[{_m(_v('CTE'),42.04)}]")
# # #     print(f"  n_easy={r.get('n_easy',0)}  n_hard={r.get('n_hard',0)}")
# # #     print(f"  vs ST-Trans: {_gap(r)}")
# # #     bt = _beat(r)
# # #     if bt: print(f"  {bt}")
# # #     print(f"  Score={_score(r):.2f}")
# # #     print(f"{'='*70}\n")
# # #     return r


# # # @torch.no_grad()
# # # def evaluate(model, loader, dev, tag="", ema=None, steps=20,
# # #              use_selector=False) -> Dict:
# # #     """Full evaluation (backward compat)."""
# # #     return evaluate_split(model, loader, dev, tag=tag, ema=ema,
# # #                           steps=steps, use_selector=use_selector)


# # # # ─────────────────────────────────────────────────────────────────────────────
# # # #  [G-F] Diversity check
# # # # ─────────────────────────────────────────────────────────────────────────────

# # # @torch.no_grad()
# # # def check_diversity(model, loader, dev, n_batches: int = 5,
# # #                     n_ensemble: int = 20, ddim_steps: int = 5) -> float:
# # #     """
# # #     Đo diversity score trên một subset của val set.
# # #     Nếu < 50 km → cảnh báo R1 (mode collapse).

# # #     Args:
# # #         n_batches:  số batches để đo (5 là đủ)
# # #         n_ensemble: số candidates (nhỏ để nhanh)
# # #         ddim_steps: DDIM steps (ít để nhanh)

# # #     Returns:
# # #         diversity_km: float
# # #     """
# # #     model.eval()
# # #     all_divs = []

# # #     for i, b in enumerate(loader):
# # #         if i >= n_batches:
# # #             break
# # #         bl = move(list(b), dev)

# # #         # Sample nhiều candidates
# # #         _, _, all_c = model.sample(bl, num_ensemble=n_ensemble,
# # #                                     ddim_steps=ddim_steps)
# # #         # all_c: [2N, T, B, 2] normalized (do speed sweep tạo 2× candidates)
# # #         # Reshape thành list
# # #         N_total = all_c.shape[0]
# # #         cands = [all_c[k] for k in range(N_total)]

# # #         div = compute_diversity_score(cands)
# # #         all_divs.append(div)

# # #     return float(np.mean(all_divs)) if all_divs else 0.0


# # # # ─────────────────────────────────────────────────────────────────────────────
# # # #  Checkpoint
# # # # ─────────────────────────────────────────────────────────────────────────────

# # # def _save_ckpt(path, ep, model, opt, sched, saver, tl, vl, extra=None):
# # #     m = _unwrap(model)
# # #     ema = getattr(m, "_ema", None)
# # #     esd = None
# # #     if ema and hasattr(ema, "shadow"):
# # #         try: esd = {k: v.cpu().clone() for k, v in ema.shadow.items()}
# # #         except: pass
# # #     d = {
# # #         "epoch": ep,
# # #         "model_state_dict": m.state_dict(),
# # #         "optimizer_state":  opt.state_dict(),
# # #         "scheduler_state":  sched.state_dict(),
# # #         "ema_shadow":       esd,
# # #         "best_score":       saver.bs,
# # #         "best_ade":         saver.ba,
# # #         "best_72h":         saver.b7,
# # #         "best_ate":         saver.bat,
# # #         "best_cte":         saver.bc,
# # #         "best_easy_ade":    saver.best_easy_ade,
# # #         "train_loss":       tl,
# # #         "val_loss":         vl,
# # #         "version":          "v59strategy",
# # #     }
# # #     if extra:
# # #         d.update(extra)
# # #     torch.save(d, path)


# # # # ─────────────────────────────────────────────────────────────────────────────
# # # #  Saver
# # # # ─────────────────────────────────────────────────────────────────────────────

# # # class Saver:
# # #     def __init__(self, patience=40, min_ep=35, enable_stop=True):
# # #         self.patience     = patience
# # #         self.min_ep       = min_ep
# # #         self.enable_stop  = enable_stop
# # #         self.cnt          = 0
# # #         self.stop         = False
# # #         self.bs           = float("inf")  # best score
# # #         self.ba           = float("inf")  # best ADE
# # #         self.b7           = float("inf")  # best 72h
# # #         self.bat          = float("inf")  # best ATE
# # #         self.bc           = float("inf")  # best CTE
# # #         self.best_easy_ade = float("inf") # [G-E] best easy ADE (EMA guard)

# # #     def update(self, r, model, out_dir, ep, opt, sched, tl, vl, tag=""):
# # #         sc  = _score(r)
# # #         ade = r.get("ADE",     1e9)
# # #         h72 = r.get("72h",     1e9)
# # #         ate = r.get("ATE",     1e9)
# # #         cte = r.get("CTE",     1e9)
# # #         easy_ade = r.get("easy_ADE", ade)  # fallback to ADE if no split

# # #         any_improved = False

# # #         for v, attr, fn in [
# # #             (ade,      "ba",  f"best_ade_{tag}.pth"),
# # #             (h72,      "b7",  f"best_72h_{tag}.pth"),
# # #             (ate,      "bat", f"best_ate_{tag}.pth"),
# # #             (cte,      "bc",  f"best_cte_{tag}.pth"),
# # #         ]:
# # #             if v < getattr(self, attr):
# # #                 setattr(self, attr, v)
# # #                 _save_ckpt(os.path.join(out_dir, fn), ep, model, opt,
# # #                            sched, self, tl, vl)
# # #                 any_improved = True

# # #         if sc < self.bs:
# # #             self.bs = sc
# # #             any_improved = True
# # #             _save_ckpt(os.path.join(out_dir, f"best_{tag or 'composite'}.pth"),
# # #                        ep, model, opt, sched, self, tl, vl)
# # #             print(f"  [BEST] {tag} ep={ep} score={sc:.2f}"
# # #                   f"  ADE={ade:.1f}  72h={h72:.0f}"
# # #                   f"  ATE={ate:.1f}  CTE={cte:.1f}")

# # #         # [G-E] Update best easy_ADE
# # #         if np.isfinite(easy_ade) and easy_ade < self.best_easy_ade:
# # #             self.best_easy_ade = easy_ade

# # #         if self.enable_stop:
# # #             if any_improved: self.cnt = 0
# # #             else:
# # #                 self.cnt += 1
# # #                 print(f"  No improve {self.cnt}/{self.patience}"
# # #                       f"  best={self.bs:.2f}  cur={sc:.2f}")
# # #             if ep >= self.min_ep and self.cnt >= self.patience:
# # #                 self.stop = True


# # # def _mksub(ds, n, bs, cf):
# # #     idx = random.Random(42).sample(range(len(ds)), min(n, len(ds)))
# # #     return DataLoader(Subset(ds, idx), batch_size=bs, shuffle=False,
# # #                       collate_fn=cf, num_workers=0, drop_last=False)


# # # # ─────────────────────────────────────────────────────────────────────────────
# # # #  Args
# # # # ─────────────────────────────────────────────────────────────────────────────

# # # def get_args():
# # #     p = argparse.ArgumentParser(
# # #         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# # #     p.add_argument("--dataset_root",       default="TCND_vn")
# # #     p.add_argument("--obs_len",            default=8,    type=int)
# # #     p.add_argument("--pred_len",           default=12,   type=int)
# # #     p.add_argument("--batch_size",         default=32,   type=int)
# # #     p.add_argument("--num_epochs",         default=100,  type=int)
# # #     p.add_argument("--learning_rate",      default=1e-4, type=float)
# # #     p.add_argument("--weight_decay",       default=1e-3, type=float)
# # #     p.add_argument("--warmup_epochs",      default=3,    type=int)
# # #     p.add_argument("--grad_clip",          default=1.0,  type=float)
# # #     p.add_argument("--use_amp",            action="store_true")
# # #     p.add_argument("--num_workers",        default=2,    type=int)
# # #     p.add_argument("--sigma_min",          default=0.02, type=float)
# # #     p.add_argument("--use_ot",             default=True, action="store_true")
# # #     p.add_argument("--no_ot",              dest="use_ot", action="store_false")
# # #     p.add_argument("--cfg_guidance_scale", default=1.5,  type=float)
# # #     p.add_argument("--n_ensemble",         default=50,   type=int)
# # #     p.add_argument("--use_ema",            default=True, action="store_true")
# # #     p.add_argument("--no_ema",             dest="use_ema", action="store_false")
# # #     p.add_argument("--ema_decay",          default=0.995, type=float)
# # #     p.add_argument("--patience",           default=40,   type=int)
# # #     p.add_argument("--min_ep",             default=35,   type=int)
# # #     p.add_argument("--val_freq",           default=3,    type=int)
# # #     p.add_argument("--val_subset_size",    default=500,  type=int)
# # #     p.add_argument("--fast_ddim",          default=10,   type=int)
# # #     p.add_argument("--full_ddim",          default=20,   type=int)
# # #     p.add_argument("--output_dir",         default="runs/v59strategy")
# # #     p.add_argument("--gpu_num",            default="0")
# # #     p.add_argument("--delim",              default=" ")
# # #     p.add_argument("--skip",               default=1,    type=int)
# # #     p.add_argument("--min_ped",            default=1,    type=int)
# # #     p.add_argument("--threshold",          default=0.002, type=float)
# # #     p.add_argument("--other_modal",        default="gph")
# # #     p.add_argument("--test_year",          default=None, type=int)
# # #     p.add_argument("--resume",             default=None)
# # #     p.add_argument("--resume_epoch",       default=None, type=int)
# # #     p.add_argument("--eval_test_after_train", default=True, action="store_true")
# # #     # GradNorm
# # #     p.add_argument("--gradnorm_alpha",     default=1.5,  type=float,
# # #                    help="GradNorm restoring force")
# # #     p.add_argument("--gradnorm_lr",        default=1e-3, type=float,
# # #                    help="GradNorm lambda learning rate")
# # #     # EMA guard
# # #     p.add_argument("--ema_guard_threshold", default=3.0, type=float,
# # #                    help="easy_ADE tăng bao nhiêu km thì rollback (km)")
# # #     # Diversity check
# # #     p.add_argument("--diversity_threshold", default=50.0, type=float,
# # #                    help="diversity_score < threshold → cảnh báo R1")
# # #     return p.parse_args()


# # # # ─────────────────────────────────────────────────────────────────────────────
# # # #  Main
# # # # ─────────────────────────────────────────────────────────────────────────────

# # # def main(args):
# # #     if torch.cuda.is_available():
# # #         os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
# # #     dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # #     os.makedirs(args.output_dir, exist_ok=True)

# # #     print("=" * 72)
# # #     print("  TC-FlowMatching v59-Strategy")
# # #     print("  [G-A] GradNorm: λ_dpe, λ_vel, λ_head, λ_spd, λ_acc tự điều chỉnh")
# # #     print("  [G-B] Easy/Hard: hard_score đa tiêu chí (curvature+speed_var+dir)")
# # #     print("  [G-C] easy_frac ≥ 50% enforcement mỗi batch")
# # #     print("  [G-D] α warm-up: 0→0.3 trong phase 2, phase 3: selector bật")
# # #     print("  [G-E] EMA guard: rollback nếu easy_ADE tăng > 3km")
# # #     print("  [G-F] Diversity check: cảnh báo nếu < 50km sau phase 1")
# # #     print("  [G-G] Split monitoring: easy_ADE + hard_ADE mỗi epoch")
# # #     print("  [G-H] Phase 4: freeze encoder, chỉ tune FM head + selector")
# # #     print(f"  Phases: 1[0-{PHASE_1_END}] 2[{PHASE_1_END+1}-{PHASE_2_END}]"
# # #           f" 3[{PHASE_2_END+1}-{PHASE_3_END}] 4[{PHASE_3_END+1}+]")
# # #     print(f"  Target: ADE<{TARGETS['ADE']} 72h<{TARGETS['72h']}"
# # #           f" ATE<{TARGETS['ATE']} CTE<{TARGETS['CTE']}")
# # #     print("=" * 72)

# # #     # Data
# # #     trd, trl = data_loader(args, {"root": args.dataset_root, "type": "train"}, test=False)
# # #     vd,  vl  = data_loader(args, {"root": args.dataset_root, "type": "val"},   test=True)
# # #     from Model.data.trajectoriesWithMe_unet_training import seq_collate
# # #     vsub = _mksub(vd, args.val_subset_size, args.batch_size, seq_collate)
# # #     print(f"  train:{len(trd)} seqs  val:{len(vd)} seqs")

# # #     # Model
# # #     model = TCFlowMatching(
# # #         pred_len=args.pred_len, obs_len=args.obs_len,
# # #         sigma_min=args.sigma_min, use_ema=args.use_ema,
# # #         ema_decay=args.ema_decay, use_ate_ot=args.use_ot,
# # #         cfg_guidance_scale=args.cfg_guidance_scale,
# # #     ).to(dev)
# # #     model.init_ema()

# # #     n_total = sum(p.numel() for p in model.parameters() if p.requires_grad)

# # #     # BUG-8: backbone vs learned weights (step_weights, selector small params)
# # #     # Backbone: numel > 100
# # #     # Selector weights cũng là model weights, clip bình thường
# # #     # Chỉ KHÔNG clip step_weights (numel = 12)
# # #     backbone_params = [p for p in model.parameters()
# # #                        if p.requires_grad and p.numel() > 100]
# # #     step_w_params   = [p for p in model.parameters()
# # #                        if p.requires_grad and p.numel() <= 12]

# # #     print(f"  params total: {n_total:,}")
# # #     print(f"  backbone (clipped at {args.grad_clip}): "
# # #           f"{sum(p.numel() for p in backbone_params):,}")
# # #     print(f"  step_weights (NOT clipped): "
# # #           f"{sum(p.numel() for p in step_w_params):,}")

# # #     # Optimizer & scheduler
# # #     opt    = optim.AdamW(model.parameters(),
# # #                           lr=args.learning_rate,
# # #                           weight_decay=args.weight_decay)
# # #     scaler = GradScaler("cuda", enabled=args.use_amp)
# # #     nstep  = len(trl)
# # #     total  = nstep * args.num_epochs
# # #     wstp   = nstep * args.warmup_epochs
# # #     sched  = get_cosine_schedule_with_warmup(opt, wstp, total, min_lr=1e-6)

# # #     # [G-A] GradNorm manager
# # #     gradnorm = GradNormManager(
# # #         terms=GRADNORM_TERMS,
# # #         alpha_gn=args.gradnorm_alpha,
# # #         lr=args.gradnorm_lr,
# # #         device=dev,
# # #     )

# # #     # [FIX-STABILITY-1] SmoothScheduler — α theo batch, không phải epoch
# # #     smooth_sched = SmoothScheduler(total_steps_per_epoch=nstep)

# # #     # Savers
# # #     fast_saver = Saver(patience=args.patience, min_ep=args.min_ep,
# # #                         enable_stop=False)
# # #     full_saver = Saver(patience=args.patience, min_ep=args.min_ep,
# # #                         enable_stop=True)

# # #     # Resume
# # #     start = 0
# # #     if args.resume and os.path.exists(args.resume):
# # #         print(f"  Loading: {args.resume}")
# # #         ck = torch.load(args.resume, map_location=dev)
# # #         m  = _unwrap(model)
# # #         ms, _ = m.load_state_dict(ck["model_state_dict"], strict=False)
# # #         if ms: print(f"  Missing: {len(ms)}")
# # #         ema = getattr(m, "_ema", None)
# # #         if ema and ck.get("ema_shadow"):
# # #             for k, v in ck["ema_shadow"].items():
# # #                 if k in ema.shadow: ema.shadow[k].copy_(v.to(dev))
# # #         try: opt.load_state_dict(ck["optimizer_state"])
# # #         except Exception as e: print(f"  Opt: {e}")
# # #         try: sched.load_state_dict(ck["scheduler_state"])
# # #         except:
# # #             for _ in range(ck.get("epoch", 0) * nstep): sched.step()
# # #         for a, attr in [("best_score","bs"), ("best_ade","ba"), ("best_72h","b7"),
# # #                         ("best_ate","bat"), ("best_cte","bc"),
# # #                         ("best_easy_ade","best_easy_ade")]:
# # #             if a in ck:
# # #                 setattr(full_saver, attr, ck[a])
# # #                 setattr(fast_saver, attr, ck[a])
# # #         start = args.resume_epoch or ck.get("epoch", 0) + 1
# # #         print(f"  Resume ep={start}  best_score={full_saver.bs:.2f}"
# # #               f"  best_easy_ADE={full_saver.best_easy_ade:.1f}")
# # #         # Restore smooth_sched global step
# # #         smooth_sched._global_step = start * nstep

# # #     try:
# # #         model = torch.compile(model, mode="reduce-overhead")
# # #         print("  torch.compile: ok")
# # #     except:
# # #         pass

# # #     # ── Phase 4 encoder list (dùng khi freeze) ────────────────────────────
# # #     def _get_encoder_params(m):
# # #         """Trả về params của encoder để freeze ở phase 4."""
# # #         enc = _unwrap(m)
# # #         params = []
# # #         # Freeze: spatial_enc, enc_1d, env_enc (các encoder đầu vào)
# # #         for mod_name in ["net.spatial_enc", "net.enc_1d", "net.env_enc",
# # #                           "net.bottleneck_pool", "net.bottleneck_proj",
# # #                           "net.decoder_proj"]:
# # #             parts = mod_name.split(".")
# # #             mod = enc
# # #             for p in parts:
# # #                 if hasattr(mod, p):
# # #                     mod = getattr(mod, p)
# # #                 else:
# # #                     mod = None
# # #                     break
# # #             if mod is not None:
# # #                 params.extend(list(mod.parameters()))
# # #         return params

# # #     _phase4_frozen = False

# # #     def _maybe_freeze_encoder(ep, m):
# # #         nonlocal _phase4_frozen
# # #         if get_phase(ep) == 4 and not _phase4_frozen:
# # #             enc_params = _get_encoder_params(m)
# # #             for p in enc_params:
# # #                 p.requires_grad_(False)
# # #             _phase4_frozen = True
# # #             print(f"  [Phase 4] Froze encoder params"
# # #                   f" (n={sum(1 for p in enc_params)})")
# # #             # Giảm LR ×0.1 cho phase 4
# # #             for g in opt.param_groups:
# # #                 g["lr"] = g["lr"] * 0.1
# # #             print(f"  [Phase 4] LR reduced to {opt.param_groups[0]['lr']:.2e}")

# # #     # ── EMA guard state ────────────────────────────────────────────────────
# # #     best_easy_ade_ema      = float("inf")
# # #     ema_rollback_count     = 0
# # #     _prev_phase            = 1
# # #     # [FIX-EMA-SMOOTH] Dùng EMA của easy_ADE thay vì raw per-epoch value.
# # #     # Vấn đề cũ: fast_eval (500 samples, 10 DDIM) có variance ±20-40km
# # #     #   → threshold 3km << variance → false rollbacks liên tục (28 lần!)
# # #     #   → smooth_sched bị kéo lùi → α không bao giờ tăng được
# # #     # Fix: EMA(easy_ADE) với weight=0.35, rollback threshold=20km
# # #     _ema_state  = {"smooth": float("inf")}
# # #     EMA_ALPHA   = 0.35   # weight cho current value (bias về recent)
# # #     EMA_THRESH  = 20.0   # km threshold để trigger rollback
# # #     # [FIX-STABILITY-3] Global hard threshold
# # #     global_hard_threshold = 0.5  # default, tính lại từ phase 2

# # #     print(f"  Training: {nstep} steps/ep, start ep={start}")
# # #     print("=" * 72)
# # #     ts = time.perf_counter()

# # #     for ep in range(start, args.num_epochs):
# # #         phase = get_phase(ep)

# # #         # [FIX-STABILITY-2] GradNorm reset khi chuyển phase
# # #         if phase != _prev_phase:
# # #             gradnorm.phase_reset(phase)
# # #             _prev_phase = phase
# # #             print(f"  [Phase {phase}] Chuyển phase tại epoch {ep}")

# # #             # [FIX-2] Reset best_easy_ade_ema và smooth EMA khi chuyển phase
# # #             best_easy_ade_ema      = float("inf")
# # #             ema_rollback_count     = 0
# # #             _ema_state["smooth"]   = float("inf")
# # #             print(f"  [EMA Guard] Reset best/smooth EMA khi chuyển phase")

# # #         # [FIX-D] Chỉ compute hard threshold 1 lần khi vào phase 2.
# # #         # hard_score_from_obs() chỉ phụ thuộc obs_traj (data cố định),
# # #         # KHÔNG phụ thuộc model weights → threshold KHÔNG thay đổi theo training.
# # #         # Tính lại mỗi epoch = ~50 forward passes × 60 epochs = lãng phí 25 phút.
# # #         if phase >= 2 and global_hard_threshold == 0.5:
# # #             global_hard_threshold = precompute_hard_threshold(trl, dev, n_batches=50)
# # #             print(f"  [FIX-D] global_hard_threshold={global_hard_threshold:.4f}"
# # #                   f" — computed once, reused for all epochs")

# # #         # [G-H] Phase 4: freeze encoder
# # #         _maybe_freeze_encoder(ep, model)

# # #         model.train()
# # #         sl = 0.0
# # #         t0 = time.perf_counter()

# # #         # Lấy lambda hiện tại từ GradNorm
# # #         lambda_dict = gradnorm.get_lambda_dict()

# # #         for i, batch in enumerate(trl):
# # #             bl    = move(list(batch), dev)
# # #             obs_t = bl[0]

# # #             # [FIX-STABILITY-1] α, easy_frac từ SmoothScheduler (per-batch)
# # #             alpha_hard    = smooth_sched.get_alpha_hard()
# # #             easy_frac     = smooth_sched.get_easy_frac()
# # #             sel_weight    = smooth_sched.get_selector_weight()
# # #             train_selector = smooth_sched.is_selector_active() and (sel_weight > 0.01)

# # #             # [FIX-STABILITY-3] Dùng global threshold thay vì per-batch p70
# # #             with torch.no_grad():
# # #                 is_hard = classify_hard_easy_global(
# # #                     obs_t[:, :, :2], global_hard_threshold).to(dev)

# # #             # [G-C] Enforce easy_frac — chỉ điều chỉnh mask, không reindex batch
# # #             is_hard = enforce_easy_frac(is_hard, easy_frac)

# # #             with autocast(device_type="cuda", enabled=args.use_amp):
# # #                 bd = model.get_loss_breakdown(
# # #                     bl,
# # #                     epoch=ep,
# # #                     alpha_hard=alpha_hard,
# # #                     is_hard=is_hard,
# # #                     train_selector=train_selector,
# # #                     lambda_dict=lambda_dict,
# # #                 )

# # #             opt.zero_grad()
# # #             scaler.scale(bd["total"]).backward()
# # #             scaler.unscale_(opt)

# # #             # [G-A] GradNorm: đo grad_norm từ p.grad SAU backward, TRƯỚC clip
# # #             # Đây là cách đúng — không gọi backward() lần 2 (graph đã giải phóng)
# # #             # Dùng grad norm của ctx_fc2.weight (shared param cuối context fusion)
# # #             try:
# # #                 shared_p_name = "net.ctx_fc2.weight"
# # #                 shared_p = None
# # #                 m_unwrapped = _unwrap(model)
# # #                 for name, p in m_unwrapped.named_parameters():
# # #                     if name == shared_p_name and p.grad is not None:
# # #                         shared_p = p
# # #                         break

# # #                 if shared_p is not None and shared_p.grad is not None:
# # #                     # Tính grad_norm của từng term: ước lượng bằng cách nhìn
# # #                     # vào current grad norm của shared_p × λ_i / λ_mean
# # #                     base_gnorm = shared_p.grad.norm().item()
# # #                     lambda_dict_cur = gradnorm.get_lambda_dict()
# # #                     lambda_mean = np.mean(list(lambda_dict_cur.values()))

# # #                     lambda_grads_approx = {}
# # #                     for t in GRADNORM_TERMS:
# # #                         λ_i = lambda_dict_cur.get(t, 1.0)
# # #                         # Ước lượng: gnorm_i ≈ base_gnorm × λ_i / λ_mean
# # #                         # (gần đúng, đủ để GradNorm cân bằng)
# # #                         lambda_grads_approx[t] = torch.tensor(
# # #                             base_gnorm * λ_i / max(lambda_mean, 1e-6),
# # #                             device=dev)

# # #                     loss_vals_float = {
# # #                         t: float(bd[t].item()) if torch.is_tensor(bd.get(t)) else float(bd.get(t, 0.0))
# # #                         for t in GRADNORM_TERMS
# # #                         if t in bd
# # #                     }
# # #                     gradnorm.update(loss_vals_float, lambda_grads_approx)
# # #                     lambda_dict = gradnorm.get_lambda_dict()
# # #             except Exception:
# # #                 pass  # GradNorm không critical

# # #             # BUG-8: clip chỉ backbone
# # #             torch.nn.utils.clip_grad_norm_(backbone_params, args.grad_clip)

# # #             scaler.step(opt)
# # #             scaler.update()
# # #             # BUG-7: unconditional
# # #             try: sched.step()
# # #             except: pass
# # #             model.ema_update()

# # #             # [FIX-STABILITY-1] Advance smooth scheduler mỗi step
# # #             smooth_sched.step()

# # #             sl += bd["total"].item()

# # #             if i % 20 == 0:
# # #                 lr     = opt.param_groups[0]["lr"]
# # #                 tot    = bd["total"].item()
# # #                 warn   = " [!>10]" if tot > 10.0 else ""
# # #                 sw_st  = _unwrap(model).step_weights.stats()
# # #                 gn_log = gradnorm.log_stats()
# # #                 n_hard_batch = int(is_hard.sum().item())

# # #                 print(
# # #                     f"  [{ep:>3}|Ph{phase}][{i:>3}/{nstep}]"
# # #                     f"  tot={tot:.3f}{warn}"
# # #                     f"  fm={bd.get('l_fm',0):.4f}"
# # #                     f"  dpe={bd.get('dpe',0):.4f}"
# # #                     f"  vel={bd.get('vel_reg',0):.4f}"
# # #                     f"  hard={bd.get('l_hard_total',0):.4f}"
# # #                     f"  {smooth_sched.log_stats()}"
# # #                     f"  n_h={n_hard_batch}/{obs_t.shape[1]}"
# # #                     f"  sw72={sw_st['sw_72h']:.2f}"
# # #                     f"  gn_dense={gn_log.get('gn_dense',0)}"
# # #                     f"  lr={lr:.2e}"
# # #                 )

# # #         avt  = sl / nstep
# # #         t_ep = time.perf_counter() - t0

# # #         # BUG-5: val với epoch=-1 → skip augmentation
# # #         model.eval()
# # #         vls = 0.0
# # #         with torch.no_grad():
# # #             for batch in vl:
# # #                 bv = move(list(batch), dev)
# # #                 with autocast(device_type="cuda", enabled=args.use_amp):
# # #                     vls += model.get_loss(bv, epoch=-1).item()
# # #         avv = vls / len(vl)

# # #         lr_cur   = opt.param_groups[0]["lr"]
# # #         gn_stats = gradnorm.log_stats()
# # #         alpha_hard_ep = smooth_sched.get_alpha_hard()
# # #         easy_frac_ep  = smooth_sched.get_easy_frac()
# # #         use_sel_eval  = smooth_sched.is_selector_active()

# # #         print(f"  Epoch {ep:>3}|Ph{phase}"
# # #               f" | train={avt:.4f} val={avv:.4f}"
# # #               f" | {smooth_sched.log_stats()}"
# # #               f" | λ_dpe={gn_stats.get('λ_l_dpe',1.0):.2f}"
# # #               f" λ_vel={gn_stats.get('λ_l_vel_reg',1.0):.2f}"
# # #               f" | lr={lr_cur:.2e} | {t_ep:.0f}s")

# # #         # [G-G] Fast eval với split monitoring (dùng fast ddim)
# # #         rf = evaluate_split(model, vsub, dev,
# # #                              tag=f"FAST ep{ep}",
# # #                              steps=args.fast_ddim,
# # #                              use_selector=use_sel_eval)
# # #         fast_saver.update(rf, model, args.output_dir, ep, opt, sched, avt, avv, tag="fast")

# # #         # [G-E] EMA guard: kiểm tra easy_ADE sau mỗi epoch
# # #         # [FIX-EMA-SMOOTH] Dùng EMA(easy_ADE) thay vì raw value.
# # #         # 28 rollbacks trong 51 epoch là do: fast_eval variance ±20-40km > threshold 3km.
# # #         # Giờ dùng EMA(α=0.35) + threshold 20km → chỉ rollback khi xu hướng thực sự xấu đi.
# # #         easy_ade_curr = rf.get("easy_ADE", rf.get("ADE", float("inf")))
# # #         if np.isfinite(easy_ade_curr):
# # #             # Update EMA
# # #             if _ema_state["smooth"] == float("inf"):
# # #                 _ema_state["smooth"] = easy_ade_curr
# # #             else:
# # #                 _ema_state["smooth"] = ((1 - EMA_ALPHA) * _ema_state["smooth"]
# # #                                         + EMA_ALPHA * easy_ade_curr)
# # #             smooth_val = _ema_state["smooth"]

# # #             if smooth_val < best_easy_ade_ema:
# # #                 best_easy_ade_ema = smooth_val
# # #                 print(f"  [EMA Guard] New best smooth_easy_ADE={smooth_val:.1f}"
# # #                       f" (raw={easy_ade_curr:.1f})")

# # #             elif (smooth_val - best_easy_ade_ema) > EMA_THRESH:
# # #                 ema_rollback_count += 1
# # #                 print(f"\n  [EMA GUARD] smooth_easy_ADE tăng bền vững "
# # #                       f"{smooth_val - best_easy_ade_ema:.1f} km"
# # #                       f" (best={best_easy_ade_ema:.1f} smooth={smooth_val:.1f}"
# # #                       f" raw={easy_ade_curr:.1f})")
# # #                 print(f"  [EMA GUARD] Rollback #{ema_rollback_count}")

# # #                 # Restore từ EMA weights
# # #                 m_raw   = _unwrap(model)
# # #                 ema_obj = getattr(m_raw, "_ema", None)
# # #                 if ema_obj:
# # #                     try:
# # #                         bk = ema_obj.apply_to(model)
# # #                         _save_ckpt(
# # #                             os.path.join(args.output_dir,
# # #                                          f"ema_guard_rollback_ep{ep}.pth"),
# # #                             ep, model, opt, sched, full_saver, avt, avv,
# # #                             extra={"ema_rollback": True,
# # #                                    "easy_ade_trigger": easy_ade_curr},
# # #                         )
# # #                         ema_obj.restore(model, bk)
# # #                         print(f"  [EMA GUARD] Saved rollback ckpt, "
# # #                               f"restored current weights for continued training")
# # #                     except Exception as e:
# # #                         print(f"  [EMA GUARD] Apply/restore failed: {e}")

# # #                 # Pullback smooth_sched (nhỏ hơn: 2 epoch thay vì 3)
# # #                 pullback_steps = nstep * 2
# # #                 smooth_sched._global_step = max(
# # #                     smooth_sched._phase2_start_step,
# # #                     smooth_sched._global_step - pullback_steps
# # #                 )
# # #                 print(f"  [EMA GUARD] α giảm về {smooth_sched.get_alpha_hard():.3f}"
# # #                       f" (step pulled back {pullback_steps} steps)")

# # #                 if ema_rollback_count >= 5:
# # #                     print(f"  [EMA GUARD] 5 rollbacks — disable hard loss push")
# # #                     if phase == 3:
# # #                         smooth_sched._global_step = min(
# # #                             smooth_sched._global_step,
# # #                             smooth_sched._phase3_start_step - 1
# # #                         )

# # #         # [G-F] Diversity check cuối phase 1
# # #         div_score = None
# # #         if ep == PHASE_1_END:
# # #             print(f"\n  {'='*50}")
# # #             print(f"  [CHECKPOINT] Cuối phase 1 — kiểm tra diversity...")
# # #             div_score = check_diversity(model, vsub, dev)
# # #             print(f"  Diversity score: {div_score:.1f} km")
# # #             if div_score < args.diversity_threshold:
# # #                 print(f"  [CẢNH BÁO R1] diversity={div_score:.1f} km"
# # #                       f" < {args.diversity_threshold} km")
# # #                 print(f"  → FM candidates không đủ đa dạng → selector sẽ không hiệu quả")
# # #                 print(f"  → Hành động: kiểm tra noise_std hoặc thêm L_diversity")
# # #             else:
# # #                 print(f"  [OK] diversity={div_score:.1f} km >= "
# # #                       f"{args.diversity_threshold} km → tiến hành phase 2")
# # #             print(f"  {'='*50}\n")

# # #         # Đánh giá đầy đủ mỗi val_freq epoch
# # #         if ep % args.val_freq == 0:
# # #             em = getattr(_unwrap(model), "_ema", None)
# # #             rr = evaluate_split(model, vl, dev,
# # #                                   tag=f"RAW ep{ep}",
# # #                                   steps=args.full_ddim,
# # #                                   use_selector=use_sel_eval)
# # #             full_saver.update(rr, model, args.output_dir, ep, opt, sched,
# # #                               avt, avv, tag="raw")

# # #             if em and ep >= 3:
# # #                 re = evaluate_split(model, vl, dev,
# # #                                      tag=f"EMA ep{ep}",
# # #                                      ema=em,
# # #                                      steps=args.full_ddim,
# # #                                      use_selector=use_sel_eval)
# # #                 full_saver.update(re, model, args.output_dir, ep, opt, sched,
# # #                                   avt, avv, tag="ema")

# # #         # Periodic checkpoint
# # #         if ep % 10 == 0 or ep == args.num_epochs - 1:
# # #             _save_ckpt(
# # #                 os.path.join(args.output_dir, f"ckpt_ep{ep:03d}.pth"),
# # #                 ep, model, opt, sched, full_saver, avt, avv,
# # #                 extra={"phase": phase,
# # #                        "alpha_hard": smooth_sched.get_alpha_hard(),
# # #                        "smooth_sched_step": smooth_sched.global_step,
# # #                        "global_hard_threshold": global_hard_threshold,
# # #                        "diversity": div_score},
# # #             )

# # #         if full_saver.stop:
# # #             print(f"  Early stop ep={ep}")
# # #             break

# # #     th = (time.perf_counter() - ts) / 3600.0
# # #     print(f"\n  Best: ADE={full_saver.ba:.1f}  72h={full_saver.b7:.0f}"
# # #           f"  ATE={full_saver.bat:.1f}  CTE={full_saver.bc:.1f}"
# # #           f"  easy_ADE={full_saver.best_easy_ade:.1f}"
# # #           f"  ({th:.2f}h)")

# # #     # Post-training test
# # #     if args.eval_test_after_train:
# # #         print("\n" + "="*72 + "\n  POST-TRAINING TEST\n" + "="*72)
# # #         try: _, tl2 = data_loader(args, {"root": args.dataset_root, "type": "test"}, test=True)
# # #         except: print("  No test → using val"); tl2 = vl

# # #         for fn, lb in [("best_raw.pth", "RAW"), ("best_ema.pth", "EMA"),
# # #                         ("best_ade_raw.pth", "BEST_ADE"),
# # #                         ("best_cte_raw.pth", "BEST_CTE")]:
# # #             pp = os.path.join(args.output_dir, fn)
# # #             if not os.path.exists(pp): continue
# # #             ck = torch.load(pp, map_location=dev)
# # #             _unwrap(model).load_state_dict(ck["model_state_dict"], strict=False)
# # #             em = getattr(_unwrap(model), "_ema", None)
# # #             if em and ck.get("ema_shadow"):
# # #                 for k, v in ck["ema_shadow"].items():
# # #                     if k in em.shadow: em.shadow[k].copy_(v.to(dev))
# # #             r = evaluate_split(model, tl2, dev, tag=f"TEST/{lb}",
# # #                                 steps=args.full_ddim,
# # #                                 use_selector=(get_phase(ck.get("epoch", 80)) >= 3))
# # #             print(f"\n  --- {lb} (ep={ck.get('epoch','?')}) ---")
# # #             for key, ref in [("ADE", 172.68), ("72h", 321.39),
# # #                               ("ATE", 142.21), ("CTE", 42.04)]:
# # #                 v = r.get(key, float("nan"))
# # #                 mk = "BEAT!" if np.isfinite(v) and v < ref else f"need {ref:.0f}"
# # #                 print(f"    {key:<8}: {v:>8.1f} km  [{mk}  gap:{v-ref:+.1f}]")
# # #             ea = r.get("easy_ADE", float("nan"))
# # #             ha = r.get("hard_ADE", float("nan"))
# # #             print(f"    easy_ADE={ea:.1f}  hard_ADE={ha:.1f}")

# # #     print("=" * 72)


# # # if __name__ == "__main__":
# # #     args = get_args()
# # #     np.random.seed(42)
# # #     torch.manual_seed(42)
# # #     if torch.cuda.is_available():
# # #         torch.cuda.manual_seed_all(42)
# # #     main(args)

# # """
# # train_v59_strategy.py — TC-FlowMatching v59-Strategy Training Script [FIXED-v2]
# # ══════════════════════════════════════════════════════════════════════════════

# # BASE: phiên bản với FIX-STABILITY-1/2/3, FIX-ALPHA-TIMING, FIX-EMA-SMOOTH, FIX-D
# # (đã chạy và có log epoch 0-51)

# # CÁC BUG MỚI ĐƯỢC FIX TRONG PHIÊN BẢN NÀY (dựa trên phân tích log thực tế):

# #   [FIX-GN-1] GradNorm "loss-ratio" KHÔNG có equilibrium → runaway có hệ thống
# #     Vấn đề cũ: r_i = raw_loss_i / mean(raw_loss_j) — KHÔNG phụ thuộc λ_i.
# #     l_dpe có raw scale lớn hơn cấu trúc so với l_vel_reg/l_heading/l_accel
# #     → r_dpe > 1 vĩnh viễn → λ_dpe luôn bị "tăng" → chạy tới ceiling (4.18)
# #     → λ_vel/head/accel bị đẩy xuống floor (0.01) → 3 objective gần như bị tắt.
# #     Log xác nhận: λ_dpe 1.00→4.18 (ep20→46), λ_vereg/heading/accel →0.01.

# #     Fix: "equal-contribution" scheme.
# #       contribution_i = λ_i * l_i  (phần đóng góp THỰC vào total loss)
# #       ratio_i = contribution_i / mean(contribution_j)
# #       ratio_i > 1 (đóng góp quá nhiều) → GIẢM λ_i
# #       ratio_i < 1 (đóng góp quá ít)    → TĂNG λ_i
# #     → có feedback âm thật: λ_i tăng → contribution_i tăng → ratio_i tăng
# #       → bị kéo giảm lại → equilibrium λ_i ≈ mean_contrib / l_i.
# #     Tự động bù trừ scale khác nhau giữa các loss term, không runaway.

# #   [FIX-GN-2] phase_reset() KHÔNG reset λ về 1.0 nữa
# #     Vấn đề cũ: λ_dpe=3.11→1.0 đột ngột tại ep20 (và tương tự ep50)
# #     → tổng loss đổi đột ngột → easy_ADE nhảy 278→369km (ep20), 278→302 (ep51).
# #     L_base (compute_st_trans_loss) không đổi giữa các phase — chỉ cộng thêm
# #     L_hard/L_sel bên ngoài GradNorm → không có lý do để reset λ.
# #     Fix: giữ nguyên λ hiện tại, chỉ reset optimizer momentum + dense window
# #     ngắn (50 step) để re-adapt nhẹ nếu landscape có dịch nhẹ.

# #   [FIX-EMA-2] EMA guard pullback có thể làm α kẹt vĩnh viễn ở 0
# #     Vấn đề cũ: pullback_steps=nstep*2, rollback dày đặc trong vùng α-ramp
# #     (ramp chỉ rộng ~35 step quanh epoch 24.5) → mỗi rollback trong vùng ramp
# #     kéo step về trước vùng ramp → α=0 lại → α=0.000 suốt cả phase 2 (30 epoch).
# #     Fix: chỉ pullback schedule nếu α hiện tại < 0.25 (còn trong vùng ramp).
# #     Nếu α đã ramp xong (>=0.25), rollback chỉ restore weights, KHÔNG pullback
# #     schedule. Giảm pullback_steps = nstep//2.

# #   [FIX-PERF-1] Xóa dead/wasteful computation mỗi step
# #     Code cũ lặp qua TOÀN BỘ named_parameters() mỗi step để tìm
# #     "net.ctx_fc2.weight", tính lambda_grads_approx rồi truyền vào update()
# #     — nhưng update() ghi rõ "lambda_grads: unused, ignore". Hoàn toàn lãng phí.
# #     Fix: xóa block này, chỉ extract loss_vals_float và gọi update() trực tiếp.

# #   [FIX-CKPT-1] _save_ckpt lưu smooth_sched_step + global_hard_threshold
# #     cho MỌI checkpoint (trước chỉ có trong periodic ckpt's "extra").

# #   [FIX-DIV-1] Diversity collapse (7.8km << 50km) — tự động xử lý
# #     Theo strategy doc mục R1: "Tăng noise std ×1.5".
# #     Fix: tự động ×1.5 sigma_min và ctx_noise_scale của model khi
# #     diversity < threshold sau phase 1.

# #   [FIX-THRESH-1] precompute threshold dùng flag riêng _threshold_computed
# #     thay vì so sánh == 0.5 (fragile).

# #   [FIX-EASYFRAC4] get_easy_frac() phase 4 dùng sigmoid (consistent với phase 2/3)

# # GIỮ NGUYÊN từ phiên bản trước:
# #   [G-A] Easy/hard pipeline với GLOBAL threshold
# #   [G-C] easy_frac enforcement
# #   [G-D] α smooth warm-up theo BATCH (FIX-ALPHA-TIMING: mid=15% of phase)
# #   [G-E] EMA guard với EMA-smoothed easy_ADE (FIX-EMA-SMOOTH)
# #   [G-F] Diversity check sau phase 1
# #   [G-G] Split monitoring
# #   [G-H] Phase 4: freeze encoder
# #   BUG-5/6/7/8 từ v59-tweaked
# # """
# # from __future__ import annotations
# # import sys, os
# # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # import argparse, math, random, time
# # from collections import defaultdict
# # from typing import Dict, List, Optional

# # import numpy as np
# # import torch
# # import torch.optim as optim
# # import torch.nn.functional as F
# # from torch.amp import autocast, GradScaler
# # from torch.utils.data import DataLoader, Subset

# # from Model.data.loader_training import data_loader
# # from Model.flow_matching_model import (
# #     TCFlowMatching,
# #     hard_score_from_obs,       # dùng trong precompute_hard_threshold
# #     classify_hard_easy,        # dùng trong evaluate_split
# #     compute_diversity_score,   # dùng trong check_diversity
# #     _norm_to_deg,
# #     _haversine_deg,
# # )

# # try:
# #     from Model.utils import get_cosine_schedule_with_warmup
# # except ImportError:
# #     from torch.optim.lr_scheduler import CosineAnnealingLR
# #     def get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps, min_lr=1e-6):
# #         return CosineAnnealingLR(opt, T_max=max(total_steps, 1), eta_min=min_lr)

# # TARGETS = {
# #     "ADE": 172.68, "72h": 321.39, "ATE": 142.21, "CTE": 42.04,
# #     "12h": 65.42,  "24h": 104.67, "48h": 205.10,
# # }
# # R_EARTH = 6371.0

# # # ── Phase boundaries ──────────────────────────────────────────────────────────
# # PHASE_1_END  = 19   # epoch 0–19:  warm-up, α=0, easy_frac=80%
# # PHASE_2_END  = 49   # epoch 20–49: hard intro, α 0→0.3, easy_frac 80→60%
# # PHASE_3_END  = 79   # epoch 50–79: selector, α=0.3, easy_frac=55%
# # # epoch 80+: fine-tune, LR×0.1, freeze encoder

# # # ── GradNorm terms (các loss term cần cân bằng) ──────────────────────────────
# # GRADNORM_TERMS = ["l_dpe", "l_vel_reg", "l_heading", "l_speed", "l_accel"]

# # # Alpha threshold để coi là "đã ramp xong" — dùng cho EMA guard pullback gate
# # ALPHA_RAMP_DONE_THRESHOLD = 0.25


# # # ─────────────────────────────────────────────────────────────────────────────
# # #  [FIX-GN-1+2] GradNorm implementation — equal-contribution scheme
# # #
# # #  Thuật toán mới:
# # #    1. contribution_i = λ_i * l_i  (phần đóng góp thực vào total loss)
# # #    2. mean_contrib = mean({contribution_i})
# # #    3. ratio_i = contribution_i / mean_contrib
# # #    4. ratio_i > 1 (đóng góp quá nhiều so với TB) → GIẢM λ_i → grad > 0
# # #       ratio_i < 1 (đóng góp quá ít)              → TĂNG λ_i → grad < 0
# # #    5. Update λ_i qua Adam với grad set thủ công
# # #    6. Re-normalize: sum(λ_i) = n_terms, clamp [0.1, 4.0]
# # #
# # #  Tại sao có equilibrium (khác bản cũ):
# # #    Nếu λ_i tăng → contribution_i = λ_i*l_i tăng → ratio_i tăng
# # #    → grad_i = (ratio_i-1)*scale tăng (dương hơn) → λ_i bị kéo giảm lại.
# # #    => Feedback ÂM thật sự → tồn tại điểm cân bằng λ_i ≈ mean_contrib/l_i,
# # #       tự động bù trừ sự khác biệt về raw scale giữa các loss term.
# # #
# # #  Bản cũ (r_i = raw_l_i/mean_raw_l): KHÔNG có λ_i trong r_i
# # #    → term có raw scale lớn hơn cấu trúc luôn có r_i>1 VĨNH VIỄN
# # #    → λ_i của term đó bị đẩy tới ceiling, không có equilibrium.
# # # ─────────────────────────────────────────────────────────────────────────────

# # class GradNormManager:
# #     """
# #     Quản lý GradNorm cho các loss terms — equal-contribution scheme.

# #     [FIX-GN-2] phase_reset(): KHÔNG reset λ về 1.0.
# #     L_base (compute_st_trans_loss) không đổi giữa các phase — chỉ cộng thêm
# #     L_hard/L_sel bên ngoài GradNorm. Reset λ→1.0 đột ngột tại ranh giới phase
# #     gây shock lớn (easy_ADE nhảy hàng chục-trăm km, xem log ep20, ep51).
# #     Chỉ reset optimizer momentum + bật dense update ngắn (50 step) để
# #     re-adapt nhẹ nếu landscape có dịch (do thêm gradient từ L_hard/L_sel
# #     chảy qua shared encoder).
# #     """
# #     def __init__(self, terms: List[str], alpha_gn: float = 1.5,
# #                  lr: float = 1e-3, device=None,
# #                  lambda_min: float = 0.1, lambda_max: float = 4.0):
# #         self.terms      = terms
# #         self.alpha_gn   = alpha_gn
# #         self.device     = device or torch.device("cpu")
# #         self.lambda_min = lambda_min
# #         self.lambda_max = lambda_max

# #         # λ_i: meta-optimization parameter (KHÔNG phải model weights)
# #         # Init = 1.0 cho tất cả terms
# #         self.lambdas = {t: torch.ones(1, device=self.device, requires_grad=True)
# #                         for t in terms}
# #         self.opt = optim.Adam(list(self.lambdas.values()), lr=lr)

# #         self._step             = 0
# #         self._update_freq      = 5      # update mỗi 5 steps (bình thường)
# #         self._dense_steps_left = 200    # dense ở lúc khởi động training
# #         self._current_phase    = 1

# #     def phase_reset(self, new_phase: int) -> None:
# #         """
# #         [FIX-GN-2] Gọi khi chuyển sang phase mới.
# #         KHÔNG reset λ — chỉ reset optimizer momentum và bật dense window
# #         ngắn (50 step, không phải 200) để re-adapt nhẹ.
# #         """
# #         if new_phase == self._current_phase:
# #             return
# #         self._current_phase = new_phase

# #         # Reset optimizer state (momentum cũ không còn phù hợp với landscape
# #         # mới) nhưng GIỮ λ hiện tại — đây là điểm khác biệt cốt lõi với bản cũ
# #         self.opt = optim.Adam(list(self.lambdas.values()),
# #                                lr=self.opt.param_groups[0]["lr"])

# #         # Dense update ngắn để re-adapt nhẹ (không phải reset hoàn toàn)
# #         self._dense_steps_left = 50

# #         cur = self.get_lambda_dict()
# #         lam_str = " ".join(f"λ_{t.replace('l_','').replace('_reg','')}"
# #                             f"={cur[t]:.2f}" for t in self.terms)
# #         print(f"  [GradNorm] Phase {new_phase}: GIỮ λ hiện tại ({lam_str}), "
# #               f"reset optimizer momentum, dense update 50 steps")

# #     def get_lambda_dict(self) -> Dict[str, float]:
# #         """Trả về λ hiện tại để truyền vào get_loss_breakdown()."""
# #         return {t: float(self.lambdas[t].item()) for t in self.terms}

# #     def update(self, loss_term_values: Dict[str, float]) -> None:
# #         """
# #         [FIX-GN-1] Cập nhật λ dùng equal-contribution scheme.

# #         Args:
# #             loss_term_values: {term: float} — loss RAW value từng term
# #                               (detached, chưa nhân λ)
# #         """
# #         self._step += 1

# #         # Adaptive update frequency
# #         if self._dense_steps_left > 0:
# #             self._dense_steps_left -= 1
# #             update_freq = 1
# #         else:
# #             update_freq = self._update_freq

# #         def _renorm():
# #             """Re-normalize: sum(λ)=n_terms, clamp [lambda_min, lambda_max]."""
# #             with torch.no_grad():
# #                 n_terms = len(self.terms)
# #                 for t in self.terms:
# #                     self.lambdas[t].clamp_(min=self.lambda_min,
# #                                             max=self.lambda_max)
# #                 total_λ = sum(float(self.lambdas[t].item()) for t in self.terms)
# #                 if total_λ > 0:
# #                     scale = n_terms / total_λ
# #                     for t in self.terms:
# #                         self.lambdas[t].mul_(scale)
# #                         self.lambdas[t].clamp_(min=self.lambda_min,
# #                                                 max=self.lambda_max)

# #         if self._step % update_freq != 0:
# #             _renorm()
# #             return

# #         if not loss_term_values:
# #             _renorm()
# #             return

# #         loss_vals = {t: v for t, v in loss_term_values.items()
# #                      if t in self.terms and np.isfinite(v) and v > 0}
# #         if len(loss_vals) < 2:
# #             _renorm()
# #             return

# #         # [FIX-GN-1] contribution_i = λ_i * l_i (phần đóng góp THỰC)
# #         lambda_now = self.get_lambda_dict()
# #         contributions = {t: lambda_now[t] * loss_vals[t] for t in loss_vals}
# #         mean_contrib = float(np.mean(list(contributions.values())))
# #         if mean_contrib <= 0:
# #             _renorm()
# #             return

# #         self.opt.zero_grad()
# #         for t in self.terms:
# #             if t not in contributions:
# #                 continue
# #             ratio = contributions[t] / (mean_contrib + 1e-8)
# #             # ratio > 1: đóng góp quá nhiều → cần GIẢM λ_i
# #             #   Adam: param -= lr*grad → muốn giảm λ_i → cần grad > 0
# #             #   grad = +(ratio - 1.0) * scale  → ratio>1 → grad>0 ✓
# #             # ratio < 1: đóng góp quá ít → cần TĂNG λ_i → grad < 0
# #             #   grad = +(ratio - 1.0) * scale  → ratio<1 → grad<0 ✓
# #             grad_val = (ratio - 1.0) * 0.1
# #             if self.lambdas[t].grad is None:
# #                 self.lambdas[t].grad = torch.zeros_like(self.lambdas[t])
# #             self.lambdas[t].grad.fill_(float(grad_val))

# #         self.opt.step()
# #         _renorm()

# #     def log_stats(self) -> Dict[str, float]:
# #         """Log λ hiện tại để monitor."""
# #         d = {f"λ_{t}": float(self.lambdas[t].item()) for t in self.terms}
# #         d["gn_dense"] = self._dense_steps_left
# #         return d


# # # ─────────────────────────────────────────────────────────────────────────────
# # #  Phase helper
# # # ─────────────────────────────────────────────────────────────────────────────

# # def get_phase(epoch: int) -> int:
# #     """
# #     Returns:
# #         1: warm-up (ep 0–19)
# #         2: hard introduction (ep 20–49)
# #         3: selector (ep 50–79)
# #         4: fine-tune (ep 80+)
# #     """
# #     if epoch <= PHASE_1_END:  return 1
# #     if epoch <= PHASE_2_END:  return 2
# #     if epoch <= PHASE_3_END:  return 3
# #     return 4


# # class SmoothScheduler:
# #     """
# #     [FIX-STABILITY-1] α smooth warm-up theo BATCH, không phải epoch.

# #     [FIX-ALPHA-TIMING] mid point ở 15% của phase thay vì 50%:
# #         mid ở epoch ~24.5 (phase2) → α≈0.15 ở ep23, α≈0.3 ở ep26
# #         → hard loss kích hoạt sớm hơn ~10 epoch so với mid=50%.

# #     Selector cũng dùng cùng logic để bật đồng bộ.
# #     """
# #     def __init__(self, total_steps_per_epoch: int):
# #         self.steps_per_epoch = total_steps_per_epoch
# #         self._global_step = 0

# #         # Điểm bắt đầu của mỗi phase (tính theo global step)
# #         self._phase2_start_step = (PHASE_1_END + 1) * total_steps_per_epoch
# #         self._phase3_start_step = (PHASE_2_END + 1) * total_steps_per_epoch
# #         self._phase4_start_step = (PHASE_3_END + 1) * total_steps_per_epoch

# #         # [FIX-ALPHA-TIMING] Temperature ngắn → α tăng SỚM trong phase 2.
# #         self._temp_p2 = 0.6 * total_steps_per_epoch / 4.4
# #         self._temp_p3 = 0.6 * total_steps_per_epoch / 4.4  # selector bật nhanh
# #         # [FIX-EASYFRAC4] Temperature cho phase 4 (giảm 55%→50% qua ~5 epoch)
# #         self._temp_p4 = 2.5 * total_steps_per_epoch / 4.4

# #     def step(self):
# #         """Gọi sau mỗi optimizer step."""
# #         self._global_step += 1

# #     @property
# #     def global_step(self):
# #         return self._global_step

# #     def get_alpha_hard(self) -> float:
# #         """
# #         α_hard: trọng số L_hard, tăng mượt qua sigmoid.
# #         Phase 1 (step < phase2_start): α = 0
# #         Phase 2 (step >= phase2_start): sigmoid 0 → 0.3, mid ở 15% phase
# #         Phase 3+: α = 0.3 cố định
# #         """
# #         s = self._global_step
# #         if s < self._phase2_start_step:
# #             return 0.0
# #         if s >= self._phase3_start_step:
# #             return 0.3

# #         mid  = self._phase2_start_step + (self._phase3_start_step
# #                                            - self._phase2_start_step) * 0.15
# #         x    = (s - mid) / max(self._temp_p2, 1.0)
# #         sig  = 1.0 / (1.0 + math.exp(-x))
# #         return 0.3 * sig

# #     def get_selector_weight(self) -> float:
# #         """
# #         selector_weight: trọng số L_sel, tăng mượt từ phase 3.
# #         Phase < 3: 0
# #         Phase 3: sigmoid 0 → 0.2
# #         Phase 4: 0.2 cố định
# #         """
# #         s = self._global_step
# #         if s < self._phase3_start_step:
# #             return 0.0
# #         if s >= self._phase4_start_step:
# #             return 0.2

# #         mid  = self._phase3_start_step + (self._phase4_start_step
# #                                            - self._phase3_start_step) / 2.0
# #         x    = (s - mid) / max(self._temp_p3, 1.0)
# #         sig  = 1.0 / (1.0 + math.exp(-x))
# #         return 0.2 * sig

# #     def get_easy_frac(self) -> float:
# #         """
# #         easy_frac: giảm mượt, không bao giờ xuống dưới 50%.
# #         Phase 1: 80%
# #         Phase 2: 80% → 60% theo sigmoid
# #         Phase 3: 60% → 55%
# #         [FIX-EASYFRAC4] Phase 4: 55% → 50% theo sigmoid (thay vì linear)
# #             Consistent với phương pháp của các phase khác.
# #             Midpoint: 5 epoch sau khi vào phase 4.
# #         """
# #         s = self._global_step
# #         if s < self._phase2_start_step:
# #             return 0.80

# #         if s < self._phase3_start_step:
# #             mid  = self._phase2_start_step + (self._phase3_start_step
# #                                                - self._phase2_start_step) / 2.0
# #             x    = (s - mid) / max(self._temp_p2, 1.0)
# #             sig  = 1.0 / (1.0 + math.exp(-x))
# #             return 0.80 - 0.20 * sig   # 80% → 60%

# #         if s < self._phase4_start_step:
# #             mid  = self._phase3_start_step + (self._phase4_start_step
# #                                                - self._phase3_start_step) / 2.0
# #             x    = (s - mid) / max(self._temp_p3, 1.0)
# #             sig  = 1.0 / (1.0 + math.exp(-x))
# #             return 0.60 - 0.05 * sig   # 60% → 55%

# #         # [FIX-EASYFRAC4] Phase 4: 55% → 50% qua sigmoid (không phải linear)
# #         phase4_mid = self._phase4_start_step + 5 * self.steps_per_epoch
# #         x   = (s - phase4_mid) / max(self._temp_p4, 1.0)
# #         sig = 1.0 / (1.0 + math.exp(-x))
# #         return max(0.50, 0.55 - 0.05 * sig)

# #     def is_selector_active(self) -> bool:
# #         """True từ phase 3 trở đi."""
# #         return self._global_step >= self._phase3_start_step

# #     def log_stats(self) -> str:
# #         return (f"α={self.get_alpha_hard():.3f}"
# #                 f" easy={self.get_easy_frac():.0%}"
# #                 f" sel={self.get_selector_weight():.3f}")


# # # Backward-compat wrappers (dùng trong checkpoint restore)
# # def get_alpha_hard(epoch: int) -> float:
# #     """Legacy: chỉ dùng khi không có SmoothScheduler."""
# #     if epoch <= PHASE_1_END: return 0.0
# #     if epoch <= PHASE_2_END:
# #         return 0.3 * (epoch - PHASE_1_END) / (PHASE_2_END - PHASE_1_END)
# #     return 0.3

# # def get_easy_frac(epoch: int) -> float:
# #     """Legacy fallback."""
# #     if epoch <= PHASE_1_END: return 0.80
# #     if epoch <= PHASE_2_END:
# #         p = (epoch - PHASE_1_END) / (PHASE_2_END - PHASE_1_END)
# #         return 0.80 - 0.20 * p
# #     if epoch <= PHASE_3_END: return 0.55
# #     return 0.50


# # # ─────────────────────────────────────────────────────────────────────────────
# # #  [FIX-STABILITY-3 + FIX-THRESH-1] Global hard threshold — tính 1 lần
# # #
# # #  hard_score_from_obs() chỉ phụ thuộc obs_traj (data cố định), KHÔNG phụ
# # #  thuộc model weights → threshold KHÔNG thay đổi theo training.
# # #  [FIX-THRESH-1] Dùng flag _threshold_computed riêng (xem main loop) thay vì
# # #  so sánh global_hard_threshold == 0.5 (fragile — nếu p70 thực tế = 0.5 thì
# # #  sẽ recompute mãi).
# # # ─────────────────────────────────────────────────────────────────────────────

# # @torch.no_grad()
# # def precompute_hard_threshold(loader, dev, n_batches: int = 50) -> float:
# #     """
# #     Tính p70 của hard_score trên subset của train set (1 lần duy nhất).
# #     """
# #     all_scores = []
# #     for i, batch in enumerate(loader):
# #         if i >= n_batches:
# #             break
# #         bl    = move(list(batch), dev)
# #         obs_t = bl[0]
# #         scores = hard_score_from_obs(obs_t[:, :, :2])  # [B]
# #         all_scores.append(scores.cpu())

# #     if not all_scores:
# #         return 0.7  # fallback

# #     all_scores_cat = torch.cat(all_scores)  # [N]
# #     threshold = float(torch.quantile(all_scores_cat, 0.70).item())
# #     n_hard = int((all_scores_cat >= threshold).sum().item())
# #     n_total = len(all_scores_cat)
# #     print(f"  [HardThreshold] global p70={threshold:.4f}"
# #           f"  n_hard={n_hard}/{n_total} ({100*n_hard/max(n_total,1):.0f}%)")
# #     return threshold


# # @torch.no_grad()
# # def classify_hard_easy_global(
# #     obs_traj_norm: torch.Tensor,
# #     global_threshold: float,
# # ) -> torch.Tensor:
# #     """Phân loại hard/easy dùng global threshold (thay vì per-batch p70)."""
# #     scores   = hard_score_from_obs(obs_traj_norm)  # [B]
# #     is_hard  = scores >= global_threshold
# #     return is_hard


# # def enforce_easy_frac(
# #     is_hard: torch.Tensor,
# #     easy_frac: float,
# # ) -> torch.Tensor:
# #     """
# #     Chỉ điều chỉnh is_hard mask, KHÔNG reindex batch_list.
# #     Nếu batch có quá nhiều hard → "demote" một số về easy (is_hard[i]=False).
# #     Batch data giữ nguyên 100%.
# #     """
# #     B        = is_hard.shape[0]
# #     max_hard = max(0, int(B * (1.0 - easy_frac)))
# #     n_hard   = int(is_hard.sum().item())

# #     if n_hard <= max_hard:
# #         return is_hard  # đã đúng tỉ lệ

# #     n_demote  = n_hard - max_hard
# #     hard_idx  = is_hard.nonzero(as_tuple=True)[0]
# #     perm      = torch.randperm(hard_idx.numel(), device=is_hard.device)
# #     to_demote = hard_idx[perm[:n_demote]]

# #     is_hard_new            = is_hard.clone()
# #     is_hard_new[to_demote] = False
# #     return is_hard_new


# # # ─────────────────────────────────────────────────────────────────────────────
# # #  Utilities
# # # ─────────────────────────────────────────────────────────────────────────────

# # def _unwrap(m):
# #     return m._orig_mod if hasattr(m, "_orig_mod") else m

# # def move(b, dev):
# #     out = list(b)
# #     for i, x in enumerate(out):
# #         if torch.is_tensor(x): out[i] = x.to(dev)
# #         elif isinstance(x, dict):
# #             out[i] = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in x.items()}
# #     return out


# # def _ntd(a):
# #     o = a.clone()
# #     o[..., 0] = (a[..., 0] * 50.0 + 1800.0) / 10.0
# #     o[..., 1] = (a[..., 1] * 50.0) / 10.0
# #     return o

# # def _hav(p1, p2):
# #     la1 = torch.deg2rad(p1[..., 1]); la2 = torch.deg2rad(p2[..., 1])
# #     dlat = torch.deg2rad(p2[..., 1] - p1[..., 1])
# #     dlon = torch.deg2rad(p2[..., 0] - p1[..., 0])
# #     a = (torch.sin(dlat/2).pow(2)
# #          + torch.cos(la1)*torch.cos(la2)*torch.sin(dlon/2).pow(2))
# #     return 2.0 * R_EARTH * torch.asin(a.clamp(1e-12, 1-1e-12).sqrt())

# # def _atecte(pd, gd):
# #     T = min(pd.shape[0], gd.shape[0])
# #     if T < 2:
# #         z = pd.new_zeros(1, pd.shape[1]); return z, z
# #     lo1=torch.deg2rad(gd[:T-1,:,0]); la1=torch.deg2rad(gd[:T-1,:,1])
# #     lo2=torch.deg2rad(gd[1:T, :,0]); la2=torch.deg2rad(gd[1:T, :,1])
# #     lo3=torch.deg2rad(pd[1:T, :,0]); la3=torch.deg2rad(pd[1:T, :,1])
# #     ya=torch.sin(lo2-lo1)*torch.cos(la2)
# #     xa=(torch.cos(la1)*torch.sin(la2)-torch.sin(la1)*torch.cos(la2)*torch.cos(lo2-lo1))
# #     be=torch.atan2(ya,xa)
# #     ye=torch.sin(lo3-lo2)*torch.cos(la3)
# #     xe=(torch.cos(la2)*torch.sin(la3)-torch.sin(la2)*torch.cos(la3)*torch.cos(lo3-lo2))
# #     bee=torch.atan2(ye,xe)
# #     tot=_hav(pd[1:T],gd[1:T]); ang=bee-be
# #     return tot*torch.cos(ang), tot*torch.sin(ang)


# # # ─────────────────────────────────────────────────────────────────────────────
# # #  [G-G] Split metrics — đo easy_ADE và hard_ADE riêng biệt
# # # ─────────────────────────────────────────────────────────────────────────────

# # class SplitAcc:
# #     """
# #     Accumulator đo easy_ADE và hard_ADE riêng biệt.
# #     Dùng để:
# #     - EMA guard: phát hiện easy forgetting sớm
# #     - Hard improvement: xác nhận hard loss đang hoạt động
# #     """
# #     def __init__(self):
# #         self.easy_d = []
# #         self.hard_d = []
# #         self.all_d  = []
# #         self.all_a  = []
# #         self.all_c  = []
# #         self.sd     = defaultdict(list)
# #         self._h     = {12: 1, 24: 3, 48: 7, 72: 11}

# #     def update(self, dist, is_hard=None, ate=None, cte=None):
# #         """
# #         Args:
# #             dist:    [T, B] distance per step
# #             is_hard: [B] bool (None → tất cả easy)
# #         """
# #         B = dist.shape[1]
# #         mean_dist = dist.mean(0)  # [B]
# #         self.all_d.extend(mean_dist.tolist())

# #         if is_hard is not None:
# #             easy_mask = ~is_hard
# #             if easy_mask.any():
# #                 self.easy_d.extend(mean_dist[easy_mask].tolist())
# #             if is_hard.any():
# #                 self.hard_d.extend(mean_dist[is_hard].tolist())
# #         else:
# #             self.easy_d.extend(mean_dist.tolist())

# #         for h, s in self._h.items():
# #             if s < dist.shape[0]:
# #                 self.sd[h].extend(dist[s].tolist())

# #         if ate is not None: self.all_a.extend(ate.abs().mean(0).tolist())
# #         if cte is not None: self.all_c.extend(cte.abs().mean(0).tolist())

# #     def compute(self):
# #         r = {
# #             "ADE":      float(np.mean(self.all_d))  if self.all_d  else float("nan"),
# #             "easy_ADE": float(np.mean(self.easy_d)) if self.easy_d else float("nan"),
# #             "hard_ADE": float(np.mean(self.hard_d)) if self.hard_d else float("nan"),
# #             "ATE":      float(np.mean(self.all_a))  if self.all_a  else float("nan"),
# #             "CTE":      float(np.mean(self.all_c))  if self.all_c  else float("nan"),
# #             "n":        len(self.all_d),
# #             "n_easy":   len(self.easy_d),
# #             "n_hard":   len(self.hard_d),
# #         }
# #         for h in self._h:
# #             v = self.sd.get(h, [])
# #             r[f"{h}h"] = float(np.mean(v)) if v else float("nan")
# #         return r


# # class Acc:
# #     """Simple accumulator (backward compat với evaluate())."""
# #     def __init__(self):
# #         self.d=[]; self.a=[]; self.c=[]; self.sd=defaultdict(list)
# #         self._h={12:1, 24:3, 48:7, 72:11}
# #     def update(self, dist, ate=None, cte=None):
# #         self.d.extend(dist.mean(0).tolist())
# #         for h,s in self._h.items():
# #             if s < dist.shape[0]: self.sd[h].extend(dist[s].tolist())
# #         if ate is not None: self.a.extend(ate.abs().mean(0).tolist())
# #         if cte is not None: self.c.extend(cte.abs().mean(0).tolist())
# #     def compute(self):
# #         r = {"ADE": float(np.mean(self.d)) if self.d else float("nan"),
# #              "ATE": float(np.mean(self.a)) if self.a else float("nan"),
# #              "CTE": float(np.mean(self.c)) if self.c else float("nan"),
# #              "n": len(self.d)}
# #         for h in self._h:
# #             v=self.sd.get(h,[]); r[f"{h}h"]=float(np.mean(v)) if v else float("nan")
# #         return r


# # def _score(r):
# #     ade=r.get("ADE",1e9); h72=r.get("72h",1e9)
# #     ate=r.get("ATE",1e9); cte=r.get("CTE",1e9)
# #     if not np.isfinite(ate): ate=ade*0.46
# #     if not np.isfinite(cte): cte=ade*0.53
# #     return 100.0*(0.05*(ade/136.)+0.10*(r.get("12h",ade)/50.)
# #                   +0.15*(r.get("24h",ade)/100.)+0.20*(r.get("48h",ade)/200.)
# #                   +0.25*(h72/300.)+0.13*(ate/80.)+0.12*(cte/94.))

# # def _beat(r):
# #     p=[]
# #     for k,t in [("ADE",172.68),("ATE",142.21),("CTE",42.04),
# #                 ("72h",321.39),("12h",65.42),("24h",104.67),("48h",205.10)]:
# #         v=r.get(k,1e9)
# #         if np.isfinite(v) and v < t: p.append(f"{k}:{v:.1f}")
# #     return "*** BEAT ST-TRANS: "+" ".join(p)+" ***" if p else ""

# # def _gap(r):
# #     out=[]
# #     for k,ref in [("ADE",172.68),("72h",321.39),("ATE",142.21),("CTE",42.04)]:
# #         v=r.get(k,float("nan"))
# #         if np.isfinite(v):
# #             out.append(f"{k}:{v:.0f}({'dn' if v<ref else 'up'}{abs(v-ref):.0f})")
# #     return " | ".join(out)


# # # ─────────────────────────────────────────────────────────────────────────────
# # #  [G-E] Easy_ADE evaluation cho EMA guard — dùng global threshold (FIX-4)
# # # ─────────────────────────────────────────────────────────────────────────────

# # @torch.no_grad()
# # def evaluate_split(model, loader, dev, tag="", ema=None, steps=10,
# #                    use_selector=False,
# #                    global_hard_threshold: Optional[float] = None) -> Dict:
# #     """
# #     Evaluate với split easy/hard ADE.

# #     Args:
# #         global_hard_threshold: float từ precompute_hard_threshold(), hoặc None
# #             → dùng per-batch p70 (phase 1 fallback, chỉ ảnh hưởng log)
# #     """
# #     bk = None
# #     if ema:
# #         try: bk = ema.apply_to(model)
# #         except: pass
# #     model.eval()
# #     acc = SplitAcc()
# #     t0  = time.perf_counter()

# #     for b in loader:
# #         bl = move(list(b), dev)
# #         obs_t = bl[0]

# #         if global_hard_threshold is not None:
# #             is_hard = classify_hard_easy_global(
# #                 obs_t[:, :, :2], global_hard_threshold).to(dev)
# #         else:
# #             is_hard = classify_hard_easy(obs_t[:, :, :2]).to(dev)

# #         result = model.sample(bl, ddim_steps=steps, use_selector=use_selector)
# #         p = result[0] if isinstance(result, (tuple, list)) else result
# #         g = bl[1]; T = min(p.shape[0], g.shape[0])
# #         pd = _ntd(p[:T]); gd = _ntd(g[:T])
# #         dist = _hav(pd, gd)
# #         at, ct = _atecte(pd, gd)
# #         acc.update(dist, is_hard=is_hard, ate=at, cte=ct)

# #     # [FIX-EMA-restore] bk luôn là dict non-empty nếu apply_to thành công
# #     # (shadow luôn có ít nhất 1 floating-point param) — nhưng dùng
# #     # `is not None` cho rõ ràng & an toàn tuyệt đối.
# #     if bk is not None:
# #         try: ema.restore(model, bk)
# #         except: pass

# #     r  = acc.compute()
# #     el = time.perf_counter() - t0

# #     def _v(k): return r.get(k, float("nan"))
# #     def _m(v, t): return "ok" if np.isfinite(v) and v < t else "no"

# #     thr_str = f"{global_hard_threshold:.4f}" if global_hard_threshold is not None else "per-batch"
# #     print(f"\n{'='*70}")
# #     print(f"  [{tag}  {el:.0f}s  threshold={thr_str}]")
# #     print(f"  ADE={_v('ADE'):.1f}[{_m(_v('ADE'),172.68)}]"
# #           f"  easy_ADE={_v('easy_ADE'):.1f}  hard_ADE={_v('hard_ADE'):.1f}")
# #     print(f"  12h={_v('12h'):.0f}  24h={_v('24h'):.0f}"
# #           f"  48h={_v('48h'):.0f}  72h={_v('72h'):.0f}[{_m(_v('72h'),321.39)}]")
# #     print(f"  ATE={_v('ATE'):.1f}[{_m(_v('ATE'),142.21)}]"
# #           f"  CTE={_v('CTE'):.1f}[{_m(_v('CTE'),42.04)}]")
# #     print(f"  n_easy={r.get('n_easy',0)}  n_hard={r.get('n_hard',0)}")
# #     print(f"  vs ST-Trans: {_gap(r)}")
# #     bt = _beat(r)
# #     if bt: print(f"  {bt}")
# #     print(f"  Score={_score(r):.2f}")
# #     print(f"{'='*70}\n")
# #     return r


# # @torch.no_grad()
# # def evaluate(model, loader, dev, tag="", ema=None, steps=20,
# #              use_selector=False,
# #              global_hard_threshold: Optional[float] = None) -> Dict:
# #     """Full evaluation (backward compat)."""
# #     return evaluate_split(model, loader, dev, tag=tag, ema=ema,
# #                           steps=steps, use_selector=use_selector,
# #                           global_hard_threshold=global_hard_threshold)


# # # ─────────────────────────────────────────────────────────────────────────────
# # #  [G-F] Diversity check
# # # ─────────────────────────────────────────────────────────────────────────────

# # @torch.no_grad()
# # def check_diversity(model, loader, dev, n_batches: int = 5,
# #                     n_ensemble: int = 20, ddim_steps: int = 5) -> float:
# #     """
# #     Đo diversity score trên một subset của val set.
# #     Nếu < 50 km → cảnh báo R1 (mode collapse).
# #     """
# #     model.eval()
# #     all_divs = []

# #     for i, b in enumerate(loader):
# #         if i >= n_batches:
# #             break
# #         bl = move(list(b), dev)

# #         _, _, all_c = model.sample(bl, num_ensemble=n_ensemble,
# #                                     ddim_steps=ddim_steps)
# #         N_total = all_c.shape[0]
# #         cands = [all_c[k] for k in range(N_total)]

# #         div = compute_diversity_score(cands)
# #         all_divs.append(div)

# #     return float(np.mean(all_divs)) if all_divs else 0.0


# # # ─────────────────────────────────────────────────────────────────────────────
# # #  [FIX-CKPT-1] Checkpoint — lưu smooth_sched_step + global_hard_threshold
# # #  trong MỌI checkpoint (trước chỉ có trong periodic ckpt's "extra")
# # # ─────────────────────────────────────────────────────────────────────────────

# # def _save_ckpt(path, ep, model, opt, sched, saver, tl, vl,
# #                smooth_sched_step: int = 0,
# #                global_hard_threshold: Optional[float] = None,
# #                extra=None):
# #     m = _unwrap(model)
# #     ema = getattr(m, "_ema", None)
# #     esd = None
# #     if ema and hasattr(ema, "shadow"):
# #         try: esd = {k: v.cpu().clone() for k, v in ema.shadow.items()}
# #         except: pass
# #     d = {
# #         "epoch": ep,
# #         "model_state_dict": m.state_dict(),
# #         "optimizer_state":  opt.state_dict(),
# #         "scheduler_state":  sched.state_dict(),
# #         "ema_shadow":       esd,
# #         "best_score":       saver.bs,
# #         "best_ade":         saver.ba,
# #         "best_72h":         saver.b7,
# #         "best_ate":         saver.bat,
# #         "best_cte":         saver.bc,
# #         "best_easy_ade":    saver.best_easy_ade,
# #         "train_loss":       tl,
# #         "val_loss":         vl,
# #         # [FIX-CKPT-1] Luôn lưu để resume chính xác smooth_sched + threshold
# #         "smooth_sched_step":     smooth_sched_step,
# #         "global_hard_threshold": global_hard_threshold,
# #         "version":          "v59strategy_fixed_v2",
# #     }
# #     if extra:
# #         d.update(extra)
# #     torch.save(d, path)


# # # ─────────────────────────────────────────────────────────────────────────────
# # #  Saver
# # # ─────────────────────────────────────────────────────────────────────────────

# # class Saver:
# #     def __init__(self, patience=40, min_ep=35, enable_stop=True):
# #         self.patience     = patience
# #         self.min_ep       = min_ep
# #         self.enable_stop  = enable_stop
# #         self.cnt          = 0
# #         self.stop         = False
# #         self.bs           = float("inf")  # best score
# #         self.ba           = float("inf")  # best ADE
# #         self.b7           = float("inf")  # best 72h
# #         self.bat          = float("inf")  # best ATE
# #         self.bc           = float("inf")  # best CTE
# #         self.best_easy_ade = float("inf") # [G-E] best easy ADE (EMA guard)

# #     def update(self, r, model, out_dir, ep, opt, sched, tl, vl, tag="",
# #                smooth_sched_step: int = 0,
# #                global_hard_threshold: Optional[float] = None):
# #         sc  = _score(r)
# #         ade = r.get("ADE",     1e9)
# #         h72 = r.get("72h",     1e9)
# #         ate = r.get("ATE",     1e9)
# #         cte = r.get("CTE",     1e9)
# #         easy_ade = r.get("easy_ADE", ade)  # fallback to ADE if no split

# #         any_improved = False

# #         for v, attr, fn in [
# #             (ade,      "ba",  f"best_ade_{tag}.pth"),
# #             (h72,      "b7",  f"best_72h_{tag}.pth"),
# #             (ate,      "bat", f"best_ate_{tag}.pth"),
# #             (cte,      "bc",  f"best_cte_{tag}.pth"),
# #         ]:
# #             if v < getattr(self, attr):
# #                 setattr(self, attr, v)
# #                 _save_ckpt(os.path.join(out_dir, fn), ep, model, opt,
# #                            sched, self, tl, vl,
# #                            smooth_sched_step=smooth_sched_step,
# #                            global_hard_threshold=global_hard_threshold)
# #                 any_improved = True

# #         if sc < self.bs:
# #             self.bs = sc
# #             any_improved = True
# #             _save_ckpt(os.path.join(out_dir, f"best_{tag or 'composite'}.pth"),
# #                        ep, model, opt, sched, self, tl, vl,
# #                        smooth_sched_step=smooth_sched_step,
# #                        global_hard_threshold=global_hard_threshold)
# #             print(f"  [BEST] {tag} ep={ep} score={sc:.2f}"
# #                   f"  ADE={ade:.1f}  72h={h72:.0f}"
# #                   f"  ATE={ate:.1f}  CTE={cte:.1f}")

# #         # [G-E] Update best easy_ADE
# #         if np.isfinite(easy_ade) and easy_ade < self.best_easy_ade:
# #             self.best_easy_ade = easy_ade

# #         if self.enable_stop:
# #             if any_improved: self.cnt = 0
# #             else:
# #                 self.cnt += 1
# #                 print(f"  No improve {self.cnt}/{self.patience}"
# #                       f"  best={self.bs:.2f}  cur={sc:.2f}")
# #             if ep >= self.min_ep and self.cnt >= self.patience:
# #                 self.stop = True


# # def _mksub(ds, n, bs, cf):
# #     idx = random.Random(42).sample(range(len(ds)), min(n, len(ds)))
# #     return DataLoader(Subset(ds, idx), batch_size=bs, shuffle=False,
# #                       collate_fn=cf, num_workers=0, drop_last=False)


# # # ─────────────────────────────────────────────────────────────────────────────
# # #  Args
# # # ─────────────────────────────────────────────────────────────────────────────

# # def get_args():
# #     p = argparse.ArgumentParser(
# #         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# #     p.add_argument("--dataset_root",       default="TCND_vn")
# #     p.add_argument("--obs_len",            default=8,    type=int)
# #     p.add_argument("--pred_len",           default=12,   type=int)
# #     p.add_argument("--batch_size",         default=32,   type=int)
# #     p.add_argument("--num_epochs",         default=100,  type=int)
# #     p.add_argument("--learning_rate",      default=1e-4, type=float)
# #     p.add_argument("--weight_decay",       default=1e-3, type=float)
# #     p.add_argument("--warmup_epochs",      default=3,    type=int)
# #     p.add_argument("--grad_clip",          default=1.0,  type=float)
# #     p.add_argument("--use_amp",            action="store_true")
# #     p.add_argument("--num_workers",        default=2,    type=int)
# #     p.add_argument("--sigma_min",          default=0.02, type=float)
# #     p.add_argument("--use_ot",             default=True, action="store_true")
# #     p.add_argument("--no_ot",              dest="use_ot", action="store_false")
# #     p.add_argument("--cfg_guidance_scale", default=1.5,  type=float)
# #     p.add_argument("--n_ensemble",         default=50,   type=int)
# #     p.add_argument("--use_ema",            default=True, action="store_true")
# #     p.add_argument("--no_ema",             dest="use_ema", action="store_false")
# #     p.add_argument("--ema_decay",          default=0.995, type=float)
# #     p.add_argument("--patience",           default=40,   type=int)
# #     p.add_argument("--min_ep",             default=35,   type=int)
# #     p.add_argument("--val_freq",           default=3,    type=int)
# #     p.add_argument("--val_subset_size",    default=500,  type=int)
# #     p.add_argument("--fast_ddim",          default=10,   type=int)
# #     p.add_argument("--full_ddim",          default=20,   type=int)
# #     p.add_argument("--output_dir",         default="runs/v59strategy")
# #     p.add_argument("--gpu_num",            default="0")
# #     p.add_argument("--delim",              default=" ")
# #     p.add_argument("--skip",               default=1,    type=int)
# #     p.add_argument("--min_ped",            default=1,    type=int)
# #     p.add_argument("--threshold",          default=0.002, type=float)
# #     p.add_argument("--other_modal",        default="gph")
# #     p.add_argument("--test_year",          default=None, type=int)
# #     p.add_argument("--resume",             default=None)
# #     p.add_argument("--resume_epoch",       default=None, type=int)
# #     p.add_argument("--eval_test_after_train", default=True, action="store_true")
# #     # GradNorm
# #     p.add_argument("--gradnorm_alpha",     default=1.5,  type=float,
# #                    help="GradNorm restoring force (giữ cho backward compat)")
# #     p.add_argument("--gradnorm_lr",        default=1e-3, type=float,
# #                    help="GradNorm lambda learning rate")
# #     # EMA guard
# #     p.add_argument("--ema_guard_threshold", default=20.0, type=float,
# #                    help="smooth_easy_ADE tăng bao nhiêu km thì rollback (km)")
# #     # Diversity check
# #     p.add_argument("--diversity_threshold", default=50.0, type=float,
# #                    help="diversity_score < threshold → cảnh báo R1")
# #     return p.parse_args()


# # # ─────────────────────────────────────────────────────────────────────────────
# # #  Main
# # # ─────────────────────────────────────────────────────────────────────────────

# # def main(args):
# #     if torch.cuda.is_available():
# #         os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
# #     dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #     os.makedirs(args.output_dir, exist_ok=True)

# #     print("=" * 72)
# #     print("  TC-FlowMatching v59-Strategy [FIXED-v2]")
# #     print("  [FIX-GN-1] GradNorm: equal-contribution scheme (có equilibrium)")
# #     print("  [FIX-GN-2] phase_reset: KHÔNG reset λ (chỉ reset optimizer)")
# #     print("  [FIX-EMA-2] EMA guard pullback: chỉ khi α<0.25 (tránh kẹt)")
# #     print("  [FIX-PERF-1] Xóa dead computation (named_parameters mỗi step)")
# #     print("  [FIX-CKPT-1] Lưu smooth_sched_step + threshold mọi checkpoint")
# #     print("  [FIX-DIV-1] Auto x1.5 noise nếu diversity < threshold")
# #     print("  [G-A] Easy/Hard: global threshold (tính 1 lần)")
# #     print("  [G-C] easy_frac ≥ 50% enforcement mỗi batch")
# #     print("  [G-D] α warm-up: sigmoid, mid=15% phase (FIX-ALPHA-TIMING)")
# #     print("  [G-E] EMA guard: EMA-smoothed easy_ADE, threshold 20km")
# #     print("  [G-F] Diversity check sau phase 1")
# #     print("  [G-G] Split monitoring: easy_ADE + hard_ADE")
# #     print("  [G-H] Phase 4: freeze encoder")
# #     print(f"  Phases: 1[0-{PHASE_1_END}] 2[{PHASE_1_END+1}-{PHASE_2_END}]"
# #           f" 3[{PHASE_2_END+1}-{PHASE_3_END}] 4[{PHASE_3_END+1}+]")
# #     print(f"  Target: ADE<{TARGETS['ADE']} 72h<{TARGETS['72h']}"
# #           f" ATE<{TARGETS['ATE']} CTE<{TARGETS['CTE']}")
# #     print("=" * 72)

# #     # Data
# #     trd, trl = data_loader(args, {"root": args.dataset_root, "type": "train"}, test=False)
# #     vd,  vl  = data_loader(args, {"root": args.dataset_root, "type": "val"},   test=True)
# #     from Model.data.trajectoriesWithMe_unet_training import seq_collate
# #     vsub = _mksub(vd, args.val_subset_size, args.batch_size, seq_collate)
# #     print(f"  train:{len(trd)} seqs  val:{len(vd)} seqs")

# #     # Model
# #     model = TCFlowMatching(
# #         pred_len=args.pred_len, obs_len=args.obs_len,
# #         sigma_min=args.sigma_min, use_ema=args.use_ema,
# #         ema_decay=args.ema_decay, use_ate_ot=args.use_ot,
# #         cfg_guidance_scale=args.cfg_guidance_scale,
# #     ).to(dev)
# #     model.init_ema()

# #     n_total = sum(p.numel() for p in model.parameters() if p.requires_grad)

# #     # BUG-8: backbone vs learned weights (step_weights, selector small params)
# #     backbone_params = [p for p in model.parameters()
# #                        if p.requires_grad and p.numel() > 100]
# #     step_w_params   = [p for p in model.parameters()
# #                        if p.requires_grad and p.numel() <= 12]

# #     print(f"  params total: {n_total:,}")
# #     print(f"  backbone (clipped at {args.grad_clip}): "
# #           f"{sum(p.numel() for p in backbone_params):,}")
# #     print(f"  step_weights (NOT clipped): "
# #           f"{sum(p.numel() for p in step_w_params):,}")

# #     # Optimizer & scheduler
# #     opt    = optim.AdamW(model.parameters(),
# #                           lr=args.learning_rate,
# #                           weight_decay=args.weight_decay)
# #     scaler = GradScaler("cuda", enabled=args.use_amp)
# #     nstep  = len(trl)
# #     total  = nstep * args.num_epochs
# #     wstp   = nstep * args.warmup_epochs
# #     sched  = get_cosine_schedule_with_warmup(opt, wstp, total, min_lr=1e-6)

# #     # [G-A] GradNorm manager — equal-contribution scheme (FIX-GN-1)
# #     gradnorm = GradNormManager(
# #         terms=GRADNORM_TERMS,
# #         alpha_gn=args.gradnorm_alpha,
# #         lr=args.gradnorm_lr,
# #         device=dev,
# #     )

# #     # SmoothScheduler — α theo batch, không phải epoch
# #     smooth_sched = SmoothScheduler(total_steps_per_epoch=nstep)

# #     # Savers
# #     fast_saver = Saver(patience=args.patience, min_ep=args.min_ep,
# #                         enable_stop=False)
# #     full_saver = Saver(patience=args.patience, min_ep=args.min_ep,
# #                         enable_stop=True)

# #     # [FIX-THRESH-1] global threshold + flag riêng (thay vì == 0.5)
# #     global_hard_threshold: Optional[float] = None
# #     _threshold_computed = False

# #     # Resume
# #     start = 0
# #     if args.resume and os.path.exists(args.resume):
# #         print(f"  Loading: {args.resume}")
# #         ck = torch.load(args.resume, map_location=dev)
# #         m  = _unwrap(model)
# #         ms, _ = m.load_state_dict(ck["model_state_dict"], strict=False)
# #         if ms: print(f"  Missing: {len(ms)}")
# #         ema = getattr(m, "_ema", None)
# #         if ema and ck.get("ema_shadow"):
# #             for k, v in ck["ema_shadow"].items():
# #                 if k in ema.shadow: ema.shadow[k].copy_(v.to(dev))
# #         try: opt.load_state_dict(ck["optimizer_state"])
# #         except Exception as e: print(f"  Opt: {e}")
# #         try: sched.load_state_dict(ck["scheduler_state"])
# #         except:
# #             for _ in range(ck.get("epoch", 0) * nstep): sched.step()
# #         for a, attr in [("best_score","bs"), ("best_ade","ba"), ("best_72h","b7"),
# #                         ("best_ate","bat"), ("best_cte","bc"),
# #                         ("best_easy_ade","best_easy_ade")]:
# #             if a in ck:
# #                 setattr(full_saver, attr, ck[a])
# #                 setattr(fast_saver, attr, ck[a])
# #         start = args.resume_epoch or ck.get("epoch", 0) + 1
# #         print(f"  Resume ep={start}  best_score={full_saver.bs:.2f}"
# #               f"  best_easy_ADE={full_saver.best_easy_ade:.1f}")

# #         # [FIX-CKPT-1] Restore smooth_sched step + threshold từ checkpoint
# #         if "smooth_sched_step" in ck and ck["smooth_sched_step"] is not None:
# #             smooth_sched._global_step = ck["smooth_sched_step"]
# #             print(f"  Restored smooth_sched_step={ck['smooth_sched_step']}")
# #         else:
# #             smooth_sched._global_step = start * nstep
# #             print(f"  Estimated smooth_sched_step={start * nstep} from epoch")

# #         if "global_hard_threshold" in ck and ck["global_hard_threshold"] is not None:
# #             global_hard_threshold = ck["global_hard_threshold"]
# #             _threshold_computed = True
# #             print(f"  Restored global_hard_threshold={global_hard_threshold:.4f}")

# #     try:
# #         model = torch.compile(model, mode="reduce-overhead")
# #         print("  torch.compile: ok")
# #     except:
# #         pass

# #     # ── Phase 4 encoder list (dùng khi freeze) ────────────────────────────
# #     def _get_encoder_params(m):
# #         """Trả về params của encoder để freeze ở phase 4."""
# #         enc = _unwrap(m)
# #         params = []
# #         for mod_name in ["net.spatial_enc", "net.enc_1d", "net.env_enc",
# #                           "net.bottleneck_pool", "net.bottleneck_proj",
# #                           "net.decoder_proj"]:
# #             parts = mod_name.split(".")
# #             mod = enc
# #             for p in parts:
# #                 if hasattr(mod, p):
# #                     mod = getattr(mod, p)
# #                 else:
# #                     mod = None
# #                     break
# #             if mod is not None:
# #                 params.extend(list(mod.parameters()))
# #         return params

# #     _phase4_frozen = False

# #     def _maybe_freeze_encoder(ep, m):
# #         nonlocal _phase4_frozen
# #         if get_phase(ep) == 4 and not _phase4_frozen:
# #             enc_params = _get_encoder_params(m)
# #             for p in enc_params:
# #                 p.requires_grad_(False)
# #             _phase4_frozen = True
# #             print(f"  [Phase 4] Froze encoder params"
# #                   f" (n={sum(1 for p in enc_params)})")
# #             for g in opt.param_groups:
# #                 g["lr"] = g["lr"] * 0.1
# #             print(f"  [Phase 4] LR reduced to {opt.param_groups[0]['lr']:.2e}")

# #     # ── EMA guard state ────────────────────────────────────────────────────
# #     best_easy_ade_ema      = float("inf")
# #     ema_rollback_count     = 0
# #     _prev_phase            = get_phase(start) if start > 0 else 1
# #     # EMA của easy_ADE — robust hơn raw value với variance fast_eval ±20-40km
# #     _ema_state  = {"smooth": float("inf")}
# #     EMA_ALPHA   = 0.35   # weight cho current value (bias về recent)
# #     EMA_THRESH  = args.ema_guard_threshold   # km threshold để trigger rollback

# #     # [FIX-DIV-1] flag để chỉ áp dụng diversity-noise-boost 1 lần
# #     _diversity_boosted = False

# #     print(f"  Training: {nstep} steps/ep, start ep={start}")
# #     print("=" * 72)
# #     ts = time.perf_counter()

# #     for ep in range(start, args.num_epochs):
# #         phase = get_phase(ep)

# #         # [FIX-GN-2] GradNorm phase transition — KHÔNG reset λ
# #         if phase != _prev_phase:
# #             gradnorm.phase_reset(phase)
# #             _prev_phase = phase
# #             print(f"  [Phase {phase}] Chuyển phase tại epoch {ep}")

# #             # Reset best_easy_ade_ema và smooth EMA khi chuyển phase
# #             # (baseline easy_ADE giữa các phase có thể khác nhau tự nhiên,
# #             #  không nên coi sự khác biệt này là "regression")
# #             best_easy_ade_ema      = float("inf")
# #             ema_rollback_count     = 0
# #             _ema_state["smooth"]   = float("inf")
# #             print(f"  [EMA Guard] Reset best/smooth EMA khi chuyển phase")

# #         # [FIX-THRESH-1] Chỉ compute hard threshold 1 lần khi vào phase 2.
# #         if phase >= 2 and not _threshold_computed:
# #             global_hard_threshold = precompute_hard_threshold(trl, dev, n_batches=50)
# #             _threshold_computed = True
# #             print(f"  [FIX-THRESH-1] global_hard_threshold={global_hard_threshold:.4f}"
# #                   f" — computed once, reused for all epochs")

# #         # [G-H] Phase 4: freeze encoder
# #         _maybe_freeze_encoder(ep, model)

# #         model.train()
# #         sl = 0.0
# #         t0 = time.perf_counter()

# #         # Lấy lambda hiện tại từ GradNorm
# #         lambda_dict = gradnorm.get_lambda_dict()

# #         for i, batch in enumerate(trl):
# #             bl    = move(list(batch), dev)
# #             obs_t = bl[0]

# #             # α, easy_frac từ SmoothScheduler (per-batch)
# #             alpha_hard    = smooth_sched.get_alpha_hard()
# #             easy_frac     = smooth_sched.get_easy_frac()
# #             sel_weight    = smooth_sched.get_selector_weight()
# #             train_selector = smooth_sched.is_selector_active() and (sel_weight > 0.01)

# #             # Dùng global threshold (phase>=2) hoặc zeros (phase 1)
# #             if phase >= 2 and global_hard_threshold is not None:
# #                 with torch.no_grad():
# #                     is_hard = classify_hard_easy_global(
# #                         obs_t[:, :, :2], global_hard_threshold).to(dev)
# #                 is_hard = enforce_easy_frac(is_hard, easy_frac)
# #             else:
# #                 # Phase 1: alpha_hard=0 → hard loss = 0, không cần phân loại
# #                 is_hard = torch.zeros(
# #                     obs_t.shape[1], dtype=torch.bool, device=dev)

# #             with autocast(device_type="cuda", enabled=args.use_amp):
# #                 bd = model.get_loss_breakdown(
# #                     bl,
# #                     epoch=ep,
# #                     alpha_hard=alpha_hard,
# #                     is_hard=is_hard,
# #                     train_selector=train_selector,
# #                     lambda_dict=lambda_dict,
# #                 )

# #             opt.zero_grad()
# #             scaler.scale(bd["total"]).backward()
# #             scaler.unscale_(opt)

# #             # [FIX-GN-1 + FIX-PERF-1] GradNorm: equal-contribution scheme.
# #             # Chỉ cần loss VALUES (raw, detached) — không cần gradient norm
# #             # của shared params (xóa hoàn toàn block named_parameters() cũ,
# #             # vốn lặp qua TOÀN BỘ params mỗi step nhưng kết quả bị ignore).
# #             try:
# #                 loss_vals_float = {}
# #                 for t in GRADNORM_TERMS:
# #                     val = bd.get(t)
# #                     if val is None:
# #                         continue
# #                     fval = float(val.item()) if torch.is_tensor(val) else float(val)
# #                     if np.isfinite(fval) and fval > 0:
# #                         loss_vals_float[t] = fval

# #                 if loss_vals_float:
# #                     gradnorm.update(loss_vals_float)
# #                     lambda_dict = gradnorm.get_lambda_dict()
# #             except Exception:
# #                 pass  # GradNorm không critical

# #             # BUG-8: clip chỉ backbone
# #             torch.nn.utils.clip_grad_norm_(backbone_params, args.grad_clip)

# #             scaler.step(opt)
# #             scaler.update()
# #             # BUG-7: unconditional
# #             try: sched.step()
# #             except: pass
# #             model.ema_update()

# #             smooth_sched.step()

# #             sl += bd["total"].item()

# #             if i % 20 == 0:
# #                 lr     = opt.param_groups[0]["lr"]
# #                 tot    = bd["total"].item()
# #                 warn   = " [!>10]" if tot > 10.0 else ""
# #                 sw_st  = _unwrap(model).step_weights.stats()
# #                 gn_log = gradnorm.log_stats()
# #                 n_hard_batch = int(is_hard.sum().item())

# #                 lam_str = " ".join(
# #                     f"λ_{t.replace('l_','').replace('_reg','')}="
# #                     f"{gn_log.get(f'λ_{t}', 1.0):.2f}"
# #                     for t in GRADNORM_TERMS
# #                 )

# #                 print(
# #                     f"  [{ep:>3}|Ph{phase}][{i:>3}/{nstep}]"
# #                     f"  tot={tot:.3f}{warn}"
# #                     f"  fm={bd.get('l_fm',0):.4f}"
# #                     f"  dpe={bd.get('dpe',0):.4f}"
# #                     f"  vel={bd.get('vel_reg',0):.4f}"
# #                     f"  hard={bd.get('l_hard_total',0):.4f}"
# #                     f"  {smooth_sched.log_stats()}"
# #                     f"  n_h={n_hard_batch}/{obs_t.shape[1]}"
# #                     f"  sw72={sw_st['sw_72h']:.2f}"
# #                     f"  gn_d={gn_log.get('gn_dense',0)}"
# #                     f"  {lam_str}"
# #                     f"  lr={lr:.2e}"
# #                 )

# #         avt  = sl / nstep
# #         t_ep = time.perf_counter() - t0

# #         # BUG-5: val với epoch=-1 → skip augmentation
# #         model.eval()
# #         vls = 0.0
# #         with torch.no_grad():
# #             for batch in vl:
# #                 bv = move(list(batch), dev)
# #                 with autocast(device_type="cuda", enabled=args.use_amp):
# #                     vls += model.get_loss(bv, epoch=-1).item()
# #         avv = vls / len(vl)

# #         lr_cur   = opt.param_groups[0]["lr"]
# #         gn_stats = gradnorm.log_stats()
# #         use_sel_eval  = smooth_sched.is_selector_active()
# #         eval_threshold = global_hard_threshold if phase >= 2 else None

# #         lam_epoch = " ".join(
# #             f"λ_{t.replace('l_','').replace('_reg','')}="
# #             f"{gn_stats.get(f'λ_{t}', 1.0):.3f}"
# #             for t in GRADNORM_TERMS
# #         )

# #         print(f"  Epoch {ep:>3}|Ph{phase}"
# #               f" | train={avt:.4f} val={avv:.4f}"
# #               f" | {smooth_sched.log_stats()}"
# #               f" | {lam_epoch}"
# #               f" | lr={lr_cur:.2e} | {t_ep:.0f}s")

# #         # [G-G] Fast eval với split monitoring (dùng fast ddim)
# #         rf = evaluate_split(model, vsub, dev,
# #                              tag=f"FAST ep{ep}",
# #                              steps=args.fast_ddim,
# #                              use_selector=use_sel_eval,
# #                              global_hard_threshold=eval_threshold)
# #         fast_saver.update(rf, model, args.output_dir, ep, opt, sched, avt, avv,
# #                           tag="fast",
# #                           smooth_sched_step=smooth_sched.global_step,
# #                           global_hard_threshold=global_hard_threshold)

# #         # ── [G-E + FIX-EMA-2] EMA guard: kiểm tra easy_ADE sau mỗi epoch ──
# #         #
# #         # [FIX-EMA-2] Vấn đề cũ: pullback_steps=nstep*2, rollback dày đặc
# #         # trong vùng α-ramp (ramp chỉ rộng ~35 step quanh điểm mid) → mỗi
# #         # rollback trong vùng ramp kéo step về TRƯỚC vùng ramp → α=0 lại
# #         # → α=0.000 suốt cả phase 2 (xem log gốc, 30 epoch toàn α=0).
# #         #
# #         # Fix:
# #         #   - pullback_steps = nstep // 2 (giảm 1 nửa so với nstep*2)
# #         #   - CHỈ pullback schedule nếu α hiện tại < ALPHA_RAMP_DONE_THRESHOLD
# #         #     (còn trong vùng ramp). Nếu α đã ramp xong (>=0.25), rollback
# #         #     chỉ restore weights, KHÔNG pullback schedule — tránh kẹt
# #         #     vĩnh viễn do rollback lặp lại sau khi đã vượt vùng ramp.
# #         easy_ade_curr = rf.get("easy_ADE", rf.get("ADE", float("inf")))
# #         if np.isfinite(easy_ade_curr):
# #             # Update EMA
# #             if _ema_state["smooth"] == float("inf"):
# #                 _ema_state["smooth"] = easy_ade_curr
# #             else:
# #                 _ema_state["smooth"] = ((1 - EMA_ALPHA) * _ema_state["smooth"]
# #                                         + EMA_ALPHA * easy_ade_curr)
# #             smooth_val = _ema_state["smooth"]

# #             if smooth_val < best_easy_ade_ema:
# #                 best_easy_ade_ema = smooth_val
# #                 print(f"  [EMA Guard] New best smooth_easy_ADE={smooth_val:.1f}"
# #                       f" (raw={easy_ade_curr:.1f})")

# #             elif (smooth_val - best_easy_ade_ema) > EMA_THRESH:
# #                 ema_rollback_count += 1
# #                 print(f"\n  [EMA GUARD] smooth_easy_ADE tăng bền vững "
# #                       f"{smooth_val - best_easy_ade_ema:.1f} km"
# #                       f" (best={best_easy_ade_ema:.1f} smooth={smooth_val:.1f}"
# #                       f" raw={easy_ade_curr:.1f})")
# #                 print(f"  [EMA GUARD] Rollback #{ema_rollback_count}")

# #                 # Restore weights từ EMA
# #                 m_raw   = _unwrap(model)
# #                 ema_obj = getattr(m_raw, "_ema", None)
# #                 if ema_obj:
# #                     try:
# #                         bk = ema_obj.apply_to(model)
# #                         _save_ckpt(
# #                             os.path.join(args.output_dir,
# #                                          f"ema_guard_rollback_ep{ep}.pth"),
# #                             ep, model, opt, sched, full_saver, avt, avv,
# #                             smooth_sched_step=smooth_sched.global_step,
# #                             global_hard_threshold=global_hard_threshold,
# #                             extra={"ema_rollback": True,
# #                                    "easy_ade_trigger": easy_ade_curr},
# #                         )
# #                         ema_obj.restore(model, bk)
# #                         print(f"  [EMA GUARD] Saved rollback ckpt, "
# #                               f"restored current weights for continued training")
# #                     except Exception as e:
# #                         print(f"  [EMA GUARD] Apply/restore failed: {e}")

# #                 # [FIX-EMA-2] Chỉ pullback schedule nếu α còn trong vùng ramp
# #                 cur_alpha = smooth_sched.get_alpha_hard()
# #                 if cur_alpha < ALPHA_RAMP_DONE_THRESHOLD:
# #                     pullback_steps = nstep // 2  # giảm so với nstep*2
# #                     smooth_sched._global_step = max(
# #                         smooth_sched._phase2_start_step,
# #                         smooth_sched._global_step - pullback_steps
# #                     )
# #                     print(f"  [EMA GUARD] α còn trong vùng ramp ({cur_alpha:.3f}"
# #                           f" < {ALPHA_RAMP_DONE_THRESHOLD}) → pullback"
# #                           f" {pullback_steps} step. α mới="
# #                           f"{smooth_sched.get_alpha_hard():.3f}")
# #                 else:
# #                     print(f"  [EMA GUARD] α đã ramp xong ({cur_alpha:.3f}"
# #                           f" >= {ALPHA_RAMP_DONE_THRESHOLD}) → KHÔNG pullback"
# #                           f" schedule, chỉ restore weights")

# #         # [G-F + FIX-DIV-1] Diversity check cuối phase 1
# #         div_score = None
# #         if ep == PHASE_1_END:
# #             print(f"\n  {'='*50}")
# #             print(f"  [CHECKPOINT] Cuối phase 1 — kiểm tra diversity...")
# #             div_score = check_diversity(model, vsub, dev)
# #             print(f"  Diversity score: {div_score:.1f} km")
# #             if div_score < args.diversity_threshold:
# #                 print(f"  [CẢNH BÁO R1] diversity={div_score:.1f} km"
# #                       f" < {args.diversity_threshold} km")
# #                 print(f"  → FM candidates không đủ đa dạng → selector sẽ kém")

# #                 # [FIX-DIV-1] Tự động xử lý theo strategy doc R1 fix #2:
# #                 # "Tăng noise std ×1.5". Bọc try/except + hasattr để KHÔNG
# #                 # làm crash training nếu tên attribute khác với kỳ vọng
# #                 # (model file có thể không có sigma_min/ctx_noise_scale là
# #                 # attribute trực tiếp trên TCFlowMatching).
# #                 if not _diversity_boosted:
# #                     try:
# #                         m_div = _unwrap(model)
# #                         boosted_attrs = []
# #                         for attr_name in ("sigma_min", "ctx_noise_scale"):
# #                             if hasattr(m_div, attr_name):
# #                                 old_v = getattr(m_div, attr_name)
# #                                 new_v = old_v * 1.5
# #                                 setattr(m_div, attr_name, new_v)
# #                                 boosted_attrs.append(
# #                                     f"{attr_name} {old_v:.4f}→{new_v:.4f}")
# #                         if boosted_attrs:
# #                             print(f"  [FIX-DIV-1] Tự động ×1.5 noise: "
# #                                   + ", ".join(boosted_attrs))
# #                             print(f"  → Ảnh hưởng initial noise của ensemble"
# #                                   f" sampling (sample()) và CFG noise đầu"
# #                                   f" DDIM, KHÔNG ảnh hưởng train CFM noise"
# #                                   f" schedule (_sigma_schedule).")
# #                         else:
# #                             print(f"  [FIX-DIV-1] Bỏ qua: model không có"
# #                                   f" attribute sigma_min/ctx_noise_scale"
# #                                   f" trực tiếp — cần xử lý thủ công"
# #                                   f" (xem strategy doc R1 fix #2).")
# #                         _diversity_boosted = True
# #                     except Exception as e:
# #                         print(f"  [FIX-DIV-1] Lỗi khi áp dụng noise boost: {e}")
# #             else:
# #                 print(f"  [OK] diversity={div_score:.1f} km >= "
# #                       f"{args.diversity_threshold} km → tiến hành phase 2")
# #             print(f"  {'='*50}\n")

# #         # Đánh giá đầy đủ mỗi val_freq epoch
# #         if ep % args.val_freq == 0:
# #             em = getattr(_unwrap(model), "_ema", None)
# #             rr = evaluate_split(model, vl, dev,
# #                                   tag=f"RAW ep{ep}",
# #                                   steps=args.full_ddim,
# #                                   use_selector=use_sel_eval,
# #                                   global_hard_threshold=eval_threshold)
# #             full_saver.update(rr, model, args.output_dir, ep, opt, sched,
# #                               avt, avv, tag="raw",
# #                               smooth_sched_step=smooth_sched.global_step,
# #                               global_hard_threshold=global_hard_threshold)

# #             if em and ep >= 3:
# #                 re = evaluate_split(model, vl, dev,
# #                                      tag=f"EMA ep{ep}",
# #                                      ema=em,
# #                                      steps=args.full_ddim,
# #                                      use_selector=use_sel_eval,
# #                                      global_hard_threshold=eval_threshold)
# #                 full_saver.update(re, model, args.output_dir, ep, opt, sched,
# #                                   avt, avv, tag="ema",
# #                                   smooth_sched_step=smooth_sched.global_step,
# #                                   global_hard_threshold=global_hard_threshold)

# #         # Periodic checkpoint
# #         if ep % 10 == 0 or ep == args.num_epochs - 1:
# #             _save_ckpt(
# #                 os.path.join(args.output_dir, f"ckpt_ep{ep:03d}.pth"),
# #                 ep, model, opt, sched, full_saver, avt, avv,
# #                 smooth_sched_step=smooth_sched.global_step,
# #                 global_hard_threshold=global_hard_threshold,
# #                 extra={"phase": phase,
# #                        "alpha_hard": smooth_sched.get_alpha_hard(),
# #                        "diversity": div_score},
# #             )

# #         if full_saver.stop:
# #             print(f"  Early stop ep={ep}")
# #             break

# #     th = (time.perf_counter() - ts) / 3600.0
# #     print(f"\n  Best: ADE={full_saver.ba:.1f}  72h={full_saver.b7:.0f}"
# #           f"  ATE={full_saver.bat:.1f}  CTE={full_saver.bc:.1f}"
# #           f"  easy_ADE={full_saver.best_easy_ade:.1f}"
# #           f"  ({th:.2f}h)")

# #     # Post-training test
# #     if args.eval_test_after_train:
# #         print("\n" + "="*72 + "\n  POST-TRAINING TEST\n" + "="*72)
# #         try: _, tl2 = data_loader(args, {"root": args.dataset_root, "type": "test"}, test=True)
# #         except: print("  No test → using val"); tl2 = vl

# #         for fn, lb in [("best_raw.pth", "RAW"), ("best_ema.pth", "EMA"),
# #                         ("best_ade_raw.pth", "BEST_ADE"),
# #                         ("best_cte_raw.pth", "BEST_CTE")]:
# #             pp = os.path.join(args.output_dir, fn)
# #             if not os.path.exists(pp): continue
# #             ck = torch.load(pp, map_location=dev)
# #             _unwrap(model).load_state_dict(ck["model_state_dict"], strict=False)
# #             em = getattr(_unwrap(model), "_ema", None)
# #             if em and ck.get("ema_shadow"):
# #                 for k, v in ck["ema_shadow"].items():
# #                     if k in em.shadow: em.shadow[k].copy_(v.to(dev))
# #             ckpt_threshold = ck.get("global_hard_threshold", None)
# #             r = evaluate_split(model, tl2, dev, tag=f"TEST/{lb}",
# #                                 steps=args.full_ddim,
# #                                 use_selector=(get_phase(ck.get("epoch", 80)) >= 3),
# #                                 global_hard_threshold=ckpt_threshold)
# #             print(f"\n  --- {lb} (ep={ck.get('epoch','?')}) ---")
# #             for key, ref in [("ADE", 172.68), ("72h", 321.39),
# #                               ("ATE", 142.21), ("CTE", 42.04)]:
# #                 v = r.get(key, float("nan"))
# #                 mk = "BEAT!" if np.isfinite(v) and v < ref else f"need {ref:.0f}"
# #                 print(f"    {key:<8}: {v:>8.1f} km  [{mk}  gap:{v-ref:+.1f}]")
# #             ea = r.get("easy_ADE", float("nan"))
# #             ha = r.get("hard_ADE", float("nan"))
# #             print(f"    easy_ADE={ea:.1f}  hard_ADE={ha:.1f}")

# #     print("=" * 72)


# # if __name__ == "__main__":
# #     args = get_args()
# #     np.random.seed(42)
# #     torch.manual_seed(42)
# #     if torch.cuda.is_available():
# #         torch.cuda.manual_seed_all(42)
# #     main(args)

# """
# train_v59_strategy.py — TC-FlowMatching v59-Strategy Training Script [FIXED-v2]
# ══════════════════════════════════════════════════════════════════════════════

# BASE: phiên bản với FIX-STABILITY-1/2/3, FIX-ALPHA-TIMING, FIX-EMA-SMOOTH, FIX-D
# (đã chạy và có log epoch 0-51)

# CÁC BUG MỚI ĐƯỢC FIX TRONG PHIÊN BẢN NÀY (dựa trên phân tích log thực tế):

#   [FIX-GN-1] GradNorm "loss-ratio" KHÔNG có equilibrium → runaway có hệ thống
#     Vấn đề cũ: r_i = raw_loss_i / mean(raw_loss_j) — KHÔNG phụ thuộc λ_i.
#     l_dpe có raw scale lớn hơn cấu trúc so với l_vel_reg/l_heading/l_accel
#     → r_dpe > 1 vĩnh viễn → λ_dpe luôn bị "tăng" → chạy tới ceiling (4.18)
#     → λ_vel/head/accel bị đẩy xuống floor (0.01) → 3 objective gần như bị tắt.
#     Log xác nhận: λ_dpe 1.00→4.18 (ep20→46), λ_vereg/heading/accel →0.01.

#     Fix: "equal-contribution" scheme.
#       contribution_i = λ_i * l_i  (phần đóng góp THỰC vào total loss)
#       ratio_i = contribution_i / mean(contribution_j)
#       ratio_i > 1 (đóng góp quá nhiều) → GIẢM λ_i
#       ratio_i < 1 (đóng góp quá ít)    → TĂNG λ_i
#     → có feedback âm thật: λ_i tăng → contribution_i tăng → ratio_i tăng
#       → bị kéo giảm lại → equilibrium λ_i ≈ mean_contrib / l_i.
#     Tự động bù trừ scale khác nhau giữa các loss term, không runaway.

#   [FIX-GN-2] phase_reset() KHÔNG reset λ về 1.0 nữa
#     Vấn đề cũ: λ_dpe=3.11→1.0 đột ngột tại ep20 (và tương tự ep50)
#     → tổng loss đổi đột ngột → easy_ADE nhảy 278→369km (ep20), 278→302 (ep51).
#     L_base (compute_st_trans_loss) không đổi giữa các phase — chỉ cộng thêm
#     L_hard/L_sel bên ngoài GradNorm → không có lý do để reset λ.
#     Fix: giữ nguyên λ hiện tại, chỉ reset optimizer momentum + dense window
#     ngắn (50 step) để re-adapt nhẹ nếu landscape có dịch nhẹ.

#   [FIX-EMA-2] EMA guard pullback có thể làm α kẹt vĩnh viễn ở 0
#     Vấn đề cũ: pullback_steps=nstep*2, rollback dày đặc trong vùng α-ramp
#     (ramp chỉ rộng ~35 step quanh epoch 24.5) → mỗi rollback trong vùng ramp
#     kéo step về trước vùng ramp → α=0 lại → α=0.000 suốt cả phase 2 (30 epoch).
#     Fix: chỉ pullback schedule nếu α hiện tại < 0.25 (còn trong vùng ramp).
#     Nếu α đã ramp xong (>=0.25), rollback chỉ restore weights, KHÔNG pullback
#     schedule. Giảm pullback_steps = nstep//2.

#   [FIX-PERF-1] Xóa dead/wasteful computation mỗi step
#     Code cũ lặp qua TOÀN BỘ named_parameters() mỗi step để tìm
#     "net.ctx_fc2.weight", tính lambda_grads_approx rồi truyền vào update()
#     — nhưng update() ghi rõ "lambda_grads: unused, ignore". Hoàn toàn lãng phí.
#     Fix: xóa block này, chỉ extract loss_vals_float và gọi update() trực tiếp.

#   [FIX-CKPT-1] _save_ckpt lưu smooth_sched_step + global_hard_threshold
#     cho MỌI checkpoint (trước chỉ có trong periodic ckpt's "extra").

#   [FIX-DIV-1] Diversity collapse (7.8km << 50km) — tự động xử lý
#     Theo strategy doc mục R1: "Tăng noise std ×1.5".
#     Fix: tự động ×1.5 sigma_min và ctx_noise_scale của model khi
#     diversity < threshold sau phase 1.

#   [FIX-THRESH-1] precompute threshold dùng flag riêng _threshold_computed
#     thay vì so sánh == 0.5 (fragile).

#   [FIX-EASYFRAC4] get_easy_frac() phase 4 dùng sigmoid (consistent với phase 2/3)

# CÁC BUG MỚI ĐƯỢC FIX TRONG v3 (dựa trên log thực tế chạy v2, epoch 0-6):

#   [FIX-GN-3] "equal-contribution" (FIX-GN-1) vẫn SAI — thay bằng
#     "anchor + target-ratio".

#     Log v2 ep0-6: λ_dpe GIẢM liên tục 1.00→0.335 (đang đi về floor 0.1),
#     λ_vel/head/accel TĂNG 1.00→1.17-1.20. val loss giảm đẹp (2.42→2.34,
#     THẤP NHẤT) nhưng easy_ADE TĂNG GẦN GẤP ĐÔI (408.9km log gốc → 705.7km)
#     và ADE trend ep2-6 ĐANG TĂNG (574→598→575→619→649). EMA Guard rollback
#     ngay tại ep6.

#     Root cause: l_dpe (Huber loss vị trí — objective CHÍNH, gắn trực tiếp
#     với ADE) có raw scale LỚN HƠN l_vel_reg/heading/accel KHÔNG PHẢI do lệch
#     scale ngẫu nhiên, mà vì đây là objective khó giảm hơn (regularizer phụ
#     dễ về gần 0). "Equal-contribution" coi đó là mất cân bằng cần sửa →
#     đẩy λ_dpe XUỐNG để "cân bằng" với 4 regularizer phụ → model dồn capacity
#     tối ưu 4 term phụ "dễ" (val loss giảm đẹp) nhưng hy sinh độ chính xác vị
#     trí → ADE tăng. Cùng pattern với log gốc ep49→50 (λ_dpe reset 3.92→1.0
#     đột ngột → easy_ADE 220→263km, "No improve" liên tục 4 epoch).

#     Fix: l_dpe = ANCHOR, λ_dpe CỐ ĐỊNH = 1.20 (= default hand-tuned trong
#     model, KHÔNG BAO GIỜ bị GradNorm điều chỉnh) → gradient cho vị trí
#     không bao giờ bị giảm dưới mức hand-tuned. 4 term phụ (vel_reg, heading,
#     speed, accel) được GradNorm điều chỉnh để duy trì TỶ LỆ ĐÓNG GÓP MỤC
#     TIÊU so với (λ_dpe·l_dpe), lấy từ default weights hand-tuned trong
#     model gốc (1.20, 1.40, 0.40, 0.05, 0.01):
#         target_contrib_i = DEFAULT_LAMBDA[i] * l_dpe
#         ratio_i = (λ_i · l_i) / target_contrib_i
#         ratio_i > 1 → giảm λ_i ; ratio_i < 1 → tăng λ_i
#     Equilibrium: λ_i ≈ DEFAULT_LAMBDA[i] · l_dpe / l_i — tự bù trừ scale RAW
#     của term i trôi theo training (đúng mục đích GradNorm), nhưng KHÔNG
#     đụng l_dpe và tôn trọng tỷ lệ quan trọng đã hand-tune giữa các term phụ.
#     Không cần renorm sum=5 nữa (anchor cố định, không có sum constraint).
#     Clamp λ_aux ∈ [0.02, 8.0] (range rộng hơn vì target_ratio chênh nhau
#     0.008–1.167 giữa các term).

#   [FIX-DIV-2] FIX-DIV-1 (×1.5 noise cố định) KHÔNG đủ.
#     Log gốc: diversity=8.3km << 50km (gap ~6x), ×1.5 → ~12.5km vẫn cách xa.
#     sigma_min và ctx_noise_scale CHỈ ảnh hưởng sample()/inference (initial
#     DDIM noise + CFG context noise 3 step đầu) — KHÔNG ảnh hưởng training
#     loss (_cfm_noisy dùng current_sigma từ _sigma_schedule(epoch), không
#     dùng self.sigma_min; forward_with_ctx lúc train không truyền noise_scale)
#     → có thể boost mạnh hơn nhiều mà không ảnh hưởng ổn định training.
#     Fix: boost LẶP — mỗi vòng ×1.8 (sigma_min và ctx_noise_scale), sau đó
#     chạy lại check_diversity(); lặp tối đa 3 vòng hoặc đến khi
#     diversity >= threshold; tổng boost factor cap ở 6.0x để tránh initial
#     noise quá lớn làm DDIM không hội tụ về trajectory hợp lý.

# GIỮ NGUYÊN từ phiên bản trước:
#   [G-A] Easy/hard pipeline với GLOBAL threshold
#   [G-C] easy_frac enforcement
#   [G-D] α smooth warm-up theo BATCH (FIX-ALPHA-TIMING: mid=15% of phase)
#   [G-E] EMA guard với EMA-smoothed easy_ADE (FIX-EMA-SMOOTH)
#   [G-F] Diversity check sau phase 1
#   [G-G] Split monitoring
#   [G-H] Phase 4: freeze encoder
#   BUG-5/6/7/8 từ v59-tweaked
# """
# from __future__ import annotations
# import sys, os
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# import argparse, math, random, time
# from collections import defaultdict
# from typing import Dict, List, Optional

# import numpy as np
# import torch
# import torch.optim as optim
# import torch.nn.functional as F
# from torch.amp import autocast, GradScaler
# from torch.utils.data import DataLoader, Subset

# from Model.data.loader_training import data_loader
# from Model.flow_matching_model import (
#     TCFlowMatching,
#     hard_score_from_obs,       # dùng trong precompute_hard_threshold
#     classify_hard_easy,        # dùng trong evaluate_split
#     compute_diversity_score,   # dùng trong check_diversity
#     _norm_to_deg,
#     _haversine_deg,
# )

# try:
#     from Model.utils import get_cosine_schedule_with_warmup
# except ImportError:
#     from torch.optim.lr_scheduler import CosineAnnealingLR
#     def get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps, min_lr=1e-6):
#         return CosineAnnealingLR(opt, T_max=max(total_steps, 1), eta_min=min_lr)

# TARGETS = {
#     "ADE": 172.68, "72h": 321.39, "ATE": 142.21, "CTE": 42.04,
#     "12h": 65.42,  "24h": 104.67, "48h": 205.10,
# }
# R_EARTH = 6371.0

# # ── Phase boundaries ──────────────────────────────────────────────────────────
# PHASE_1_END  = 19   # epoch 0–19:  warm-up, α=0, easy_frac=80%
# PHASE_2_END  = 49   # epoch 20–49: hard intro, α 0→0.3, easy_frac 80→60%
# PHASE_3_END  = 79   # epoch 50–79: selector, α=0.3, easy_frac=55%
# # epoch 80+: fine-tune, LR×0.1, freeze encoder

# # ── GradNorm terms (các loss term cần cân bằng) ──────────────────────────────
# GRADNORM_TERMS = ["l_dpe", "l_vel_reg", "l_heading", "l_speed", "l_accel"]

# # [FIX-GN-3] l_dpe là ANCHOR — λ_dpe cố định, KHÔNG bị GradNorm điều chỉnh.
# # Giá trị = default hand-tuned trong compute_st_trans_loss (total_default).
# ANCHOR_TERM  = "l_dpe"
# AUX_TERMS    = ["l_vel_reg", "l_heading", "l_speed", "l_accel"]
# DEFAULT_LAMBDA = {
#     "l_dpe":     1.20,
#     "l_vel_reg": 1.40,
#     "l_heading": 0.40,
#     "l_speed":   0.05,
#     "l_accel":   0.01,
# }
# AUX_LAMBDA_MIN = 0.02
# AUX_LAMBDA_MAX = 8.0

# # Alpha threshold để coi là "đã ramp xong" — dùng cho EMA guard pullback gate
# ALPHA_RAMP_DONE_THRESHOLD = 0.25


# # ─────────────────────────────────────────────────────────────────────────────
# #  [FIX-GN-3] GradNorm implementation — anchor + target-ratio scheme
# #
# #  Thuật toán:
# #    - l_dpe = ANCHOR. λ_dpe CỐ ĐỊNH = DEFAULT_LAMBDA["l_dpe"] = 1.20.
# #      → gradient cho l_dpe (objective chính, gắn trực tiếp ADE) không
# #        bao giờ bị GradNorm giảm xuống dưới mức hand-tuned.
# #    - Với mỗi term phụ i ∈ AUX_TERMS:
# #        contribution_i    = λ_i * l_i
# #        target_contrib_i  = DEFAULT_LAMBDA[i] * l_dpe
# #        ratio_i           = contribution_i / target_contrib_i
# #        ratio_i > 1 (đóng góp vượt tỷ lệ mục tiêu) → GIẢM λ_i → grad > 0
# #        ratio_i < 1 (đóng góp dưới tỷ lệ mục tiêu) → TĂNG λ_i → grad < 0
# #    - Equilibrium: λ_i ≈ DEFAULT_LAMBDA[i] * l_dpe / l_i
# #      Nếu l_i == l_dpe thì λ_i == DEFAULT_LAMBDA[i] (đúng hand-tuned gốc).
# #      Nếu l_i trôi scale theo training, λ_i tự bù trừ để giữ TỶ LỆ ĐÓNG GÓP
# #      (không phải giá trị λ) bằng đúng tỷ lệ hand-tuned.
# #    - Không có sum constraint → không cần renorm. Mỗi λ_i clamp riêng vào
# #      [AUX_LAMBDA_MIN, AUX_LAMBDA_MAX] (range rộng vì DEFAULT_LAMBDA chênh
# #      nhau tới ~150x giữa l_vel_reg và l_accel).
# # ─────────────────────────────────────────────────────────────────────────────

# class GradNormManager:
#     """
#     [FIX-GN-3] Quản lý GradNorm — anchor + target-ratio scheme.

#     l_dpe là ANCHOR (λ cố định = DEFAULT_LAMBDA["l_dpe"] = 1.20, không nằm
#     trong self.lambdas, không bao giờ bị update). Chỉ AUX_TERMS
#     (l_vel_reg, l_heading, l_speed, l_accel) được GradNorm điều chỉnh,
#     với equilibrium λ_i ≈ DEFAULT_LAMBDA[i] * l_dpe / l_i.

#     [FIX-GN-2] phase_reset(): KHÔNG reset λ — chỉ reset optimizer momentum
#     + bật dense update ngắn (50 step) để re-adapt nhẹ nếu landscape có dịch
#     (do thêm gradient từ L_hard/L_sel chảy qua shared encoder ở phase mới).
#     """
#     def __init__(self, terms: List[str], alpha_gn: float = 1.5,
#                  lr: float = 1e-3, device=None,
#                  lambda_min: float = AUX_LAMBDA_MIN,
#                  lambda_max: float = AUX_LAMBDA_MAX):
#         # terms: full GRADNORM_TERMS list (bao gồm anchor) — giữ để
#         # tương thích các nơi khác lặp qua GRADNORM_TERMS khi log/extract.
#         self.terms      = terms
#         self.aux_terms  = [t for t in terms if t != ANCHOR_TERM]
#         self.alpha_gn   = alpha_gn
#         self.device     = device or torch.device("cpu")
#         self.lambda_min = lambda_min
#         self.lambda_max = lambda_max

#         # λ_i cho AUX_TERMS, init = DEFAULT_LAMBDA[i] (equilibrium đúng nếu
#         # l_i == l_dpe ngay từ đầu — điểm khởi đầu hợp lý hơn 1.0 đồng nhất).
#         self.lambdas = {t: torch.full((1,), DEFAULT_LAMBDA[t],
#                                        device=self.device, requires_grad=True)
#                         for t in self.aux_terms}
#         self.opt = optim.Adam(list(self.lambdas.values()), lr=lr)

#         self._step             = 0
#         self._update_freq      = 5      # update mỗi 5 steps (bình thường)
#         self._dense_steps_left = 200    # dense ở lúc khởi động training
#         self._current_phase    = 1

#     def phase_reset(self, new_phase: int) -> None:
#         """
#         [FIX-GN-2] Gọi khi chuyển sang phase mới.
#         KHÔNG reset λ — chỉ reset optimizer momentum và bật dense window
#         ngắn (50 step, không phải 200) để re-adapt nhẹ.
#         """
#         if new_phase == self._current_phase:
#             return
#         self._current_phase = new_phase

#         self.opt = optim.Adam(list(self.lambdas.values()),
#                                lr=self.opt.param_groups[0]["lr"])
#         self._dense_steps_left = 50

#         cur = self.get_lambda_dict()
#         lam_str = " ".join(f"λ_{t.replace('l_','').replace('_reg','')}"
#                             f"={cur[t]:.3f}" for t in self.terms)
#         print(f"  [GradNorm] Phase {new_phase}: GIỮ λ hiện tại ({lam_str}), "
#               f"λ_dpe cố định={DEFAULT_LAMBDA[ANCHOR_TERM]:.2f}, "
#               f"reset optimizer momentum, dense update 50 steps")

#     def get_lambda_dict(self) -> Dict[str, float]:
#         """Trả về λ hiện tại (đủ GRADNORM_TERMS) để truyền vào
#         get_loss_breakdown(). λ_dpe luôn = DEFAULT_LAMBDA["l_dpe"] (cố định)."""
#         d = {t: float(self.lambdas[t].item()) for t in self.aux_terms}
#         d[ANCHOR_TERM] = DEFAULT_LAMBDA[ANCHOR_TERM]
#         return d

#     def update(self, loss_term_values: Dict[str, float]) -> None:
#         """
#         [FIX-GN-3] Cập nhật λ_aux dùng anchor + target-ratio scheme.

#         Args:
#             loss_term_values: {term: float} — loss RAW value từng term
#                               (detached, chưa nhân λ), PHẢI bao gồm
#                               ANCHOR_TERM ("l_dpe") để dùng làm tham chiếu.
#         """
#         self._step += 1

#         if self._dense_steps_left > 0:
#             self._dense_steps_left -= 1
#             update_freq = 1
#         else:
#             update_freq = self._update_freq

#         def _clamp():
#             with torch.no_grad():
#                 for t in self.aux_terms:
#                     self.lambdas[t].clamp_(min=self.lambda_min,
#                                             max=self.lambda_max)

#         if self._step % update_freq != 0:
#             _clamp()
#             return

#         if not loss_term_values:
#             _clamp()
#             return

#         l_anchor = loss_term_values.get(ANCHOR_TERM)
#         if l_anchor is None or not np.isfinite(l_anchor) or l_anchor <= 0:
#             _clamp()
#             return

#         loss_vals = {t: v for t, v in loss_term_values.items()
#                      if t in self.aux_terms and np.isfinite(v) and v > 0}
#         if not loss_vals:
#             _clamp()
#             return

#         self.opt.zero_grad()
#         for t in self.aux_terms:
#             if t not in loss_vals:
#                 continue
#             lam_t            = float(self.lambdas[t].item())
#             contribution_t   = lam_t * loss_vals[t]
#             target_contrib_t = DEFAULT_LAMBDA[t] * l_anchor
#             ratio = contribution_t / (target_contrib_t + 1e-8)
#             # ratio > 1: đóng góp vượt tỷ lệ mục tiêu → GIẢM λ_i → grad > 0
#             # ratio < 1: đóng góp dưới tỷ lệ mục tiêu → TĂNG λ_i → grad < 0
#             grad_val = (ratio - 1.0) * 0.1
#             if self.lambdas[t].grad is None:
#                 self.lambdas[t].grad = torch.zeros_like(self.lambdas[t])
#             self.lambdas[t].grad.fill_(float(grad_val))

#         self.opt.step()
#         _clamp()

#     def log_stats(self) -> Dict[str, float]:
#         """Log λ hiện tại để monitor. λ_l_dpe luôn = giá trị anchor cố định."""
#         d = {f"λ_{t}": float(self.lambdas[t].item()) for t in self.aux_terms}
#         d[f"λ_{ANCHOR_TERM}"] = DEFAULT_LAMBDA[ANCHOR_TERM]
#         d["gn_dense"] = self._dense_steps_left
#         return d


# # ─────────────────────────────────────────────────────────────────────────────
# #  Phase helper
# # ─────────────────────────────────────────────────────────────────────────────

# def get_phase(epoch: int) -> int:
#     """
#     Returns:
#         1: warm-up (ep 0–19)
#         2: hard introduction (ep 20–49)
#         3: selector (ep 50–79)
#         4: fine-tune (ep 80+)
#     """
#     if epoch <= PHASE_1_END:  return 1
#     if epoch <= PHASE_2_END:  return 2
#     if epoch <= PHASE_3_END:  return 3
#     return 4


# class SmoothScheduler:
#     """
#     [FIX-STABILITY-1] α smooth warm-up theo BATCH, không phải epoch.

#     [FIX-ALPHA-TIMING] mid point ở 15% của phase thay vì 50%:
#         mid ở epoch ~24.5 (phase2) → α≈0.15 ở ep23, α≈0.3 ở ep26
#         → hard loss kích hoạt sớm hơn ~10 epoch so với mid=50%.

#     Selector cũng dùng cùng logic để bật đồng bộ.
#     """
#     def __init__(self, total_steps_per_epoch: int):
#         self.steps_per_epoch = total_steps_per_epoch
#         self._global_step = 0

#         # Điểm bắt đầu của mỗi phase (tính theo global step)
#         self._phase2_start_step = (PHASE_1_END + 1) * total_steps_per_epoch
#         self._phase3_start_step = (PHASE_2_END + 1) * total_steps_per_epoch
#         self._phase4_start_step = (PHASE_3_END + 1) * total_steps_per_epoch

#         # [FIX-ALPHA-TIMING] Temperature ngắn → α tăng SỚM trong phase 2.
#         self._temp_p2 = 0.6 * total_steps_per_epoch / 4.4
#         self._temp_p3 = 0.6 * total_steps_per_epoch / 4.4  # selector bật nhanh
#         # [FIX-EASYFRAC4] Temperature cho phase 4 (giảm 55%→50% qua ~5 epoch)
#         self._temp_p4 = 2.5 * total_steps_per_epoch / 4.4

#     def step(self):
#         """Gọi sau mỗi optimizer step."""
#         self._global_step += 1

#     @property
#     def global_step(self):
#         return self._global_step

#     def get_alpha_hard(self) -> float:
#         """
#         α_hard: trọng số L_hard, tăng mượt qua sigmoid.
#         Phase 1 (step < phase2_start): α = 0
#         Phase 2 (step >= phase2_start): sigmoid 0 → 0.3, mid ở 15% phase
#         Phase 3+: α = 0.3 cố định
#         """
#         s = self._global_step
#         if s < self._phase2_start_step:
#             return 0.0
#         if s >= self._phase3_start_step:
#             return 0.3

#         mid  = self._phase2_start_step + (self._phase3_start_step
#                                            - self._phase2_start_step) * 0.15
#         x    = (s - mid) / max(self._temp_p2, 1.0)
#         sig  = 1.0 / (1.0 + math.exp(-x))
#         return 0.3 * sig

#     def get_selector_weight(self) -> float:
#         """
#         selector_weight: trọng số L_sel, tăng mượt từ phase 3.
#         Phase < 3: 0
#         Phase 3: sigmoid 0 → 0.2
#         Phase 4: 0.2 cố định
#         """
#         s = self._global_step
#         if s < self._phase3_start_step:
#             return 0.0
#         if s >= self._phase4_start_step:
#             return 0.2

#         mid  = self._phase3_start_step + (self._phase4_start_step
#                                            - self._phase3_start_step) / 2.0
#         x    = (s - mid) / max(self._temp_p3, 1.0)
#         sig  = 1.0 / (1.0 + math.exp(-x))
#         return 0.2 * sig

#     def get_easy_frac(self) -> float:
#         """
#         easy_frac: giảm mượt, không bao giờ xuống dưới 50%.
#         Phase 1: 80%
#         Phase 2: 80% → 60% theo sigmoid
#         Phase 3: 60% → 55%
#         [FIX-EASYFRAC4] Phase 4: 55% → 50% theo sigmoid (thay vì linear)
#             Consistent với phương pháp của các phase khác.
#             Midpoint: 5 epoch sau khi vào phase 4.
#         """
#         s = self._global_step
#         if s < self._phase2_start_step:
#             return 0.80

#         if s < self._phase3_start_step:
#             mid  = self._phase2_start_step + (self._phase3_start_step
#                                                - self._phase2_start_step) / 2.0
#             x    = (s - mid) / max(self._temp_p2, 1.0)
#             sig  = 1.0 / (1.0 + math.exp(-x))
#             return 0.80 - 0.20 * sig   # 80% → 60%

#         if s < self._phase4_start_step:
#             mid  = self._phase3_start_step + (self._phase4_start_step
#                                                - self._phase3_start_step) / 2.0
#             x    = (s - mid) / max(self._temp_p3, 1.0)
#             sig  = 1.0 / (1.0 + math.exp(-x))
#             return 0.60 - 0.05 * sig   # 60% → 55%

#         # [FIX-EASYFRAC4] Phase 4: 55% → 50% qua sigmoid (không phải linear)
#         phase4_mid = self._phase4_start_step + 5 * self.steps_per_epoch
#         x   = (s - phase4_mid) / max(self._temp_p4, 1.0)
#         sig = 1.0 / (1.0 + math.exp(-x))
#         return max(0.50, 0.55 - 0.05 * sig)

#     def get_diversity_weight(self, target: float = 0.15) -> float:
#         """
#         [FIX-DIV-3] diversity_loss_weight: trọng số L_diversity (hinge,
#         ngoài GradNorm), tăng mượt từ phase 2 — CÙNG schedule shape với
#         get_alpha_hard (sigmoid, mid ở 15% phase 2).
#         Phase 1: 0 (giữ warm-up ổn định, KHÔNG thêm forward pass thứ 2)
#         Phase 2: sigmoid 0 → target
#         Phase 3+: target cố định
#         """
#         s = self._global_step
#         if s < self._phase2_start_step:
#             return 0.0
#         if s >= self._phase3_start_step:
#             return target
#         mid  = self._phase2_start_step + (self._phase3_start_step
#                                            - self._phase2_start_step) * 0.15
#         x    = (s - mid) / max(self._temp_p2, 1.0)
#         sig  = 1.0 / (1.0 + math.exp(-x))
#         return target * sig

#     def is_selector_active(self) -> bool:
#         """True từ phase 3 trở đi."""
#         return self._global_step >= self._phase3_start_step

#     def log_stats(self) -> str:
#         return (f"α={self.get_alpha_hard():.3f}"
#                 f" easy={self.get_easy_frac():.0%}"
#                 f" sel={self.get_selector_weight():.3f}")


# # Backward-compat wrappers (dùng trong checkpoint restore)
# def get_alpha_hard(epoch: int) -> float:
#     """Legacy: chỉ dùng khi không có SmoothScheduler."""
#     if epoch <= PHASE_1_END: return 0.0
#     if epoch <= PHASE_2_END:
#         return 0.3 * (epoch - PHASE_1_END) / (PHASE_2_END - PHASE_1_END)
#     return 0.3

# def get_easy_frac(epoch: int) -> float:
#     """Legacy fallback."""
#     if epoch <= PHASE_1_END: return 0.80
#     if epoch <= PHASE_2_END:
#         p = (epoch - PHASE_1_END) / (PHASE_2_END - PHASE_1_END)
#         return 0.80 - 0.20 * p
#     if epoch <= PHASE_3_END: return 0.55
#     return 0.50


# # ─────────────────────────────────────────────────────────────────────────────
# #  [FIX-STABILITY-3 + FIX-THRESH-1] Global hard threshold — tính 1 lần
# #
# #  hard_score_from_obs() chỉ phụ thuộc obs_traj (data cố định), KHÔNG phụ
# #  thuộc model weights → threshold KHÔNG thay đổi theo training.
# #  [FIX-THRESH-1] Dùng flag _threshold_computed riêng (xem main loop) thay vì
# #  so sánh global_hard_threshold == 0.5 (fragile — nếu p70 thực tế = 0.5 thì
# #  sẽ recompute mãi).
# # ─────────────────────────────────────────────────────────────────────────────

# @torch.no_grad()
# def precompute_hard_threshold(loader, dev, n_batches: int = 50) -> float:
#     """
#     Tính p70 của hard_score trên subset của train set (1 lần duy nhất).
#     """
#     all_scores = []
#     for i, batch in enumerate(loader):
#         if i >= n_batches:
#             break
#         bl    = move(list(batch), dev)
#         obs_t = bl[0]
#         scores = hard_score_from_obs(obs_t[:, :, :2])  # [B]
#         all_scores.append(scores.cpu())

#     if not all_scores:
#         return 0.7  # fallback

#     all_scores_cat = torch.cat(all_scores)  # [N]
#     threshold = float(torch.quantile(all_scores_cat, 0.70).item())
#     n_hard = int((all_scores_cat >= threshold).sum().item())
#     n_total = len(all_scores_cat)
#     print(f"  [HardThreshold] global p70={threshold:.4f}"
#           f"  n_hard={n_hard}/{n_total} ({100*n_hard/max(n_total,1):.0f}%)")
#     return threshold


# @torch.no_grad()
# def classify_hard_easy_global(
#     obs_traj_norm: torch.Tensor,
#     global_threshold: float,
# ) -> torch.Tensor:
#     """Phân loại hard/easy dùng global threshold (thay vì per-batch p70)."""
#     scores   = hard_score_from_obs(obs_traj_norm)  # [B]
#     is_hard  = scores >= global_threshold
#     return is_hard


# def enforce_easy_frac(
#     is_hard: torch.Tensor,
#     easy_frac: float,
# ) -> torch.Tensor:
#     """
#     Chỉ điều chỉnh is_hard mask, KHÔNG reindex batch_list.
#     Nếu batch có quá nhiều hard → "demote" một số về easy (is_hard[i]=False).
#     Batch data giữ nguyên 100%.
#     """
#     B        = is_hard.shape[0]
#     max_hard = max(0, int(B * (1.0 - easy_frac)))
#     n_hard   = int(is_hard.sum().item())

#     if n_hard <= max_hard:
#         return is_hard  # đã đúng tỉ lệ

#     n_demote  = n_hard - max_hard
#     hard_idx  = is_hard.nonzero(as_tuple=True)[0]
#     perm      = torch.randperm(hard_idx.numel(), device=is_hard.device)
#     to_demote = hard_idx[perm[:n_demote]]

#     is_hard_new            = is_hard.clone()
#     is_hard_new[to_demote] = False
#     return is_hard_new


# # ─────────────────────────────────────────────────────────────────────────────
# #  Utilities
# # ─────────────────────────────────────────────────────────────────────────────

# def _unwrap(m):
#     return m._orig_mod if hasattr(m, "_orig_mod") else m

# def move(b, dev):
#     out = list(b)
#     for i, x in enumerate(out):
#         if torch.is_tensor(x): out[i] = x.to(dev)
#         elif isinstance(x, dict):
#             out[i] = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in x.items()}
#     return out


# def _ntd(a):
#     o = a.clone()
#     o[..., 0] = (a[..., 0] * 50.0 + 1800.0) / 10.0
#     o[..., 1] = (a[..., 1] * 50.0) / 10.0
#     return o

# def _hav(p1, p2):
#     la1 = torch.deg2rad(p1[..., 1]); la2 = torch.deg2rad(p2[..., 1])
#     dlat = torch.deg2rad(p2[..., 1] - p1[..., 1])
#     dlon = torch.deg2rad(p2[..., 0] - p1[..., 0])
#     a = (torch.sin(dlat/2).pow(2)
#          + torch.cos(la1)*torch.cos(la2)*torch.sin(dlon/2).pow(2))
#     return 2.0 * R_EARTH * torch.asin(a.clamp(1e-12, 1-1e-12).sqrt())

# def _atecte(pd, gd):
#     T = min(pd.shape[0], gd.shape[0])
#     if T < 2:
#         z = pd.new_zeros(1, pd.shape[1]); return z, z
#     lo1=torch.deg2rad(gd[:T-1,:,0]); la1=torch.deg2rad(gd[:T-1,:,1])
#     lo2=torch.deg2rad(gd[1:T, :,0]); la2=torch.deg2rad(gd[1:T, :,1])
#     lo3=torch.deg2rad(pd[1:T, :,0]); la3=torch.deg2rad(pd[1:T, :,1])
#     ya=torch.sin(lo2-lo1)*torch.cos(la2)
#     xa=(torch.cos(la1)*torch.sin(la2)-torch.sin(la1)*torch.cos(la2)*torch.cos(lo2-lo1))
#     be=torch.atan2(ya,xa)
#     ye=torch.sin(lo3-lo2)*torch.cos(la3)
#     xe=(torch.cos(la2)*torch.sin(la3)-torch.sin(la2)*torch.cos(la3)*torch.cos(lo3-lo2))
#     bee=torch.atan2(ye,xe)
#     tot=_hav(pd[1:T],gd[1:T]); ang=bee-be
#     return tot*torch.cos(ang), tot*torch.sin(ang)


# # ─────────────────────────────────────────────────────────────────────────────
# #  [G-G] Split metrics — đo easy_ADE và hard_ADE riêng biệt
# # ─────────────────────────────────────────────────────────────────────────────

# class SplitAcc:
#     """
#     Accumulator đo easy_ADE và hard_ADE riêng biệt.
#     Dùng để:
#     - EMA guard: phát hiện easy forgetting sớm
#     - Hard improvement: xác nhận hard loss đang hoạt động
#     """
#     def __init__(self):
#         self.easy_d = []
#         self.hard_d = []
#         self.all_d  = []
#         self.all_a  = []
#         self.all_c  = []
#         self.sd     = defaultdict(list)
#         self._h     = {12: 1, 24: 3, 48: 7, 72: 11}

#     def update(self, dist, is_hard=None, ate=None, cte=None):
#         """
#         Args:
#             dist:    [T, B] distance per step
#             is_hard: [B] bool (None → tất cả easy)
#         """
#         B = dist.shape[1]
#         mean_dist = dist.mean(0)  # [B]
#         self.all_d.extend(mean_dist.tolist())

#         if is_hard is not None:
#             easy_mask = ~is_hard
#             if easy_mask.any():
#                 self.easy_d.extend(mean_dist[easy_mask].tolist())
#             if is_hard.any():
#                 self.hard_d.extend(mean_dist[is_hard].tolist())
#         else:
#             self.easy_d.extend(mean_dist.tolist())

#         for h, s in self._h.items():
#             if s < dist.shape[0]:
#                 self.sd[h].extend(dist[s].tolist())

#         if ate is not None: self.all_a.extend(ate.abs().mean(0).tolist())
#         if cte is not None: self.all_c.extend(cte.abs().mean(0).tolist())

#     def compute(self):
#         r = {
#             "ADE":      float(np.mean(self.all_d))  if self.all_d  else float("nan"),
#             "easy_ADE": float(np.mean(self.easy_d)) if self.easy_d else float("nan"),
#             "hard_ADE": float(np.mean(self.hard_d)) if self.hard_d else float("nan"),
#             "ATE":      float(np.mean(self.all_a))  if self.all_a  else float("nan"),
#             "CTE":      float(np.mean(self.all_c))  if self.all_c  else float("nan"),
#             "n":        len(self.all_d),
#             "n_easy":   len(self.easy_d),
#             "n_hard":   len(self.hard_d),
#         }
#         for h in self._h:
#             v = self.sd.get(h, [])
#             r[f"{h}h"] = float(np.mean(v)) if v else float("nan")
#         return r


# class Acc:
#     """Simple accumulator (backward compat với evaluate())."""
#     def __init__(self):
#         self.d=[]; self.a=[]; self.c=[]; self.sd=defaultdict(list)
#         self._h={12:1, 24:3, 48:7, 72:11}
#     def update(self, dist, ate=None, cte=None):
#         self.d.extend(dist.mean(0).tolist())
#         for h,s in self._h.items():
#             if s < dist.shape[0]: self.sd[h].extend(dist[s].tolist())
#         if ate is not None: self.a.extend(ate.abs().mean(0).tolist())
#         if cte is not None: self.c.extend(cte.abs().mean(0).tolist())
#     def compute(self):
#         r = {"ADE": float(np.mean(self.d)) if self.d else float("nan"),
#              "ATE": float(np.mean(self.a)) if self.a else float("nan"),
#              "CTE": float(np.mean(self.c)) if self.c else float("nan"),
#              "n": len(self.d)}
#         for h in self._h:
#             v=self.sd.get(h,[]); r[f"{h}h"]=float(np.mean(v)) if v else float("nan")
#         return r


# def _score(r):
#     ade=r.get("ADE",1e9); h72=r.get("72h",1e9)
#     ate=r.get("ATE",1e9); cte=r.get("CTE",1e9)
#     if not np.isfinite(ate): ate=ade*0.46
#     if not np.isfinite(cte): cte=ade*0.53
#     return 100.0*(0.05*(ade/136.)+0.10*(r.get("12h",ade)/50.)
#                   +0.15*(r.get("24h",ade)/100.)+0.20*(r.get("48h",ade)/200.)
#                   +0.25*(h72/300.)+0.13*(ate/80.)+0.12*(cte/94.))

# def _beat(r):
#     p=[]
#     for k,t in [("ADE",172.68),("ATE",142.21),("CTE",42.04),
#                 ("72h",321.39),("12h",65.42),("24h",104.67),("48h",205.10)]:
#         v=r.get(k,1e9)
#         if np.isfinite(v) and v < t: p.append(f"{k}:{v:.1f}")
#     return "*** BEAT ST-TRANS: "+" ".join(p)+" ***" if p else ""

# def _gap(r):
#     out=[]
#     for k,ref in [("ADE",172.68),("72h",321.39),("ATE",142.21),("CTE",42.04)]:
#         v=r.get(k,float("nan"))
#         if np.isfinite(v):
#             out.append(f"{k}:{v:.0f}({'dn' if v<ref else 'up'}{abs(v-ref):.0f})")
#     return " | ".join(out)


# # ─────────────────────────────────────────────────────────────────────────────
# #  [G-E] Easy_ADE evaluation cho EMA guard — dùng global threshold (FIX-4)
# # ─────────────────────────────────────────────────────────────────────────────

# @torch.no_grad()
# def evaluate_split(model, loader, dev, tag="", ema=None, steps=10,
#                    use_selector=False,
#                    global_hard_threshold: Optional[float] = None) -> Dict:
#     """
#     Evaluate với split easy/hard ADE.

#     Args:
#         global_hard_threshold: float từ precompute_hard_threshold(), hoặc None
#             → dùng per-batch p70 (phase 1 fallback, chỉ ảnh hưởng log)
#     """
#     bk = None
#     if ema:
#         try: bk = ema.apply_to(model)
#         except: pass
#     model.eval()
#     acc = SplitAcc()
#     t0  = time.perf_counter()

#     for b in loader:
#         bl = move(list(b), dev)
#         obs_t = bl[0]

#         if global_hard_threshold is not None:
#             is_hard = classify_hard_easy_global(
#                 obs_t[:, :, :2], global_hard_threshold).to(dev)
#         else:
#             is_hard = classify_hard_easy(obs_t[:, :, :2]).to(dev)

#         result = model.sample(bl, ddim_steps=steps, use_selector=use_selector)
#         p = result[0] if isinstance(result, (tuple, list)) else result
#         g = bl[1]; T = min(p.shape[0], g.shape[0])
#         pd = _ntd(p[:T]); gd = _ntd(g[:T])
#         dist = _hav(pd, gd)
#         at, ct = _atecte(pd, gd)
#         acc.update(dist, is_hard=is_hard, ate=at, cte=ct)

#     # [FIX-EMA-restore] bk luôn là dict non-empty nếu apply_to thành công
#     # (shadow luôn có ít nhất 1 floating-point param) — nhưng dùng
#     # `is not None` cho rõ ràng & an toàn tuyệt đối.
#     if bk is not None:
#         try: ema.restore(model, bk)
#         except: pass

#     r  = acc.compute()
#     el = time.perf_counter() - t0

#     def _v(k): return r.get(k, float("nan"))
#     def _m(v, t): return "ok" if np.isfinite(v) and v < t else "no"

#     thr_str = f"{global_hard_threshold:.4f}" if global_hard_threshold is not None else "per-batch"
#     print(f"\n{'='*70}")
#     print(f"  [{tag}  {el:.0f}s  threshold={thr_str}]")
#     print(f"  ADE={_v('ADE'):.1f}[{_m(_v('ADE'),172.68)}]"
#           f"  easy_ADE={_v('easy_ADE'):.1f}  hard_ADE={_v('hard_ADE'):.1f}")
#     print(f"  12h={_v('12h'):.0f}  24h={_v('24h'):.0f}"
#           f"  48h={_v('48h'):.0f}  72h={_v('72h'):.0f}[{_m(_v('72h'),321.39)}]")
#     print(f"  ATE={_v('ATE'):.1f}[{_m(_v('ATE'),142.21)}]"
#           f"  CTE={_v('CTE'):.1f}[{_m(_v('CTE'),42.04)}]")
#     print(f"  n_easy={r.get('n_easy',0)}  n_hard={r.get('n_hard',0)}")
#     print(f"  vs ST-Trans: {_gap(r)}")
#     bt = _beat(r)
#     if bt: print(f"  {bt}")
#     print(f"  Score={_score(r):.2f}")
#     print(f"{'='*70}\n")
#     return r


# @torch.no_grad()
# def evaluate(model, loader, dev, tag="", ema=None, steps=20,
#              use_selector=False,
#              global_hard_threshold: Optional[float] = None) -> Dict:
#     """Full evaluation (backward compat)."""
#     return evaluate_split(model, loader, dev, tag=tag, ema=ema,
#                           steps=steps, use_selector=use_selector,
#                           global_hard_threshold=global_hard_threshold)


# # ─────────────────────────────────────────────────────────────────────────────
# #  [G-F] Diversity check
# # ─────────────────────────────────────────────────────────────────────────────

# @torch.no_grad()
# def check_diversity(model, loader, dev, n_batches: int = 5,
#                     n_ensemble: int = 20, ddim_steps: int = 5) -> float:
#     """
#     Đo diversity score trên một subset của val set.
#     Nếu < 50 km → cảnh báo R1 (mode collapse).
#     """
#     model.eval()
#     all_divs = []

#     for i, b in enumerate(loader):
#         if i >= n_batches:
#             break
#         bl = move(list(b), dev)

#         _, _, all_c = model.sample(bl, num_ensemble=n_ensemble,
#                                     ddim_steps=ddim_steps)
#         N_total = all_c.shape[0]
#         cands = [all_c[k] for k in range(N_total)]

#         div = compute_diversity_score(cands)
#         all_divs.append(div)

#     return float(np.mean(all_divs)) if all_divs else 0.0


# # ─────────────────────────────────────────────────────────────────────────────
# #  [FIX-CKPT-1] Checkpoint — lưu smooth_sched_step + global_hard_threshold
# #  trong MỌI checkpoint (trước chỉ có trong periodic ckpt's "extra")
# # ─────────────────────────────────────────────────────────────────────────────

# def _save_ckpt(path, ep, model, opt, sched, saver, tl, vl,
#                smooth_sched_step: int = 0,
#                global_hard_threshold: Optional[float] = None,
#                diversity_boost_factor: float = 1.0,
#                extra=None):
#     m = _unwrap(model)
#     ema = getattr(m, "_ema", None)
#     esd = None
#     if ema and hasattr(ema, "shadow"):
#         try: esd = {k: v.cpu().clone() for k, v in ema.shadow.items()}
#         except: pass
#     d = {
#         "epoch": ep,
#         "model_state_dict": m.state_dict(),
#         "optimizer_state":  opt.state_dict(),
#         "scheduler_state":  sched.state_dict(),
#         "ema_shadow":       esd,
#         "best_score":       saver.bs,
#         "best_ade":         saver.ba,
#         "best_72h":         saver.b7,
#         "best_ate":         saver.bat,
#         "best_cte":         saver.bc,
#         "best_easy_ade":    saver.best_easy_ade,
#         "train_loss":       tl,
#         "val_loss":         vl,
#         # [FIX-CKPT-1] Luôn lưu để resume chính xác smooth_sched + threshold
#         "smooth_sched_step":     smooth_sched_step,
#         "global_hard_threshold": global_hard_threshold,
#         # [FIX-DIV-2] Lưu hệ số boost diversity để resume áp lại đúng
#         # sigma_min/ctx_noise_scale (đây là plain float attr, KHÔNG nằm
#         # trong model.state_dict() nên sẽ mất nếu không lưu riêng).
#         "diversity_boost_factor": diversity_boost_factor,
#         "version":          "v59strategy_fixed_v3",
#     }
#     if extra:
#         d.update(extra)
#     torch.save(d, path)


# # ─────────────────────────────────────────────────────────────────────────────
# #  Saver
# # ─────────────────────────────────────────────────────────────────────────────

# class Saver:
#     def __init__(self, patience=40, min_ep=35, enable_stop=True):
#         self.patience     = patience
#         self.min_ep       = min_ep
#         self.enable_stop  = enable_stop
#         self.cnt          = 0
#         self.stop         = False
#         self.bs           = float("inf")  # best score
#         self.ba           = float("inf")  # best ADE
#         self.b7           = float("inf")  # best 72h
#         self.bat          = float("inf")  # best ATE
#         self.bc           = float("inf")  # best CTE
#         self.best_easy_ade = float("inf") # [G-E] best easy ADE (EMA guard)

#     def update(self, r, model, out_dir, ep, opt, sched, tl, vl, tag="",
#                smooth_sched_step: int = 0,
#                global_hard_threshold: Optional[float] = None,
#                diversity_boost_factor: float = 1.0):
#         sc  = _score(r)
#         ade = r.get("ADE",     1e9)
#         h72 = r.get("72h",     1e9)
#         ate = r.get("ATE",     1e9)
#         cte = r.get("CTE",     1e9)
#         easy_ade = r.get("easy_ADE", ade)  # fallback to ADE if no split

#         any_improved = False

#         for v, attr, fn in [
#             (ade,      "ba",  f"best_ade_{tag}.pth"),
#             (h72,      "b7",  f"best_72h_{tag}.pth"),
#             (ate,      "bat", f"best_ate_{tag}.pth"),
#             (cte,      "bc",  f"best_cte_{tag}.pth"),
#         ]:
#             if v < getattr(self, attr):
#                 setattr(self, attr, v)
#                 _save_ckpt(os.path.join(out_dir, fn), ep, model, opt,
#                            sched, self, tl, vl,
#                            smooth_sched_step=smooth_sched_step,
#                            global_hard_threshold=global_hard_threshold,
#                            diversity_boost_factor=diversity_boost_factor)
#                 any_improved = True

#         if sc < self.bs:
#             self.bs = sc
#             any_improved = True
#             _save_ckpt(os.path.join(out_dir, f"best_{tag or 'composite'}.pth"),
#                        ep, model, opt, sched, self, tl, vl,
#                        smooth_sched_step=smooth_sched_step,
#                        global_hard_threshold=global_hard_threshold,
#                        diversity_boost_factor=diversity_boost_factor)
#             print(f"  [BEST] {tag} ep={ep} score={sc:.2f}"
#                   f"  ADE={ade:.1f}  72h={h72:.0f}"
#                   f"  ATE={ate:.1f}  CTE={cte:.1f}")

#         # [G-E] Update best easy_ADE
#         if np.isfinite(easy_ade) and easy_ade < self.best_easy_ade:
#             self.best_easy_ade = easy_ade

#         if self.enable_stop:
#             if any_improved: self.cnt = 0
#             else:
#                 self.cnt += 1
#                 print(f"  No improve {self.cnt}/{self.patience}"
#                       f"  best={self.bs:.2f}  cur={sc:.2f}")
#             if ep >= self.min_ep and self.cnt >= self.patience:
#                 self.stop = True


# def _mksub(ds, n, bs, cf):
#     idx = random.Random(42).sample(range(len(ds)), min(n, len(ds)))
#     return DataLoader(Subset(ds, idx), batch_size=bs, shuffle=False,
#                       collate_fn=cf, num_workers=0, drop_last=False)


# # ─────────────────────────────────────────────────────────────────────────────
# #  Args
# # ─────────────────────────────────────────────────────────────────────────────

# def get_args():
#     p = argparse.ArgumentParser(
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     p.add_argument("--dataset_root",       default="TCND_vn")
#     p.add_argument("--obs_len",            default=8,    type=int)
#     p.add_argument("--pred_len",           default=12,   type=int)
#     p.add_argument("--batch_size",         default=32,   type=int)
#     p.add_argument("--num_epochs",         default=100,  type=int)
#     p.add_argument("--learning_rate",      default=1e-4, type=float)
#     p.add_argument("--weight_decay",       default=1e-3, type=float)
#     p.add_argument("--warmup_epochs",      default=3,    type=int)
#     p.add_argument("--grad_clip",          default=1.0,  type=float)
#     p.add_argument("--use_amp",            action="store_true")
#     p.add_argument("--num_workers",        default=2,    type=int)
#     p.add_argument("--sigma_min",          default=0.02, type=float)
#     p.add_argument("--use_ot",             default=True, action="store_true")
#     p.add_argument("--no_ot",              dest="use_ot", action="store_false")
#     p.add_argument("--cfg_guidance_scale", default=1.5,  type=float)
#     p.add_argument("--n_ensemble",         default=50,   type=int)
#     p.add_argument("--use_ema",            default=True, action="store_true")
#     p.add_argument("--no_ema",             dest="use_ema", action="store_false")
#     p.add_argument("--ema_decay",          default=0.995, type=float)
#     p.add_argument("--patience",           default=40,   type=int)
#     p.add_argument("--min_ep",             default=35,   type=int)
#     p.add_argument("--val_freq",           default=3,    type=int)
#     p.add_argument("--val_subset_size",    default=500,  type=int)
#     p.add_argument("--fast_ddim",          default=10,   type=int)
#     p.add_argument("--full_ddim",          default=20,   type=int)
#     p.add_argument("--output_dir",         default="runs/v59strategy")
#     p.add_argument("--gpu_num",            default="0")
#     p.add_argument("--delim",              default=" ")
#     p.add_argument("--skip",               default=1,    type=int)
#     p.add_argument("--min_ped",            default=1,    type=int)
#     p.add_argument("--threshold",          default=0.002, type=float)
#     p.add_argument("--other_modal",        default="gph")
#     p.add_argument("--test_year",          default=None, type=int)
#     p.add_argument("--resume",             default=None)
#     p.add_argument("--resume_epoch",       default=None, type=int)
#     p.add_argument("--eval_test_after_train", default=True, action="store_true")
#     # GradNorm
#     p.add_argument("--gradnorm_alpha",     default=1.5,  type=float,
#                    help="GradNorm restoring force (giữ cho backward compat)")
#     p.add_argument("--gradnorm_lr",        default=1e-3, type=float,
#                    help="GradNorm lambda learning rate")
#     # EMA guard
#     p.add_argument("--ema_guard_threshold", default=20.0, type=float,
#                    help="smooth_easy_ADE tăng bao nhiêu km thì rollback (km)")
#     # Diversity check
#     p.add_argument("--diversity_threshold", default=50.0, type=float,
#                    help="diversity_score < threshold → cảnh báo R1")
#     # [FIX-DIV-2] Boost lặp sigma_min/ctx_noise_scale nếu diversity thấp.
#     # CẢNH BÁO TRADE-OFF: sigma_min/ctx_noise_scale chỉ ảnh hưởng sample()
#     # (inference) — KHÔNG ảnh hưởng train loss — nhưng boost quá mạnh có
#     # thể làm TỪNG candidate noisy hơn → có thể làm FAST/RAW ADE/CTE epoch
#     # 20+ xấu hơn epoch 19 (đánh đổi accuracy-per-candidate để lấy diversity
#     # cho selector). Nếu thấy easy_ADE ep20 tệ hơn rõ rệt so với ep19 do
#     # nguyên nhân này (không phải do phase-transition — đã fix ở FIX-GN-2),
#     # giảm --diversity_boost_max ở lần chạy sau (resume sẽ áp lại factor đã
#     # lưu, nên cần chạy lại từ ckpt trước ep19 hoặc từ đầu để đổi factor).
#     p.add_argument("--diversity_boost_step", default=1.8, type=float,
#                    help="hệ số nhân sigma_min/ctx_noise_scale mỗi vòng boost")
#     p.add_argument("--diversity_boost_max", default=6.0, type=float,
#                    help="cap tổng hệ số boost (so với giá trị gốc)")
#     p.add_argument("--diversity_boost_iters", default=3, type=int,
#                    help="số vòng boost+recheck tối đa")

#     # [FIX-DIV-3] L_diversity training loss (hinge, ngoài GradNorm).
#     # An toàn: bounded ∈[0, target_norm], =0 khi diversity đủ (self-limiting).
#     # Tốn thêm 1 forward pass qua transformer decoder (phần FNO3D/encoder
#     # KHÔNG chạy lại) khi active — chỉ active từ phase 2 (sigmoid ramp),
#     # tắt hoàn toàn ở phase 1 (mặc định weight=0 → KHÔNG tốn compute thêm).
#     p.add_argument("--use_diversity_loss", action="store_true", default=True,
#                    help="bật L_diversity training loss (FIX-DIV-3)")
#     p.add_argument("--no_diversity_loss", dest="use_diversity_loss",
#                    action="store_false",
#                    help="tắt L_diversity training loss hoàn toàn")
#     p.add_argument("--diversity_loss_weight", default=0.15, type=float,
#                    help="trọng số đích (sau ramp) cho L_diversity hinge")
#     p.add_argument("--diversity_target_km", default=None, type=float,
#                    help="target diversity (km) cho L_diversity hinge;"
#                         " mặc định = --diversity_threshold")
#     return p.parse_args()


# # ─────────────────────────────────────────────────────────────────────────────
# #  Main
# # ─────────────────────────────────────────────────────────────────────────────

# def main(args):
#     if torch.cuda.is_available():
#         os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
#     dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     os.makedirs(args.output_dir, exist_ok=True)

#     # [FIX-DIV-3] target diversity (km) cho L_diversity hinge — mặc định
#     # dùng cùng giá trị với --diversity_threshold (consistency: cùng
#     # ngưỡng "đủ đa dạng" cho cả check_diversity() và L_diversity).
#     _diversity_target_km = (args.diversity_target_km
#                             if args.diversity_target_km is not None
#                             else args.diversity_threshold)

#     print("=" * 72)
#     print("  TC-FlowMatching v59-Strategy [FIXED-v4]")
#     print("  [FIX-GN-3] GradNorm: anchor λ_dpe=1.20 cố định + target-ratio aux")
#     print("  [FIX-GN-2] phase_reset: KHÔNG reset λ (chỉ reset optimizer)")
#     print("  [FIX-EMA-2] EMA guard pullback: chỉ khi α<0.25 (tránh kẹt)")
#     print("  [FIX-PERF-1] Xóa dead computation (named_parameters mỗi step)")
#     print("  [FIX-CKPT-1] Lưu smooth_sched_step + threshold mọi checkpoint")
#     print("  [FIX-DIV-2] Boost lặp (cap configurable) + recheck diversity +"
#           " resume-persist")
#     print(f"  [FIX-DIV-3] L_diversity hinge (target={_diversity_target_km:.0f}km,"
#           f" w_target={args.diversity_loss_weight if args.use_diversity_loss else 0.0:.2f},"
#           f" ramp từ phase2) — "
#           f"{'ON' if args.use_diversity_loss else 'OFF'}")
#     print("  [G-A] Easy/Hard: global threshold (tính 1 lần)")
#     print("  [G-C] easy_frac ≥ 50% enforcement mỗi batch")
#     print("  [G-D] α warm-up: sigmoid, mid=15% phase (FIX-ALPHA-TIMING)")
#     print("  [G-E] EMA guard: EMA-smoothed easy_ADE, threshold 20km")
#     print("  [G-F] Diversity check sau phase 1")
#     print("  [G-G] Split monitoring: easy_ADE + hard_ADE")
#     print("  [G-H] Phase 4: freeze encoder")
#     print(f"  Phases: 1[0-{PHASE_1_END}] 2[{PHASE_1_END+1}-{PHASE_2_END}]"
#           f" 3[{PHASE_2_END+1}-{PHASE_3_END}] 4[{PHASE_3_END+1}+]")
#     print(f"  Target: ADE<{TARGETS['ADE']} 72h<{TARGETS['72h']}"
#           f" ATE<{TARGETS['ATE']} CTE<{TARGETS['CTE']}")
#     print("=" * 72)

#     # Data
#     trd, trl = data_loader(args, {"root": args.dataset_root, "type": "train"}, test=False)
#     vd,  vl  = data_loader(args, {"root": args.dataset_root, "type": "val"},   test=True)
#     from Model.data.trajectoriesWithMe_unet_training import seq_collate
#     vsub = _mksub(vd, args.val_subset_size, args.batch_size, seq_collate)
#     print(f"  train:{len(trd)} seqs  val:{len(vd)} seqs")

#     # Model
#     model = TCFlowMatching(
#         pred_len=args.pred_len, obs_len=args.obs_len,
#         sigma_min=args.sigma_min, use_ema=args.use_ema,
#         ema_decay=args.ema_decay, use_ate_ot=args.use_ot,
#         cfg_guidance_scale=args.cfg_guidance_scale,
#     ).to(dev)
#     model.init_ema()

#     n_total = sum(p.numel() for p in model.parameters() if p.requires_grad)

#     # BUG-8: backbone vs learned weights (step_weights, selector small params)
#     backbone_params = [p for p in model.parameters()
#                        if p.requires_grad and p.numel() > 100]
#     step_w_params   = [p for p in model.parameters()
#                        if p.requires_grad and p.numel() <= 12]

#     print(f"  params total: {n_total:,}")
#     print(f"  backbone (clipped at {args.grad_clip}): "
#           f"{sum(p.numel() for p in backbone_params):,}")
#     print(f"  step_weights (NOT clipped): "
#           f"{sum(p.numel() for p in step_w_params):,}")

#     # Optimizer & scheduler
#     opt    = optim.AdamW(model.parameters(),
#                           lr=args.learning_rate,
#                           weight_decay=args.weight_decay)
#     scaler = GradScaler("cuda", enabled=args.use_amp)
#     nstep  = len(trl)
#     total  = nstep * args.num_epochs
#     wstp   = nstep * args.warmup_epochs
#     sched  = get_cosine_schedule_with_warmup(opt, wstp, total, min_lr=1e-6)

#     # [G-A] GradNorm manager — equal-contribution scheme (FIX-GN-1)
#     gradnorm = GradNormManager(
#         terms=GRADNORM_TERMS,
#         alpha_gn=args.gradnorm_alpha,
#         lr=args.gradnorm_lr,
#         device=dev,
#     )

#     # SmoothScheduler — α theo batch, không phải epoch
#     smooth_sched = SmoothScheduler(total_steps_per_epoch=nstep)

#     # Savers
#     fast_saver = Saver(patience=args.patience, min_ep=args.min_ep,
#                         enable_stop=False)
#     full_saver = Saver(patience=args.patience, min_ep=args.min_ep,
#                         enable_stop=True)

#     # [FIX-THRESH-1] global threshold + flag riêng (thay vì == 0.5)
#     global_hard_threshold: Optional[float] = None
#     _threshold_computed = False

#     # [FIX-DIV-2] hệ số boost diversity hiện tại (1.0 = chưa boost).
#     # Cần khai báo TRƯỚC resume vì có thể được restore từ checkpoint.
#     diversity_boost_factor: float = 1.0

#     # Resume
#     start = 0
#     if args.resume and os.path.exists(args.resume):
#         print(f"  Loading: {args.resume}")
#         ck = torch.load(args.resume, map_location=dev)
#         m  = _unwrap(model)
#         ms, _ = m.load_state_dict(ck["model_state_dict"], strict=False)
#         if ms: print(f"  Missing: {len(ms)}")
#         ema = getattr(m, "_ema", None)
#         if ema and ck.get("ema_shadow"):
#             for k, v in ck["ema_shadow"].items():
#                 if k in ema.shadow: ema.shadow[k].copy_(v.to(dev))
#         try: opt.load_state_dict(ck["optimizer_state"])
#         except Exception as e: print(f"  Opt: {e}")
#         try: sched.load_state_dict(ck["scheduler_state"])
#         except:
#             for _ in range(ck.get("epoch", 0) * nstep): sched.step()
#         for a, attr in [("best_score","bs"), ("best_ade","ba"), ("best_72h","b7"),
#                         ("best_ate","bat"), ("best_cte","bc"),
#                         ("best_easy_ade","best_easy_ade")]:
#             if a in ck:
#                 setattr(full_saver, attr, ck[a])
#                 setattr(fast_saver, attr, ck[a])
#         start = args.resume_epoch or ck.get("epoch", 0) + 1
#         print(f"  Resume ep={start}  best_score={full_saver.bs:.2f}"
#               f"  best_easy_ADE={full_saver.best_easy_ade:.1f}")

#         # [FIX-CKPT-1] Restore smooth_sched step + threshold từ checkpoint
#         if "smooth_sched_step" in ck and ck["smooth_sched_step"] is not None:
#             smooth_sched._global_step = ck["smooth_sched_step"]
#             print(f"  Restored smooth_sched_step={ck['smooth_sched_step']}")
#         else:
#             smooth_sched._global_step = start * nstep
#             print(f"  Estimated smooth_sched_step={start * nstep} from epoch")

#         if "global_hard_threshold" in ck and ck["global_hard_threshold"] is not None:
#             global_hard_threshold = ck["global_hard_threshold"]
#             _threshold_computed = True
#             print(f"  Restored global_hard_threshold={global_hard_threshold:.4f}")

#         # [FIX-DIV-2] Restore hệ số boost diversity. sigma_min/ctx_noise_scale
#         # là plain float attribute trên model — KHÔNG nằm trong state_dict()
#         # nên model vừa load_state_dict ở trên vẫn đang ở giá trị CONSTRUCTOR
#         # (args.sigma_min, ctx_noise_scale=0.01 default). Nếu checkpoint có
#         # ghi nhận đã boost (factor > 1), áp lại NGAY tại đây — nếu không,
#         # boost sẽ bị mất vĩnh viễn vì block diversity-check chỉ chạy đúng
#         # 1 lần tại ep==PHASE_1_END (sẽ không chạy lại khi resume ở ep>19).
#         ck_div_factor = ck.get("diversity_boost_factor", 1.0)
#         if ck_div_factor and ck_div_factor > 1.0 + 1e-6:
#             diversity_boost_factor = float(ck_div_factor)
#             try:
#                 applied = []
#                 for attr_name, base_val in (("sigma_min", args.sigma_min),
#                                              ("ctx_noise_scale", 0.01)):
#                     if hasattr(m, attr_name):
#                         setattr(m, attr_name, base_val * diversity_boost_factor)
#                         applied.append(f"{attr_name}="
#                                        f"{getattr(m, attr_name):.4f}")
#                 print(f"  [FIX-DIV-2] Restored diversity boost factor="
#                       f"{diversity_boost_factor:.2f}x ({', '.join(applied)})")
#             except Exception as e:
#                 print(f"  [FIX-DIV-2] Lỗi khi restore diversity boost: {e}")

#     try:
#         model = torch.compile(model, mode="reduce-overhead")
#         print("  torch.compile: ok")
#     except:
#         pass

#     # ── Phase 4 encoder list (dùng khi freeze) ────────────────────────────
#     def _get_encoder_params(m):
#         """Trả về params của encoder để freeze ở phase 4."""
#         enc = _unwrap(m)
#         params = []
#         for mod_name in ["net.spatial_enc", "net.enc_1d", "net.env_enc",
#                           "net.bottleneck_pool", "net.bottleneck_proj",
#                           "net.decoder_proj"]:
#             parts = mod_name.split(".")
#             mod = enc
#             for p in parts:
#                 if hasattr(mod, p):
#                     mod = getattr(mod, p)
#                 else:
#                     mod = None
#                     break
#             if mod is not None:
#                 params.extend(list(mod.parameters()))
#         return params

#     _phase4_frozen = False

#     def _maybe_freeze_encoder(ep, m):
#         nonlocal _phase4_frozen
#         if get_phase(ep) == 4 and not _phase4_frozen:
#             enc_params = _get_encoder_params(m)
#             for p in enc_params:
#                 p.requires_grad_(False)
#             _phase4_frozen = True
#             print(f"  [Phase 4] Froze encoder params"
#                   f" (n={sum(1 for p in enc_params)})")
#             for g in opt.param_groups:
#                 g["lr"] = g["lr"] * 0.1
#             print(f"  [Phase 4] LR reduced to {opt.param_groups[0]['lr']:.2e}")

#     # ── EMA guard state ────────────────────────────────────────────────────
#     best_easy_ade_ema      = float("inf")
#     ema_rollback_count     = 0
#     _prev_phase            = get_phase(start) if start > 0 else 1
#     # EMA của easy_ADE — robust hơn raw value với variance fast_eval ±20-40km
#     _ema_state  = {"smooth": float("inf")}
#     EMA_ALPHA   = 0.35   # weight cho current value (bias về recent)
#     EMA_THRESH  = args.ema_guard_threshold   # km threshold để trigger rollback

#     # [FIX-DIV-2] flag để chỉ áp dụng diversity-noise-boost 1 lần.
#     # Nếu đã restore boost factor>1 từ checkpoint (resume), coi như đã
#     # boosted — không chạy lại block boost (vốn cũng chỉ chạy ở ep==19).
#     _diversity_boosted = diversity_boost_factor > 1.0 + 1e-6

#     print(f"  Training: {nstep} steps/ep, start ep={start}")
#     print("=" * 72)
#     ts = time.perf_counter()

#     for ep in range(start, args.num_epochs):
#         phase = get_phase(ep)

#         # [FIX-GN-2] GradNorm phase transition — KHÔNG reset λ
#         if phase != _prev_phase:
#             gradnorm.phase_reset(phase)
#             _prev_phase = phase
#             print(f"  [Phase {phase}] Chuyển phase tại epoch {ep}")

#             # Reset best_easy_ade_ema và smooth EMA khi chuyển phase
#             # (baseline easy_ADE giữa các phase có thể khác nhau tự nhiên,
#             #  không nên coi sự khác biệt này là "regression")
#             best_easy_ade_ema      = float("inf")
#             ema_rollback_count     = 0
#             _ema_state["smooth"]   = float("inf")
#             print(f"  [EMA Guard] Reset best/smooth EMA khi chuyển phase")

#         # [FIX-THRESH-1] Chỉ compute hard threshold 1 lần khi vào phase 2.
#         if phase >= 2 and not _threshold_computed:
#             global_hard_threshold = precompute_hard_threshold(trl, dev, n_batches=50)
#             _threshold_computed = True
#             print(f"  [FIX-THRESH-1] global_hard_threshold={global_hard_threshold:.4f}"
#                   f" — computed once, reused for all epochs")

#         # [G-H] Phase 4: freeze encoder
#         _maybe_freeze_encoder(ep, model)

#         model.train()
#         sl = 0.0
#         t0 = time.perf_counter()
#         # [FIX-DIV-3] accumulate diversity_km_train (chỉ tính step finite)
#         div_km_sum, div_km_cnt = 0.0, 0

#         # Lấy lambda hiện tại từ GradNorm
#         lambda_dict = gradnorm.get_lambda_dict()

#         for i, batch in enumerate(trl):
#             bl    = move(list(batch), dev)
#             obs_t = bl[0]

#             # α, easy_frac từ SmoothScheduler (per-batch)
#             alpha_hard    = smooth_sched.get_alpha_hard()
#             easy_frac     = smooth_sched.get_easy_frac()
#             sel_weight    = smooth_sched.get_selector_weight()
#             train_selector = smooth_sched.is_selector_active() and (sel_weight > 0.01)

#             # Dùng global threshold (phase>=2) hoặc zeros (phase 1)
#             if phase >= 2 and global_hard_threshold is not None:
#                 with torch.no_grad():
#                     is_hard = classify_hard_easy_global(
#                         obs_t[:, :, :2], global_hard_threshold).to(dev)
#                 is_hard = enforce_easy_frac(is_hard, easy_frac)
#             else:
#                 # Phase 1: alpha_hard=0 → hard loss = 0, không cần phân loại
#                 is_hard = torch.zeros(
#                     obs_t.shape[1], dtype=torch.bool, device=dev)

#             # [FIX-DIV-3] L_diversity weight (0 ở phase 1, ramp từ phase 2)
#             div_loss_w = (smooth_sched.get_diversity_weight(
#                               target=args.diversity_loss_weight)
#                           if args.use_diversity_loss else 0.0)

#             with autocast(device_type="cuda", enabled=args.use_amp):
#                 bd = model.get_loss_breakdown(
#                     bl,
#                     epoch=ep,
#                     alpha_hard=alpha_hard,
#                     is_hard=is_hard,
#                     train_selector=train_selector,
#                     lambda_dict=lambda_dict,
#                     diversity_loss_weight=div_loss_w,
#                     diversity_target_km=_diversity_target_km,
#                 )

#             opt.zero_grad()
#             scaler.scale(bd["total"]).backward()
#             scaler.unscale_(opt)

#             # [FIX-GN-1 + FIX-PERF-1] GradNorm: equal-contribution scheme.
#             # Chỉ cần loss VALUES (raw, detached) — không cần gradient norm
#             # của shared params (xóa hoàn toàn block named_parameters() cũ,
#             # vốn lặp qua TOÀN BỘ params mỗi step nhưng kết quả bị ignore).
#             try:
#                 loss_vals_float = {}
#                 for t in GRADNORM_TERMS:
#                     val = bd.get(t)
#                     if val is None:
#                         continue
#                     fval = float(val.item()) if torch.is_tensor(val) else float(val)
#                     if np.isfinite(fval) and fval > 0:
#                         loss_vals_float[t] = fval

#                 if loss_vals_float:
#                     gradnorm.update(loss_vals_float)
#                     lambda_dict = gradnorm.get_lambda_dict()
#             except Exception:
#                 pass  # GradNorm không critical

#             # BUG-8: clip chỉ backbone
#             torch.nn.utils.clip_grad_norm_(backbone_params, args.grad_clip)

#             scaler.step(opt)
#             scaler.update()
#             # BUG-7: unconditional
#             try: sched.step()
#             except: pass
#             model.ema_update()

#             smooth_sched.step()

#             sl += bd["total"].item()

#             # [FIX-DIV-3] track training-time diversity (nan khi tắt/lỗi)
#             _dkm = bd.get("diversity_km_train", float("nan"))
#             if np.isfinite(_dkm):
#                 div_km_sum += _dkm
#                 div_km_cnt += 1

#             if i % 20 == 0:
#                 lr     = opt.param_groups[0]["lr"]
#                 tot    = bd["total"].item()
#                 warn   = " [!>10]" if tot > 10.0 else ""
#                 sw_st  = _unwrap(model).step_weights.stats()
#                 gn_log = gradnorm.log_stats()
#                 n_hard_batch = int(is_hard.sum().item())

#                 lam_str = " ".join(
#                     f"λ_{t.replace('l_','').replace('_reg','')}="
#                     f"{gn_log.get(f'λ_{t}', 1.0):.2f}"
#                     for t in GRADNORM_TERMS
#                 )

#                 print(
#                     f"  [{ep:>3}|Ph{phase}][{i:>3}/{nstep}]"
#                     f"  tot={tot:.3f}{warn}"
#                     f"  fm={bd.get('l_fm',0):.4f}"
#                     f"  dpe={bd.get('dpe',0):.4f}"
#                     f"  vel={bd.get('vel_reg',0):.4f}"
#                     f"  hard={bd.get('l_hard_total',0):.4f}"
#                     f"  {smooth_sched.log_stats()}"
#                     f"  n_h={n_hard_batch}/{obs_t.shape[1]}"
#                     f"  sw72={sw_st['sw_72h']:.2f}"
#                     f"  gn_d={gn_log.get('gn_dense',0)}"
#                     f"  {lam_str}"
#                     f"  lr={lr:.2e}"
#                     f"  div_w={div_loss_w:.3f}"
#                     f"  div_km={_dkm:.1f}"
#                 )

#         avt  = sl / nstep
#         t_ep = time.perf_counter() - t0

#         # BUG-5: val với epoch=-1 → skip augmentation
#         model.eval()
#         vls = 0.0
#         with torch.no_grad():
#             for batch in vl:
#                 bv = move(list(batch), dev)
#                 with autocast(device_type="cuda", enabled=args.use_amp):
#                     vls += model.get_loss(bv, epoch=-1).item()
#         avv = vls / len(vl)

#         lr_cur   = opt.param_groups[0]["lr"]
#         gn_stats = gradnorm.log_stats()
#         use_sel_eval  = smooth_sched.is_selector_active()
#         eval_threshold = global_hard_threshold if phase >= 2 else None

#         lam_epoch = " ".join(
#             f"λ_{t.replace('l_','').replace('_reg','')}="
#             f"{gn_stats.get(f'λ_{t}', 1.0):.3f}"
#             for t in GRADNORM_TERMS
#         )

#         # [FIX-DIV-3] avg training-time diversity (nan nếu weight=0 cả epoch)
#         avg_div_km_train = (div_km_sum / div_km_cnt) if div_km_cnt > 0 else float("nan")

#         print(f"  Epoch {ep:>3}|Ph{phase}"
#               f" | train={avt:.4f} val={avv:.4f}"
#               f" | {smooth_sched.log_stats()}"
#               f" | {lam_epoch}"
#               f" | div_km_train={avg_div_km_train:.1f}"
#               f" | lr={lr_cur:.2e} | {t_ep:.0f}s")

#         # [G-G] Fast eval với split monitoring (dùng fast ddim)
#         rf = evaluate_split(model, vsub, dev,
#                              tag=f"FAST ep{ep}",
#                              steps=args.fast_ddim,
#                              use_selector=use_sel_eval,
#                              global_hard_threshold=eval_threshold)
#         fast_saver.update(rf, model, args.output_dir, ep, opt, sched, avt, avv,
#                           tag="fast",
#                           smooth_sched_step=smooth_sched.global_step,
#                           global_hard_threshold=global_hard_threshold,
#                           diversity_boost_factor=diversity_boost_factor)

#         # ── [G-E + FIX-EMA-2] EMA guard: kiểm tra easy_ADE sau mỗi epoch ──
#         #
#         # [FIX-EMA-2] Vấn đề cũ: pullback_steps=nstep*2, rollback dày đặc
#         # trong vùng α-ramp (ramp chỉ rộng ~35 step quanh điểm mid) → mỗi
#         # rollback trong vùng ramp kéo step về TRƯỚC vùng ramp → α=0 lại
#         # → α=0.000 suốt cả phase 2 (xem log gốc, 30 epoch toàn α=0).
#         #
#         # Fix:
#         #   - pullback_steps = nstep // 2 (giảm 1 nửa so với nstep*2)
#         #   - CHỈ pullback schedule nếu α hiện tại < ALPHA_RAMP_DONE_THRESHOLD
#         #     (còn trong vùng ramp). Nếu α đã ramp xong (>=0.25), rollback
#         #     chỉ restore weights, KHÔNG pullback schedule — tránh kẹt
#         #     vĩnh viễn do rollback lặp lại sau khi đã vượt vùng ramp.
#         easy_ade_curr = rf.get("easy_ADE", rf.get("ADE", float("inf")))
#         if np.isfinite(easy_ade_curr):
#             # Update EMA
#             if _ema_state["smooth"] == float("inf"):
#                 _ema_state["smooth"] = easy_ade_curr
#             else:
#                 _ema_state["smooth"] = ((1 - EMA_ALPHA) * _ema_state["smooth"]
#                                         + EMA_ALPHA * easy_ade_curr)
#             smooth_val = _ema_state["smooth"]

#             if smooth_val < best_easy_ade_ema:
#                 best_easy_ade_ema = smooth_val
#                 print(f"  [EMA Guard] New best smooth_easy_ADE={smooth_val:.1f}"
#                       f" (raw={easy_ade_curr:.1f})")

#             elif (smooth_val - best_easy_ade_ema) > EMA_THRESH:
#                 ema_rollback_count += 1
#                 print(f"\n  [EMA GUARD] smooth_easy_ADE tăng bền vững "
#                       f"{smooth_val - best_easy_ade_ema:.1f} km"
#                       f" (best={best_easy_ade_ema:.1f} smooth={smooth_val:.1f}"
#                       f" raw={easy_ade_curr:.1f})")
#                 print(f"  [EMA GUARD] Rollback #{ema_rollback_count}")

#                 # Restore weights từ EMA
#                 m_raw   = _unwrap(model)
#                 ema_obj = getattr(m_raw, "_ema", None)
#                 if ema_obj:
#                     try:
#                         bk = ema_obj.apply_to(model)
#                         _save_ckpt(
#                             os.path.join(args.output_dir,
#                                          f"ema_guard_rollback_ep{ep}.pth"),
#                             ep, model, opt, sched, full_saver, avt, avv,
#                             smooth_sched_step=smooth_sched.global_step,
#                             global_hard_threshold=global_hard_threshold,
#                             diversity_boost_factor=diversity_boost_factor,
#                             extra={"ema_rollback": True,
#                                    "easy_ade_trigger": easy_ade_curr},
#                         )
#                         ema_obj.restore(model, bk)
#                         print(f"  [EMA GUARD] Saved rollback ckpt, "
#                               f"restored current weights for continued training")
#                     except Exception as e:
#                         print(f"  [EMA GUARD] Apply/restore failed: {e}")

#                 # [FIX-EMA-2] Chỉ pullback schedule nếu α còn trong vùng ramp
#                 cur_alpha = smooth_sched.get_alpha_hard()
#                 if cur_alpha < ALPHA_RAMP_DONE_THRESHOLD:
#                     pullback_steps = nstep // 2  # giảm so với nstep*2
#                     smooth_sched._global_step = max(
#                         smooth_sched._phase2_start_step,
#                         smooth_sched._global_step - pullback_steps
#                     )
#                     print(f"  [EMA GUARD] α còn trong vùng ramp ({cur_alpha:.3f}"
#                           f" < {ALPHA_RAMP_DONE_THRESHOLD}) → pullback"
#                           f" {pullback_steps} step. α mới="
#                           f"{smooth_sched.get_alpha_hard():.3f}")
#                 else:
#                     print(f"  [EMA GUARD] α đã ramp xong ({cur_alpha:.3f}"
#                           f" >= {ALPHA_RAMP_DONE_THRESHOLD}) → KHÔNG pullback"
#                           f" schedule, chỉ restore weights")

#         # [G-F + FIX-DIV-2] Diversity check cuối phase 1
#         div_score = None
#         if ep == PHASE_1_END:
#             print(f"\n  {'='*50}")
#             print(f"  [CHECKPOINT] Cuối phase 1 — kiểm tra diversity...")
#             baseline_easy_ade = rf.get("easy_ADE", float("nan"))
#             print(f"  (baseline FAST easy_ADE trước boost: "
#                   f"{baseline_easy_ade:.1f}km — so sánh với ep{ep+1}"
#                   f" FAST easy_ADE để kiểm tra trade-off boost)")
#             div_score = check_diversity(model, vsub, dev)
#             print(f"  Diversity score: {div_score:.1f} km")
#             if div_score < args.diversity_threshold:
#                 print(f"  [CẢNH BÁO R1] diversity={div_score:.1f} km"
#                       f" < {args.diversity_threshold} km")
#                 print(f"  → FM candidates không đủ đa dạng → selector sẽ kém")

#                 # [FIX-DIV-2] sigma_min/ctx_noise_scale CHỈ ảnh hưởng
#                 # sample() (initial DDIM noise + CFG context noise 3 step
#                 # đầu) — KHÔNG ảnh hưởng train CFM loss (_cfm_noisy dùng
#                 # current_sigma từ _sigma_schedule(epoch), không dùng
#                 # self.sigma_min; forward_with_ctx lúc train không truyền
#                 # noise_scale). → an toàn để boost MẠNH hơn FIX-DIV-1 (×1.5).
#                 # Boost LẶP + recheck thật, cap tổng diversity_boost_max,
#                 # tối đa diversity_boost_iters vòng.
#                 #
#                 # TRADE-OFF: boost làm TỪNG candidate noisy hơn → có thể
#                 # làm FAST/RAW ADE/CTE từ ep20 xấu hơn ep19 (đổi accuracy
#                 # per-candidate lấy diversity cho selector phase 3). So
#                 # sánh easy_ADE ep19 vs ep20 (FAST eval) sau khi có log:
#                 # nếu ep20 tệ hơn rõ rệt KHÔNG do phase-shock (đã fix ở
#                 # FIX-GN-2/FIX-EMA-2), giảm --diversity_boost_max lần sau.
#                 if not _diversity_boosted:
#                     try:
#                         m_div = _unwrap(model)
#                         boost_attrs = [a for a in ("sigma_min", "ctx_noise_scale")
#                                        if hasattr(m_div, a)]
#                         if not boost_attrs:
#                             print(f"  [FIX-DIV-2] Bỏ qua: model không có"
#                                   f" attribute sigma_min/ctx_noise_scale"
#                                   f" trực tiếp — cần xử lý thủ công"
#                                   f" (vd thêm L_diversity loss trong model).")
#                         else:
#                             base_vals = {a: getattr(m_div, a) for a in boost_attrs}
#                             BOOST_STEP     = args.diversity_boost_step
#                             MAX_TOTAL_BOOST = args.diversity_boost_max
#                             MAX_ITERS      = args.diversity_boost_iters
#                             total_boost = 1.0
#                             for it in range(1, MAX_ITERS + 1):
#                                 total_boost = min(total_boost * BOOST_STEP,
#                                                    MAX_TOTAL_BOOST)
#                                 for a in boost_attrs:
#                                     setattr(m_div, a, base_vals[a] * total_boost)
#                                 div_score = check_diversity(model, vsub, dev)
#                                 attr_str = ", ".join(
#                                     f"{a}={getattr(m_div, a):.4f}"
#                                     for a in boost_attrs)
#                                 print(f"  [FIX-DIV-2] vòng {it}: "
#                                       f"total_boost={total_boost:.2f}x "
#                                       f"({attr_str}) → diversity={div_score:.1f}km")
#                                 if (div_score >= args.diversity_threshold
#                                         or total_boost >= MAX_TOTAL_BOOST):
#                                     break
#                             # [FIX-DIV-2] Lưu factor để thread vào checkpoint
#                             # (resume sẽ áp lại, xem khối resume phía trên).
#                             diversity_boost_factor = total_boost
#                             if div_score >= args.diversity_threshold:
#                                 print(f"  [FIX-DIV-2] OK: diversity={div_score:.1f}km"
#                                       f" >= {args.diversity_threshold}km"
#                                       f" sau boost {total_boost:.2f}x")
#                             else:
#                                 print(f"  [FIX-DIV-2] VẪN CHƯA ĐỦ sau boost"
#                                       f" {total_boost:.2f}x (cap {MAX_TOTAL_BOOST}x):"
#                                       f" diversity={div_score:.1f}km"
#                                       f" < {args.diversity_threshold}km."
#                                       f" → cần can thiệp sâu hơn (L_diversity"
#                                       f" loss trong model, hoặc tăng"
#                                       f" n_ensemble/num_ensemble).")
#                         _diversity_boosted = True
#                     except Exception as e:
#                         print(f"  [FIX-DIV-2] Lỗi khi áp dụng noise boost: {e}")
#             else:
#                 print(f"  [OK] diversity={div_score:.1f} km >= "
#                       f"{args.diversity_threshold} km → tiến hành phase 2")
#             print(f"  {'='*50}\n")


#         # Đánh giá đầy đủ mỗi val_freq epoch
#         if ep % args.val_freq == 0:
#             em = getattr(_unwrap(model), "_ema", None)
#             rr = evaluate_split(model, vl, dev,
#                                   tag=f"RAW ep{ep}",
#                                   steps=args.full_ddim,
#                                   use_selector=use_sel_eval,
#                                   global_hard_threshold=eval_threshold)
#             full_saver.update(rr, model, args.output_dir, ep, opt, sched,
#                               avt, avv, tag="raw",
#                               smooth_sched_step=smooth_sched.global_step,
#                               global_hard_threshold=global_hard_threshold,
#                               diversity_boost_factor=diversity_boost_factor)

#             if em and ep >= 3:
#                 re = evaluate_split(model, vl, dev,
#                                      tag=f"EMA ep{ep}",
#                                      ema=em,
#                                      steps=args.full_ddim,
#                                      use_selector=use_sel_eval,
#                                      global_hard_threshold=eval_threshold)
#                 full_saver.update(re, model, args.output_dir, ep, opt, sched,
#                                   avt, avv, tag="ema",
#                                   smooth_sched_step=smooth_sched.global_step,
#                                   global_hard_threshold=global_hard_threshold,
#                                   diversity_boost_factor=diversity_boost_factor)

#         # Periodic checkpoint
#         if ep % 10 == 0 or ep == args.num_epochs - 1:
#             _save_ckpt(
#                 os.path.join(args.output_dir, f"ckpt_ep{ep:03d}.pth"),
#                 ep, model, opt, sched, full_saver, avt, avv,
#                 smooth_sched_step=smooth_sched.global_step,
#                 global_hard_threshold=global_hard_threshold,
#                 diversity_boost_factor=diversity_boost_factor,
#                 extra={"phase": phase,
#                        "alpha_hard": smooth_sched.get_alpha_hard(),
#                        "diversity": div_score},
#             )

#         if full_saver.stop:
#             print(f"  Early stop ep={ep}")
#             break

#     th = (time.perf_counter() - ts) / 3600.0
#     print(f"\n  Best: ADE={full_saver.ba:.1f}  72h={full_saver.b7:.0f}"
#           f"  ATE={full_saver.bat:.1f}  CTE={full_saver.bc:.1f}"
#           f"  easy_ADE={full_saver.best_easy_ade:.1f}"
#           f"  ({th:.2f}h)")

#     # Post-training test
#     if args.eval_test_after_train:
#         print("\n" + "="*72 + "\n  POST-TRAINING TEST\n" + "="*72)
#         try: _, tl2 = data_loader(args, {"root": args.dataset_root, "type": "test"}, test=True)
#         except: print("  No test → using val"); tl2 = vl

#         for fn, lb in [("best_raw.pth", "RAW"), ("best_ema.pth", "EMA"),
#                         ("best_ade_raw.pth", "BEST_ADE"),
#                         ("best_cte_raw.pth", "BEST_CTE")]:
#             pp = os.path.join(args.output_dir, fn)
#             if not os.path.exists(pp): continue
#             ck = torch.load(pp, map_location=dev)
#             _unwrap(model).load_state_dict(ck["model_state_dict"], strict=False)
#             em = getattr(_unwrap(model), "_ema", None)
#             if em and ck.get("ema_shadow"):
#                 for k, v in ck["ema_shadow"].items():
#                     if k in em.shadow: em.shadow[k].copy_(v.to(dev))
#             ckpt_threshold = ck.get("global_hard_threshold", None)
#             r = evaluate_split(model, tl2, dev, tag=f"TEST/{lb}",
#                                 steps=args.full_ddim,
#                                 use_selector=(get_phase(ck.get("epoch", 80)) >= 3),
#                                 global_hard_threshold=ckpt_threshold)
#             print(f"\n  --- {lb} (ep={ck.get('epoch','?')}) ---")
#             for key, ref in [("ADE", 172.68), ("72h", 321.39),
#                               ("ATE", 142.21), ("CTE", 42.04)]:
#                 v = r.get(key, float("nan"))
#                 mk = "BEAT!" if np.isfinite(v) and v < ref else f"need {ref:.0f}"
#                 print(f"    {key:<8}: {v:>8.1f} km  [{mk}  gap:{v-ref:+.1f}]")
#             ea = r.get("easy_ADE", float("nan"))
#             ha = r.get("hard_ADE", float("nan"))
#             print(f"    easy_ADE={ea:.1f}  hard_ADE={ha:.1f}")

#     print("=" * 72)


# if __name__ == "__main__":
#     args = get_args()
#     np.random.seed(42)
#     torch.manual_seed(42)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(42)
#     main(args)

"""
train_v59_strategy.py — TC-FlowMatching v59-Strategy Training Script [FIXED-v2]
══════════════════════════════════════════════════════════════════════════════

BASE: phiên bản với FIX-STABILITY-1/2/3, FIX-ALPHA-TIMING, FIX-EMA-SMOOTH, FIX-D
(đã chạy và có log epoch 0-51)

CÁC BUG MỚI ĐƯỢC FIX TRONG PHIÊN BẢN NÀY (dựa trên phân tích log thực tế):

  [FIX-GN-1] GradNorm "loss-ratio" KHÔNG có equilibrium → runaway có hệ thống
    Vấn đề cũ: r_i = raw_loss_i / mean(raw_loss_j) — KHÔNG phụ thuộc λ_i.
    l_dpe có raw scale lớn hơn cấu trúc so với l_vel_reg/l_heading/l_accel
    → r_dpe > 1 vĩnh viễn → λ_dpe luôn bị "tăng" → chạy tới ceiling (4.18)
    → λ_vel/head/accel bị đẩy xuống floor (0.01) → 3 objective gần như bị tắt.
    Log xác nhận: λ_dpe 1.00→4.18 (ep20→46), λ_vereg/heading/accel →0.01.

    Fix: "equal-contribution" scheme.
      contribution_i = λ_i * l_i  (phần đóng góp THỰC vào total loss)
      ratio_i = contribution_i / mean(contribution_j)
      ratio_i > 1 (đóng góp quá nhiều) → GIẢM λ_i
      ratio_i < 1 (đóng góp quá ít)    → TĂNG λ_i
    → có feedback âm thật: λ_i tăng → contribution_i tăng → ratio_i tăng
      → bị kéo giảm lại → equilibrium λ_i ≈ mean_contrib / l_i.
    Tự động bù trừ scale khác nhau giữa các loss term, không runaway.

  [FIX-GN-2] phase_reset() KHÔNG reset λ về 1.0 nữa
    Vấn đề cũ: λ_dpe=3.11→1.0 đột ngột tại ep20 (và tương tự ep50)
    → tổng loss đổi đột ngột → easy_ADE nhảy 278→369km (ep20), 278→302 (ep51).
    L_base (compute_st_trans_loss) không đổi giữa các phase — chỉ cộng thêm
    L_hard/L_sel bên ngoài GradNorm → không có lý do để reset λ.
    Fix: giữ nguyên λ hiện tại, chỉ reset optimizer momentum + dense window
    ngắn (50 step) để re-adapt nhẹ nếu landscape có dịch nhẹ.

  [FIX-EMA-2] EMA guard pullback có thể làm α kẹt vĩnh viễn ở 0
    Vấn đề cũ: pullback_steps=nstep*2, rollback dày đặc trong vùng α-ramp
    (ramp chỉ rộng ~35 step quanh epoch 24.5) → mỗi rollback trong vùng ramp
    kéo step về trước vùng ramp → α=0 lại → α=0.000 suốt cả phase 2 (30 epoch).
    Fix: chỉ pullback schedule nếu α hiện tại < 0.25 (còn trong vùng ramp).
    Nếu α đã ramp xong (>=0.25), rollback chỉ restore weights, KHÔNG pullback
    schedule. Giảm pullback_steps = nstep//2.

  [FIX-PERF-1] Xóa dead/wasteful computation mỗi step
    Code cũ lặp qua TOÀN BỘ named_parameters() mỗi step để tìm
    "net.ctx_fc2.weight", tính lambda_grads_approx rồi truyền vào update()
    — nhưng update() ghi rõ "lambda_grads: unused, ignore". Hoàn toàn lãng phí.
    Fix: xóa block này, chỉ extract loss_vals_float và gọi update() trực tiếp.

  [FIX-CKPT-1] _save_ckpt lưu smooth_sched_step + global_hard_threshold
    cho MỌI checkpoint (trước chỉ có trong periodic ckpt's "extra").

  [FIX-DIV-1] Diversity collapse (7.8km << 50km) — tự động xử lý
    Theo strategy doc mục R1: "Tăng noise std ×1.5".
    Fix: tự động ×1.5 sigma_min và ctx_noise_scale của model khi
    diversity < threshold sau phase 1.

  [FIX-THRESH-1] precompute threshold dùng flag riêng _threshold_computed
    thay vì so sánh == 0.5 (fragile).

  [FIX-EASYFRAC4] get_easy_frac() phase 4 dùng sigmoid (consistent với phase 2/3)

CÁC BUG MỚI ĐƯỢC FIX TRONG v3 (dựa trên log thực tế chạy v2, epoch 0-6):

  [FIX-GN-3] "equal-contribution" (FIX-GN-1) vẫn SAI — thay bằng
    "anchor + target-ratio".

    Log v2 ep0-6: λ_dpe GIẢM liên tục 1.00→0.335 (đang đi về floor 0.1),
    λ_vel/head/accel TĂNG 1.00→1.17-1.20. val loss giảm đẹp (2.42→2.34,
    THẤP NHẤT) nhưng easy_ADE TĂNG GẦN GẤP ĐÔI (408.9km log gốc → 705.7km)
    và ADE trend ep2-6 ĐANG TĂNG (574→598→575→619→649). EMA Guard rollback
    ngay tại ep6.

    Root cause: l_dpe (Huber loss vị trí — objective CHÍNH, gắn trực tiếp
    với ADE) có raw scale LỚN HƠN l_vel_reg/heading/accel KHÔNG PHẢI do lệch
    scale ngẫu nhiên, mà vì đây là objective khó giảm hơn (regularizer phụ
    dễ về gần 0). "Equal-contribution" coi đó là mất cân bằng cần sửa →
    đẩy λ_dpe XUỐNG để "cân bằng" với 4 regularizer phụ → model dồn capacity
    tối ưu 4 term phụ "dễ" (val loss giảm đẹp) nhưng hy sinh độ chính xác vị
    trí → ADE tăng. Cùng pattern với log gốc ep49→50 (λ_dpe reset 3.92→1.0
    đột ngột → easy_ADE 220→263km, "No improve" liên tục 4 epoch).

    Fix: l_dpe = ANCHOR, λ_dpe CỐ ĐỊNH = 1.20 (= default hand-tuned trong
    model, KHÔNG BAO GIỜ bị GradNorm điều chỉnh) → gradient cho vị trí
    không bao giờ bị giảm dưới mức hand-tuned. 4 term phụ (vel_reg, heading,
    speed, accel) được GradNorm điều chỉnh để duy trì TỶ LỆ ĐÓNG GÓP MỤC
    TIÊU so với (λ_dpe·l_dpe), lấy từ default weights hand-tuned trong
    model gốc (1.20, 1.40, 0.40, 0.05, 0.01):
        target_contrib_i = DEFAULT_LAMBDA[i] * l_dpe
        ratio_i = (λ_i · l_i) / target_contrib_i
        ratio_i > 1 → giảm λ_i ; ratio_i < 1 → tăng λ_i
    Equilibrium: λ_i ≈ DEFAULT_LAMBDA[i] · l_dpe / l_i — tự bù trừ scale RAW
    của term i trôi theo training (đúng mục đích GradNorm), nhưng KHÔNG
    đụng l_dpe và tôn trọng tỷ lệ quan trọng đã hand-tune giữa các term phụ.
    Không cần renorm sum=5 nữa (anchor cố định, không có sum constraint).
    Clamp λ_aux ∈ [0.02, 8.0] (range rộng hơn vì target_ratio chênh nhau
    0.008–1.167 giữa các term).

  [FIX-DIV-2] FIX-DIV-1 (×1.5 noise cố định) KHÔNG đủ.
    Log gốc: diversity=8.3km << 50km (gap ~6x), ×1.5 → ~12.5km vẫn cách xa.
    sigma_min và ctx_noise_scale CHỈ ảnh hưởng sample()/inference (initial
    DDIM noise + CFG context noise 3 step đầu) — KHÔNG ảnh hưởng training
    loss (_cfm_noisy dùng current_sigma từ _sigma_schedule(epoch), không
    dùng self.sigma_min; forward_with_ctx lúc train không truyền noise_scale)
    → có thể boost mạnh hơn nhiều mà không ảnh hưởng ổn định training.
    Fix: boost LẶP — mỗi vòng ×1.8 (sigma_min và ctx_noise_scale), sau đó
    chạy lại check_diversity(); lặp tối đa 3 vòng hoặc đến khi
    diversity >= threshold; tổng boost factor cap ở 6.0x để tránh initial
    noise quá lớn làm DDIM không hội tụ về trajectory hợp lý.

GIỮ NGUYÊN từ phiên bản trước:
  [G-A] Easy/hard pipeline với GLOBAL threshold
  [G-C] easy_frac enforcement
  [G-D] α smooth warm-up theo BATCH (FIX-ALPHA-TIMING: mid=15% of phase)
  [G-E] EMA guard với EMA-smoothed easy_ADE (FIX-EMA-SMOOTH)
  [G-F] Diversity check sau phase 1
  [G-G] Split monitoring
  [G-H] Phase 4: freeze encoder
  BUG-5/6/7/8 từ v59-tweaked
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse, math, random, time
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset

from Model.data.loader_training import data_loader
from Model.flow_matching_model import (
    TCFlowMatching,
    hard_score_from_obs,       # dùng trong precompute_hard_threshold
    classify_hard_easy,        # dùng trong evaluate_split
    compute_diversity_score,   # dùng trong check_diversity
    _norm_to_deg,
    _haversine_deg,
)

try:
    from Model.utils import get_cosine_schedule_with_warmup
except ImportError:
    from torch.optim.lr_scheduler import CosineAnnealingLR
    def get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps, min_lr=1e-6):
        return CosineAnnealingLR(opt, T_max=max(total_steps, 1), eta_min=min_lr)

TARGETS = {
    "ADE": 172.68, "72h": 321.39, "ATE": 142.21, "CTE": 42.04,
    "12h": 65.42,  "24h": 104.67, "48h": 205.10,
}
R_EARTH = 6371.0

# ── Phase boundaries ──────────────────────────────────────────────────────────
PHASE_1_END  = 19   # epoch 0–19:  warm-up, α=0, easy_frac=80%
PHASE_2_END  = 49   # epoch 20–49: hard intro, α 0→0.3, easy_frac 80→60%
PHASE_3_END  = 79   # epoch 50–79: selector, α=0.3, easy_frac=55%
# epoch 80+: fine-tune, LR×0.1, freeze encoder

# ── GradNorm terms (các loss term cần cân bằng) ──────────────────────────────
GRADNORM_TERMS = ["l_dpe", "l_vel_reg", "l_heading", "l_speed", "l_accel"]

# [FIX-GN-3] l_dpe là ANCHOR — λ_dpe cố định, KHÔNG bị GradNorm điều chỉnh.
# Giá trị = default hand-tuned trong compute_st_trans_loss (total_default).
ANCHOR_TERM  = "l_dpe"
AUX_TERMS    = ["l_vel_reg", "l_heading", "l_speed", "l_accel"]
DEFAULT_LAMBDA = {
    "l_dpe":     1.20,
    "l_vel_reg": 1.40,
    "l_heading": 0.40,
    "l_speed":   0.05,
    "l_accel":   0.01,
}
AUX_LAMBDA_MIN = 0.02
AUX_LAMBDA_MAX = 8.0

# Alpha threshold để coi là "đã ramp xong" — dùng cho EMA guard pullback gate
ALPHA_RAMP_DONE_THRESHOLD = 0.25


# ─────────────────────────────────────────────────────────────────────────────
#  [FIX-GN-3] GradNorm implementation — anchor + target-ratio scheme
#
#  Thuật toán:
#    - l_dpe = ANCHOR. λ_dpe CỐ ĐỊNH = DEFAULT_LAMBDA["l_dpe"] = 1.20.
#      → gradient cho l_dpe (objective chính, gắn trực tiếp ADE) không
#        bao giờ bị GradNorm giảm xuống dưới mức hand-tuned.
#    - Với mỗi term phụ i ∈ AUX_TERMS:
#        contribution_i    = λ_i * l_i
#        target_contrib_i  = DEFAULT_LAMBDA[i] * l_dpe
#        ratio_i           = contribution_i / target_contrib_i
#        ratio_i > 1 (đóng góp vượt tỷ lệ mục tiêu) → GIẢM λ_i → grad > 0
#        ratio_i < 1 (đóng góp dưới tỷ lệ mục tiêu) → TĂNG λ_i → grad < 0
#    - Equilibrium: λ_i ≈ DEFAULT_LAMBDA[i] * l_dpe / l_i
#      Nếu l_i == l_dpe thì λ_i == DEFAULT_LAMBDA[i] (đúng hand-tuned gốc).
#      Nếu l_i trôi scale theo training, λ_i tự bù trừ để giữ TỶ LỆ ĐÓNG GÓP
#      (không phải giá trị λ) bằng đúng tỷ lệ hand-tuned.
#    - Không có sum constraint → không cần renorm. Mỗi λ_i clamp riêng vào
#      [AUX_LAMBDA_MIN, AUX_LAMBDA_MAX] (range rộng vì DEFAULT_LAMBDA chênh
#      nhau tới ~150x giữa l_vel_reg và l_accel).
# ─────────────────────────────────────────────────────────────────────────────

class GradNormManager:
    """
    [FIX-GN-3] Quản lý GradNorm — anchor + target-ratio scheme.

    l_dpe là ANCHOR (λ cố định = DEFAULT_LAMBDA["l_dpe"] = 1.20, không nằm
    trong self.lambdas, không bao giờ bị update). Chỉ AUX_TERMS
    (l_vel_reg, l_heading, l_speed, l_accel) được GradNorm điều chỉnh,
    với equilibrium λ_i ≈ DEFAULT_LAMBDA[i] * l_dpe / l_i.

    [FIX-GN-2] phase_reset(): KHÔNG reset λ — chỉ reset optimizer momentum
    + bật dense update ngắn (50 step) để re-adapt nhẹ nếu landscape có dịch
    (do thêm gradient từ L_hard/L_sel chảy qua shared encoder ở phase mới).
    """
    def __init__(self, terms: List[str], alpha_gn: float = 1.5,
                 lr: float = 1e-3, device=None,
                 lambda_min: float = AUX_LAMBDA_MIN,
                 lambda_max: float = AUX_LAMBDA_MAX):
        # terms: full GRADNORM_TERMS list (bao gồm anchor) — giữ để
        # tương thích các nơi khác lặp qua GRADNORM_TERMS khi log/extract.
        self.terms      = terms
        self.aux_terms  = [t for t in terms if t != ANCHOR_TERM]
        self.alpha_gn   = alpha_gn
        self.device     = device or torch.device("cpu")
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max

        # λ_i cho AUX_TERMS, init = DEFAULT_LAMBDA[i] (equilibrium đúng nếu
        # l_i == l_dpe ngay từ đầu — điểm khởi đầu hợp lý hơn 1.0 đồng nhất).
        self.lambdas = {t: torch.full((1,), DEFAULT_LAMBDA[t],
                                       device=self.device, requires_grad=True)
                        for t in self.aux_terms}
        self.opt = optim.Adam(list(self.lambdas.values()), lr=lr)

        self._step             = 0
        self._update_freq      = 5      # update mỗi 5 steps (bình thường)
        self._dense_steps_left = 200    # dense ở lúc khởi động training
        self._current_phase    = 1

    # ─────────────────────────────────────────────────────────────────
    # [FIX-GN-4] state_dict/load_state_dict — persist λ_aux, optimizer
    # GradNorm, _step, _dense_steps_left, _current_phase qua checkpoint.
    #
    # BUG trước fix: GradNormManager được khởi tạo lại từ constructor
    # mỗi lần script chạy (kể cả khi resume) → λ_aux reset về
    # DEFAULT_LAMBDA (1.40/0.40/0.05/0.01), _dense_steps_left reset về
    # 200, _step reset về 0 → 30 epoch điều chỉnh λ bị MẤT HOÀN TOÀN,
    # GradNorm phải "học lại từ đầu" sau mỗi resume (dense window 200
    # step lại chạy lại, λ lại tăng dần từ default y như epoch 0).
    # ─────────────────────────────────────────────────────────────────
    def state_dict(self) -> Dict:
        return {
            "lambdas": {t: v.detach().cpu().clone()
                        for t, v in self.lambdas.items()},
            "opt": self.opt.state_dict(),
            "step": self._step,
            "dense_steps_left": self._dense_steps_left,
            "current_phase": self._current_phase,
        }

    def load_state_dict(self, sd: Dict) -> None:
        try:
            for t, v in sd["lambdas"].items():
                if t in self.lambdas:
                    with torch.no_grad():
                        self.lambdas[t].copy_(v.to(self.device))
            # Re-tạo optimizer trên cùng tensor self.lambdas (đã copy_
            # giá trị ở trên, KHÔNG thay reference) rồi load lại state
            # (momentum Adam) — tránh mismatch param-group nếu thứ tự
            # self.aux_terms khác giữa lần lưu và lần load.
            self.opt = optim.Adam(list(self.lambdas.values()),
                                   lr=self.opt.param_groups[0]["lr"])
            try:
                self.opt.load_state_dict(sd["opt"])
            except Exception:
                pass  # momentum Adam không critical, lambdas đã đúng
            self._step             = sd.get("step", 0)
            self._dense_steps_left = sd.get("dense_steps_left", 0)
            self._current_phase    = sd.get("current_phase", self._current_phase)
        except Exception as e:
            print(f"  [FIX-GN-4] Lỗi khi restore GradNorm state: {e}")

    def phase_reset(self, new_phase: int) -> None:
        """
        [FIX-GN-2] Gọi khi chuyển sang phase mới.
        KHÔNG reset λ — chỉ reset optimizer momentum và bật dense window
        ngắn (50 step, không phải 200) để re-adapt nhẹ.
        """
        if new_phase == self._current_phase:
            return
        self._current_phase = new_phase

        self.opt = optim.Adam(list(self.lambdas.values()),
                               lr=self.opt.param_groups[0]["lr"])
        self._dense_steps_left = 50

        cur = self.get_lambda_dict()
        lam_str = " ".join(f"λ_{t.replace('l_','').replace('_reg','')}"
                            f"={cur[t]:.3f}" for t in self.terms)
        print(f"  [GradNorm] Phase {new_phase}: GIỮ λ hiện tại ({lam_str}), "
              f"λ_dpe cố định={DEFAULT_LAMBDA[ANCHOR_TERM]:.2f}, "
              f"reset optimizer momentum, dense update 50 steps")

    def get_lambda_dict(self) -> Dict[str, float]:
        """Trả về λ hiện tại (đủ GRADNORM_TERMS) để truyền vào
        get_loss_breakdown(). λ_dpe luôn = DEFAULT_LAMBDA["l_dpe"] (cố định)."""
        d = {t: float(self.lambdas[t].item()) for t in self.aux_terms}
        d[ANCHOR_TERM] = DEFAULT_LAMBDA[ANCHOR_TERM]
        return d

    def update(self, loss_term_values: Dict[str, float]) -> None:
        """
        [FIX-GN-3] Cập nhật λ_aux dùng anchor + target-ratio scheme.

        Args:
            loss_term_values: {term: float} — loss RAW value từng term
                              (detached, chưa nhân λ), PHẢI bao gồm
                              ANCHOR_TERM ("l_dpe") để dùng làm tham chiếu.
        """
        self._step += 1

        if self._dense_steps_left > 0:
            self._dense_steps_left -= 1
            update_freq = 1
        else:
            update_freq = self._update_freq

        def _clamp():
            with torch.no_grad():
                for t in self.aux_terms:
                    self.lambdas[t].clamp_(min=self.lambda_min,
                                            max=self.lambda_max)

        if self._step % update_freq != 0:
            _clamp()
            return

        if not loss_term_values:
            _clamp()
            return

        l_anchor = loss_term_values.get(ANCHOR_TERM)
        if l_anchor is None or not np.isfinite(l_anchor) or l_anchor <= 0:
            _clamp()
            return

        loss_vals = {t: v for t, v in loss_term_values.items()
                     if t in self.aux_terms and np.isfinite(v) and v > 0}
        if not loss_vals:
            _clamp()
            return

        self.opt.zero_grad()
        for t in self.aux_terms:
            if t not in loss_vals:
                continue
            lam_t            = float(self.lambdas[t].item())
            contribution_t   = lam_t * loss_vals[t]
            target_contrib_t = DEFAULT_LAMBDA[t] * l_anchor
            ratio = contribution_t / (target_contrib_t + 1e-8)
            # ratio > 1: đóng góp vượt tỷ lệ mục tiêu → GIẢM λ_i → grad > 0
            # ratio < 1: đóng góp dưới tỷ lệ mục tiêu → TĂNG λ_i → grad < 0
            grad_val = (ratio - 1.0) * 0.1
            if self.lambdas[t].grad is None:
                self.lambdas[t].grad = torch.zeros_like(self.lambdas[t])
            self.lambdas[t].grad.fill_(float(grad_val))

        self.opt.step()
        _clamp()

    def log_stats(self) -> Dict[str, float]:
        """Log λ hiện tại để monitor. λ_l_dpe luôn = giá trị anchor cố định."""
        d = {f"λ_{t}": float(self.lambdas[t].item()) for t in self.aux_terms}
        d[f"λ_{ANCHOR_TERM}"] = DEFAULT_LAMBDA[ANCHOR_TERM]
        d["gn_dense"] = self._dense_steps_left
        return d


# ─────────────────────────────────────────────────────────────────────────────
#  Phase helper
# ─────────────────────────────────────────────────────────────────────────────

def get_phase(epoch: int) -> int:
    """
    Returns:
        1: warm-up (ep 0–19)
        2: hard introduction (ep 20–49)
        3: selector (ep 50–79)
        4: fine-tune (ep 80+)
    """
    if epoch <= PHASE_1_END:  return 1
    if epoch <= PHASE_2_END:  return 2
    if epoch <= PHASE_3_END:  return 3
    return 4


class SmoothScheduler:
    """
    [FIX-STABILITY-1] α smooth warm-up theo BATCH, không phải epoch.

    [FIX-ALPHA-TIMING] mid point ở 15% của phase thay vì 50%:
        mid ở epoch ~24.5 (phase2) → α≈0.15 ở ep23, α≈0.3 ở ep26
        → hard loss kích hoạt sớm hơn ~10 epoch so với mid=50%.

    Selector cũng dùng cùng logic để bật đồng bộ.
    """
    def __init__(self, total_steps_per_epoch: int):
        self.steps_per_epoch = total_steps_per_epoch
        self._global_step = 0

        # Điểm bắt đầu của mỗi phase (tính theo global step)
        self._phase2_start_step = (PHASE_1_END + 1) * total_steps_per_epoch
        self._phase3_start_step = (PHASE_2_END + 1) * total_steps_per_epoch
        self._phase4_start_step = (PHASE_3_END + 1) * total_steps_per_epoch

        # [FIX-ALPHA-TIMING] Temperature ngắn → α tăng SỚM trong phase 2.
        self._temp_p2 = 0.6 * total_steps_per_epoch / 4.4
        self._temp_p3 = 0.6 * total_steps_per_epoch / 4.4  # selector bật nhanh
        # [FIX-EASYFRAC4] Temperature cho phase 4 (giảm 55%→50% qua ~5 epoch)
        self._temp_p4 = 2.5 * total_steps_per_epoch / 4.4

    def step(self):
        """Gọi sau mỗi optimizer step."""
        self._global_step += 1

    @property
    def global_step(self):
        return self._global_step

    def get_alpha_hard(self) -> float:
        """
        α_hard: trọng số L_hard, tăng mượt qua sigmoid.
        Phase 1 (step < phase2_start): α = 0
        Phase 2 (step >= phase2_start): sigmoid 0 → 0.3, mid ở 15% phase
        Phase 3+: α = 0.3 cố định
        """
        s = self._global_step
        if s < self._phase2_start_step:
            return 0.0
        if s >= self._phase3_start_step:
            return 0.3

        mid  = self._phase2_start_step + (self._phase3_start_step
                                           - self._phase2_start_step) * 0.15
        x    = (s - mid) / max(self._temp_p2, 1.0)
        sig  = 1.0 / (1.0 + math.exp(-x))
        return 0.3 * sig

    def get_selector_weight(self) -> float:
        """
        selector_weight: trọng số L_sel, tăng mượt từ phase 3.
        Phase < 3: 0
        Phase 3: sigmoid 0 → 0.2
        Phase 4: 0.2 cố định
        """
        s = self._global_step
        if s < self._phase3_start_step:
            return 0.0
        if s >= self._phase4_start_step:
            return 0.2

        mid  = self._phase3_start_step + (self._phase4_start_step
                                           - self._phase3_start_step) / 2.0
        x    = (s - mid) / max(self._temp_p3, 1.0)
        sig  = 1.0 / (1.0 + math.exp(-x))
        return 0.2 * sig

    def get_easy_frac(self) -> float:
        """
        easy_frac: giảm mượt, không bao giờ xuống dưới 50%.
        Phase 1: 80%
        Phase 2: 80% → 60% theo sigmoid
        Phase 3: 60% → 55%
        [FIX-EASYFRAC4] Phase 4: 55% → 50% theo sigmoid (thay vì linear)
            Consistent với phương pháp của các phase khác.
            Midpoint: 5 epoch sau khi vào phase 4.
        """
        s = self._global_step
        if s < self._phase2_start_step:
            return 0.80

        if s < self._phase3_start_step:
            mid  = self._phase2_start_step + (self._phase3_start_step
                                               - self._phase2_start_step) / 2.0
            x    = (s - mid) / max(self._temp_p2, 1.0)
            sig  = 1.0 / (1.0 + math.exp(-x))
            return 0.80 - 0.20 * sig   # 80% → 60%

        if s < self._phase4_start_step:
            mid  = self._phase3_start_step + (self._phase4_start_step
                                               - self._phase3_start_step) / 2.0
            x    = (s - mid) / max(self._temp_p3, 1.0)
            sig  = 1.0 / (1.0 + math.exp(-x))
            return 0.60 - 0.05 * sig   # 60% → 55%

        # [FIX-EASYFRAC4] Phase 4: 55% → 50% qua sigmoid (không phải linear)
        phase4_mid = self._phase4_start_step + 5 * self.steps_per_epoch
        x   = (s - phase4_mid) / max(self._temp_p4, 1.0)
        sig = 1.0 / (1.0 + math.exp(-x))
        return max(0.50, 0.55 - 0.05 * sig)

    def get_diversity_weight(self, target: float = 0.15) -> float:
        """
        [FIX-DIV-3] diversity_loss_weight: trọng số L_diversity (hinge,
        ngoài GradNorm), tăng mượt từ phase 2 — CÙNG schedule shape với
        get_alpha_hard (sigmoid, mid ở 15% phase 2).
        Phase 1: 0 (giữ warm-up ổn định, KHÔNG thêm forward pass thứ 2)
        Phase 2: sigmoid 0 → target
        Phase 3+: target cố định
        """
        s = self._global_step
        if s < self._phase2_start_step:
            return 0.0
        if s >= self._phase3_start_step:
            return target
        mid  = self._phase2_start_step + (self._phase3_start_step
                                           - self._phase2_start_step) * 0.15
        x    = (s - mid) / max(self._temp_p2, 1.0)
        sig  = 1.0 / (1.0 + math.exp(-x))
        return target * sig

    def is_selector_active(self) -> bool:
        """True từ phase 3 trở đi."""
        return self._global_step >= self._phase3_start_step

    def log_stats(self) -> str:
        return (f"α={self.get_alpha_hard():.3f}"
                f" easy={self.get_easy_frac():.0%}"
                f" sel={self.get_selector_weight():.3f}")


# Backward-compat wrappers (dùng trong checkpoint restore)
def get_alpha_hard(epoch: int) -> float:
    """Legacy: chỉ dùng khi không có SmoothScheduler."""
    if epoch <= PHASE_1_END: return 0.0
    if epoch <= PHASE_2_END:
        return 0.3 * (epoch - PHASE_1_END) / (PHASE_2_END - PHASE_1_END)
    return 0.3

def get_easy_frac(epoch: int) -> float:
    """Legacy fallback."""
    if epoch <= PHASE_1_END: return 0.80
    if epoch <= PHASE_2_END:
        p = (epoch - PHASE_1_END) / (PHASE_2_END - PHASE_1_END)
        return 0.80 - 0.20 * p
    if epoch <= PHASE_3_END: return 0.55
    return 0.50


# ─────────────────────────────────────────────────────────────────────────────
#  [FIX-STABILITY-3 + FIX-THRESH-1] Global hard threshold — tính 1 lần
#
#  hard_score_from_obs() chỉ phụ thuộc obs_traj (data cố định), KHÔNG phụ
#  thuộc model weights → threshold KHÔNG thay đổi theo training.
#  [FIX-THRESH-1] Dùng flag _threshold_computed riêng (xem main loop) thay vì
#  so sánh global_hard_threshold == 0.5 (fragile — nếu p70 thực tế = 0.5 thì
#  sẽ recompute mãi).
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def precompute_hard_threshold(loader, dev, n_batches: int = 50) -> float:
    """
    Tính p70 của hard_score trên subset của train set (1 lần duy nhất).
    """
    all_scores = []
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        bl    = move(list(batch), dev)
        obs_t = bl[0]
        scores = hard_score_from_obs(obs_t[:, :, :2])  # [B]
        all_scores.append(scores.cpu())

    if not all_scores:
        return 0.7  # fallback

    all_scores_cat = torch.cat(all_scores)  # [N]
    threshold = float(torch.quantile(all_scores_cat, 0.70).item())
    n_hard = int((all_scores_cat >= threshold).sum().item())
    n_total = len(all_scores_cat)
    print(f"  [HardThreshold] global p70={threshold:.4f}"
          f"  n_hard={n_hard}/{n_total} ({100*n_hard/max(n_total,1):.0f}%)")
    return threshold


@torch.no_grad()
def classify_hard_easy_global(
    obs_traj_norm: torch.Tensor,
    global_threshold: float,
) -> torch.Tensor:
    """Phân loại hard/easy dùng global threshold (thay vì per-batch p70)."""
    scores   = hard_score_from_obs(obs_traj_norm)  # [B]
    is_hard  = scores >= global_threshold
    return is_hard


def enforce_easy_frac(
    is_hard: torch.Tensor,
    easy_frac: float,
) -> torch.Tensor:
    """
    Chỉ điều chỉnh is_hard mask, KHÔNG reindex batch_list.
    Nếu batch có quá nhiều hard → "demote" một số về easy (is_hard[i]=False).
    Batch data giữ nguyên 100%.
    """
    B        = is_hard.shape[0]
    max_hard = max(0, int(B * (1.0 - easy_frac)))
    n_hard   = int(is_hard.sum().item())

    if n_hard <= max_hard:
        return is_hard  # đã đúng tỉ lệ

    n_demote  = n_hard - max_hard
    hard_idx  = is_hard.nonzero(as_tuple=True)[0]
    perm      = torch.randperm(hard_idx.numel(), device=is_hard.device)
    to_demote = hard_idx[perm[:n_demote]]

    is_hard_new            = is_hard.clone()
    is_hard_new[to_demote] = False
    return is_hard_new


# ─────────────────────────────────────────────────────────────────────────────
#  Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _unwrap(m):
    return m._orig_mod if hasattr(m, "_orig_mod") else m

def move(b, dev):
    out = list(b)
    for i, x in enumerate(out):
        if torch.is_tensor(x): out[i] = x.to(dev)
        elif isinstance(x, dict):
            out[i] = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in x.items()}
    return out


def _ntd(a):
    o = a.clone()
    o[..., 0] = (a[..., 0] * 50.0 + 1800.0) / 10.0
    o[..., 1] = (a[..., 1] * 50.0) / 10.0
    return o

def _hav(p1, p2):
    la1 = torch.deg2rad(p1[..., 1]); la2 = torch.deg2rad(p2[..., 1])
    dlat = torch.deg2rad(p2[..., 1] - p1[..., 1])
    dlon = torch.deg2rad(p2[..., 0] - p1[..., 0])
    a = (torch.sin(dlat/2).pow(2)
         + torch.cos(la1)*torch.cos(la2)*torch.sin(dlon/2).pow(2))
    return 2.0 * R_EARTH * torch.asin(a.clamp(1e-12, 1-1e-12).sqrt())

def _atecte(pd, gd):
    T = min(pd.shape[0], gd.shape[0])
    if T < 2:
        z = pd.new_zeros(1, pd.shape[1]); return z, z
    lo1=torch.deg2rad(gd[:T-1,:,0]); la1=torch.deg2rad(gd[:T-1,:,1])
    lo2=torch.deg2rad(gd[1:T, :,0]); la2=torch.deg2rad(gd[1:T, :,1])
    lo3=torch.deg2rad(pd[1:T, :,0]); la3=torch.deg2rad(pd[1:T, :,1])
    ya=torch.sin(lo2-lo1)*torch.cos(la2)
    xa=(torch.cos(la1)*torch.sin(la2)-torch.sin(la1)*torch.cos(la2)*torch.cos(lo2-lo1))
    be=torch.atan2(ya,xa)
    ye=torch.sin(lo3-lo2)*torch.cos(la3)
    xe=(torch.cos(la2)*torch.sin(la3)-torch.sin(la2)*torch.cos(la3)*torch.cos(lo3-lo2))
    bee=torch.atan2(ye,xe)
    tot=_hav(pd[1:T],gd[1:T]); ang=bee-be
    return tot*torch.cos(ang), tot*torch.sin(ang)


# ─────────────────────────────────────────────────────────────────────────────
#  [G-G] Split metrics — đo easy_ADE và hard_ADE riêng biệt
# ─────────────────────────────────────────────────────────────────────────────

class SplitAcc:
    """
    Accumulator đo easy_ADE và hard_ADE riêng biệt.
    Dùng để:
    - EMA guard: phát hiện easy forgetting sớm
    - Hard improvement: xác nhận hard loss đang hoạt động
    """
    def __init__(self):
        self.easy_d = []
        self.hard_d = []
        self.all_d  = []
        self.all_a  = []
        self.all_c  = []
        self.sd     = defaultdict(list)
        self._h     = {12: 1, 24: 3, 48: 7, 72: 11}

    def update(self, dist, is_hard=None, ate=None, cte=None):
        """
        Args:
            dist:    [T, B] distance per step
            is_hard: [B] bool (None → tất cả easy)
        """
        B = dist.shape[1]
        mean_dist = dist.mean(0)  # [B]
        self.all_d.extend(mean_dist.tolist())

        if is_hard is not None:
            easy_mask = ~is_hard
            if easy_mask.any():
                self.easy_d.extend(mean_dist[easy_mask].tolist())
            if is_hard.any():
                self.hard_d.extend(mean_dist[is_hard].tolist())
        else:
            self.easy_d.extend(mean_dist.tolist())

        for h, s in self._h.items():
            if s < dist.shape[0]:
                self.sd[h].extend(dist[s].tolist())

        if ate is not None: self.all_a.extend(ate.abs().mean(0).tolist())
        if cte is not None: self.all_c.extend(cte.abs().mean(0).tolist())

    def compute(self):
        r = {
            "ADE":      float(np.mean(self.all_d))  if self.all_d  else float("nan"),
            "easy_ADE": float(np.mean(self.easy_d)) if self.easy_d else float("nan"),
            "hard_ADE": float(np.mean(self.hard_d)) if self.hard_d else float("nan"),
            "ATE":      float(np.mean(self.all_a))  if self.all_a  else float("nan"),
            "CTE":      float(np.mean(self.all_c))  if self.all_c  else float("nan"),
            "n":        len(self.all_d),
            "n_easy":   len(self.easy_d),
            "n_hard":   len(self.hard_d),
        }
        for h in self._h:
            v = self.sd.get(h, [])
            r[f"{h}h"] = float(np.mean(v)) if v else float("nan")
        return r


class Acc:
    """Simple accumulator (backward compat với evaluate())."""
    def __init__(self):
        self.d=[]; self.a=[]; self.c=[]; self.sd=defaultdict(list)
        self._h={12:1, 24:3, 48:7, 72:11}
    def update(self, dist, ate=None, cte=None):
        self.d.extend(dist.mean(0).tolist())
        for h,s in self._h.items():
            if s < dist.shape[0]: self.sd[h].extend(dist[s].tolist())
        if ate is not None: self.a.extend(ate.abs().mean(0).tolist())
        if cte is not None: self.c.extend(cte.abs().mean(0).tolist())
    def compute(self):
        r = {"ADE": float(np.mean(self.d)) if self.d else float("nan"),
             "ATE": float(np.mean(self.a)) if self.a else float("nan"),
             "CTE": float(np.mean(self.c)) if self.c else float("nan"),
             "n": len(self.d)}
        for h in self._h:
            v=self.sd.get(h,[]); r[f"{h}h"]=float(np.mean(v)) if v else float("nan")
        return r


def _score(r):
    ade=r.get("ADE",1e9); h72=r.get("72h",1e9)
    ate=r.get("ATE",1e9); cte=r.get("CTE",1e9)
    if not np.isfinite(ate): ate=ade*0.46
    if not np.isfinite(cte): cte=ade*0.53
    return 100.0*(0.05*(ade/136.)+0.10*(r.get("12h",ade)/50.)
                  +0.15*(r.get("24h",ade)/100.)+0.20*(r.get("48h",ade)/200.)
                  +0.25*(h72/300.)+0.13*(ate/80.)+0.12*(cte/94.))

def _beat(r):
    p=[]
    for k,t in [("ADE",172.68),("ATE",142.21),("CTE",42.04),
                ("72h",321.39),("12h",65.42),("24h",104.67),("48h",205.10)]:
        v=r.get(k,1e9)
        if np.isfinite(v) and v < t: p.append(f"{k}:{v:.1f}")
    return "*** BEAT ST-TRANS: "+" ".join(p)+" ***" if p else ""

def _gap(r):
    out=[]
    for k,ref in [("ADE",172.68),("72h",321.39),("ATE",142.21),("CTE",42.04)]:
        v=r.get(k,float("nan"))
        if np.isfinite(v):
            out.append(f"{k}:{v:.0f}({'dn' if v<ref else 'up'}{abs(v-ref):.0f})")
    return " | ".join(out)


# ─────────────────────────────────────────────────────────────────────────────
#  [G-E] Easy_ADE evaluation cho EMA guard — dùng global threshold (FIX-4)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_split(model, loader, dev, tag="", ema=None, steps=10,
                   use_selector=False,
                   global_hard_threshold: Optional[float] = None) -> Dict:
    """
    Evaluate với split easy/hard ADE.

    Args:
        global_hard_threshold: float từ precompute_hard_threshold(), hoặc None
            → dùng per-batch p70 (phase 1 fallback, chỉ ảnh hưởng log)
    """
    bk = None
    if ema:
        try: bk = ema.apply_to(model)
        except: pass
    model.eval()
    acc = SplitAcc()
    t0  = time.perf_counter()

    for b in loader:
        bl = move(list(b), dev)
        obs_t = bl[0]

        if global_hard_threshold is not None:
            is_hard = classify_hard_easy_global(
                obs_t[:, :, :2], global_hard_threshold).to(dev)
        else:
            is_hard = classify_hard_easy(obs_t[:, :, :2]).to(dev)

        result = model.sample(bl, ddim_steps=steps, use_selector=use_selector)
        p = result[0] if isinstance(result, (tuple, list)) else result
        g = bl[1]; T = min(p.shape[0], g.shape[0])
        pd = _ntd(p[:T]); gd = _ntd(g[:T])
        dist = _hav(pd, gd)
        at, ct = _atecte(pd, gd)
        acc.update(dist, is_hard=is_hard, ate=at, cte=ct)

    # [FIX-EMA-restore] bk luôn là dict non-empty nếu apply_to thành công
    # (shadow luôn có ít nhất 1 floating-point param) — nhưng dùng
    # `is not None` cho rõ ràng & an toàn tuyệt đối.
    if bk is not None:
        try: ema.restore(model, bk)
        except: pass

    r  = acc.compute()
    el = time.perf_counter() - t0

    def _v(k): return r.get(k, float("nan"))
    def _m(v, t): return "ok" if np.isfinite(v) and v < t else "no"

    thr_str = f"{global_hard_threshold:.4f}" if global_hard_threshold is not None else "per-batch"
    print(f"\n{'='*70}")
    print(f"  [{tag}  {el:.0f}s  threshold={thr_str}]")
    print(f"  ADE={_v('ADE'):.1f}[{_m(_v('ADE'),172.68)}]"
          f"  easy_ADE={_v('easy_ADE'):.1f}  hard_ADE={_v('hard_ADE'):.1f}")
    print(f"  12h={_v('12h'):.0f}  24h={_v('24h'):.0f}"
          f"  48h={_v('48h'):.0f}  72h={_v('72h'):.0f}[{_m(_v('72h'),321.39)}]")
    print(f"  ATE={_v('ATE'):.1f}[{_m(_v('ATE'),142.21)}]"
          f"  CTE={_v('CTE'):.1f}[{_m(_v('CTE'),42.04)}]")
    print(f"  n_easy={r.get('n_easy',0)}  n_hard={r.get('n_hard',0)}")
    print(f"  vs ST-Trans: {_gap(r)}")
    bt = _beat(r)
    if bt: print(f"  {bt}")
    print(f"  Score={_score(r):.2f}")
    print(f"{'='*70}\n")
    return r


@torch.no_grad()
def evaluate(model, loader, dev, tag="", ema=None, steps=20,
             use_selector=False,
             global_hard_threshold: Optional[float] = None) -> Dict:
    """Full evaluation (backward compat)."""
    return evaluate_split(model, loader, dev, tag=tag, ema=ema,
                          steps=steps, use_selector=use_selector,
                          global_hard_threshold=global_hard_threshold)


# ─────────────────────────────────────────────────────────────────────────────
#  [G-F] Diversity check
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def check_diversity(model, loader, dev, n_batches: int = 5,
                    n_ensemble: int = 20, ddim_steps: int = 5) -> float:
    """
    Đo diversity score trên một subset của val set.
    Nếu < 50 km → cảnh báo R1 (mode collapse).
    """
    model.eval()
    all_divs = []

    for i, b in enumerate(loader):
        if i >= n_batches:
            break
        bl = move(list(b), dev)

        _, _, all_c = model.sample(bl, num_ensemble=n_ensemble,
                                    ddim_steps=ddim_steps)
        N_total = all_c.shape[0]
        cands = [all_c[k] for k in range(N_total)]

        div = compute_diversity_score(cands)
        all_divs.append(div)

    return float(np.mean(all_divs)) if all_divs else 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  [FIX-CKPT-1] Checkpoint — lưu smooth_sched_step + global_hard_threshold
#  trong MỌI checkpoint (trước chỉ có trong periodic ckpt's "extra")
# ─────────────────────────────────────────────────────────────────────────────

def _save_ckpt(path, ep, model, opt, sched, saver, tl, vl,
               smooth_sched_step: int = 0,
               global_hard_threshold: Optional[float] = None,
               diversity_boost_factor: float = 1.0,
               gradnorm_state: Optional[Dict] = None,
               extra=None):
    m = _unwrap(model)
    ema = getattr(m, "_ema", None)
    esd = None
    if ema and hasattr(ema, "shadow"):
        try: esd = {k: v.cpu().clone() for k, v in ema.shadow.items()}
        except: pass
    d = {
        "epoch": ep,
        "model_state_dict": m.state_dict(),
        "optimizer_state":  opt.state_dict(),
        "scheduler_state":  sched.state_dict(),
        "ema_shadow":       esd,
        "best_score":       saver.bs,
        "best_ade":         saver.ba,
        "best_72h":         saver.b7,
        "best_ate":         saver.bat,
        "best_cte":         saver.bc,
        "best_easy_ade":    saver.best_easy_ade,
        "train_loss":       tl,
        "val_loss":         vl,
        # [FIX-CKPT-1] Luôn lưu để resume chính xác smooth_sched + threshold
        "smooth_sched_step":     smooth_sched_step,
        "global_hard_threshold": global_hard_threshold,
        # [FIX-DIV-2] Lưu hệ số boost diversity để resume áp lại đúng
        # sigma_min/ctx_noise_scale (đây là plain float attr, KHÔNG nằm
        # trong model.state_dict() nên sẽ mất nếu không lưu riêng).
        "diversity_boost_factor": diversity_boost_factor,
        # [FIX-GN-4] Lưu state GradNormManager (λ_aux, optimizer riêng,
        # _step, _dense_steps_left, _current_phase) — KHÔNG nằm trong
        # model.state_dict(), nên nếu không lưu sẽ mất khi resume và
        # GradNorm phải "học lại từ đầu" (λ_aux reset về DEFAULT_LAMBDA,
        # dense window 200 step chạy lại).
        "gradnorm_state":   gradnorm_state,
        "version":          "v59strategy_fixed_v4",
    }
    if extra:
        d.update(extra)
    torch.save(d, path)


# ─────────────────────────────────────────────────────────────────────────────
#  Saver
# ─────────────────────────────────────────────────────────────────────────────

class Saver:
    def __init__(self, patience=40, min_ep=35, enable_stop=True):
        self.patience     = patience
        self.min_ep       = min_ep
        self.enable_stop  = enable_stop
        self.cnt          = 0
        self.stop         = False
        self.bs           = float("inf")  # best score
        self.ba           = float("inf")  # best ADE
        self.b7           = float("inf")  # best 72h
        self.bat          = float("inf")  # best ATE
        self.bc           = float("inf")  # best CTE
        self.best_easy_ade = float("inf") # [G-E] best easy ADE (EMA guard)

    def update(self, r, model, out_dir, ep, opt, sched, tl, vl, tag="",
               smooth_sched_step: int = 0,
               global_hard_threshold: Optional[float] = None,
               diversity_boost_factor: float = 1.0,
               gradnorm_state: Optional[Dict] = None):
        sc  = _score(r)
        ade = r.get("ADE",     1e9)
        h72 = r.get("72h",     1e9)
        ate = r.get("ATE",     1e9)
        cte = r.get("CTE",     1e9)
        easy_ade = r.get("easy_ADE", ade)  # fallback to ADE if no split

        any_improved = False

        for v, attr, fn in [
            (ade,      "ba",  f"best_ade_{tag}.pth"),
            (h72,      "b7",  f"best_72h_{tag}.pth"),
            (ate,      "bat", f"best_ate_{tag}.pth"),
            (cte,      "bc",  f"best_cte_{tag}.pth"),
        ]:
            if v < getattr(self, attr):
                setattr(self, attr, v)
                _save_ckpt(os.path.join(out_dir, fn), ep, model, opt,
                           sched, self, tl, vl,
                           smooth_sched_step=smooth_sched_step,
                           global_hard_threshold=global_hard_threshold,
                           diversity_boost_factor=diversity_boost_factor,
                           gradnorm_state=gradnorm_state)
                any_improved = True

        if sc < self.bs:
            self.bs = sc
            any_improved = True
            _save_ckpt(os.path.join(out_dir, f"best_{tag or 'composite'}.pth"),
                       ep, model, opt, sched, self, tl, vl,
                       smooth_sched_step=smooth_sched_step,
                       global_hard_threshold=global_hard_threshold,
                       diversity_boost_factor=diversity_boost_factor,
                       gradnorm_state=gradnorm_state)
            print(f"  [BEST] {tag} ep={ep} score={sc:.2f}"
                  f"  ADE={ade:.1f}  72h={h72:.0f}"
                  f"  ATE={ate:.1f}  CTE={cte:.1f}")

        # [G-E] Update best easy_ADE
        if np.isfinite(easy_ade) and easy_ade < self.best_easy_ade:
            self.best_easy_ade = easy_ade

        if self.enable_stop:
            if any_improved: self.cnt = 0
            else:
                self.cnt += 1
                print(f"  No improve {self.cnt}/{self.patience}"
                      f"  best={self.bs:.2f}  cur={sc:.2f}")
            if ep >= self.min_ep and self.cnt >= self.patience:
                self.stop = True


def _mksub(ds, n, bs, cf):
    idx = random.Random(42).sample(range(len(ds)), min(n, len(ds)))
    return DataLoader(Subset(ds, idx), batch_size=bs, shuffle=False,
                      collate_fn=cf, num_workers=0, drop_last=False)


# ─────────────────────────────────────────────────────────────────────────────
#  Args
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset_root",       default="TCND_vn")
    p.add_argument("--obs_len",            default=8,    type=int)
    p.add_argument("--pred_len",           default=12,   type=int)
    p.add_argument("--batch_size",         default=32,   type=int)
    p.add_argument("--num_epochs",         default=100,  type=int)
    p.add_argument("--learning_rate",      default=1e-4, type=float)
    p.add_argument("--weight_decay",       default=1e-3, type=float)
    p.add_argument("--warmup_epochs",      default=3,    type=int)
    p.add_argument("--grad_clip",          default=1.0,  type=float)
    p.add_argument("--use_amp",            action="store_true")
    p.add_argument("--num_workers",        default=2,    type=int)
    p.add_argument("--sigma_min",          default=0.02, type=float)
    p.add_argument("--use_ot",             default=True, action="store_true")
    p.add_argument("--no_ot",              dest="use_ot", action="store_false")
    p.add_argument("--cfg_guidance_scale", default=1.5,  type=float)
    p.add_argument("--n_ensemble",         default=50,   type=int)
    p.add_argument("--use_ema",            default=True, action="store_true")
    p.add_argument("--no_ema",             dest="use_ema", action="store_false")
    p.add_argument("--ema_decay",          default=0.995, type=float)
    p.add_argument("--patience",           default=40,   type=int)
    p.add_argument("--min_ep",             default=35,   type=int)
    p.add_argument("--val_freq",           default=3,    type=int)
    p.add_argument("--val_subset_size",    default=500,  type=int)
    p.add_argument("--fast_ddim",          default=10,   type=int)
    p.add_argument("--full_ddim",          default=20,   type=int)
    p.add_argument("--output_dir",         default="runs/v59strategy")
    p.add_argument("--gpu_num",            default="0")
    p.add_argument("--delim",              default=" ")
    p.add_argument("--skip",               default=1,    type=int)
    p.add_argument("--min_ped",            default=1,    type=int)
    p.add_argument("--threshold",          default=0.002, type=float)
    p.add_argument("--other_modal",        default="gph")
    p.add_argument("--test_year",          default=None, type=int)
    p.add_argument("--resume",             default=None)
    p.add_argument("--resume_epoch",       default=None, type=int)
    p.add_argument("--eval_test_after_train", default=True, action="store_true")
    # GradNorm
    p.add_argument("--gradnorm_alpha",     default=1.5,  type=float,
                   help="GradNorm restoring force (giữ cho backward compat)")
    p.add_argument("--gradnorm_lr",        default=1e-3, type=float,
                   help="GradNorm lambda learning rate")
    # EMA guard
    p.add_argument("--ema_guard_threshold", default=20.0, type=float,
                   help="smooth_easy_ADE tăng bao nhiêu km thì rollback (km)")
    # Diversity check
    p.add_argument("--diversity_threshold", default=50.0, type=float,
                   help="diversity_score < threshold → cảnh báo R1")
    # [FIX-DIV-2] Boost lặp sigma_min/ctx_noise_scale nếu diversity thấp.
    # CẢNH BÁO TRADE-OFF: sigma_min/ctx_noise_scale chỉ ảnh hưởng sample()
    # (inference) — KHÔNG ảnh hưởng train loss — nhưng boost quá mạnh có
    # thể làm TỪNG candidate noisy hơn → có thể làm FAST/RAW ADE/CTE epoch
    # 20+ xấu hơn epoch 19 (đánh đổi accuracy-per-candidate để lấy diversity
    # cho selector). Nếu thấy easy_ADE ep20 tệ hơn rõ rệt so với ep19 do
    # nguyên nhân này (không phải do phase-transition — đã fix ở FIX-GN-2),
    # giảm --diversity_boost_max ở lần chạy sau (resume sẽ áp lại factor đã
    # lưu, nên cần chạy lại từ ckpt trước ep19 hoặc từ đầu để đổi factor).
    p.add_argument("--diversity_boost_step", default=1.8, type=float,
                   help="hệ số nhân sigma_min/ctx_noise_scale mỗi vòng boost")
    p.add_argument("--diversity_boost_max", default=6.0, type=float,
                   help="cap tổng hệ số boost (so với giá trị gốc)")
    p.add_argument("--diversity_boost_iters", default=3, type=int,
                   help="số vòng boost+recheck tối đa")
    # [FIX-DIV-4] Override hệ số boost đã lưu trong checkpoint khi resume.
    p.add_argument("--diversity_boost_override", default=None, type=float,
                   help="ghi đè diversity_boost_factor khi resume (vd 3.0)."
                        " None = giữ nguyên factor đã lưu trong checkpoint.")

    # [FIX-DIV-3] L_diversity training loss (hinge, ngoài GradNorm).
    # An toàn: bounded ∈[0, target_norm], =0 khi diversity đủ (self-limiting).
    # Tốn thêm 1 forward pass qua transformer decoder (phần FNO3D/encoder
    # KHÔNG chạy lại) khi active — chỉ active từ phase 2 (sigmoid ramp),
    # tắt hoàn toàn ở phase 1 (mặc định weight=0 → KHÔNG tốn compute thêm).
    p.add_argument("--use_diversity_loss", action="store_true", default=True,
                   help="bật L_diversity training loss (FIX-DIV-3)")
    p.add_argument("--no_diversity_loss", dest="use_diversity_loss",
                   action="store_false",
                   help="tắt L_diversity training loss hoàn toàn")
    p.add_argument("--diversity_loss_weight", default=0.15, type=float,
                   help="trọng số đích (sau ramp) cho L_diversity hinge")
    p.add_argument("--diversity_target_km", default=None, type=float,
                   help="target diversity (km) cho L_diversity hinge;"
                        " mặc định = --diversity_threshold")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # [FIX-DIV-3] target diversity (km) cho L_diversity hinge — mặc định
    # dùng cùng giá trị với --diversity_threshold (consistency: cùng
    # ngưỡng "đủ đa dạng" cho cả check_diversity() và L_diversity).
    _diversity_target_km = (args.diversity_target_km
                            if args.diversity_target_km is not None
                            else args.diversity_threshold)

    print("=" * 72)
    print("  TC-FlowMatching v59-Strategy [FIXED-v4]")
    print("  [FIX-GN-3] GradNorm: anchor λ_dpe=1.20 cố định + target-ratio aux")
    print("  [FIX-GN-2] phase_reset: KHÔNG reset λ (chỉ reset optimizer)")
    print("  [FIX-EMA-2] EMA guard pullback: chỉ khi α<0.25 (tránh kẹt)")
    print("  [FIX-PERF-1] Xóa dead computation (named_parameters mỗi step)")
    print("  [FIX-CKPT-1] Lưu smooth_sched_step + threshold mọi checkpoint")
    print("  [FIX-DIV-2] Boost lặp (cap configurable) + recheck diversity +"
          " resume-persist")
    print(f"  [FIX-DIV-3] L_diversity hinge (target={_diversity_target_km:.0f}km,"
          f" w_target={args.diversity_loss_weight if args.use_diversity_loss else 0.0:.2f},"
          f" ramp từ phase2) — "
          f"{'ON' if args.use_diversity_loss else 'OFF'}")
    print("  [G-A] Easy/Hard: global threshold (tính 1 lần)")
    print("  [G-C] easy_frac ≥ 50% enforcement mỗi batch")
    print("  [G-D] α warm-up: sigmoid, mid=15% phase (FIX-ALPHA-TIMING)")
    print("  [G-E] EMA guard: EMA-smoothed easy_ADE, threshold 20km")
    print("  [G-F] Diversity check sau phase 1")
    print("  [G-G] Split monitoring: easy_ADE + hard_ADE")
    print("  [G-H] Phase 4: freeze encoder")
    print(f"  Phases: 1[0-{PHASE_1_END}] 2[{PHASE_1_END+1}-{PHASE_2_END}]"
          f" 3[{PHASE_2_END+1}-{PHASE_3_END}] 4[{PHASE_3_END+1}+]")
    print(f"  Target: ADE<{TARGETS['ADE']} 72h<{TARGETS['72h']}"
          f" ATE<{TARGETS['ATE']} CTE<{TARGETS['CTE']}")
    print("=" * 72)

    # Data
    trd, trl = data_loader(args, {"root": args.dataset_root, "type": "train"}, test=False)
    vd,  vl  = data_loader(args, {"root": args.dataset_root, "type": "val"},   test=True)
    from Model.data.trajectoriesWithMe_unet_training import seq_collate
    vsub = _mksub(vd, args.val_subset_size, args.batch_size, seq_collate)
    print(f"  train:{len(trd)} seqs  val:{len(vd)} seqs")

    # Model
    model = TCFlowMatching(
        pred_len=args.pred_len, obs_len=args.obs_len,
        sigma_min=args.sigma_min, use_ema=args.use_ema,
        ema_decay=args.ema_decay, use_ate_ot=args.use_ot,
        cfg_guidance_scale=args.cfg_guidance_scale,
    ).to(dev)
    model.init_ema()

    n_total = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # BUG-8: backbone vs learned weights (step_weights, selector small params)
    backbone_params = [p for p in model.parameters()
                       if p.requires_grad and p.numel() > 100]
    step_w_params   = [p for p in model.parameters()
                       if p.requires_grad and p.numel() <= 12]

    print(f"  params total: {n_total:,}")
    print(f"  backbone (clipped at {args.grad_clip}): "
          f"{sum(p.numel() for p in backbone_params):,}")
    print(f"  step_weights (NOT clipped): "
          f"{sum(p.numel() for p in step_w_params):,}")

    # Optimizer & scheduler
    opt    = optim.AdamW(model.parameters(),
                          lr=args.learning_rate,
                          weight_decay=args.weight_decay)
    scaler = GradScaler("cuda", enabled=args.use_amp)
    nstep  = len(trl)
    total  = nstep * args.num_epochs
    wstp   = nstep * args.warmup_epochs
    sched  = get_cosine_schedule_with_warmup(opt, wstp, total, min_lr=1e-6)

    # [G-A] GradNorm manager — equal-contribution scheme (FIX-GN-1)
    gradnorm = GradNormManager(
        terms=GRADNORM_TERMS,
        alpha_gn=args.gradnorm_alpha,
        lr=args.gradnorm_lr,
        device=dev,
    )

    # SmoothScheduler — α theo batch, không phải epoch
    smooth_sched = SmoothScheduler(total_steps_per_epoch=nstep)

    # Savers
    fast_saver = Saver(patience=args.patience, min_ep=args.min_ep,
                        enable_stop=False)
    full_saver = Saver(patience=args.patience, min_ep=args.min_ep,
                        enable_stop=True)

    # [FIX-THRESH-1] global threshold + flag riêng (thay vì == 0.5)
    global_hard_threshold: Optional[float] = None
    _threshold_computed = False

    # [FIX-DIV-2] hệ số boost diversity hiện tại (1.0 = chưa boost).
    # Cần khai báo TRƯỚC resume vì có thể được restore từ checkpoint.
    diversity_boost_factor: float = 1.0

    # Resume
    start = 0
    if args.resume and os.path.exists(args.resume):
        print(f"  Loading: {args.resume}")
        ck = torch.load(args.resume, map_location=dev)
        m  = _unwrap(model)
        ms, _ = m.load_state_dict(ck["model_state_dict"], strict=False)
        if ms: print(f"  Missing: {len(ms)}")
        ema = getattr(m, "_ema", None)
        if ema and ck.get("ema_shadow"):
            for k, v in ck["ema_shadow"].items():
                if k in ema.shadow: ema.shadow[k].copy_(v.to(dev))
        try: opt.load_state_dict(ck["optimizer_state"])
        except Exception as e: print(f"  Opt: {e}")
        try: sched.load_state_dict(ck["scheduler_state"])
        except:
            for _ in range(ck.get("epoch", 0) * nstep): sched.step()
        for a, attr in [("best_score","bs"), ("best_ade","ba"), ("best_72h","b7"),
                        ("best_ate","bat"), ("best_cte","bc"),
                        ("best_easy_ade","best_easy_ade")]:
            if a in ck:
                setattr(full_saver, attr, ck[a])
                setattr(fast_saver, attr, ck[a])
        start = args.resume_epoch or ck.get("epoch", 0) + 1
        print(f"  Resume ep={start}  best_score={full_saver.bs:.2f}"
              f"  best_easy_ADE={full_saver.best_easy_ade:.1f}")

        # [FIX-CKPT-1] Restore smooth_sched step + threshold từ checkpoint
        if "smooth_sched_step" in ck and ck["smooth_sched_step"] is not None:
            smooth_sched._global_step = ck["smooth_sched_step"]
            print(f"  Restored smooth_sched_step={ck['smooth_sched_step']}")
        else:
            smooth_sched._global_step = start * nstep
            print(f"  Estimated smooth_sched_step={start * nstep} from epoch")

        if "global_hard_threshold" in ck and ck["global_hard_threshold"] is not None:
            global_hard_threshold = ck["global_hard_threshold"]
            _threshold_computed = True
            print(f"  Restored global_hard_threshold={global_hard_threshold:.4f}")

        # [FIX-GN-4] Restore GradNorm state (λ_aux, optimizer riêng, _step,
        # _dense_steps_left, _current_phase). Checkpoint cũ (trước fix này)
        # không có key "gradnorm_state" -> bỏ qua, GradNorm khởi động lại
        # từ DEFAULT_LAMBDA + dense window 200 step (behavior cũ, không
        # crash, chỉ là không tối ưu — chấp nhận được cho checkpoint cũ).
        if ck.get("gradnorm_state") is not None:
            gradnorm.load_state_dict(ck["gradnorm_state"])
            lam_str = " ".join(
                f"λ_{t.replace('l_','').replace('_reg','')}="
                f"{gradnorm.lambdas[t].item():.3f}"
                for t in gradnorm.aux_terms)
            print(f"  [FIX-GN-4] Restored GradNorm state: {lam_str}"
                  f"  step={gradnorm._step}"
                  f"  dense_left={gradnorm._dense_steps_left}"
                  f"  phase={gradnorm._current_phase}")
        else:
            print(f"  [FIX-GN-4] Checkpoint không có gradnorm_state"
                  f" (checkpoint cũ trước fix) -> GradNorm khởi động lại"
                  f" từ DEFAULT_LAMBDA + dense window 200 step.")

        # [FIX-DIV-2] Restore hệ số boost diversity. sigma_min/ctx_noise_scale
        # là plain float attribute trên model — KHÔNG nằm trong state_dict()
        # nên model vừa load_state_dict ở trên vẫn đang ở giá trị CONSTRUCTOR
        # (args.sigma_min, ctx_noise_scale=0.01 default). Nếu checkpoint có
        # ghi nhận đã boost (factor > 1), áp lại NGAY tại đây — nếu không,
        # boost sẽ bị mất vĩnh viễn vì block diversity-check chỉ chạy đúng
        # 1 lần tại ep==PHASE_1_END (sẽ không chạy lại khi resume ở ep>19).
        ck_div_factor = ck.get("diversity_boost_factor", 1.0)
        if ck_div_factor and ck_div_factor > 1.0 + 1e-6:
            diversity_boost_factor = float(ck_div_factor)
            try:
                applied = []
                for attr_name, base_val in (("sigma_min", args.sigma_min),
                                             ("ctx_noise_scale", 0.01)):
                    if hasattr(m, attr_name):
                        setattr(m, attr_name, base_val * diversity_boost_factor)
                        applied.append(f"{attr_name}="
                                       f"{getattr(m, attr_name):.4f}")
                print(f"  [FIX-DIV-2] Restored diversity boost factor="
                      f"{diversity_boost_factor:.2f}x ({', '.join(applied)})")
            except Exception as e:
                print(f"  [FIX-DIV-2] Lỗi khi restore diversity boost: {e}")

        # [FIX-DIV-4] Override diversity_boost_factor sau khi restore.
        # Áp dụng dù checkpoint đã có factor>1 (giảm/tăng mức boost hiện
        # tại) hay factor==1.0 (set boost mới ngay từ resume này).
        # sigma_min/ctx_noise_scale luôn được tính lại từ BASE values
        # (args.sigma_min, 0.01) * override — không nhân dồn lên giá trị
        # đã boost trước đó, tránh double-boost.
        if args.diversity_boost_override is not None:
            old_factor = diversity_boost_factor
            new_factor = float(args.diversity_boost_override)
            if new_factor < 1.0:
                print(f"  [FIX-DIV-4] CẢNH BÁO: --diversity_boost_override="
                      f"{new_factor:.2f} < 1.0 — không hợp lệ (boost factor"
                      f" phải >= 1.0). Bỏ qua override, giữ factor="
                      f"{old_factor:.2f}x.")
            elif abs(new_factor - old_factor) < 1e-6:
                print(f"  [FIX-DIV-4] --diversity_boost_override="
                      f"{new_factor:.2f}x == factor hiện tại, không đổi.")
            else:
                diversity_boost_factor = new_factor
                try:
                    applied = []
                    for attr_name, base_val in (("sigma_min", args.sigma_min),
                                                  ("ctx_noise_scale", 0.01)):
                        if hasattr(m, attr_name):
                            setattr(m, attr_name, base_val * diversity_boost_factor)
                            applied.append(f"{attr_name}="
                                           f"{getattr(m, attr_name):.4f}")
                    init_noise_km_old = args.sigma_min * 2.5 * old_factor * 555.0
                    init_noise_km_new = args.sigma_min * 2.5 * new_factor * 555.0
                    print(f"  [FIX-DIV-4] Override diversity boost factor: "
                          f"{old_factor:.2f}x -> {new_factor:.2f}x "
                          f"({', '.join(applied)})")
                    print(f"  [FIX-DIV-4] Initial DDIM noise std: "
                          f"~{init_noise_km_old:.0f}km -> ~{init_noise_km_new:.0f}km"
                          f"  (training-time current_sigma noise ~19-50km"
                          f" tuỳ epoch)")
                    # diversity_boost_factor mới sẽ được ghi vào MỌI
                    # checkpoint lưu từ epoch này trở đi (threaded qua
                    # _save_ckpt/Saver.update ở training loop bên dưới),
                    # nên lần resume KẾ TIẾP sẽ tự dùng factor mới —
                    # không cần truyền lại --diversity_boost_override.
                except Exception as e:
                    print(f"  [FIX-DIV-4] Lỗi khi override diversity boost: {e}")

    try:
        model = torch.compile(model, mode="reduce-overhead")
        print("  torch.compile: ok")
    except:
        pass

    # ── Phase 4 encoder list (dùng khi freeze) ────────────────────────────
    def _get_encoder_params(m):
        """Trả về params của encoder để freeze ở phase 4."""
        enc = _unwrap(m)
        params = []
        for mod_name in ["net.spatial_enc", "net.enc_1d", "net.env_enc",
                          "net.bottleneck_pool", "net.bottleneck_proj",
                          "net.decoder_proj"]:
            parts = mod_name.split(".")
            mod = enc
            for p in parts:
                if hasattr(mod, p):
                    mod = getattr(mod, p)
                else:
                    mod = None
                    break
            if mod is not None:
                params.extend(list(mod.parameters()))
        return params

    _phase4_frozen = False

    def _maybe_freeze_encoder(ep, m):
        nonlocal _phase4_frozen
        if get_phase(ep) == 4 and not _phase4_frozen:
            enc_params = _get_encoder_params(m)
            for p in enc_params:
                p.requires_grad_(False)
            _phase4_frozen = True
            print(f"  [Phase 4] Froze encoder params"
                  f" (n={sum(1 for p in enc_params)})")
            for g in opt.param_groups:
                g["lr"] = g["lr"] * 0.1
            print(f"  [Phase 4] LR reduced to {opt.param_groups[0]['lr']:.2e}")

    # ── EMA guard state ────────────────────────────────────────────────────
    best_easy_ade_ema      = float("inf")
    ema_rollback_count     = 0
    _prev_phase            = get_phase(start) if start > 0 else 1
    # EMA của easy_ADE — robust hơn raw value với variance fast_eval ±20-40km
    _ema_state  = {"smooth": float("inf")}
    EMA_ALPHA   = 0.35   # weight cho current value (bias về recent)
    EMA_THRESH  = args.ema_guard_threshold   # km threshold để trigger rollback

    # [FIX-DIV-2] flag để chỉ áp dụng diversity-noise-boost 1 lần.
    # Nếu đã restore boost factor>1 từ checkpoint (resume), coi như đã
    # boosted — không chạy lại block boost (vốn cũng chỉ chạy ở ep==19).
    _diversity_boosted = diversity_boost_factor > 1.0 + 1e-6

    print(f"  Training: {nstep} steps/ep, start ep={start}")
    print("=" * 72)
    ts = time.perf_counter()

    for ep in range(start, args.num_epochs):
        phase = get_phase(ep)

        # [FIX-GN-2] GradNorm phase transition — KHÔNG reset λ
        if phase != _prev_phase:
            gradnorm.phase_reset(phase)
            _prev_phase = phase
            print(f"  [Phase {phase}] Chuyển phase tại epoch {ep}")

            # Reset best_easy_ade_ema và smooth EMA khi chuyển phase
            # (baseline easy_ADE giữa các phase có thể khác nhau tự nhiên,
            #  không nên coi sự khác biệt này là "regression")
            best_easy_ade_ema      = float("inf")
            ema_rollback_count     = 0
            _ema_state["smooth"]   = float("inf")
            print(f"  [EMA Guard] Reset best/smooth EMA khi chuyển phase")

        # [FIX-THRESH-1] Chỉ compute hard threshold 1 lần khi vào phase 2.
        if phase >= 2 and not _threshold_computed:
            global_hard_threshold = precompute_hard_threshold(trl, dev, n_batches=50)
            _threshold_computed = True
            print(f"  [FIX-THRESH-1] global_hard_threshold={global_hard_threshold:.4f}"
                  f" — computed once, reused for all epochs")

        # [G-H] Phase 4: freeze encoder
        _maybe_freeze_encoder(ep, model)

        model.train()
        sl = 0.0
        t0 = time.perf_counter()
        # [FIX-DIV-3] accumulate diversity_km_train (chỉ tính step finite)
        div_km_sum, div_km_cnt = 0.0, 0

        # Lấy lambda hiện tại từ GradNorm
        lambda_dict = gradnorm.get_lambda_dict()

        for i, batch in enumerate(trl):
            bl    = move(list(batch), dev)
            obs_t = bl[0]

            # α, easy_frac từ SmoothScheduler (per-batch)
            alpha_hard    = smooth_sched.get_alpha_hard()
            easy_frac     = smooth_sched.get_easy_frac()
            sel_weight    = smooth_sched.get_selector_weight()
            train_selector = smooth_sched.is_selector_active() and (sel_weight > 0.01)

            # Dùng global threshold (phase>=2) hoặc zeros (phase 1)
            if phase >= 2 and global_hard_threshold is not None:
                with torch.no_grad():
                    is_hard = classify_hard_easy_global(
                        obs_t[:, :, :2], global_hard_threshold).to(dev)
                is_hard = enforce_easy_frac(is_hard, easy_frac)
            else:
                # Phase 1: alpha_hard=0 → hard loss = 0, không cần phân loại
                is_hard = torch.zeros(
                    obs_t.shape[1], dtype=torch.bool, device=dev)

            # [FIX-DIV-3] L_diversity weight (0 ở phase 1, ramp từ phase 2)
            div_loss_w = (smooth_sched.get_diversity_weight(
                              target=args.diversity_loss_weight)
                          if args.use_diversity_loss else 0.0)

            with autocast(device_type="cuda", enabled=args.use_amp):
                bd = model.get_loss_breakdown(
                    bl,
                    epoch=ep,
                    alpha_hard=alpha_hard,
                    is_hard=is_hard,
                    train_selector=train_selector,
                    lambda_dict=lambda_dict,
                    diversity_loss_weight=div_loss_w,
                    diversity_target_km=_diversity_target_km,
                )

            opt.zero_grad()
            scaler.scale(bd["total"]).backward()
            scaler.unscale_(opt)

            # [FIX-GN-1 + FIX-PERF-1] GradNorm: equal-contribution scheme.
            # Chỉ cần loss VALUES (raw, detached) — không cần gradient norm
            # của shared params (xóa hoàn toàn block named_parameters() cũ,
            # vốn lặp qua TOÀN BỘ params mỗi step nhưng kết quả bị ignore).
            try:
                loss_vals_float = {}
                for t in GRADNORM_TERMS:
                    val = bd.get(t)
                    if val is None:
                        continue
                    fval = float(val.item()) if torch.is_tensor(val) else float(val)
                    if np.isfinite(fval) and fval > 0:
                        loss_vals_float[t] = fval

                if loss_vals_float:
                    gradnorm.update(loss_vals_float)
                    lambda_dict = gradnorm.get_lambda_dict()
            except Exception:
                pass  # GradNorm không critical

            # BUG-8: clip chỉ backbone
            torch.nn.utils.clip_grad_norm_(backbone_params, args.grad_clip)

            scaler.step(opt)
            scaler.update()
            # BUG-7: unconditional
            try: sched.step()
            except: pass
            model.ema_update()

            smooth_sched.step()

            sl += bd["total"].item()

            # [FIX-DIV-3] track training-time diversity (nan khi tắt/lỗi)
            _dkm = bd.get("diversity_km_train", float("nan"))
            if np.isfinite(_dkm):
                div_km_sum += _dkm
                div_km_cnt += 1

            if i % 20 == 0:
                lr     = opt.param_groups[0]["lr"]
                tot    = bd["total"].item()
                warn   = " [!>10]" if tot > 10.0 else ""
                sw_st  = _unwrap(model).step_weights.stats()
                gn_log = gradnorm.log_stats()
                n_hard_batch = int(is_hard.sum().item())

                lam_str = " ".join(
                    f"λ_{t.replace('l_','').replace('_reg','')}="
                    f"{gn_log.get(f'λ_{t}', 1.0):.2f}"
                    for t in GRADNORM_TERMS
                )

                print(
                    f"  [{ep:>3}|Ph{phase}][{i:>3}/{nstep}]"
                    f"  tot={tot:.3f}{warn}"
                    f"  fm={bd.get('l_fm',0):.4f}"
                    f"  dpe={bd.get('dpe',0):.4f}"
                    f"  vel={bd.get('vel_reg',0):.4f}"
                    f"  hard={bd.get('l_hard_total',0):.4f}"
                    f"  {smooth_sched.log_stats()}"
                    f"  n_h={n_hard_batch}/{obs_t.shape[1]}"
                    f"  sw72={sw_st['sw_72h']:.2f}"
                    f"  gn_d={gn_log.get('gn_dense',0)}"
                    f"  {lam_str}"
                    f"  lr={lr:.2e}"
                    f"  div_w={div_loss_w:.3f}"
                    f"  div_km={_dkm:.1f}"
                )

        avt  = sl / nstep
        t_ep = time.perf_counter() - t0

        # BUG-5: val với epoch=-1 → skip augmentation
        model.eval()
        vls = 0.0
        with torch.no_grad():
            for batch in vl:
                bv = move(list(batch), dev)
                with autocast(device_type="cuda", enabled=args.use_amp):
                    vls += model.get_loss(bv, epoch=-1).item()
        avv = vls / len(vl)

        lr_cur   = opt.param_groups[0]["lr"]
        gn_stats = gradnorm.log_stats()
        use_sel_eval  = smooth_sched.is_selector_active()
        eval_threshold = global_hard_threshold if phase >= 2 else None

        lam_epoch = " ".join(
            f"λ_{t.replace('l_','').replace('_reg','')}="
            f"{gn_stats.get(f'λ_{t}', 1.0):.3f}"
            for t in GRADNORM_TERMS
        )

        # [FIX-DIV-3] avg training-time diversity (nan nếu weight=0 cả epoch)
        avg_div_km_train = (div_km_sum / div_km_cnt) if div_km_cnt > 0 else float("nan")

        print(f"  Epoch {ep:>3}|Ph{phase}"
              f" | train={avt:.4f} val={avv:.4f}"
              f" | {smooth_sched.log_stats()}"
              f" | {lam_epoch}"
              f" | div_km_train={avg_div_km_train:.1f}"
              f" | lr={lr_cur:.2e} | {t_ep:.0f}s")

        # [G-G] Fast eval với split monitoring (dùng fast ddim)
        rf = evaluate_split(model, vsub, dev,
                             tag=f"FAST ep{ep}",
                             steps=args.fast_ddim,
                             use_selector=use_sel_eval,
                             global_hard_threshold=eval_threshold)
        fast_saver.update(rf, model, args.output_dir, ep, opt, sched, avt, avv,
                          tag="fast",
                          smooth_sched_step=smooth_sched.global_step,
                          global_hard_threshold=global_hard_threshold,
                          diversity_boost_factor=diversity_boost_factor,
                          gradnorm_state=gradnorm.state_dict())

        # ── [G-E + FIX-EMA-2] EMA guard: kiểm tra easy_ADE sau mỗi epoch ──
        #
        # [FIX-EMA-2] Vấn đề cũ: pullback_steps=nstep*2, rollback dày đặc
        # trong vùng α-ramp (ramp chỉ rộng ~35 step quanh điểm mid) → mỗi
        # rollback trong vùng ramp kéo step về TRƯỚC vùng ramp → α=0 lại
        # → α=0.000 suốt cả phase 2 (xem log gốc, 30 epoch toàn α=0).
        #
        # Fix:
        #   - pullback_steps = nstep // 2 (giảm 1 nửa so với nstep*2)
        #   - CHỈ pullback schedule nếu α hiện tại < ALPHA_RAMP_DONE_THRESHOLD
        #     (còn trong vùng ramp). Nếu α đã ramp xong (>=0.25), rollback
        #     chỉ restore weights, KHÔNG pullback schedule — tránh kẹt
        #     vĩnh viễn do rollback lặp lại sau khi đã vượt vùng ramp.
        easy_ade_curr = rf.get("easy_ADE", rf.get("ADE", float("inf")))
        if np.isfinite(easy_ade_curr):
            # Update EMA
            if _ema_state["smooth"] == float("inf"):
                _ema_state["smooth"] = easy_ade_curr
            else:
                _ema_state["smooth"] = ((1 - EMA_ALPHA) * _ema_state["smooth"]
                                        + EMA_ALPHA * easy_ade_curr)
            smooth_val = _ema_state["smooth"]

            if smooth_val < best_easy_ade_ema:
                best_easy_ade_ema = smooth_val
                print(f"  [EMA Guard] New best smooth_easy_ADE={smooth_val:.1f}"
                      f" (raw={easy_ade_curr:.1f})")

            elif (smooth_val - best_easy_ade_ema) > EMA_THRESH:
                ema_rollback_count += 1
                print(f"\n  [EMA GUARD] smooth_easy_ADE tăng bền vững "
                      f"{smooth_val - best_easy_ade_ema:.1f} km"
                      f" (best={best_easy_ade_ema:.1f} smooth={smooth_val:.1f}"
                      f" raw={easy_ade_curr:.1f})")
                print(f"  [EMA GUARD] Rollback #{ema_rollback_count}")

                # Restore weights từ EMA
                m_raw   = _unwrap(model)
                ema_obj = getattr(m_raw, "_ema", None)
                if ema_obj:
                    try:
                        bk = ema_obj.apply_to(model)
                        _save_ckpt(
                            os.path.join(args.output_dir,
                                         f"ema_guard_rollback_ep{ep}.pth"),
                            ep, model, opt, sched, full_saver, avt, avv,
                            smooth_sched_step=smooth_sched.global_step,
                            global_hard_threshold=global_hard_threshold,
                            diversity_boost_factor=diversity_boost_factor,
                            gradnorm_state=gradnorm.state_dict(),
                            extra={"ema_rollback": True,
                                   "easy_ade_trigger": easy_ade_curr},
                        )
                        ema_obj.restore(model, bk)
                        print(f"  [EMA GUARD] Saved rollback ckpt, "
                              f"restored current weights for continued training")
                    except Exception as e:
                        print(f"  [EMA GUARD] Apply/restore failed: {e}")

                # [FIX-EMA-2] Chỉ pullback schedule nếu α còn trong vùng ramp
                cur_alpha = smooth_sched.get_alpha_hard()
                if cur_alpha < ALPHA_RAMP_DONE_THRESHOLD:
                    pullback_steps = nstep // 2  # giảm so với nstep*2
                    smooth_sched._global_step = max(
                        smooth_sched._phase2_start_step,
                        smooth_sched._global_step - pullback_steps
                    )
                    print(f"  [EMA GUARD] α còn trong vùng ramp ({cur_alpha:.3f}"
                          f" < {ALPHA_RAMP_DONE_THRESHOLD}) → pullback"
                          f" {pullback_steps} step. α mới="
                          f"{smooth_sched.get_alpha_hard():.3f}")
                else:
                    print(f"  [EMA GUARD] α đã ramp xong ({cur_alpha:.3f}"
                          f" >= {ALPHA_RAMP_DONE_THRESHOLD}) → KHÔNG pullback"
                          f" schedule, chỉ restore weights")

        # [G-F + FIX-DIV-2] Diversity check cuối phase 1
        div_score = None
        if ep == PHASE_1_END:
            print(f"\n  {'='*50}")
            print(f"  [CHECKPOINT] Cuối phase 1 — kiểm tra diversity...")
            baseline_easy_ade = rf.get("easy_ADE", float("nan"))
            print(f"  (baseline FAST easy_ADE trước boost: "
                  f"{baseline_easy_ade:.1f}km — so sánh với ep{ep+1}"
                  f" FAST easy_ADE để kiểm tra trade-off boost)")
            div_score = check_diversity(model, vsub, dev)
            print(f"  Diversity score: {div_score:.1f} km")
            if div_score < args.diversity_threshold:
                print(f"  [CẢNH BÁO R1] diversity={div_score:.1f} km"
                      f" < {args.diversity_threshold} km")
                print(f"  → FM candidates không đủ đa dạng → selector sẽ kém")

                # [FIX-DIV-2] sigma_min/ctx_noise_scale CHỈ ảnh hưởng
                # sample() (initial DDIM noise + CFG context noise 3 step
                # đầu) — KHÔNG ảnh hưởng train CFM loss (_cfm_noisy dùng
                # current_sigma từ _sigma_schedule(epoch), không dùng
                # self.sigma_min; forward_with_ctx lúc train không truyền
                # noise_scale). → an toàn để boost MẠNH hơn FIX-DIV-1 (×1.5).
                # Boost LẶP + recheck thật, cap tổng diversity_boost_max,
                # tối đa diversity_boost_iters vòng.
                #
                # TRADE-OFF: boost làm TỪNG candidate noisy hơn → có thể
                # làm FAST/RAW ADE/CTE từ ep20 xấu hơn ep19 (đổi accuracy
                # per-candidate lấy diversity cho selector phase 3). So
                # sánh easy_ADE ep19 vs ep20 (FAST eval) sau khi có log:
                # nếu ep20 tệ hơn rõ rệt KHÔNG do phase-shock (đã fix ở
                # FIX-GN-2/FIX-EMA-2), giảm --diversity_boost_max lần sau.
                if not _diversity_boosted:
                    try:
                        m_div = _unwrap(model)
                        boost_attrs = [a for a in ("sigma_min", "ctx_noise_scale")
                                       if hasattr(m_div, a)]
                        if not boost_attrs:
                            print(f"  [FIX-DIV-2] Bỏ qua: model không có"
                                  f" attribute sigma_min/ctx_noise_scale"
                                  f" trực tiếp — cần xử lý thủ công"
                                  f" (vd thêm L_diversity loss trong model).")
                        else:
                            base_vals = {a: getattr(m_div, a) for a in boost_attrs}
                            BOOST_STEP     = args.diversity_boost_step
                            MAX_TOTAL_BOOST = args.diversity_boost_max
                            MAX_ITERS      = args.diversity_boost_iters
                            total_boost = 1.0
                            for it in range(1, MAX_ITERS + 1):
                                total_boost = min(total_boost * BOOST_STEP,
                                                   MAX_TOTAL_BOOST)
                                for a in boost_attrs:
                                    setattr(m_div, a, base_vals[a] * total_boost)
                                div_score = check_diversity(model, vsub, dev)
                                attr_str = ", ".join(
                                    f"{a}={getattr(m_div, a):.4f}"
                                    for a in boost_attrs)
                                print(f"  [FIX-DIV-2] vòng {it}: "
                                      f"total_boost={total_boost:.2f}x "
                                      f"({attr_str}) → diversity={div_score:.1f}km")
                                if (div_score >= args.diversity_threshold
                                        or total_boost >= MAX_TOTAL_BOOST):
                                    break
                            # [FIX-DIV-2] Lưu factor để thread vào checkpoint
                            # (resume sẽ áp lại, xem khối resume phía trên).
                            diversity_boost_factor = total_boost
                            if div_score >= args.diversity_threshold:
                                print(f"  [FIX-DIV-2] OK: diversity={div_score:.1f}km"
                                      f" >= {args.diversity_threshold}km"
                                      f" sau boost {total_boost:.2f}x")
                            else:
                                print(f"  [FIX-DIV-2] VẪN CHƯA ĐỦ sau boost"
                                      f" {total_boost:.2f}x (cap {MAX_TOTAL_BOOST}x):"
                                      f" diversity={div_score:.1f}km"
                                      f" < {args.diversity_threshold}km."
                                      f" → cần can thiệp sâu hơn (L_diversity"
                                      f" loss trong model, hoặc tăng"
                                      f" n_ensemble/num_ensemble).")
                        _diversity_boosted = True
                    except Exception as e:
                        print(f"  [FIX-DIV-2] Lỗi khi áp dụng noise boost: {e}")
            else:
                print(f"  [OK] diversity={div_score:.1f} km >= "
                      f"{args.diversity_threshold} km → tiến hành phase 2")
            print(f"  {'='*50}\n")


        # Đánh giá đầy đủ mỗi val_freq epoch
        if ep % args.val_freq == 0:
            em = getattr(_unwrap(model), "_ema", None)
            rr = evaluate_split(model, vl, dev,
                                  tag=f"RAW ep{ep}",
                                  steps=args.full_ddim,
                                  use_selector=use_sel_eval,
                                  global_hard_threshold=eval_threshold)
            full_saver.update(rr, model, args.output_dir, ep, opt, sched,
                              avt, avv, tag="raw",
                              smooth_sched_step=smooth_sched.global_step,
                              global_hard_threshold=global_hard_threshold,
                              diversity_boost_factor=diversity_boost_factor,
                              gradnorm_state=gradnorm.state_dict())

            if em and ep >= 3:
                re = evaluate_split(model, vl, dev,
                                     tag=f"EMA ep{ep}",
                                     ema=em,
                                     steps=args.full_ddim,
                                     use_selector=use_sel_eval,
                                     global_hard_threshold=eval_threshold)
                full_saver.update(re, model, args.output_dir, ep, opt, sched,
                                  avt, avv, tag="ema",
                                  smooth_sched_step=smooth_sched.global_step,
                                  global_hard_threshold=global_hard_threshold,
                                  diversity_boost_factor=diversity_boost_factor,
                                  gradnorm_state=gradnorm.state_dict())

        # Periodic checkpoint
        if ep % 10 == 0 or ep == args.num_epochs - 1:
            _save_ckpt(
                os.path.join(args.output_dir, f"ckpt_ep{ep:03d}.pth"),
                ep, model, opt, sched, full_saver, avt, avv,
                smooth_sched_step=smooth_sched.global_step,
                global_hard_threshold=global_hard_threshold,
                diversity_boost_factor=diversity_boost_factor,
                gradnorm_state=gradnorm.state_dict(),
                extra={"phase": phase,
                       "alpha_hard": smooth_sched.get_alpha_hard(),
                       "diversity": div_score},
            )

        if full_saver.stop:
            print(f"  Early stop ep={ep}")
            break

    th = (time.perf_counter() - ts) / 3600.0
    print(f"\n  Best: ADE={full_saver.ba:.1f}  72h={full_saver.b7:.0f}"
          f"  ATE={full_saver.bat:.1f}  CTE={full_saver.bc:.1f}"
          f"  easy_ADE={full_saver.best_easy_ade:.1f}"
          f"  ({th:.2f}h)")

    # Post-training test
    if args.eval_test_after_train:
        print("\n" + "="*72 + "\n  POST-TRAINING TEST\n" + "="*72)
        try: _, tl2 = data_loader(args, {"root": args.dataset_root, "type": "test"}, test=True)
        except: print("  No test → using val"); tl2 = vl

        for fn, lb in [("best_raw.pth", "RAW"), ("best_ema.pth", "EMA"),
                        ("best_ade_raw.pth", "BEST_ADE"),
                        ("best_cte_raw.pth", "BEST_CTE")]:
            pp = os.path.join(args.output_dir, fn)
            if not os.path.exists(pp): continue
            ck = torch.load(pp, map_location=dev)
            _unwrap(model).load_state_dict(ck["model_state_dict"], strict=False)
            em = getattr(_unwrap(model), "_ema", None)
            if em and ck.get("ema_shadow"):
                for k, v in ck["ema_shadow"].items():
                    if k in em.shadow: em.shadow[k].copy_(v.to(dev))
            ckpt_threshold = ck.get("global_hard_threshold", None)
            r = evaluate_split(model, tl2, dev, tag=f"TEST/{lb}",
                                steps=args.full_ddim,
                                use_selector=(get_phase(ck.get("epoch", 80)) >= 3),
                                global_hard_threshold=ckpt_threshold)
            print(f"\n  --- {lb} (ep={ck.get('epoch','?')}) ---")
            for key, ref in [("ADE", 172.68), ("72h", 321.39),
                              ("ATE", 142.21), ("CTE", 42.04)]:
                v = r.get(key, float("nan"))
                mk = "BEAT!" if np.isfinite(v) and v < ref else f"need {ref:.0f}"
                print(f"    {key:<8}: {v:>8.1f} km  [{mk}  gap:{v-ref:+.1f}]")
            ea = r.get("easy_ADE", float("nan"))
            ha = r.get("hard_ADE", float("nan"))
            print(f"    easy_ADE={ea:.1f}  hard_ADE={ha:.1f}")

    print("=" * 72)


if __name__ == "__main__":
    args = get_args()
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    main(args)