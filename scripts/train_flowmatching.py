# # # """
# # # train_v59_strategy.py — TC-FlowMatching v59-Strategy Training Script [FIXED-v2]
# # # ══════════════════════════════════════════════════════════════════════════════

# # # BASE: phiên bản với FIX-STABILITY-1/2/3, FIX-ALPHA-TIMING, FIX-EMA-SMOOTH, FIX-D
# # # (đã chạy và có log epoch 0-51)

# # # CÁC BUG MỚI ĐƯỢC FIX TRONG PHIÊN BẢN NÀY (dựa trên phân tích log thực tế):

# # #   [FIX-GN-1] GradNorm "loss-ratio" KHÔNG có equilibrium → runaway có hệ thống
# # #     Vấn đề cũ: r_i = raw_loss_i / mean(raw_loss_j) — KHÔNG phụ thuộc λ_i.
# # #     l_dpe có raw scale lớn hơn cấu trúc so với l_vel_reg/l_heading/l_accel
# # #     → r_dpe > 1 vĩnh viễn → λ_dpe luôn bị "tăng" → chạy tới ceiling (4.18)
# # #     → λ_vel/head/accel bị đẩy xuống floor (0.01) → 3 objective gần như bị tắt.
# # #     Log xác nhận: λ_dpe 1.00→4.18 (ep20→46), λ_vereg/heading/accel →0.01.

# # #     Fix: "equal-contribution" scheme.
# # #       contribution_i = λ_i * l_i  (phần đóng góp THỰC vào total loss)
# # #       ratio_i = contribution_i / mean(contribution_j)
# # #       ratio_i > 1 (đóng góp quá nhiều) → GIẢM λ_i
# # #       ratio_i < 1 (đóng góp quá ít)    → TĂNG λ_i
# # #     → có feedback âm thật: λ_i tăng → contribution_i tăng → ratio_i tăng
# # #       → bị kéo giảm lại → equilibrium λ_i ≈ mean_contrib / l_i.
# # #     Tự động bù trừ scale khác nhau giữa các loss term, không runaway.

# # #   [FIX-GN-2] phase_reset() KHÔNG reset λ về 1.0 nữa
# # #     Vấn đề cũ: λ_dpe=3.11→1.0 đột ngột tại ep20 (và tương tự ep50)
# # #     → tổng loss đổi đột ngột → easy_ADE nhảy 278→369km (ep20), 278→302 (ep51).
# # #     L_base (compute_st_trans_loss) không đổi giữa các phase — chỉ cộng thêm
# # #     L_hard/L_sel bên ngoài GradNorm → không có lý do để reset λ.
# # #     Fix: giữ nguyên λ hiện tại, chỉ reset optimizer momentum + dense window
# # #     ngắn (50 step) để re-adapt nhẹ nếu landscape có dịch nhẹ.

# # #   [FIX-EMA-2] EMA guard pullback có thể làm α kẹt vĩnh viễn ở 0
# # #     Vấn đề cũ: pullback_steps=nstep*2, rollback dày đặc trong vùng α-ramp
# # #     (ramp chỉ rộng ~35 step quanh epoch 24.5) → mỗi rollback trong vùng ramp
# # #     kéo step về trước vùng ramp → α=0 lại → α=0.000 suốt cả phase 2 (30 epoch).
# # #     Fix: chỉ pullback schedule nếu α hiện tại < 0.25 (còn trong vùng ramp).
# # #     Nếu α đã ramp xong (>=0.25), rollback chỉ restore weights, KHÔNG pullback
# # #     schedule. Giảm pullback_steps = nstep//2.

# # #   [FIX-PERF-1] Xóa dead/wasteful computation mỗi step
# # #     Code cũ lặp qua TOÀN BỘ named_parameters() mỗi step để tìm
# # #     "net.ctx_fc2.weight", tính lambda_grads_approx rồi truyền vào update()
# # #     — nhưng update() ghi rõ "lambda_grads: unused, ignore". Hoàn toàn lãng phí.
# # #     Fix: xóa block này, chỉ extract loss_vals_float và gọi update() trực tiếp.

# # #   [FIX-CKPT-1] _save_ckpt lưu smooth_sched_step + global_hard_threshold
# # #     cho MỌI checkpoint (trước chỉ có trong periodic ckpt's "extra").

# # #   [FIX-DIV-1] Diversity collapse (7.8km << 50km) — tự động xử lý
# # #     Theo strategy doc mục R1: "Tăng noise std ×1.5".
# # #     Fix: tự động ×1.5 sigma_min và ctx_noise_scale của model khi
# # #     diversity < threshold sau phase 1.

# # #   [FIX-THRESH-1] precompute threshold dùng flag riêng _threshold_computed
# # #     thay vì so sánh == 0.5 (fragile).

# # #   [FIX-EASYFRAC4] get_easy_frac() phase 4 dùng sigmoid (consistent với phase 2/3)

# # # CÁC BUG MỚI ĐƯỢC FIX TRONG v3 (dựa trên log thực tế chạy v2, epoch 0-6):

# # #   [FIX-GN-3] "equal-contribution" (FIX-GN-1) vẫn SAI — thay bằng
# # #     "anchor + target-ratio".

# # #     Log v2 ep0-6: λ_dpe GIẢM liên tục 1.00→0.335 (đang đi về floor 0.1),
# # #     λ_vel/head/accel TĂNG 1.00→1.17-1.20. val loss giảm đẹp (2.42→2.34,
# # #     THẤP NHẤT) nhưng easy_ADE TĂNG GẦN GẤP ĐÔI (408.9km log gốc → 705.7km)
# # #     và ADE trend ep2-6 ĐANG TĂNG (574→598→575→619→649). EMA Guard rollback
# # #     ngay tại ep6.

# # #     Root cause: l_dpe (Huber loss vị trí — objective CHÍNH, gắn trực tiếp
# # #     với ADE) có raw scale LỚN HƠN l_vel_reg/heading/accel KHÔNG PHẢI do lệch
# # #     scale ngẫu nhiên, mà vì đây là objective khó giảm hơn (regularizer phụ
# # #     dễ về gần 0). "Equal-contribution" coi đó là mất cân bằng cần sửa →
# # #     đẩy λ_dpe XUỐNG để "cân bằng" với 4 regularizer phụ → model dồn capacity
# # #     tối ưu 4 term phụ "dễ" (val loss giảm đẹp) nhưng hy sinh độ chính xác vị
# # #     trí → ADE tăng. Cùng pattern với log gốc ep49→50 (λ_dpe reset 3.92→1.0
# # #     đột ngột → easy_ADE 220→263km, "No improve" liên tục 4 epoch).

# # #     Fix: l_dpe = ANCHOR, λ_dpe CỐ ĐỊNH = 1.20 (= default hand-tuned trong
# # #     model, KHÔNG BAO GIỜ bị GradNorm điều chỉnh) → gradient cho vị trí
# # #     không bao giờ bị giảm dưới mức hand-tuned. 4 term phụ (vel_reg, heading,
# # #     speed, accel) được GradNorm điều chỉnh để duy trì TỶ LỆ ĐÓNG GÓP MỤC
# # #     TIÊU so với (λ_dpe·l_dpe), lấy từ default weights hand-tuned trong
# # #     model gốc (1.20, 1.40, 0.40, 0.05, 0.01):
# # #         target_contrib_i = DEFAULT_LAMBDA[i] * l_dpe
# # #         ratio_i = (λ_i · l_i) / target_contrib_i
# # #         ratio_i > 1 → giảm λ_i ; ratio_i < 1 → tăng λ_i
# # #     Equilibrium: λ_i ≈ DEFAULT_LAMBDA[i] · l_dpe / l_i — tự bù trừ scale RAW
# # #     của term i trôi theo training (đúng mục đích GradNorm), nhưng KHÔNG
# # #     đụng l_dpe và tôn trọng tỷ lệ quan trọng đã hand-tune giữa các term phụ.
# # #     Không cần renorm sum=5 nữa (anchor cố định, không có sum constraint).
# # #     Clamp λ_aux ∈ [0.02, 8.0] (range rộng hơn vì target_ratio chênh nhau
# # #     0.008–1.167 giữa các term).

# # #   [FIX-DIV-2] FIX-DIV-1 (×1.5 noise cố định) KHÔNG đủ.
# # #     Log gốc: diversity=8.3km << 50km (gap ~6x), ×1.5 → ~12.5km vẫn cách xa.
# # #     sigma_min và ctx_noise_scale CHỈ ảnh hưởng sample()/inference (initial
# # #     DDIM noise + CFG context noise 3 step đầu) — KHÔNG ảnh hưởng training
# # #     loss (_cfm_noisy dùng current_sigma từ _sigma_schedule(epoch), không
# # #     dùng self.sigma_min; forward_with_ctx lúc train không truyền noise_scale)
# # #     → có thể boost mạnh hơn nhiều mà không ảnh hưởng ổn định training.
# # #     Fix: boost LẶP — mỗi vòng ×1.8 (sigma_min và ctx_noise_scale), sau đó
# # #     chạy lại check_diversity(); lặp tối đa 3 vòng hoặc đến khi
# # #     diversity >= threshold; tổng boost factor cap ở 6.0x để tránh initial
# # #     noise quá lớn làm DDIM không hội tụ về trajectory hợp lý.

# # # GIỮ NGUYÊN từ phiên bản trước:
# # #   [G-A] Easy/hard pipeline với GLOBAL threshold
# # #   [G-C] easy_frac enforcement
# # #   [G-D] α smooth warm-up theo BATCH (FIX-ALPHA-TIMING: mid=15% of phase)
# # #   [G-E] EMA guard với EMA-smoothed easy_ADE (FIX-EMA-SMOOTH)
# # #   [G-F] Diversity check sau phase 1
# # #   [G-G] Split monitoring
# # #   [G-H] Phase 4: freeze encoder
# # #   BUG-5/6/7/8 từ v59-tweaked
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

# # # # [FIX-GN-3] l_dpe là ANCHOR — λ_dpe cố định, KHÔNG bị GradNorm điều chỉnh.
# # # # Giá trị = default hand-tuned trong compute_st_trans_loss (total_default).
# # # ANCHOR_TERM  = "l_dpe"
# # # AUX_TERMS    = ["l_vel_reg", "l_heading", "l_speed", "l_accel"]
# # # DEFAULT_LAMBDA = {
# # #     "l_dpe":     1.20,
# # #     "l_vel_reg": 1.40,
# # #     "l_heading": 0.40,
# # #     "l_speed":   0.05,
# # #     "l_accel":   0.01,
# # # }
# # # AUX_LAMBDA_MIN = 0.02
# # # AUX_LAMBDA_MAX = 8.0

# # # # Alpha threshold để coi là "đã ramp xong" — dùng cho EMA guard pullback gate
# # # ALPHA_RAMP_DONE_THRESHOLD = 0.25


# # # # ─────────────────────────────────────────────────────────────────────────────
# # # #  [FIX-GN-3] GradNorm implementation — anchor + target-ratio scheme
# # # #
# # # #  Thuật toán:
# # # #    - l_dpe = ANCHOR. λ_dpe CỐ ĐỊNH = DEFAULT_LAMBDA["l_dpe"] = 1.20.
# # # #      → gradient cho l_dpe (objective chính, gắn trực tiếp ADE) không
# # # #        bao giờ bị GradNorm giảm xuống dưới mức hand-tuned.
# # # #    - Với mỗi term phụ i ∈ AUX_TERMS:
# # # #        contribution_i    = λ_i * l_i
# # # #        target_contrib_i  = DEFAULT_LAMBDA[i] * l_dpe
# # # #        ratio_i           = contribution_i / target_contrib_i
# # # #        ratio_i > 1 (đóng góp vượt tỷ lệ mục tiêu) → GIẢM λ_i → grad > 0
# # # #        ratio_i < 1 (đóng góp dưới tỷ lệ mục tiêu) → TĂNG λ_i → grad < 0
# # # #    - Equilibrium: λ_i ≈ DEFAULT_LAMBDA[i] * l_dpe / l_i
# # # #      Nếu l_i == l_dpe thì λ_i == DEFAULT_LAMBDA[i] (đúng hand-tuned gốc).
# # # #      Nếu l_i trôi scale theo training, λ_i tự bù trừ để giữ TỶ LỆ ĐÓNG GÓP
# # # #      (không phải giá trị λ) bằng đúng tỷ lệ hand-tuned.
# # # #    - Không có sum constraint → không cần renorm. Mỗi λ_i clamp riêng vào
# # # #      [AUX_LAMBDA_MIN, AUX_LAMBDA_MAX] (range rộng vì DEFAULT_LAMBDA chênh
# # # #      nhau tới ~150x giữa l_vel_reg và l_accel).
# # # # ─────────────────────────────────────────────────────────────────────────────

# # # class GradNormManager:
# # #     """
# # #     [FIX-GN-3] Quản lý GradNorm — anchor + target-ratio scheme.

# # #     l_dpe là ANCHOR (λ cố định = DEFAULT_LAMBDA["l_dpe"] = 1.20, không nằm
# # #     trong self.lambdas, không bao giờ bị update). Chỉ AUX_TERMS
# # #     (l_vel_reg, l_heading, l_speed, l_accel) được GradNorm điều chỉnh,
# # #     với equilibrium λ_i ≈ DEFAULT_LAMBDA[i] * l_dpe / l_i.

# # #     [FIX-GN-2] phase_reset(): KHÔNG reset λ — chỉ reset optimizer momentum
# # #     + bật dense update ngắn (50 step) để re-adapt nhẹ nếu landscape có dịch
# # #     (do thêm gradient từ L_hard/L_sel chảy qua shared encoder ở phase mới).
# # #     """
# # #     def __init__(self, terms: List[str], alpha_gn: float = 1.5,
# # #                  lr: float = 1e-3, device=None,
# # #                  lambda_min: float = AUX_LAMBDA_MIN,
# # #                  lambda_max: float = AUX_LAMBDA_MAX):
# # #         # terms: full GRADNORM_TERMS list (bao gồm anchor) — giữ để
# # #         # tương thích các nơi khác lặp qua GRADNORM_TERMS khi log/extract.
# # #         self.terms      = terms
# # #         self.aux_terms  = [t for t in terms if t != ANCHOR_TERM]
# # #         self.alpha_gn   = alpha_gn
# # #         self.device     = device or torch.device("cpu")
# # #         self.lambda_min = lambda_min
# # #         self.lambda_max = lambda_max

# # #         # λ_i cho AUX_TERMS, init = DEFAULT_LAMBDA[i] (equilibrium đúng nếu
# # #         # l_i == l_dpe ngay từ đầu — điểm khởi đầu hợp lý hơn 1.0 đồng nhất).
# # #         self.lambdas = {t: torch.full((1,), DEFAULT_LAMBDA[t],
# # #                                        device=self.device, requires_grad=True)
# # #                         for t in self.aux_terms}
# # #         self.opt = optim.Adam(list(self.lambdas.values()), lr=lr)

# # #         self._step             = 0
# # #         self._update_freq      = 5      # update mỗi 5 steps (bình thường)
# # #         self._dense_steps_left = 200    # dense ở lúc khởi động training
# # #         self._current_phase    = 1

# # #     # ─────────────────────────────────────────────────────────────────
# # #     # [FIX-GN-4] state_dict/load_state_dict — persist λ_aux, optimizer
# # #     # GradNorm, _step, _dense_steps_left, _current_phase qua checkpoint.
# # #     #
# # #     # BUG trước fix: GradNormManager được khởi tạo lại từ constructor
# # #     # mỗi lần script chạy (kể cả khi resume) → λ_aux reset về
# # #     # DEFAULT_LAMBDA (1.40/0.40/0.05/0.01), _dense_steps_left reset về
# # #     # 200, _step reset về 0 → 30 epoch điều chỉnh λ bị MẤT HOÀN TOÀN,
# # #     # GradNorm phải "học lại từ đầu" sau mỗi resume (dense window 200
# # #     # step lại chạy lại, λ lại tăng dần từ default y như epoch 0).
# # #     # ─────────────────────────────────────────────────────────────────
# # #     def state_dict(self) -> Dict:
# # #         return {
# # #             "lambdas": {t: v.detach().cpu().clone()
# # #                         for t, v in self.lambdas.items()},
# # #             "opt": self.opt.state_dict(),
# # #             "step": self._step,
# # #             "dense_steps_left": self._dense_steps_left,
# # #             "current_phase": self._current_phase,
# # #         }

# # #     def load_state_dict(self, sd: Dict) -> None:
# # #         try:
# # #             for t, v in sd["lambdas"].items():
# # #                 if t in self.lambdas:
# # #                     with torch.no_grad():
# # #                         self.lambdas[t].copy_(v.to(self.device))
# # #             # Re-tạo optimizer trên cùng tensor self.lambdas (đã copy_
# # #             # giá trị ở trên, KHÔNG thay reference) rồi load lại state
# # #             # (momentum Adam) — tránh mismatch param-group nếu thứ tự
# # #             # self.aux_terms khác giữa lần lưu và lần load.
# # #             self.opt = optim.Adam(list(self.lambdas.values()),
# # #                                    lr=self.opt.param_groups[0]["lr"])
# # #             try:
# # #                 self.opt.load_state_dict(sd["opt"])
# # #             except Exception:
# # #                 pass  # momentum Adam không critical, lambdas đã đúng
# # #             self._step             = sd.get("step", 0)
# # #             self._dense_steps_left = sd.get("dense_steps_left", 0)
# # #             self._current_phase    = sd.get("current_phase", self._current_phase)
# # #         except Exception as e:
# # #             print(f"  [FIX-GN-4] Lỗi khi restore GradNorm state: {e}")

# # #     def phase_reset(self, new_phase: int) -> None:
# # #         """
# # #         [FIX-GN-2] Gọi khi chuyển sang phase mới.
# # #         KHÔNG reset λ — chỉ reset optimizer momentum và bật dense window
# # #         ngắn (50 step, không phải 200) để re-adapt nhẹ.
# # #         """
# # #         if new_phase == self._current_phase:
# # #             return
# # #         self._current_phase = new_phase

# # #         self.opt = optim.Adam(list(self.lambdas.values()),
# # #                                lr=self.opt.param_groups[0]["lr"])
# # #         self._dense_steps_left = 50

# # #         cur = self.get_lambda_dict()
# # #         lam_str = " ".join(f"λ_{t.replace('l_','').replace('_reg','')}"
# # #                             f"={cur[t]:.3f}" for t in self.terms)
# # #         print(f"  [GradNorm] Phase {new_phase}: GIỮ λ hiện tại ({lam_str}), "
# # #               f"λ_dpe cố định={DEFAULT_LAMBDA[ANCHOR_TERM]:.2f}, "
# # #               f"reset optimizer momentum, dense update 50 steps")

# # #     def get_lambda_dict(self) -> Dict[str, float]:
# # #         """Trả về λ hiện tại (đủ GRADNORM_TERMS) để truyền vào
# # #         get_loss_breakdown(). λ_dpe luôn = DEFAULT_LAMBDA["l_dpe"] (cố định)."""
# # #         d = {t: float(self.lambdas[t].item()) for t in self.aux_terms}
# # #         d[ANCHOR_TERM] = DEFAULT_LAMBDA[ANCHOR_TERM]
# # #         return d

# # #     def update(self, loss_term_values: Dict[str, float]) -> None:
# # #         """
# # #         [FIX-GN-3] Cập nhật λ_aux dùng anchor + target-ratio scheme.

# # #         Args:
# # #             loss_term_values: {term: float} — loss RAW value từng term
# # #                               (detached, chưa nhân λ), PHẢI bao gồm
# # #                               ANCHOR_TERM ("l_dpe") để dùng làm tham chiếu.
# # #         """
# # #         self._step += 1

# # #         if self._dense_steps_left > 0:
# # #             self._dense_steps_left -= 1
# # #             update_freq = 1
# # #         else:
# # #             update_freq = self._update_freq

# # #         def _clamp():
# # #             with torch.no_grad():
# # #                 for t in self.aux_terms:
# # #                     self.lambdas[t].clamp_(min=self.lambda_min,
# # #                                             max=self.lambda_max)

# # #         if self._step % update_freq != 0:
# # #             _clamp()
# # #             return

# # #         if not loss_term_values:
# # #             _clamp()
# # #             return

# # #         l_anchor = loss_term_values.get(ANCHOR_TERM)
# # #         if l_anchor is None or not np.isfinite(l_anchor) or l_anchor <= 0:
# # #             _clamp()
# # #             return

# # #         loss_vals = {t: v for t, v in loss_term_values.items()
# # #                      if t in self.aux_terms and np.isfinite(v) and v > 0}
# # #         if not loss_vals:
# # #             _clamp()
# # #             return

# # #         self.opt.zero_grad()
# # #         for t in self.aux_terms:
# # #             if t not in loss_vals:
# # #                 continue
# # #             lam_t            = float(self.lambdas[t].item())
# # #             contribution_t   = lam_t * loss_vals[t]
# # #             target_contrib_t = DEFAULT_LAMBDA[t] * l_anchor
# # #             ratio = contribution_t / (target_contrib_t + 1e-8)
# # #             # ratio > 1: đóng góp vượt tỷ lệ mục tiêu → GIẢM λ_i → grad > 0
# # #             # ratio < 1: đóng góp dưới tỷ lệ mục tiêu → TĂNG λ_i → grad < 0
# # #             grad_val = (ratio - 1.0) * 0.1
# # #             if self.lambdas[t].grad is None:
# # #                 self.lambdas[t].grad = torch.zeros_like(self.lambdas[t])
# # #             self.lambdas[t].grad.fill_(float(grad_val))

# # #         self.opt.step()
# # #         _clamp()

# # #     def log_stats(self) -> Dict[str, float]:
# # #         """Log λ hiện tại để monitor. λ_l_dpe luôn = giá trị anchor cố định."""
# # #         d = {f"λ_{t}": float(self.lambdas[t].item()) for t in self.aux_terms}
# # #         d[f"λ_{ANCHOR_TERM}"] = DEFAULT_LAMBDA[ANCHOR_TERM]
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

# # #     [FIX-ALPHA-TIMING] mid point ở 15% của phase thay vì 50%:
# # #         mid ở epoch ~24.5 (phase2) → α≈0.15 ở ep23, α≈0.3 ở ep26
# # #         → hard loss kích hoạt sớm hơn ~10 epoch so với mid=50%.

# # #     Selector cũng dùng cùng logic để bật đồng bộ.
# # #     """
# # #     def __init__(self, total_steps_per_epoch: int):
# # #         self.steps_per_epoch = total_steps_per_epoch
# # #         self._global_step = 0

# # #         # Điểm bắt đầu của mỗi phase (tính theo global step)
# # #         self._phase2_start_step = (PHASE_1_END + 1) * total_steps_per_epoch
# # #         self._phase3_start_step = (PHASE_2_END + 1) * total_steps_per_epoch
# # #         self._phase4_start_step = (PHASE_3_END + 1) * total_steps_per_epoch

# # #         # [FIX-ALPHA-TIMING] Temperature ngắn → α tăng SỚM trong phase 2.
# # #         self._temp_p2 = 0.6 * total_steps_per_epoch / 4.4
# # #         self._temp_p3 = 0.6 * total_steps_per_epoch / 4.4  # selector bật nhanh
# # #         # [FIX-EASYFRAC4] Temperature cho phase 4 (giảm 55%→50% qua ~5 epoch)
# # #         self._temp_p4 = 2.5 * total_steps_per_epoch / 4.4

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
# # #         Phase 2 (step >= phase2_start): sigmoid 0 → 0.3, mid ở 15% phase
# # #         Phase 3+: α = 0.3 cố định
# # #         """
# # #         s = self._global_step
# # #         if s < self._phase2_start_step:
# # #             return 0.0
# # #         if s >= self._phase3_start_step:
# # #             return 0.3

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
# # #         [FIX-EASYFRAC4] Phase 4: 55% → 50% theo sigmoid (thay vì linear)
# # #             Consistent với phương pháp của các phase khác.
# # #             Midpoint: 5 epoch sau khi vào phase 4.
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

# # #         # [FIX-EASYFRAC4] Phase 4: 55% → 50% qua sigmoid (không phải linear)
# # #         phase4_mid = self._phase4_start_step + 5 * self.steps_per_epoch
# # #         x   = (s - phase4_mid) / max(self._temp_p4, 1.0)
# # #         sig = 1.0 / (1.0 + math.exp(-x))
# # #         return max(0.50, 0.55 - 0.05 * sig)

# # #     def get_diversity_weight(self, target: float = 0.15) -> float:
# # #         """
# # #         [FIX-DIV-3] diversity_loss_weight: trọng số L_diversity (hinge,
# # #         ngoài GradNorm), tăng mượt từ phase 2 — CÙNG schedule shape với
# # #         get_alpha_hard (sigmoid, mid ở 15% phase 2).
# # #         Phase 1: 0 (giữ warm-up ổn định, KHÔNG thêm forward pass thứ 2)
# # #         Phase 2: sigmoid 0 → target
# # #         Phase 3+: target cố định
# # #         """
# # #         s = self._global_step
# # #         if s < self._phase2_start_step:
# # #             return 0.0
# # #         if s >= self._phase3_start_step:
# # #             return target
# # #         mid  = self._phase2_start_step + (self._phase3_start_step
# # #                                            - self._phase2_start_step) * 0.15
# # #         x    = (s - mid) / max(self._temp_p2, 1.0)
# # #         sig  = 1.0 / (1.0 + math.exp(-x))
# # #         return target * sig

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
# # # #  [FIX-STABILITY-3 + FIX-THRESH-1] Global hard threshold — tính 1 lần
# # # #
# # # #  hard_score_from_obs() chỉ phụ thuộc obs_traj (data cố định), KHÔNG phụ
# # # #  thuộc model weights → threshold KHÔNG thay đổi theo training.
# # # #  [FIX-THRESH-1] Dùng flag _threshold_computed riêng (xem main loop) thay vì
# # # #  so sánh global_hard_threshold == 0.5 (fragile — nếu p70 thực tế = 0.5 thì
# # # #  sẽ recompute mãi).
# # # # ─────────────────────────────────────────────────────────────────────────────

# # # @torch.no_grad()
# # # def precompute_hard_threshold(loader, dev, n_batches: int = 50) -> float:
# # #     """
# # #     Tính p70 của hard_score trên subset của train set (1 lần duy nhất).
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
# # #     """Phân loại hard/easy dùng global threshold (thay vì per-batch p70)."""
# # #     scores   = hard_score_from_obs(obs_traj_norm)  # [B]
# # #     is_hard  = scores >= global_threshold
# # #     return is_hard


# # # def enforce_easy_frac(
# # #     is_hard: torch.Tensor,
# # #     easy_frac: float,
# # # ) -> torch.Tensor:
# # #     """
# # #     Chỉ điều chỉnh is_hard mask, KHÔNG reindex batch_list.
# # #     Nếu batch có quá nhiều hard → "demote" một số về easy (is_hard[i]=False).
# # #     Batch data giữ nguyên 100%.
# # #     """
# # #     B        = is_hard.shape[0]
# # #     max_hard = max(0, int(B * (1.0 - easy_frac)))
# # #     n_hard   = int(is_hard.sum().item())

# # #     if n_hard <= max_hard:
# # #         return is_hard  # đã đúng tỉ lệ

# # #     n_demote  = n_hard - max_hard
# # #     hard_idx  = is_hard.nonzero(as_tuple=True)[0]
# # #     perm      = torch.randperm(hard_idx.numel(), device=is_hard.device)
# # #     to_demote = hard_idx[perm[:n_demote]]

# # #     is_hard_new            = is_hard.clone()
# # #     is_hard_new[to_demote] = False
# # #     return is_hard_new


# # # # ─────────────────────────────────────────────────────────────────────────────
# # # #  Utilities
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
# # # #  [G-E] Easy_ADE evaluation cho EMA guard — dùng global threshold (FIX-4)
# # # # ─────────────────────────────────────────────────────────────────────────────

# # # @torch.no_grad()
# # # def evaluate_split(model, loader, dev, tag="", ema=None, steps=10,
# # #                    use_selector=False,
# # #                    global_hard_threshold: Optional[float] = None) -> Dict:
# # #     """
# # #     Evaluate với split easy/hard ADE.

# # #     Args:
# # #         global_hard_threshold: float từ precompute_hard_threshold(), hoặc None
# # #             → dùng per-batch p70 (phase 1 fallback, chỉ ảnh hưởng log)
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

# # #         if global_hard_threshold is not None:
# # #             is_hard = classify_hard_easy_global(
# # #                 obs_t[:, :, :2], global_hard_threshold).to(dev)
# # #         else:
# # #             is_hard = classify_hard_easy(obs_t[:, :, :2]).to(dev)

# # #         result = model.sample(bl, ddim_steps=steps, use_selector=use_selector)
# # #         p = result[0] if isinstance(result, (tuple, list)) else result
# # #         g = bl[1]; T = min(p.shape[0], g.shape[0])
# # #         pd = _ntd(p[:T]); gd = _ntd(g[:T])
# # #         dist = _hav(pd, gd)
# # #         at, ct = _atecte(pd, gd)
# # #         acc.update(dist, is_hard=is_hard, ate=at, cte=ct)

# # #     # [FIX-EMA-restore] bk luôn là dict non-empty nếu apply_to thành công
# # #     # (shadow luôn có ít nhất 1 floating-point param) — nhưng dùng
# # #     # `is not None` cho rõ ràng & an toàn tuyệt đối.
# # #     if bk is not None:
# # #         try: ema.restore(model, bk)
# # #         except: pass

# # #     r  = acc.compute()
# # #     el = time.perf_counter() - t0

# # #     def _v(k): return r.get(k, float("nan"))
# # #     def _m(v, t): return "ok" if np.isfinite(v) and v < t else "no"

# # #     thr_str = f"{global_hard_threshold:.4f}" if global_hard_threshold is not None else "per-batch"
# # #     print(f"\n{'='*70}")
# # #     print(f"  [{tag}  {el:.0f}s  threshold={thr_str}]")
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
# # #              use_selector=False,
# # #              global_hard_threshold: Optional[float] = None) -> Dict:
# # #     """Full evaluation (backward compat)."""
# # #     return evaluate_split(model, loader, dev, tag=tag, ema=ema,
# # #                           steps=steps, use_selector=use_selector,
# # #                           global_hard_threshold=global_hard_threshold)


# # # # ─────────────────────────────────────────────────────────────────────────────
# # # #  [G-F] Diversity check
# # # # ─────────────────────────────────────────────────────────────────────────────

# # # @torch.no_grad()
# # # def check_diversity(model, loader, dev, n_batches: int = 5,
# # #                     n_ensemble: int = 20, ddim_steps: int = 5) -> float:
# # #     """
# # #     Đo diversity score trên một subset của val set.
# # #     Nếu < 50 km → cảnh báo R1 (mode collapse).
# # #     """
# # #     model.eval()
# # #     all_divs = []

# # #     for i, b in enumerate(loader):
# # #         if i >= n_batches:
# # #             break
# # #         bl = move(list(b), dev)

# # #         _, _, all_c = model.sample(bl, num_ensemble=n_ensemble,
# # #                                     ddim_steps=ddim_steps)
# # #         N_total = all_c.shape[0]
# # #         cands = [all_c[k] for k in range(N_total)]

# # #         div = compute_diversity_score(cands)
# # #         all_divs.append(div)

# # #     return float(np.mean(all_divs)) if all_divs else 0.0


# # # # ─────────────────────────────────────────────────────────────────────────────
# # # #  [FIX-CKPT-1] Checkpoint — lưu smooth_sched_step + global_hard_threshold
# # # #  trong MỌI checkpoint (trước chỉ có trong periodic ckpt's "extra")
# # # # ─────────────────────────────────────────────────────────────────────────────

# # # def _save_ckpt(path, ep, model, opt, sched, saver, tl, vl,
# # #                smooth_sched_step: int = 0,
# # #                global_hard_threshold: Optional[float] = None,
# # #                diversity_boost_factor: float = 1.0,
# # #                gradnorm_state: Optional[Dict] = None,
# # #                T_max_orig: Optional[int] = None,
# # #                phase4_frozen: bool = False,
# # #                extra=None):
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
# # #         # [FIX-CKPT-1] Luôn lưu để resume chính xác smooth_sched + threshold
# # #         "smooth_sched_step":     smooth_sched_step,
# # #         "global_hard_threshold": global_hard_threshold,
# # #         # [FIX-DIV-2] Lưu hệ số boost diversity để resume áp lại đúng
# # #         # sigma_min/ctx_noise_scale (đây là plain float attr, KHÔNG nằm
# # #         # trong model.state_dict() nên sẽ mất nếu không lưu riêng).
# # #         "diversity_boost_factor": diversity_boost_factor,
# # #         # [FIX-GN-4] Lưu state GradNormManager (λ_aux, optimizer riêng,
# # #         # _step, _dense_steps_left, _current_phase) — KHÔNG nằm trong
# # #         # model.state_dict(), nên nếu không lưu sẽ mất khi resume và
# # #         # GradNorm phải "học lại từ đầu" (λ_aux reset về DEFAULT_LAMBDA,
# # #         # dense window 200 step chạy lại).
# # #         "gradnorm_state":   gradnorm_state,
# # #         # [FIX-LR-1] Lưu T_max gốc để resume đúng schedule (không tính lại)
# # #         "T_max_orig":       T_max_orig,
# # #         # [FIX-PHASE4-1] Lưu trạng thái _phase4_frozen để resume không freeze lại
# # #         "phase4_frozen":    phase4_frozen,
# # #         "version":          "v59strategy_fixed_v5",
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

# # #     def update(self, r, model, out_dir, ep, opt, sched, tl, vl, tag="",
# # #                smooth_sched_step: int = 0,
# # #                global_hard_threshold: Optional[float] = None,
# # #                diversity_boost_factor: float = 1.0,
# # #                gradnorm_state: Optional[Dict] = None,
# # #                T_max_orig: Optional[int] = None,
# # #                phase4_frozen: bool = False):
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
# # #                            sched, self, tl, vl,
# # #                            smooth_sched_step=smooth_sched_step,
# # #                            global_hard_threshold=global_hard_threshold,
# # #                            diversity_boost_factor=diversity_boost_factor,
# # #                            gradnorm_state=gradnorm_state,
# # #                            T_max_orig=T_max_orig,
# # #                            phase4_frozen=phase4_frozen)
# # #                 any_improved = True

# # #         if sc < self.bs:
# # #             self.bs = sc
# # #             any_improved = True
# # #             _save_ckpt(os.path.join(out_dir, f"best_{tag or 'composite'}.pth"),
# # #                        ep, model, opt, sched, self, tl, vl,
# # #                        smooth_sched_step=smooth_sched_step,
# # #                        global_hard_threshold=global_hard_threshold,
# # #                        diversity_boost_factor=diversity_boost_factor,
# # #                        gradnorm_state=gradnorm_state,
# # #                        T_max_orig=T_max_orig,
# # #                        phase4_frozen=phase4_frozen)
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
# # #                    help="GradNorm restoring force (giữ cho backward compat)")
# # #     p.add_argument("--gradnorm_lr",        default=1e-3, type=float,
# # #                    help="GradNorm lambda learning rate")
# # #     # EMA guard
# # #     p.add_argument("--ema_guard_threshold", default=20.0, type=float,
# # #                    help="smooth_easy_ADE tăng bao nhiêu km thì rollback (km)")
# # #     # Diversity check
# # #     p.add_argument("--diversity_threshold", default=50.0, type=float,
# # #                    help="diversity_score < threshold → cảnh báo R1")
# # #     # [FIX-DIV-2] Boost lặp sigma_min/ctx_noise_scale nếu diversity thấp.
# # #     # CẢNH BÁO TRADE-OFF: sigma_min/ctx_noise_scale chỉ ảnh hưởng sample()
# # #     # (inference) — KHÔNG ảnh hưởng train loss — nhưng boost quá mạnh có
# # #     # thể làm TỪNG candidate noisy hơn → có thể làm FAST/RAW ADE/CTE epoch
# # #     # 20+ xấu hơn epoch 19 (đánh đổi accuracy-per-candidate để lấy diversity
# # #     # cho selector). Nếu thấy easy_ADE ep20 tệ hơn rõ rệt so với ep19 do
# # #     # nguyên nhân này (không phải do phase-transition — đã fix ở FIX-GN-2),
# # #     # giảm --diversity_boost_max ở lần chạy sau (resume sẽ áp lại factor đã
# # #     # lưu, nên cần chạy lại từ ckpt trước ep19 hoặc từ đầu để đổi factor).
# # #     p.add_argument("--diversity_boost_step", default=1.8, type=float,
# # #                    help="hệ số nhân sigma_min/ctx_noise_scale mỗi vòng boost")
# # #     p.add_argument("--diversity_boost_max", default=6.0, type=float,
# # #                    help="cap tổng hệ số boost (so với giá trị gốc)")
# # #     p.add_argument("--diversity_boost_iters", default=3, type=int,
# # #                    help="số vòng boost+recheck tối đa")
# # #     # [FIX-DIV-4] Override hệ số boost đã lưu trong checkpoint khi resume.
# # #     p.add_argument("--diversity_boost_override", default=None, type=float,
# # #                    help="ghi đè diversity_boost_factor khi resume (vd 3.0)."
# # #                         " None = giữ nguyên factor đã lưu trong checkpoint.")

# # #     # [FIX-DIV-3] L_diversity training loss (hinge, ngoài GradNorm).
# # #     # An toàn: bounded ∈[0, target_norm], =0 khi diversity đủ (self-limiting).
# # #     # Tốn thêm 1 forward pass qua transformer decoder (phần FNO3D/encoder
# # #     # KHÔNG chạy lại) khi active — chỉ active từ phase 2 (sigmoid ramp),
# # #     # tắt hoàn toàn ở phase 1 (mặc định weight=0 → KHÔNG tốn compute thêm).
# # #     p.add_argument("--use_diversity_loss", action="store_true", default=True,
# # #                    help="bật L_diversity training loss (FIX-DIV-3)")
# # #     p.add_argument("--no_diversity_loss", dest="use_diversity_loss",
# # #                    action="store_false",
# # #                    help="tắt L_diversity training loss hoàn toàn")
# # #     p.add_argument("--diversity_loss_weight", default=0.15, type=float,
# # #                    help="trọng số đích (sau ramp) cho L_diversity hinge")
# # #     p.add_argument("--diversity_target_km", default=None, type=float,
# # #                    help="target diversity (km) cho L_diversity hinge;"
# # #                         " mặc định = --diversity_threshold")
# # #     return p.parse_args()


# # # # ─────────────────────────────────────────────────────────────────────────────
# # # #  Main
# # # # ─────────────────────────────────────────────────────────────────────────────

# # # def main(args):
# # #     if torch.cuda.is_available():
# # #         os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
# # #     dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # #     os.makedirs(args.output_dir, exist_ok=True)

# # #     # [FIX-DIV-3] target diversity (km) cho L_diversity hinge — mặc định
# # #     # dùng cùng giá trị với --diversity_threshold (consistency: cùng
# # #     # ngưỡng "đủ đa dạng" cho cả check_diversity() và L_diversity).
# # #     _diversity_target_km = (args.diversity_target_km
# # #                             if args.diversity_target_km is not None
# # #                             else args.diversity_threshold)

# # #     print("=" * 72)
# # #     print("  TC-FlowMatching v59-Strategy [FIXED-v4]")
# # #     print("  [FIX-GN-3] GradNorm: anchor λ_dpe=1.20 cố định + target-ratio aux")
# # #     print("  [FIX-GN-2] phase_reset: KHÔNG reset λ (chỉ reset optimizer)")
# # #     print("  [FIX-EMA-2] EMA guard pullback: chỉ khi α<0.25 (tránh kẹt)")
# # #     print("  [FIX-PERF-1] Xóa dead computation (named_parameters mỗi step)")
# # #     print("  [FIX-CKPT-1] Lưu smooth_sched_step + threshold mọi checkpoint")
# # #     print("  [FIX-DIV-2] Boost lặp (cap configurable) + recheck diversity +"
# # #           " resume-persist")
# # #     print(f"  [FIX-DIV-3] L_diversity hinge (target={_diversity_target_km:.0f}km,"
# # #           f" w_target={args.diversity_loss_weight if args.use_diversity_loss else 0.0:.2f},"
# # #           f" ramp từ phase2) — "
# # #           f"{'ON' if args.use_diversity_loss else 'OFF'}")
# # #     print("  [G-A] Easy/Hard: global threshold (tính 1 lần)")
# # #     print("  [G-C] easy_frac ≥ 50% enforcement mỗi batch")
# # #     print("  [G-D] α warm-up: sigmoid, mid=15% phase (FIX-ALPHA-TIMING)")
# # #     print("  [G-E] EMA guard: EMA-smoothed easy_ADE, threshold 20km")
# # #     print("  [G-F] Diversity check sau phase 1")
# # #     print("  [G-G] Split monitoring: easy_ADE + hard_ADE")
# # #     print("  [G-H] Phase 4: freeze encoder")
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
# # #     # [FIX-LR-1] Lưu T_max gốc — để resume dùng lại thay vì tính từ --num_epochs mới
# # #     _T_max_orig = total

# # #     # [G-A] GradNorm manager — equal-contribution scheme (FIX-GN-1)
# # #     gradnorm = GradNormManager(
# # #         terms=GRADNORM_TERMS,
# # #         alpha_gn=args.gradnorm_alpha,
# # #         lr=args.gradnorm_lr,
# # #         device=dev,
# # #     )

# # #     # SmoothScheduler — α theo batch, không phải epoch
# # #     smooth_sched = SmoothScheduler(total_steps_per_epoch=nstep)

# # #     # Savers
# # #     fast_saver = Saver(patience=args.patience, min_ep=args.min_ep,
# # #                         enable_stop=False)
# # #     full_saver = Saver(patience=args.patience, min_ep=args.min_ep,
# # #                         enable_stop=True)

# # #     # [FIX-THRESH-1] global threshold + flag riêng (thay vì == 0.5)
# # #     global_hard_threshold: Optional[float] = None
# # #     _threshold_computed = False

# # #     # [FIX-DIV-2] hệ số boost diversity hiện tại (1.0 = chưa boost).
# # #     # Cần khai báo TRƯỚC resume vì có thể được restore từ checkpoint.
# # #     diversity_boost_factor: float = 1.0

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

# # #         # [FIX-LR-1] Khôi phục T_max gốc nếu checkpoint có, tránh LR tính lại sai
# # #         T_max_from_ckpt = ck.get("T_max_orig")
# # #         if T_max_from_ckpt is not None:
# # #             _T_max_orig = T_max_from_ckpt
# # #             print(f"  [FIX-LR-1] Restored T_max_orig={_T_max_orig} from checkpoint")

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

# # #         # [FIX-CKPT-1] Restore smooth_sched step + threshold từ checkpoint
# # #         if "smooth_sched_step" in ck and ck["smooth_sched_step"] is not None:
# # #             smooth_sched._global_step = ck["smooth_sched_step"]
# # #             print(f"  Restored smooth_sched_step={ck['smooth_sched_step']}")
# # #         else:
# # #             smooth_sched._global_step = start * nstep
# # #             print(f"  Estimated smooth_sched_step={start * nstep} from epoch")

# # #         if "global_hard_threshold" in ck and ck["global_hard_threshold"] is not None:
# # #             global_hard_threshold = ck["global_hard_threshold"]
# # #             _threshold_computed = True
# # #             print(f"  Restored global_hard_threshold={global_hard_threshold:.4f}")

# # #         # [FIX-GN-4] Restore GradNorm state (λ_aux, optimizer riêng, _step,
# # #         # _dense_steps_left, _current_phase). Checkpoint cũ (trước fix này)
# # #         # không có key "gradnorm_state" -> bỏ qua, GradNorm khởi động lại
# # #         # từ DEFAULT_LAMBDA + dense window 200 step (behavior cũ, không
# # #         # crash, chỉ là không tối ưu — chấp nhận được cho checkpoint cũ).
# # #         if ck.get("gradnorm_state") is not None:
# # #             gradnorm.load_state_dict(ck["gradnorm_state"])
# # #             lam_str = " ".join(
# # #                 f"λ_{t.replace('l_','').replace('_reg','')}="
# # #                 f"{gradnorm.lambdas[t].item():.3f}"
# # #                 for t in gradnorm.aux_terms)
# # #             print(f"  [FIX-GN-4] Restored GradNorm state: {lam_str}"
# # #                   f"  step={gradnorm._step}"
# # #                   f"  dense_left={gradnorm._dense_steps_left}"
# # #                   f"  phase={gradnorm._current_phase}")
# # #         else:
# # #             print(f"  [FIX-GN-4] Checkpoint không có gradnorm_state"
# # #                   f" (checkpoint cũ trước fix) -> GradNorm khởi động lại"
# # #                   f" từ DEFAULT_LAMBDA + dense window 200 step.")

# # #         # [FIX-DIV-2] Restore hệ số boost diversity. sigma_min/ctx_noise_scale
# # #         # là plain float attribute trên model — KHÔNG nằm trong state_dict()
# # #         # nên model vừa load_state_dict ở trên vẫn đang ở giá trị CONSTRUCTOR
# # #         # (args.sigma_min, ctx_noise_scale=0.01 default). Nếu checkpoint có
# # #         # ghi nhận đã boost (factor > 1), áp lại NGAY tại đây — nếu không,
# # #         # boost sẽ bị mất vĩnh viễn vì block diversity-check chỉ chạy đúng
# # #         # 1 lần tại ep==PHASE_1_END (sẽ không chạy lại khi resume ở ep>19).
# # #         ck_div_factor = ck.get("diversity_boost_factor", 1.0)
# # #         if ck_div_factor and ck_div_factor > 1.0 + 1e-6:
# # #             diversity_boost_factor = float(ck_div_factor)
# # #             try:
# # #                 applied = []
# # #                 for attr_name, base_val in (("sigma_min", args.sigma_min),
# # #                                              ("ctx_noise_scale", 0.01)):
# # #                     if hasattr(m, attr_name):
# # #                         setattr(m, attr_name, base_val * diversity_boost_factor)
# # #                         applied.append(f"{attr_name}="
# # #                                        f"{getattr(m, attr_name):.4f}")
# # #                 print(f"  [FIX-DIV-2] Restored diversity boost factor="
# # #                       f"{diversity_boost_factor:.2f}x ({', '.join(applied)})")
# # #             except Exception as e:
# # #                 print(f"  [FIX-DIV-2] Lỗi khi restore diversity boost: {e}")

# # #         # [FIX-PHASE4-1] Khôi phục _phase4_frozen từ checkpoint
# # #         if "phase4_frozen" in ck and ck["phase4_frozen"]:
# # #             _phase4_frozen = True
# # #             print(f"  [FIX-PHASE4-1] Restored _phase4_frozen=True from checkpoint"
# # #                   f" — encoder đã frozen trước đó, sẽ không freeze lại")

# # #         # [FIX-DIV-4] Override diversity_boost_factor sau khi restore.
# # #         # Áp dụng dù checkpoint đã có factor>1 (giảm/tăng mức boost hiện
# # #         # tại) hay factor==1.0 (set boost mới ngay từ resume này).
# # #         # sigma_min/ctx_noise_scale luôn được tính lại từ BASE values
# # #         # (args.sigma_min, 0.01) * override — không nhân dồn lên giá trị
# # #         # đã boost trước đó, tránh double-boost.
# # #         if args.diversity_boost_override is not None:
# # #             old_factor = diversity_boost_factor
# # #             new_factor = float(args.diversity_boost_override)
# # #             if new_factor < 1.0:
# # #                 print(f"  [FIX-DIV-4] CẢNH BÁO: --diversity_boost_override="
# # #                       f"{new_factor:.2f} < 1.0 — không hợp lệ (boost factor"
# # #                       f" phải >= 1.0). Bỏ qua override, giữ factor="
# # #                       f"{old_factor:.2f}x.")
# # #             elif abs(new_factor - old_factor) < 1e-6:
# # #                 print(f"  [FIX-DIV-4] --diversity_boost_override="
# # #                       f"{new_factor:.2f}x == factor hiện tại, không đổi.")
# # #             else:
# # #                 diversity_boost_factor = new_factor
# # #                 try:
# # #                     applied = []
# # #                     for attr_name, base_val in (("sigma_min", args.sigma_min),
# # #                                                   ("ctx_noise_scale", 0.01)):
# # #                         if hasattr(m, attr_name):
# # #                             setattr(m, attr_name, base_val * diversity_boost_factor)
# # #                             applied.append(f"{attr_name}="
# # #                                            f"{getattr(m, attr_name):.4f}")
# # #                     init_noise_km_old = args.sigma_min * 2.5 * old_factor * 555.0
# # #                     init_noise_km_new = args.sigma_min * 2.5 * new_factor * 555.0
# # #                     print(f"  [FIX-DIV-4] Override diversity boost factor: "
# # #                           f"{old_factor:.2f}x -> {new_factor:.2f}x "
# # #                           f"({', '.join(applied)})")
# # #                     print(f"  [FIX-DIV-4] Initial DDIM noise std: "
# # #                           f"~{init_noise_km_old:.0f}km -> ~{init_noise_km_new:.0f}km"
# # #                           f"  (training-time current_sigma noise ~19-50km"
# # #                           f" tuỳ epoch)")
# # #                     # diversity_boost_factor mới sẽ được ghi vào MỌI
# # #                     # checkpoint lưu từ epoch này trở đi (threaded qua
# # #                     # _save_ckpt/Saver.update ở training loop bên dưới),
# # #                     # nên lần resume KẾ TIẾP sẽ tự dùng factor mới —
# # #                     # không cần truyền lại --diversity_boost_override.
# # #                 except Exception as e:
# # #                     print(f"  [FIX-DIV-4] Lỗi khi override diversity boost: {e}")

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
# # #             for g in opt.param_groups:
# # #                 g["lr"] = g["lr"] * 0.1
# # #             print(f"  [Phase 4] LR reduced to {opt.param_groups[0]['lr']:.2e}")

# # #     # ── EMA guard state ────────────────────────────────────────────────────
# # #     best_easy_ade_ema      = float("inf")
# # #     ema_rollback_count     = 0
# # #     _prev_phase            = get_phase(start) if start > 0 else 1
# # #     # EMA của easy_ADE — robust hơn raw value với variance fast_eval ±20-40km
# # #     _ema_state  = {"smooth": float("inf")}
# # #     EMA_ALPHA   = 0.35   # weight cho current value (bias về recent)
# # #     EMA_THRESH  = args.ema_guard_threshold   # km threshold để trigger rollback

# # #     # [FIX-DIV-2] flag để chỉ áp dụng diversity-noise-boost 1 lần.
# # #     # Nếu đã restore boost factor>1 từ checkpoint (resume), coi như đã
# # #     # boosted — không chạy lại block boost (vốn cũng chỉ chạy ở ep==19).
# # #     _diversity_boosted = diversity_boost_factor > 1.0 + 1e-6

# # #     print(f"  Training: {nstep} steps/ep, start ep={start}")
# # #     print("=" * 72)
# # #     ts = time.perf_counter()

# # #     for ep in range(start, args.num_epochs):
# # #         phase = get_phase(ep)

# # #         # [FIX-GN-2] GradNorm phase transition — KHÔNG reset λ
# # #         if phase != _prev_phase:
# # #             gradnorm.phase_reset(phase)
# # #             _prev_phase = phase
# # #             print(f"  [Phase {phase}] Chuyển phase tại epoch {ep}")

# # #             # Reset best_easy_ade_ema và smooth EMA khi chuyển phase
# # #             # (baseline easy_ADE giữa các phase có thể khác nhau tự nhiên,
# # #             #  không nên coi sự khác biệt này là "regression")
# # #             best_easy_ade_ema      = float("inf")
# # #             ema_rollback_count     = 0
# # #             _ema_state["smooth"]   = float("inf")
# # #             print(f"  [EMA Guard] Reset best/smooth EMA khi chuyển phase")

# # #         # [FIX-THRESH-1] Chỉ compute hard threshold 1 lần khi vào phase 2.
# # #         if phase >= 2 and not _threshold_computed:
# # #             global_hard_threshold = precompute_hard_threshold(trl, dev, n_batches=50)
# # #             _threshold_computed = True
# # #             print(f"  [FIX-THRESH-1] global_hard_threshold={global_hard_threshold:.4f}"
# # #                   f" — computed once, reused for all epochs")

# # #         # [G-H] Phase 4: freeze encoder
# # #         _maybe_freeze_encoder(ep, model)

# # #         model.train()
# # #         sl = 0.0
# # #         t0 = time.perf_counter()
# # #         # [FIX-DIV-3] accumulate diversity_km_train (chỉ tính step finite)
# # #         div_km_sum, div_km_cnt = 0.0, 0

# # #         # Lấy lambda hiện tại từ GradNorm
# # #         lambda_dict = gradnorm.get_lambda_dict()

# # #         for i, batch in enumerate(trl):
# # #             bl    = move(list(batch), dev)
# # #             obs_t = bl[0]

# # #             # α, easy_frac từ SmoothScheduler (per-batch)
# # #             alpha_hard    = smooth_sched.get_alpha_hard()
# # #             easy_frac     = smooth_sched.get_easy_frac()
# # #             sel_weight    = smooth_sched.get_selector_weight()
# # #             train_selector = smooth_sched.is_selector_active() and (sel_weight > 0.01)

# # #             # Dùng global threshold (phase>=2) hoặc zeros (phase 1)
# # #             if phase >= 2 and global_hard_threshold is not None:
# # #                 with torch.no_grad():
# # #                     is_hard = classify_hard_easy_global(
# # #                         obs_t[:, :, :2], global_hard_threshold).to(dev)
# # #                 is_hard = enforce_easy_frac(is_hard, easy_frac)
# # #             else:
# # #                 # Phase 1: alpha_hard=0 → hard loss = 0, không cần phân loại
# # #                 is_hard = torch.zeros(
# # #                     obs_t.shape[1], dtype=torch.bool, device=dev)

# # #             # [FIX-DIV-3] L_diversity weight (0 ở phase 1, ramp từ phase 2)
# # #             div_loss_w = (smooth_sched.get_diversity_weight(
# # #                               target=args.diversity_loss_weight)
# # #                           if args.use_diversity_loss else 0.0)

# # #             with autocast(device_type="cuda", enabled=args.use_amp):
# # #                 bd = model.get_loss_breakdown(
# # #                     bl,
# # #                     epoch=ep,
# # #                     alpha_hard=alpha_hard,
# # #                     is_hard=is_hard,
# # #                     train_selector=train_selector,
# # #                     lambda_dict=lambda_dict,
# # #                     diversity_loss_weight=div_loss_w,
# # #                     diversity_target_km=_diversity_target_km,
# # #                 )

# # #             opt.zero_grad()
# # #             scaler.scale(bd["total"]).backward()
# # #             scaler.unscale_(opt)

# # #             # [FIX-GN-1 + FIX-PERF-1] GradNorm: equal-contribution scheme.
# # #             # Chỉ cần loss VALUES (raw, detached) — không cần gradient norm
# # #             # của shared params (xóa hoàn toàn block named_parameters() cũ,
# # #             # vốn lặp qua TOÀN BỘ params mỗi step nhưng kết quả bị ignore).
# # #             try:
# # #                 loss_vals_float = {}
# # #                 for t in GRADNORM_TERMS:
# # #                     val = bd.get(t)
# # #                     if val is None:
# # #                         continue
# # #                     fval = float(val.item()) if torch.is_tensor(val) else float(val)
# # #                     if np.isfinite(fval) and fval > 0:
# # #                         loss_vals_float[t] = fval

# # #                 if loss_vals_float:
# # #                     gradnorm.update(loss_vals_float)
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

# # #             smooth_sched.step()

# # #             sl += bd["total"].item()

# # #             # [FIX-DIV-3] track training-time diversity (nan khi tắt/lỗi)
# # #             _dkm = bd.get("diversity_km_train", float("nan"))
# # #             if np.isfinite(_dkm):
# # #                 div_km_sum += _dkm
# # #                 div_km_cnt += 1

# # #             if i % 20 == 0:
# # #                 lr     = opt.param_groups[0]["lr"]
# # #                 tot    = bd["total"].item()
# # #                 warn   = " [!>10]" if tot > 10.0 else ""
# # #                 sw_st  = _unwrap(model).step_weights.stats()
# # #                 gn_log = gradnorm.log_stats()
# # #                 n_hard_batch = int(is_hard.sum().item())

# # #                 lam_str = " ".join(
# # #                     f"λ_{t.replace('l_','').replace('_reg','')}="
# # #                     f"{gn_log.get(f'λ_{t}', 1.0):.2f}"
# # #                     for t in GRADNORM_TERMS
# # #                 )

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
# # #                     f"  gn_d={gn_log.get('gn_dense',0)}"
# # #                     f"  {lam_str}"
# # #                     f"  lr={lr:.2e}"
# # #                     f"  div_w={div_loss_w:.3f}"
# # #                     f"  div_km={_dkm:.1f}"
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
# # #         use_sel_eval  = smooth_sched.is_selector_active()
# # #         eval_threshold = global_hard_threshold if phase >= 2 else None

# # #         lam_epoch = " ".join(
# # #             f"λ_{t.replace('l_','').replace('_reg','')}="
# # #             f"{gn_stats.get(f'λ_{t}', 1.0):.3f}"
# # #             for t in GRADNORM_TERMS
# # #         )

# # #         # [FIX-DIV-3] avg training-time diversity (nan nếu weight=0 cả epoch)
# # #         avg_div_km_train = (div_km_sum / div_km_cnt) if div_km_cnt > 0 else float("nan")

# # #         print(f"  Epoch {ep:>3}|Ph{phase}"
# # #               f" | train={avt:.4f} val={avv:.4f}"
# # #               f" | {smooth_sched.log_stats()}"
# # #               f" | {lam_epoch}"
# # #               f" | div_km_train={avg_div_km_train:.1f}"
# # #               f" | lr={lr_cur:.2e} | {t_ep:.0f}s")

# # #         # [G-G] Fast eval với split monitoring (dùng fast ddim)
# # #         rf = evaluate_split(model, vsub, dev,
# # #                              tag=f"FAST ep{ep}",
# # #                              steps=args.fast_ddim,
# # #                              use_selector=use_sel_eval,
# # #                              global_hard_threshold=eval_threshold)
# # #         fast_saver.update(rf, model, args.output_dir, ep, opt, sched, avt, avv,
# # #                           tag="fast",
# # #                           smooth_sched_step=smooth_sched.global_step,
# # #                           global_hard_threshold=global_hard_threshold,
# # #                           diversity_boost_factor=diversity_boost_factor,
# # #                           gradnorm_state=gradnorm.state_dict(),
# # #                           T_max_orig=_T_max_orig,
# # #                           phase4_frozen=_phase4_frozen)

# # #         # ── [G-E + FIX-EMA-2] EMA guard: kiểm tra easy_ADE sau mỗi epoch ──
# # #         #
# # #         # [FIX-EMA-2] Vấn đề cũ: pullback_steps=nstep*2, rollback dày đặc
# # #         # trong vùng α-ramp (ramp chỉ rộng ~35 step quanh điểm mid) → mỗi
# # #         # rollback trong vùng ramp kéo step về TRƯỚC vùng ramp → α=0 lại
# # #         # → α=0.000 suốt cả phase 2 (xem log gốc, 30 epoch toàn α=0).
# # #         #
# # #         # Fix:
# # #         #   - pullback_steps = nstep // 2 (giảm 1 nửa so với nstep*2)
# # #         #   - CHỈ pullback schedule nếu α hiện tại < ALPHA_RAMP_DONE_THRESHOLD
# # #         #     (còn trong vùng ramp). Nếu α đã ramp xong (>=0.25), rollback
# # #         #     chỉ restore weights, KHÔNG pullback schedule — tránh kẹt
# # #         #     vĩnh viễn do rollback lặp lại sau khi đã vượt vùng ramp.
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

# # #                 # Restore weights từ EMA
# # #                 m_raw   = _unwrap(model)
# # #                 ema_obj = getattr(m_raw, "_ema", None)
# # #                 if ema_obj:
# # #                     try:
# # #                         bk = ema_obj.apply_to(model)
# # #                         _save_ckpt(
# # #                             os.path.join(args.output_dir,
# # #                                          f"ema_guard_rollback_ep{ep}.pth"),
# # #                             ep, model, opt, sched, full_saver, avt, avv,
# # #                             smooth_sched_step=smooth_sched.global_step,
# # #                             global_hard_threshold=global_hard_threshold,
# # #                             diversity_boost_factor=diversity_boost_factor,
# # #                             gradnorm_state=gradnorm.state_dict(),
# # #                             extra={"ema_rollback": True,
# # #                                    "easy_ade_trigger": easy_ade_curr},
# # #                         )
# # #                         ema_obj.restore(model, bk)
# # #                         print(f"  [EMA GUARD] Saved rollback ckpt, "
# # #                               f"restored current weights for continued training")
# # #                     except Exception as e:
# # #                         print(f"  [EMA GUARD] Apply/restore failed: {e}")

# # #                 # [FIX-EMA-2] Chỉ pullback schedule nếu α còn trong vùng ramp
# # #                 cur_alpha = smooth_sched.get_alpha_hard()
# # #                 if cur_alpha < ALPHA_RAMP_DONE_THRESHOLD:
# # #                     pullback_steps = nstep // 2  # giảm so với nstep*2
# # #                     smooth_sched._global_step = max(
# # #                         smooth_sched._phase2_start_step,
# # #                         smooth_sched._global_step - pullback_steps
# # #                     )
# # #                     print(f"  [EMA GUARD] α còn trong vùng ramp ({cur_alpha:.3f}"
# # #                           f" < {ALPHA_RAMP_DONE_THRESHOLD}) → pullback"
# # #                           f" {pullback_steps} step. α mới="
# # #                           f"{smooth_sched.get_alpha_hard():.3f}")
# # #                 else:
# # #                     print(f"  [EMA GUARD] α đã ramp xong ({cur_alpha:.3f}"
# # #                           f" >= {ALPHA_RAMP_DONE_THRESHOLD}) → KHÔNG pullback"
# # #                           f" schedule, chỉ restore weights")

# # #         # [G-F + FIX-DIV-2] Diversity check cuối phase 1
# # #         div_score = None
# # #         if ep == PHASE_1_END:
# # #             print(f"\n  {'='*50}")
# # #             print(f"  [CHECKPOINT] Cuối phase 1 — kiểm tra diversity...")
# # #             baseline_easy_ade = rf.get("easy_ADE", float("nan"))
# # #             print(f"  (baseline FAST easy_ADE trước boost: "
# # #                   f"{baseline_easy_ade:.1f}km — so sánh với ep{ep+1}"
# # #                   f" FAST easy_ADE để kiểm tra trade-off boost)")
# # #             div_score = check_diversity(model, vsub, dev)
# # #             print(f"  Diversity score: {div_score:.1f} km")
# # #             if div_score < args.diversity_threshold:
# # #                 print(f"  [CẢNH BÁO R1] diversity={div_score:.1f} km"
# # #                       f" < {args.diversity_threshold} km")
# # #                 print(f"  → FM candidates không đủ đa dạng → selector sẽ kém")

# # #                 # [FIX-DIV-2] sigma_min/ctx_noise_scale CHỈ ảnh hưởng
# # #                 # sample() (initial DDIM noise + CFG context noise 3 step
# # #                 # đầu) — KHÔNG ảnh hưởng train CFM loss (_cfm_noisy dùng
# # #                 # current_sigma từ _sigma_schedule(epoch), không dùng
# # #                 # self.sigma_min; forward_with_ctx lúc train không truyền
# # #                 # noise_scale). → an toàn để boost MẠNH hơn FIX-DIV-1 (×1.5).
# # #                 # Boost LẶP + recheck thật, cap tổng diversity_boost_max,
# # #                 # tối đa diversity_boost_iters vòng.
# # #                 #
# # #                 # TRADE-OFF: boost làm TỪNG candidate noisy hơn → có thể
# # #                 # làm FAST/RAW ADE/CTE từ ep20 xấu hơn ep19 (đổi accuracy
# # #                 # per-candidate lấy diversity cho selector phase 3). So
# # #                 # sánh easy_ADE ep19 vs ep20 (FAST eval) sau khi có log:
# # #                 # nếu ep20 tệ hơn rõ rệt KHÔNG do phase-shock (đã fix ở
# # #                 # FIX-GN-2/FIX-EMA-2), giảm --diversity_boost_max lần sau.
# # #                 if not _diversity_boosted:
# # #                     try:
# # #                         m_div = _unwrap(model)
# # #                         boost_attrs = [a for a in ("sigma_min", "ctx_noise_scale")
# # #                                        if hasattr(m_div, a)]
# # #                         if not boost_attrs:
# # #                             print(f"  [FIX-DIV-2] Bỏ qua: model không có"
# # #                                   f" attribute sigma_min/ctx_noise_scale"
# # #                                   f" trực tiếp — cần xử lý thủ công"
# # #                                   f" (vd thêm L_diversity loss trong model).")
# # #                         else:
# # #                             base_vals = {a: getattr(m_div, a) for a in boost_attrs}
# # #                             BOOST_STEP     = args.diversity_boost_step
# # #                             MAX_TOTAL_BOOST = args.diversity_boost_max
# # #                             MAX_ITERS      = args.diversity_boost_iters
# # #                             total_boost = 1.0
# # #                             for it in range(1, MAX_ITERS + 1):
# # #                                 total_boost = min(total_boost * BOOST_STEP,
# # #                                                    MAX_TOTAL_BOOST)
# # #                                 for a in boost_attrs:
# # #                                     setattr(m_div, a, base_vals[a] * total_boost)
# # #                                 div_score = check_diversity(model, vsub, dev)
# # #                                 attr_str = ", ".join(
# # #                                     f"{a}={getattr(m_div, a):.4f}"
# # #                                     for a in boost_attrs)
# # #                                 print(f"  [FIX-DIV-2] vòng {it}: "
# # #                                       f"total_boost={total_boost:.2f}x "
# # #                                       f"({attr_str}) → diversity={div_score:.1f}km")
# # #                                 if (div_score >= args.diversity_threshold
# # #                                         or total_boost >= MAX_TOTAL_BOOST):
# # #                                     break
# # #                             # [FIX-DIV-2] Lưu factor để thread vào checkpoint
# # #                             # (resume sẽ áp lại, xem khối resume phía trên).
# # #                             diversity_boost_factor = total_boost
# # #                             if div_score >= args.diversity_threshold:
# # #                                 print(f"  [FIX-DIV-2] OK: diversity={div_score:.1f}km"
# # #                                       f" >= {args.diversity_threshold}km"
# # #                                       f" sau boost {total_boost:.2f}x")
# # #                             else:
# # #                                 print(f"  [FIX-DIV-2] VẪN CHƯA ĐỦ sau boost"
# # #                                       f" {total_boost:.2f}x (cap {MAX_TOTAL_BOOST}x):"
# # #                                       f" diversity={div_score:.1f}km"
# # #                                       f" < {args.diversity_threshold}km."
# # #                                       f" → cần can thiệp sâu hơn (L_diversity"
# # #                                       f" loss trong model, hoặc tăng"
# # #                                       f" n_ensemble/num_ensemble).")
# # #                         _diversity_boosted = True
# # #                     except Exception as e:
# # #                         print(f"  [FIX-DIV-2] Lỗi khi áp dụng noise boost: {e}")
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
# # #                                   use_selector=use_sel_eval,
# # #                                   global_hard_threshold=eval_threshold)
# # #             full_saver.update(rr, model, args.output_dir, ep, opt, sched,
# # #                               avt, avv, tag="raw",
# # #                               smooth_sched_step=smooth_sched.global_step,
# # #                               global_hard_threshold=global_hard_threshold,
# # #                               diversity_boost_factor=diversity_boost_factor,
# # #                               gradnorm_state=gradnorm.state_dict(),
# # #                               T_max_orig=_T_max_orig,
# # #                               phase4_frozen=_phase4_frozen)

# # #             if em and ep >= 3:
# # #                 re = evaluate_split(model, vl, dev,
# # #                                      tag=f"EMA ep{ep}",
# # #                                      ema=em,
# # #                                      steps=args.full_ddim,
# # #                                      use_selector=use_sel_eval,
# # #                                      global_hard_threshold=eval_threshold)
# # #                 full_saver.update(re, model, args.output_dir, ep, opt, sched,
# # #                                   avt, avv, tag="ema",
# # #                                   smooth_sched_step=smooth_sched.global_step,
# # #                                   global_hard_threshold=global_hard_threshold,
# # #                                   diversity_boost_factor=diversity_boost_factor,
# # #                                   gradnorm_state=gradnorm.state_dict(),
# # #                                   T_max_orig=_T_max_orig,
# # #                                   phase4_frozen=_phase4_frozen)

# # #         # Periodic checkpoint
# # #         if ep % 10 == 0 or ep == args.num_epochs - 1:
# # #             _save_ckpt(
# # #                 os.path.join(args.output_dir, f"ckpt_ep{ep:03d}.pth"),
# # #                 ep, model, opt, sched, full_saver, avt, avv,
# # #                 smooth_sched_step=smooth_sched.global_step,
# # #                 global_hard_threshold=global_hard_threshold,
# # #                 diversity_boost_factor=diversity_boost_factor,
# # #                 gradnorm_state=gradnorm.state_dict(),
# # #                 T_max_orig=_T_max_orig,
# # #                 phase4_frozen=_phase4_frozen,
# # #                 extra={"phase": phase,
# # #                        "alpha_hard": smooth_sched.get_alpha_hard(),
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
# # #             ckpt_threshold = ck.get("global_hard_threshold", None)
# # #             r = evaluate_split(model, tl2, dev, tag=f"TEST/{lb}",
# # #                                 steps=args.full_ddim,
# # #                                 use_selector=(get_phase(ck.get("epoch", 80)) >= 3),
# # #                                 global_hard_threshold=ckpt_threshold)
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

# # CÁC BUG MỚI ĐƯỢC FIX TRONG v3 (dựa trên log thực tế chạy v2, epoch 0-6):

# #   [FIX-GN-3] "equal-contribution" (FIX-GN-1) vẫn SAI — thay bằng
# #     "anchor + target-ratio".

# #     Log v2 ep0-6: λ_dpe GIẢM liên tục 1.00→0.335 (đang đi về floor 0.1),
# #     λ_vel/head/accel TĂNG 1.00→1.17-1.20. val loss giảm đẹp (2.42→2.34,
# #     THẤP NHẤT) nhưng easy_ADE TĂNG GẦN GẤP ĐÔI (408.9km log gốc → 705.7km)
# #     và ADE trend ep2-6 ĐANG TĂNG (574→598→575→619→649). EMA Guard rollback
# #     ngay tại ep6.

# #     Root cause: l_dpe (Huber loss vị trí — objective CHÍNH, gắn trực tiếp
# #     với ADE) có raw scale LỚN HƠN l_vel_reg/heading/accel KHÔNG PHẢI do lệch
# #     scale ngẫu nhiên, mà vì đây là objective khó giảm hơn (regularizer phụ
# #     dễ về gần 0). "Equal-contribution" coi đó là mất cân bằng cần sửa →
# #     đẩy λ_dpe XUỐNG để "cân bằng" với 4 regularizer phụ → model dồn capacity
# #     tối ưu 4 term phụ "dễ" (val loss giảm đẹp) nhưng hy sinh độ chính xác vị
# #     trí → ADE tăng. Cùng pattern với log gốc ep49→50 (λ_dpe reset 3.92→1.0
# #     đột ngột → easy_ADE 220→263km, "No improve" liên tục 4 epoch).

# #     Fix: l_dpe = ANCHOR, λ_dpe CỐ ĐỊNH = 1.20 (= default hand-tuned trong
# #     model, KHÔNG BAO GIỜ bị GradNorm điều chỉnh) → gradient cho vị trí
# #     không bao giờ bị giảm dưới mức hand-tuned. 4 term phụ (vel_reg, heading,
# #     speed, accel) được GradNorm điều chỉnh để duy trì TỶ LỆ ĐÓNG GÓP MỤC
# #     TIÊU so với (λ_dpe·l_dpe), lấy từ default weights hand-tuned trong
# #     model gốc (1.20, 1.40, 0.40, 0.05, 0.01):
# #         target_contrib_i = DEFAULT_LAMBDA[i] * l_dpe
# #         ratio_i = (λ_i · l_i) / target_contrib_i
# #         ratio_i > 1 → giảm λ_i ; ratio_i < 1 → tăng λ_i
# #     Equilibrium: λ_i ≈ DEFAULT_LAMBDA[i] · l_dpe / l_i — tự bù trừ scale RAW
# #     của term i trôi theo training (đúng mục đích GradNorm), nhưng KHÔNG
# #     đụng l_dpe và tôn trọng tỷ lệ quan trọng đã hand-tune giữa các term phụ.
# #     Không cần renorm sum=5 nữa (anchor cố định, không có sum constraint).
# #     Clamp λ_aux ∈ [0.02, 8.0] (range rộng hơn vì target_ratio chênh nhau
# #     0.008–1.167 giữa các term).

# #   [FIX-DIV-2] FIX-DIV-1 (×1.5 noise cố định) KHÔNG đủ.
# #     Log gốc: diversity=8.3km << 50km (gap ~6x), ×1.5 → ~12.5km vẫn cách xa.
# #     sigma_min và ctx_noise_scale CHỈ ảnh hưởng sample()/inference (initial
# #     DDIM noise + CFG context noise 3 step đầu) — KHÔNG ảnh hưởng training
# #     loss (_cfm_noisy dùng current_sigma từ _sigma_schedule(epoch), không
# #     dùng self.sigma_min; forward_with_ctx lúc train không truyền noise_scale)
# #     → có thể boost mạnh hơn nhiều mà không ảnh hưởng ổn định training.
# #     Fix: boost LẶP — mỗi vòng ×1.8 (sigma_min và ctx_noise_scale), sau đó
# #     chạy lại check_diversity(); lặp tối đa 3 vòng hoặc đến khi
# #     diversity >= threshold; tổng boost factor cap ở 6.0x để tránh initial
# #     noise quá lớn làm DDIM không hội tụ về trajectory hợp lý.

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
# #     classify_hard_easy_global, # [FIX-SEL-THRESH] dùng chung train+inference
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

# # # [FIX-GN-3] l_dpe là ANCHOR — λ_dpe cố định, KHÔNG bị GradNorm điều chỉnh.
# # # Giá trị = default hand-tuned trong compute_st_trans_loss (total_default).
# # ANCHOR_TERM  = "l_dpe"
# # AUX_TERMS    = ["l_vel_reg", "l_heading", "l_speed", "l_accel"]
# # DEFAULT_LAMBDA = {
# #     "l_dpe":     1.20,
# #     "l_vel_reg": 1.40,
# #     "l_heading": 0.40,
# #     "l_speed":   0.05,
# #     "l_accel":   0.01,
# # }
# # AUX_LAMBDA_MIN = 0.02
# # AUX_LAMBDA_MAX = 8.0

# # # Alpha threshold để coi là "đã ramp xong" — dùng cho EMA guard pullback gate
# # ALPHA_RAMP_DONE_THRESHOLD = 0.25


# # # ─────────────────────────────────────────────────────────────────────────────
# # #  [FIX-GN-3] GradNorm implementation — anchor + target-ratio scheme
# # #
# # #  Thuật toán:
# # #    - l_dpe = ANCHOR. λ_dpe CỐ ĐỊNH = DEFAULT_LAMBDA["l_dpe"] = 1.20.
# # #      → gradient cho l_dpe (objective chính, gắn trực tiếp ADE) không
# # #        bao giờ bị GradNorm giảm xuống dưới mức hand-tuned.
# # #    - Với mỗi term phụ i ∈ AUX_TERMS:
# # #        contribution_i    = λ_i * l_i
# # #        target_contrib_i  = DEFAULT_LAMBDA[i] * l_dpe
# # #        ratio_i           = contribution_i / target_contrib_i
# # #        ratio_i > 1 (đóng góp vượt tỷ lệ mục tiêu) → GIẢM λ_i → grad > 0
# # #        ratio_i < 1 (đóng góp dưới tỷ lệ mục tiêu) → TĂNG λ_i → grad < 0
# # #    - Equilibrium: λ_i ≈ DEFAULT_LAMBDA[i] * l_dpe / l_i
# # #      Nếu l_i == l_dpe thì λ_i == DEFAULT_LAMBDA[i] (đúng hand-tuned gốc).
# # #      Nếu l_i trôi scale theo training, λ_i tự bù trừ để giữ TỶ LỆ ĐÓNG GÓP
# # #      (không phải giá trị λ) bằng đúng tỷ lệ hand-tuned.
# # #    - Không có sum constraint → không cần renorm. Mỗi λ_i clamp riêng vào
# # #      [AUX_LAMBDA_MIN, AUX_LAMBDA_MAX] (range rộng vì DEFAULT_LAMBDA chênh
# # #      nhau tới ~150x giữa l_vel_reg và l_accel).
# # # ─────────────────────────────────────────────────────────────────────────────

# # class GradNormManager:
# #     """
# #     [FIX-GN-3] Quản lý GradNorm — anchor + target-ratio scheme.

# #     l_dpe là ANCHOR (λ cố định = DEFAULT_LAMBDA["l_dpe"] = 1.20, không nằm
# #     trong self.lambdas, không bao giờ bị update). Chỉ AUX_TERMS
# #     (l_vel_reg, l_heading, l_speed, l_accel) được GradNorm điều chỉnh,
# #     với equilibrium λ_i ≈ DEFAULT_LAMBDA[i] * l_dpe / l_i.

# #     [FIX-GN-2] phase_reset(): KHÔNG reset λ — chỉ reset optimizer momentum
# #     + bật dense update ngắn (50 step) để re-adapt nhẹ nếu landscape có dịch
# #     (do thêm gradient từ L_hard/L_sel chảy qua shared encoder ở phase mới).
# #     """
# #     def __init__(self, terms: List[str], alpha_gn: float = 1.5,
# #                  lr: float = 1e-3, device=None,
# #                  lambda_min: float = AUX_LAMBDA_MIN,
# #                  lambda_max: float = AUX_LAMBDA_MAX):
# #         # terms: full GRADNORM_TERMS list (bao gồm anchor) — giữ để
# #         # tương thích các nơi khác lặp qua GRADNORM_TERMS khi log/extract.
# #         self.terms      = terms
# #         self.aux_terms  = [t for t in terms if t != ANCHOR_TERM]
# #         self.alpha_gn   = alpha_gn
# #         self.device     = device or torch.device("cpu")
# #         self.lambda_min = lambda_min
# #         self.lambda_max = lambda_max

# #         # λ_i cho AUX_TERMS, init = DEFAULT_LAMBDA[i] (equilibrium đúng nếu
# #         # l_i == l_dpe ngay từ đầu — điểm khởi đầu hợp lý hơn 1.0 đồng nhất).
# #         self.lambdas = {t: torch.full((1,), DEFAULT_LAMBDA[t],
# #                                        device=self.device, requires_grad=True)
# #                         for t in self.aux_terms}
# #         self.opt = optim.Adam(list(self.lambdas.values()), lr=lr)

# #         self._step             = 0
# #         self._update_freq      = 5      # update mỗi 5 steps (bình thường)
# #         self._dense_steps_left = 200    # dense ở lúc khởi động training
# #         self._current_phase    = 1

# #     # ─────────────────────────────────────────────────────────────────
# #     # [FIX-GN-4] state_dict/load_state_dict — persist λ_aux, optimizer
# #     # GradNorm, _step, _dense_steps_left, _current_phase qua checkpoint.
# #     #
# #     # BUG trước fix: GradNormManager được khởi tạo lại từ constructor
# #     # mỗi lần script chạy (kể cả khi resume) → λ_aux reset về
# #     # DEFAULT_LAMBDA (1.40/0.40/0.05/0.01), _dense_steps_left reset về
# #     # 200, _step reset về 0 → 30 epoch điều chỉnh λ bị MẤT HOÀN TOÀN,
# #     # GradNorm phải "học lại từ đầu" sau mỗi resume (dense window 200
# #     # step lại chạy lại, λ lại tăng dần từ default y như epoch 0).
# #     # ─────────────────────────────────────────────────────────────────
# #     def state_dict(self) -> Dict:
# #         return {
# #             "lambdas": {t: v.detach().cpu().clone()
# #                         for t, v in self.lambdas.items()},
# #             "opt": self.opt.state_dict(),
# #             "step": self._step,
# #             "dense_steps_left": self._dense_steps_left,
# #             "current_phase": self._current_phase,
# #         }

# #     def load_state_dict(self, sd: Dict) -> None:
# #         try:
# #             for t, v in sd["lambdas"].items():
# #                 if t in self.lambdas:
# #                     with torch.no_grad():
# #                         self.lambdas[t].copy_(v.to(self.device))
# #             # Re-tạo optimizer trên cùng tensor self.lambdas (đã copy_
# #             # giá trị ở trên, KHÔNG thay reference) rồi load lại state
# #             # (momentum Adam) — tránh mismatch param-group nếu thứ tự
# #             # self.aux_terms khác giữa lần lưu và lần load.
# #             self.opt = optim.Adam(list(self.lambdas.values()),
# #                                    lr=self.opt.param_groups[0]["lr"])
# #             try:
# #                 self.opt.load_state_dict(sd["opt"])
# #             except Exception:
# #                 pass  # momentum Adam không critical, lambdas đã đúng
# #             self._step             = sd.get("step", 0)
# #             self._dense_steps_left = sd.get("dense_steps_left", 0)
# #             self._current_phase    = sd.get("current_phase", self._current_phase)
# #         except Exception as e:
# #             print(f"  [FIX-GN-4] Lỗi khi restore GradNorm state: {e}")

# #     def phase_reset(self, new_phase: int) -> None:
# #         """
# #         [FIX-GN-2] Gọi khi chuyển sang phase mới.
# #         KHÔNG reset λ — chỉ reset optimizer momentum và bật dense window
# #         ngắn (50 step, không phải 200) để re-adapt nhẹ.
# #         """
# #         if new_phase == self._current_phase:
# #             return
# #         self._current_phase = new_phase

# #         self.opt = optim.Adam(list(self.lambdas.values()),
# #                                lr=self.opt.param_groups[0]["lr"])
# #         self._dense_steps_left = 50

# #         cur = self.get_lambda_dict()
# #         lam_str = " ".join(f"λ_{t.replace('l_','').replace('_reg','')}"
# #                             f"={cur[t]:.3f}" for t in self.terms)
# #         print(f"  [GradNorm] Phase {new_phase}: GIỮ λ hiện tại ({lam_str}), "
# #               f"λ_dpe cố định={DEFAULT_LAMBDA[ANCHOR_TERM]:.2f}, "
# #               f"reset optimizer momentum, dense update 50 steps")

# #     def get_lambda_dict(self) -> Dict[str, float]:
# #         """Trả về λ hiện tại (đủ GRADNORM_TERMS) để truyền vào
# #         get_loss_breakdown(). λ_dpe luôn = DEFAULT_LAMBDA["l_dpe"] (cố định)."""
# #         d = {t: float(self.lambdas[t].item()) for t in self.aux_terms}
# #         d[ANCHOR_TERM] = DEFAULT_LAMBDA[ANCHOR_TERM]
# #         return d

# #     def update(self, loss_term_values: Dict[str, float]) -> None:
# #         """
# #         [FIX-GN-3] Cập nhật λ_aux dùng anchor + target-ratio scheme.

# #         Args:
# #             loss_term_values: {term: float} — loss RAW value từng term
# #                               (detached, chưa nhân λ), PHẢI bao gồm
# #                               ANCHOR_TERM ("l_dpe") để dùng làm tham chiếu.
# #         """
# #         self._step += 1

# #         if self._dense_steps_left > 0:
# #             self._dense_steps_left -= 1
# #             update_freq = 1
# #         else:
# #             update_freq = self._update_freq

# #         def _clamp():
# #             with torch.no_grad():
# #                 for t in self.aux_terms:
# #                     self.lambdas[t].clamp_(min=self.lambda_min,
# #                                             max=self.lambda_max)

# #         if self._step % update_freq != 0:
# #             _clamp()
# #             return

# #         if not loss_term_values:
# #             _clamp()
# #             return

# #         l_anchor = loss_term_values.get(ANCHOR_TERM)
# #         if l_anchor is None or not np.isfinite(l_anchor) or l_anchor <= 0:
# #             _clamp()
# #             return

# #         loss_vals = {t: v for t, v in loss_term_values.items()
# #                      if t in self.aux_terms and np.isfinite(v) and v > 0}
# #         if not loss_vals:
# #             _clamp()
# #             return

# #         self.opt.zero_grad()
# #         for t in self.aux_terms:
# #             if t not in loss_vals:
# #                 continue
# #             lam_t            = float(self.lambdas[t].item())
# #             contribution_t   = lam_t * loss_vals[t]
# #             target_contrib_t = DEFAULT_LAMBDA[t] * l_anchor
# #             ratio = contribution_t / (target_contrib_t + 1e-8)
# #             # ratio > 1: đóng góp vượt tỷ lệ mục tiêu → GIẢM λ_i → grad > 0
# #             # ratio < 1: đóng góp dưới tỷ lệ mục tiêu → TĂNG λ_i → grad < 0
# #             grad_val = (ratio - 1.0) * 0.1
# #             if self.lambdas[t].grad is None:
# #                 self.lambdas[t].grad = torch.zeros_like(self.lambdas[t])
# #             self.lambdas[t].grad.fill_(float(grad_val))

# #         self.opt.step()
# #         _clamp()

# #     def log_stats(self) -> Dict[str, float]:
# #         """Log λ hiện tại để monitor. λ_l_dpe luôn = giá trị anchor cố định."""
# #         d = {f"λ_{t}": float(self.lambdas[t].item()) for t in self.aux_terms}
# #         d[f"λ_{ANCHOR_TERM}"] = DEFAULT_LAMBDA[ANCHOR_TERM]
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

# #     def get_diversity_weight(self, target: float = 0.15) -> float:
# #         """
# #         [FIX-DIV-3] diversity_loss_weight: trọng số L_diversity (hinge,
# #         ngoài GradNorm), tăng mượt từ phase 2 — CÙNG schedule shape với
# #         get_alpha_hard (sigmoid, mid ở 15% phase 2).
# #         Phase 1: 0 (giữ warm-up ổn định, KHÔNG thêm forward pass thứ 2)
# #         Phase 2: sigmoid 0 → target
# #         Phase 3+: target cố định
# #         """
# #         s = self._global_step
# #         if s < self._phase2_start_step:
# #             return 0.0
# #         if s >= self._phase3_start_step:
# #             return target
# #         mid  = self._phase2_start_step + (self._phase3_start_step
# #                                            - self._phase2_start_step) * 0.15
# #         x    = (s - mid) / max(self._temp_p2, 1.0)
# #         sig  = 1.0 / (1.0 + math.exp(-x))
# #         return target * sig

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


# # # [FIX-SEL-THRESH] classify_hard_easy_global() được import từ
# # # Model.flow_matching_model (xem khối import ở đầu file) thay vì định
# # # nghĩa riêng ở đây — đảm bảo train script và sample() (inference) luôn
# # # dùng ĐÚNG MỘT hàm, không có 2 bản định nghĩa độc lập có thể trôi lệch
# # # nhau qua các lần sửa code.


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
# #                    global_hard_threshold: Optional[float] = None,
# #                    blend_strength: float = 0.20) -> Dict:
# #     """
# #     Evaluate với split easy/hard ADE.

# #     Args:
# #         global_hard_threshold: float từ precompute_hard_threshold(), hoặc None
# #             → dùng per-batch p70 (phase 1 fallback, chỉ ảnh hưởng log)
# #         blend_strength: [DIAG] tham số chẩn đoán — truyền thẳng vào
# #             model.sample() để đo độ nhạy của _persistence_blend lên ADE.
# #             Mặc định 0.20 = behavior cũ, không đổi gì khi không truyền.
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

# #         result = model.sample(bl, ddim_steps=steps, use_selector=use_selector,
# #                               blend_strength=blend_strength,
# #                               global_hard_threshold=global_hard_threshold)
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
# #                diversity_boost_factor: float = 1.0,
# #                gradnorm_state: Optional[Dict] = None,
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
# #         # [FIX-DIV-2] Lưu hệ số boost diversity để resume áp lại đúng
# #         # sigma_min/ctx_noise_scale (đây là plain float attr, KHÔNG nằm
# #         # trong model.state_dict() nên sẽ mất nếu không lưu riêng).
# #         "diversity_boost_factor": diversity_boost_factor,
# #         # [FIX-GN-4] Lưu state GradNormManager (λ_aux, optimizer riêng,
# #         # _step, _dense_steps_left, _current_phase) — KHÔNG nằm trong
# #         # model.state_dict(), nên nếu không lưu sẽ mất khi resume và
# #         # GradNorm phải "học lại từ đầu" (λ_aux reset về DEFAULT_LAMBDA,
# #         # dense window 200 step chạy lại).
# #         "gradnorm_state":   gradnorm_state,
# #         "version":          "v59strategy_fixed_v4",
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
# #                global_hard_threshold: Optional[float] = None,
# #                diversity_boost_factor: float = 1.0,
# #                gradnorm_state: Optional[Dict] = None):
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
# #                            global_hard_threshold=global_hard_threshold,
# #                            diversity_boost_factor=diversity_boost_factor,
# #                            gradnorm_state=gradnorm_state)
# #                 any_improved = True

# #         if sc < self.bs:
# #             self.bs = sc
# #             any_improved = True
# #             _save_ckpt(os.path.join(out_dir, f"best_{tag or 'composite'}.pth"),
# #                        ep, model, opt, sched, self, tl, vl,
# #                        smooth_sched_step=smooth_sched_step,
# #                        global_hard_threshold=global_hard_threshold,
# #                        diversity_boost_factor=diversity_boost_factor,
# #                        gradnorm_state=gradnorm_state)
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
# #     # [FIX-DIV-2] Boost lặp sigma_min/ctx_noise_scale nếu diversity thấp.
# #     # CẢNH BÁO TRADE-OFF: sigma_min/ctx_noise_scale chỉ ảnh hưởng sample()
# #     # (inference) — KHÔNG ảnh hưởng train loss — nhưng boost quá mạnh có
# #     # thể làm TỪNG candidate noisy hơn → có thể làm FAST/RAW ADE/CTE epoch
# #     # 20+ xấu hơn epoch 19 (đánh đổi accuracy-per-candidate để lấy diversity
# #     # cho selector). Nếu thấy easy_ADE ep20 tệ hơn rõ rệt so với ep19 do
# #     # nguyên nhân này (không phải do phase-transition — đã fix ở FIX-GN-2),
# #     # giảm --diversity_boost_max ở lần chạy sau (resume sẽ áp lại factor đã
# #     # lưu, nên cần chạy lại từ ckpt trước ep19 hoặc từ đầu để đổi factor).
# #     p.add_argument("--diversity_boost_step", default=1.8, type=float,
# #                    help="hệ số nhân sigma_min/ctx_noise_scale mỗi vòng boost")
# #     p.add_argument("--diversity_boost_max", default=6.0, type=float,
# #                    help="cap tổng hệ số boost (so với giá trị gốc)")
# #     p.add_argument("--diversity_boost_iters", default=3, type=int,
# #                    help="số vòng boost+recheck tối đa")
# #     # [FIX-DIV-4] Override hệ số boost đã lưu trong checkpoint khi resume.
# #     p.add_argument("--diversity_boost_override", default=None, type=float,
# #                    help="ghi đè diversity_boost_factor khi resume (vd 3.0)."
# #                         " None = giữ nguyên factor đã lưu trong checkpoint.")

# #     # [FIX-DIV-3] L_diversity training loss (hinge, ngoài GradNorm).
# #     # An toàn: bounded ∈[0, target_norm], =0 khi diversity đủ (self-limiting).
# #     # Tốn thêm 1 forward pass qua transformer decoder (phần FNO3D/encoder
# #     # KHÔNG chạy lại) khi active — chỉ active từ phase 2 (sigmoid ramp),
# #     # tắt hoàn toàn ở phase 1 (mặc định weight=0 → KHÔNG tốn compute thêm).
# #     p.add_argument("--use_diversity_loss", action="store_true", default=True,
# #                    help="bật L_diversity training loss (FIX-DIV-3)")
# #     p.add_argument("--no_diversity_loss", dest="use_diversity_loss",
# #                    action="store_false",
# #                    help="tắt L_diversity training loss hoàn toàn")
# #     p.add_argument("--diversity_loss_weight", default=0.15, type=float,
# #                    help="trọng số đích (sau ramp) cho L_diversity hinge")
# #     p.add_argument("--diversity_target_km", default=None, type=float,
# #                    help="target diversity (km) cho L_diversity hinge;"
# #                         " mặc định = --diversity_threshold")
# #     return p.parse_args()


# # # ─────────────────────────────────────────────────────────────────────────────
# # #  Main
# # # ─────────────────────────────────────────────────────────────────────────────

# # def main(args):
# #     if torch.cuda.is_available():
# #         os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
# #     dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #     os.makedirs(args.output_dir, exist_ok=True)

# #     # [FIX-DIV-3] target diversity (km) cho L_diversity hinge — mặc định
# #     # dùng cùng giá trị với --diversity_threshold (consistency: cùng
# #     # ngưỡng "đủ đa dạng" cho cả check_diversity() và L_diversity).
# #     _diversity_target_km = (args.diversity_target_km
# #                             if args.diversity_target_km is not None
# #                             else args.diversity_threshold)

# #     print("=" * 72)
# #     print("  TC-FlowMatching v59-Strategy [FIXED-v4]")
# #     print("  [FIX-GN-3] GradNorm: anchor λ_dpe=1.20 cố định + target-ratio aux")
# #     print("  [FIX-GN-2] phase_reset: KHÔNG reset λ (chỉ reset optimizer)")
# #     print("  [FIX-EMA-2] EMA guard pullback: chỉ khi α<0.25 (tránh kẹt)")
# #     print("  [FIX-PERF-1] Xóa dead computation (named_parameters mỗi step)")
# #     print("  [FIX-CKPT-1] Lưu smooth_sched_step + threshold mọi checkpoint")
# #     print("  [FIX-DIV-2] Boost lặp (cap configurable) + recheck diversity +"
# #           " resume-persist")
# #     print(f"  [FIX-DIV-3] L_diversity hinge (target={_diversity_target_km:.0f}km,"
# #           f" w_target={args.diversity_loss_weight if args.use_diversity_loss else 0.0:.2f},"
# #           f" ramp từ phase2) — "
# #           f"{'ON' if args.use_diversity_loss else 'OFF'}")
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
# #     from Model.data.trajectoriesWithMe_unet_training import (
# #         seq_collate, check_env_features_in_batch,
# #     )
# #     vsub = _mksub(vd, args.val_subset_size, args.batch_size, seq_collate)
# #     print(f"  train:{len(trd)} seqs  val:{len(vd)} seqs")

# #     # [FIX-ENV-DIAG] Kiểm tra ngay từ đầu xem u500_center/v500_center (dữ
# #     # liệu FIX-SCORE-ENV phụ thuộc vào để chấm điểm candidate theo steering
# #     # môi trường) có thực sự khác 0 trong batch thật hay không. Đây là hàm
# #     # chẩn đoán đã có sẵn trong trajectoriesWithMe_unet_training.py
# #     # (FIX-DATA-30) nhưng TRƯỚC ĐÂY KHÔNG được train script gọi ở đâu cả —
# #     # nghĩa là chưa từng có xác nhận thực nghiệm liệu dữ liệu gió tầng
# #     # 500hPa có tồn tại đúng hay không (có thể toàn 0 nếu CSV thiếu cột
# #     # hoặc .npy env thiếu field, trường hợp đó FIX-SCORE-ENV vẫn an toàn —
# #     # không crash — nhưng hoàn toàn vô dụng vì steer_vec sẽ luôn ≈0).
# #     try:
# #         _diag_batch = next(iter(trl))
# #         check_env_features_in_batch(list(_diag_batch), tag="STARTUP")
# #     except Exception as e:
# #         print(f"  [FIX-ENV-DIAG] Không thể chạy diagnostic: {e}")

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

# #     # [FIX-DIV-2] hệ số boost diversity hiện tại (1.0 = chưa boost).
# #     # Cần khai báo TRƯỚC resume vì có thể được restore từ checkpoint.
# #     diversity_boost_factor: float = 1.0

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

# #         # [FIX-LR-3] Khôi phục T_max/num_training_steps gốc và áp dụng ĐÚNG
# #         # CƠ CHẾ THẬT của scheduler đang dùng.
# #         #
# #         # Quan trọng: get_cosine_schedule_with_warmup() có 2 khả năng tùy
# #         # môi trường:
# #         #   (a) Model.utils tồn tại (trường hợp THẬT trên Kaggle, đã xác
# #         #       nhận bằng cách đọc trực tiếp file utils.py) → trả về
# #         #       torch.optim.lr_scheduler.LambdaLR, với num_training_steps
# #         #       là một CLOSURE VARIABLE đóng cứng vào hàm lr_lambda ngay
# #         #       lúc gọi — KHÔNG có attribute .T_max. Phiên sửa trước
# #         #       (FIX-LR-2) đã set `sched.T_max = ...` — với LambdaLR đây
# #         #       là DEAD CODE (set một attribute object không tồn tại,
# #         #       không gây lỗi nhưng KHÔNG có tác dụng gì, vì LambdaLR chỉ
# #         #       gọi self.lr_lambdas[0](self.last_epoch), không bao giờ
# #         #       đọc lại .T_max). Bug LR-cạn-sớm KHÔNG được sửa thật.
# #         #   (b) Model.utils import lỗi → fallback CosineAnnealingLR THẬT,
# #         #       có attribute .T_max, set lại trực tiếp là đúng cơ chế.
# #         #
# #         # Fix đúng: kiểm tra loại scheduler, áp dụng cách phù hợp cho từng
# #         # loại — tạo lại closure lr_lambda mới với num_training_steps đã
# #         # khôi phục cho LambdaLR; set .T_max trực tiếp cho CosineAnnealingLR.
# #         T_max_from_ckpt = ck.get("T_max_orig")
# #         if T_max_from_ckpt is not None:
# #             _T_max_orig = T_max_from_ckpt
# #             if isinstance(sched, torch.optim.lr_scheduler.LambdaLR):
# #                 # Tạo lại đúng công thức lr_lambda (giống Model.utils thật)
# #                 # với num_training_steps = _T_max_orig đã khôi phục, thay
# #                 # thế trực tiếp closure cũ (sai) bằng closure mới (đúng).
# #                 _wstp_used = wstp  # warmup ít khả năng đổi giữa resume
# #                 def _make_correct_lr_lambda(n_warmup, n_total, min_lr=1e-6):
# #                     def _lr_lambda(current_step):
# #                         if current_step < n_warmup:
# #                             return float(current_step) / float(max(1, n_warmup))
# #                         progress = float(current_step - n_warmup) / float(
# #                             max(1, n_total - n_warmup))
# #                         cosine_val = 0.5 * (1.0 + math.cos(math.pi * progress))
# #                         return max(min_lr, cosine_val)
# #                     return _lr_lambda
# #                 sched.lr_lambdas[0] = _make_correct_lr_lambda(
# #                     _wstp_used, _T_max_orig, min_lr=1e-6)
# #                 print(f"  [FIX-LR-3] Scheduler=LambdaLR — đã thay closure"
# #                       f" lr_lambda với num_training_steps={_T_max_orig}"
# #                       f" (trước: {total}, lệch={'CÓ' if _T_max_orig != total else 'không'})")
# #             elif hasattr(sched, "T_max"):
# #                 sched.T_max = _T_max_orig
# #                 print(f"  [FIX-LR-3] Scheduler=CosineAnnealingLR (fallback)"
# #                       f" — đã gán sched.T_max={_T_max_orig} (trước: {total})")
# #             else:
# #                 print(f"  [FIX-LR-3] CẢNH BÁO: loại scheduler không nhận"
# #                       f" diện được ({type(sched).__name__}) — KHÔNG áp dụng"
# #                       f" được fix T_max, LR có thể vẫn cạn sớm nếu"
# #                       f" --num_epochs đổi giữa các lần resume.")
# #             if _T_max_orig != total:
# #                 print(f"  [FIX-LR-3] CẢNH BÁO: --num_epochs hiện tại "
# #                       f"(total={total}) khác T_max/num_training_steps gốc "
# #                       f"đã lưu ({_T_max_orig}) — đã tự sửa, LR sẽ theo "
# #                       f"đúng schedule gốc, không bị nén/giãn.")
# #         try: sched.load_state_dict(ck["scheduler_state"])
# #         except:
# #             for _ in range(ck.get("epoch", 0) * nstep): sched.step()
# #         # [FIX-LR-3] load_state_dict() của LambdaLR KHÔNG khôi phục
# #         # lr_lambdas (chỉ last_epoch/base_lrs) nên closure mới vẫn giữ
# #         # nguyên sau dòng trên — không cần set lại. Với CosineAnnealingLR
# #         # fallback, đảm bảo lần cuối T_max vẫn đúng (phòng trường hợp
# #         # state_dict cũ có field T_max khác).
# #         if T_max_from_ckpt is not None and hasattr(sched, "T_max"):
# #             sched.T_max = _T_max_orig
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

# #         # [FIX-GN-4] Restore GradNorm state (λ_aux, optimizer riêng, _step,
# #         # _dense_steps_left, _current_phase). Checkpoint cũ (trước fix này)
# #         # không có key "gradnorm_state" -> bỏ qua, GradNorm khởi động lại
# #         # từ DEFAULT_LAMBDA + dense window 200 step (behavior cũ, không
# #         # crash, chỉ là không tối ưu — chấp nhận được cho checkpoint cũ).
# #         if ck.get("gradnorm_state") is not None:
# #             gradnorm.load_state_dict(ck["gradnorm_state"])
# #             lam_str = " ".join(
# #                 f"λ_{t.replace('l_','').replace('_reg','')}="
# #                 f"{gradnorm.lambdas[t].item():.3f}"
# #                 for t in gradnorm.aux_terms)
# #             print(f"  [FIX-GN-4] Restored GradNorm state: {lam_str}"
# #                   f"  step={gradnorm._step}"
# #                   f"  dense_left={gradnorm._dense_steps_left}"
# #                   f"  phase={gradnorm._current_phase}")
# #         else:
# #             print(f"  [FIX-GN-4] Checkpoint không có gradnorm_state"
# #                   f" (checkpoint cũ trước fix) -> GradNorm khởi động lại"
# #                   f" từ DEFAULT_LAMBDA + dense window 200 step.")

# #         # [FIX-DIV-2] Restore hệ số boost diversity. sigma_min/ctx_noise_scale
# #         # là plain float attribute trên model — KHÔNG nằm trong state_dict()
# #         # nên model vừa load_state_dict ở trên vẫn đang ở giá trị CONSTRUCTOR
# #         # (args.sigma_min, ctx_noise_scale=0.01 default). Nếu checkpoint có
# #         # ghi nhận đã boost (factor > 1), áp lại NGAY tại đây — nếu không,
# #         # boost sẽ bị mất vĩnh viễn vì block diversity-check chỉ chạy đúng
# #         # 1 lần tại ep==PHASE_1_END (sẽ không chạy lại khi resume ở ep>19).
# #         ck_div_factor = ck.get("diversity_boost_factor", 1.0)
# #         if ck_div_factor and ck_div_factor > 1.0 + 1e-6:
# #             diversity_boost_factor = float(ck_div_factor)
# #             try:
# #                 applied = []
# #                 for attr_name, base_val in (("sigma_min", args.sigma_min),
# #                                              ("ctx_noise_scale", 0.01)):
# #                     if hasattr(m, attr_name):
# #                         setattr(m, attr_name, base_val * diversity_boost_factor)
# #                         applied.append(f"{attr_name}="
# #                                        f"{getattr(m, attr_name):.4f}")
# #                 print(f"  [FIX-DIV-2] Restored diversity boost factor="
# #                       f"{diversity_boost_factor:.2f}x ({', '.join(applied)})")
# #             except Exception as e:
# #                 print(f"  [FIX-DIV-2] Lỗi khi restore diversity boost: {e}")

# #         # [FIX-DIV-4] Override diversity_boost_factor sau khi restore.
# #         # Áp dụng dù checkpoint đã có factor>1 (giảm/tăng mức boost hiện
# #         # tại) hay factor==1.0 (set boost mới ngay từ resume này).
# #         # sigma_min/ctx_noise_scale luôn được tính lại từ BASE values
# #         # (args.sigma_min, 0.01) * override — không nhân dồn lên giá trị
# #         # đã boost trước đó, tránh double-boost.
# #         if args.diversity_boost_override is not None:
# #             old_factor = diversity_boost_factor
# #             new_factor = float(args.diversity_boost_override)
# #             if new_factor < 1.0:
# #                 print(f"  [FIX-DIV-4] CẢNH BÁO: --diversity_boost_override="
# #                       f"{new_factor:.2f} < 1.0 — không hợp lệ (boost factor"
# #                       f" phải >= 1.0). Bỏ qua override, giữ factor="
# #                       f"{old_factor:.2f}x.")
# #             elif abs(new_factor - old_factor) < 1e-6:
# #                 print(f"  [FIX-DIV-4] --diversity_boost_override="
# #                       f"{new_factor:.2f}x == factor hiện tại, không đổi.")
# #             else:
# #                 diversity_boost_factor = new_factor
# #                 try:
# #                     applied = []
# #                     for attr_name, base_val in (("sigma_min", args.sigma_min),
# #                                                   ("ctx_noise_scale", 0.01)):
# #                         if hasattr(m, attr_name):
# #                             setattr(m, attr_name, base_val * diversity_boost_factor)
# #                             applied.append(f"{attr_name}="
# #                                            f"{getattr(m, attr_name):.4f}")
# #                     init_noise_km_old = args.sigma_min * 2.5 * old_factor * 555.0
# #                     init_noise_km_new = args.sigma_min * 2.5 * new_factor * 555.0
# #                     print(f"  [FIX-DIV-4] Override diversity boost factor: "
# #                           f"{old_factor:.2f}x -> {new_factor:.2f}x "
# #                           f"({', '.join(applied)})")
# #                     print(f"  [FIX-DIV-4] Initial DDIM noise std: "
# #                           f"~{init_noise_km_old:.0f}km -> ~{init_noise_km_new:.0f}km"
# #                           f"  (training-time current_sigma noise ~19-50km"
# #                           f" tuỳ epoch)")
# #                     # diversity_boost_factor mới sẽ được ghi vào MỌI
# #                     # checkpoint lưu từ epoch này trở đi (threaded qua
# #                     # _save_ckpt/Saver.update ở training loop bên dưới),
# #                     # nên lần resume KẾ TIẾP sẽ tự dùng factor mới —
# #                     # không cần truyền lại --diversity_boost_override.
# #                 except Exception as e:
# #                     print(f"  [FIX-DIV-4] Lỗi khi override diversity boost: {e}")

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

# #     # [FIX-DIV-2] flag để chỉ áp dụng diversity-noise-boost 1 lần.
# #     # Nếu đã restore boost factor>1 từ checkpoint (resume), coi như đã
# #     # boosted — không chạy lại block boost (vốn cũng chỉ chạy ở ep==19).
# #     _diversity_boosted = diversity_boost_factor > 1.0 + 1e-6

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
# #         # [FIX-DIV-3] accumulate diversity_km_train (chỉ tính step finite)
# #         div_km_sum, div_km_cnt = 0.0, 0

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

# #             # [FIX-DIV-3] L_diversity weight (0 ở phase 1, ramp từ phase 2)
# #             div_loss_w = (smooth_sched.get_diversity_weight(
# #                               target=args.diversity_loss_weight)
# #                           if args.use_diversity_loss else 0.0)

# #             with autocast(device_type="cuda", enabled=args.use_amp):
# #                 bd = model.get_loss_breakdown(
# #                     bl,
# #                     epoch=ep,
# #                     alpha_hard=alpha_hard,
# #                     is_hard=is_hard,
# #                     train_selector=train_selector,
# #                     lambda_dict=lambda_dict,
# #                     diversity_loss_weight=div_loss_w,
# #                     diversity_target_km=_diversity_target_km,
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

# #             # [FIX-DIV-3] track training-time diversity (nan khi tắt/lỗi)
# #             _dkm = bd.get("diversity_km_train", float("nan"))
# #             if np.isfinite(_dkm):
# #                 div_km_sum += _dkm
# #                 div_km_cnt += 1

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
# #                     f"  div_w={div_loss_w:.3f}"
# #                     f"  div_km={_dkm:.1f}"
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

# #         # [FIX-DIV-3] avg training-time diversity (nan nếu weight=0 cả epoch)
# #         avg_div_km_train = (div_km_sum / div_km_cnt) if div_km_cnt > 0 else float("nan")

# #         print(f"  Epoch {ep:>3}|Ph{phase}"
# #               f" | train={avt:.4f} val={avv:.4f}"
# #               f" | {smooth_sched.log_stats()}"
# #               f" | {lam_epoch}"
# #               f" | div_km_train={avg_div_km_train:.1f}"
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
# #                           global_hard_threshold=global_hard_threshold,
# #                           diversity_boost_factor=diversity_boost_factor,
# #                           gradnorm_state=gradnorm.state_dict())

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
# #                             diversity_boost_factor=diversity_boost_factor,
# #                             gradnorm_state=gradnorm.state_dict(),
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

# #         # [G-F + FIX-DIV-2] Diversity check cuối phase 1
# #         div_score = None
# #         if ep == PHASE_1_END:
# #             print(f"\n  {'='*50}")
# #             print(f"  [CHECKPOINT] Cuối phase 1 — kiểm tra diversity...")
# #             baseline_easy_ade = rf.get("easy_ADE", float("nan"))
# #             print(f"  (baseline FAST easy_ADE trước boost: "
# #                   f"{baseline_easy_ade:.1f}km — so sánh với ep{ep+1}"
# #                   f" FAST easy_ADE để kiểm tra trade-off boost)")
# #             div_score = check_diversity(model, vsub, dev)
# #             print(f"  Diversity score: {div_score:.1f} km")
# #             if div_score < args.diversity_threshold:
# #                 print(f"  [CẢNH BÁO R1] diversity={div_score:.1f} km"
# #                       f" < {args.diversity_threshold} km")
# #                 print(f"  → FM candidates không đủ đa dạng → selector sẽ kém")

# #                 # [FIX-DIV-2] sigma_min/ctx_noise_scale CHỈ ảnh hưởng
# #                 # sample() (initial DDIM noise + CFG context noise 3 step
# #                 # đầu) — KHÔNG ảnh hưởng train CFM loss (_cfm_noisy dùng
# #                 # current_sigma từ _sigma_schedule(epoch), không dùng
# #                 # self.sigma_min; forward_with_ctx lúc train không truyền
# #                 # noise_scale). → an toàn để boost MẠNH hơn FIX-DIV-1 (×1.5).
# #                 # Boost LẶP + recheck thật, cap tổng diversity_boost_max,
# #                 # tối đa diversity_boost_iters vòng.
# #                 #
# #                 # TRADE-OFF: boost làm TỪNG candidate noisy hơn → có thể
# #                 # làm FAST/RAW ADE/CTE từ ep20 xấu hơn ep19 (đổi accuracy
# #                 # per-candidate lấy diversity cho selector phase 3). So
# #                 # sánh easy_ADE ep19 vs ep20 (FAST eval) sau khi có log:
# #                 # nếu ep20 tệ hơn rõ rệt KHÔNG do phase-shock (đã fix ở
# #                 # FIX-GN-2/FIX-EMA-2), giảm --diversity_boost_max lần sau.
# #                 if not _diversity_boosted:
# #                     try:
# #                         m_div = _unwrap(model)
# #                         boost_attrs = [a for a in ("sigma_min", "ctx_noise_scale")
# #                                        if hasattr(m_div, a)]
# #                         if not boost_attrs:
# #                             print(f"  [FIX-DIV-2] Bỏ qua: model không có"
# #                                   f" attribute sigma_min/ctx_noise_scale"
# #                                   f" trực tiếp — cần xử lý thủ công"
# #                                   f" (vd thêm L_diversity loss trong model).")
# #                         else:
# #                             base_vals = {a: getattr(m_div, a) for a in boost_attrs}
# #                             BOOST_STEP     = args.diversity_boost_step
# #                             MAX_TOTAL_BOOST = args.diversity_boost_max
# #                             MAX_ITERS      = args.diversity_boost_iters
# #                             total_boost = 1.0
# #                             for it in range(1, MAX_ITERS + 1):
# #                                 total_boost = min(total_boost * BOOST_STEP,
# #                                                    MAX_TOTAL_BOOST)
# #                                 for a in boost_attrs:
# #                                     setattr(m_div, a, base_vals[a] * total_boost)
# #                                 div_score = check_diversity(model, vsub, dev)
# #                                 attr_str = ", ".join(
# #                                     f"{a}={getattr(m_div, a):.4f}"
# #                                     for a in boost_attrs)
# #                                 print(f"  [FIX-DIV-2] vòng {it}: "
# #                                       f"total_boost={total_boost:.2f}x "
# #                                       f"({attr_str}) → diversity={div_score:.1f}km")
# #                                 if (div_score >= args.diversity_threshold
# #                                         or total_boost >= MAX_TOTAL_BOOST):
# #                                     break
# #                             # [FIX-DIV-2] Lưu factor để thread vào checkpoint
# #                             # (resume sẽ áp lại, xem khối resume phía trên).
# #                             diversity_boost_factor = total_boost
# #                             if div_score >= args.diversity_threshold:
# #                                 print(f"  [FIX-DIV-2] OK: diversity={div_score:.1f}km"
# #                                       f" >= {args.diversity_threshold}km"
# #                                       f" sau boost {total_boost:.2f}x")
# #                             else:
# #                                 print(f"  [FIX-DIV-2] VẪN CHƯA ĐỦ sau boost"
# #                                       f" {total_boost:.2f}x (cap {MAX_TOTAL_BOOST}x):"
# #                                       f" diversity={div_score:.1f}km"
# #                                       f" < {args.diversity_threshold}km."
# #                                       f" → cần can thiệp sâu hơn (L_diversity"
# #                                       f" loss trong model, hoặc tăng"
# #                                       f" n_ensemble/num_ensemble).")
# #                         _diversity_boosted = True
# #                     except Exception as e:
# #                         print(f"  [FIX-DIV-2] Lỗi khi áp dụng noise boost: {e}")
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
# #                               global_hard_threshold=global_hard_threshold,
# #                               diversity_boost_factor=diversity_boost_factor,
# #                               gradnorm_state=gradnorm.state_dict())

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
# #                                   global_hard_threshold=global_hard_threshold,
# #                                   diversity_boost_factor=diversity_boost_factor,
# #                                   gradnorm_state=gradnorm.state_dict())

# #         # Periodic checkpoint
# #         if ep % 10 == 0 or ep == args.num_epochs - 1:
# #             _save_ckpt(
# #                 os.path.join(args.output_dir, f"ckpt_ep{ep:03d}.pth"),
# #                 ep, model, opt, sched, full_saver, avt, avv,
# #                 smooth_sched_step=smooth_sched.global_step,
# #                 global_hard_threshold=global_hard_threshold,
# #                 diversity_boost_factor=diversity_boost_factor,
# #                 gradnorm_state=gradnorm.state_dict(),
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
# scripts/train_fm.py  ——  TC-FlowMatching v2 Training
# ═══════════════════════════════════════════════════════

# Tương ứng với flow_matching_model.py (FM đúng cách).

# THIẾT KẾ TRAINING ĐƠN GIẢN:
#   - 1 optimizer, 1 scheduler (Cosine + warmup)
#   - Không GradNorm (chỉ có 2 loss terms)
#   - Không phase transitions phức tạp
#   - Không alpha_hard, không selector training
#   - EMA + early stopping trên val ADE

# HYPERPARAMETERS MẶC ĐỊNH:
#   lr = 2e-4   (nhỏ hơn bản cũ vì FM transformer deeper)
#   batch_size = 64
#   num_epochs = 150
#   warmup = 5 epoch
#   sigma_max = 0.3, sigma_min = 0.1 (schedule tự động)
#   lambda_reg = 0.2 (ramp từ epoch 10-30)
#   n_ensemble = 20, ddim_steps = 8

# CHẠY:
#   python scripts/train_fm.py \
#     --dataset_root /kaggle/input/.../tc-ofm \
#     --output_dir   runs/fm_v2 \
#     --num_epochs   150 \
#     --batch_size   64

# RESUME:
#   python scripts/train_fm.py \
#     --resume runs/fm_v2/best_model.pth \
#     --dataset_root ...
# """
# from __future__ import annotations
# import sys, os
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# import argparse, math, random, time
# from collections import defaultdict
# from typing import Dict, Optional

# import numpy as np
# import torch
# import torch.optim as optim
# from torch.amp import autocast, GradScaler
# from torch.utils.data import DataLoader, Subset

# from Model.data.loader_training import data_loader
# from Model.data.trajectoriesWithMe_unet_training import seq_collate
# from Model.flow_matching_model import (
#     TCFlowMatching, _norm_to_deg, _haversine_deg, EMAModel,
# )

# # ─────────────────────────────────────────────────────────────────────────────
# #  Constants
# # ─────────────────────────────────────────────────────────────────────────────
# R_EARTH = 6371.0
# HORIZON_STEPS = {12: 1, 24: 3, 48: 7, 72: 11}

# ST_TRANS_TARGETS = {
#     "ADE": 172.68, "72h": 321.39, "ATE": 142.21, "CTE": 42.04,
#     "12h": 65.42,  "24h": 104.67, "48h": 205.10,
# }


# # ─────────────────────────────────────────────────────────────────────────────
# #  Utilities
# # ─────────────────────────────────────────────────────────────────────────────

# def _unwrap(m):
#     return m._orig_mod if hasattr(m, "_orig_mod") else m

# def move(batch, device):
#     out = list(batch)
#     for i, x in enumerate(out):
#         if torch.is_tensor(x):
#             out[i] = x.to(device)
#         elif isinstance(x, dict):
#             out[i] = {k: v.to(device) if torch.is_tensor(v) else v
#                       for k, v in x.items()}
#     return out

# def make_subset_loader(dataset, n, batch_size, collate_fn):
#     idx = random.Random(42).sample(range(len(dataset)), min(n, len(dataset)))
#     return DataLoader(Subset(dataset, idx), batch_size=batch_size,
#                       shuffle=False, collate_fn=collate_fn,
#                       num_workers=0, drop_last=False)

# def _ate_cte(pred_deg, gt_deg):
#     """ATE, CTE per step"""
#     T = min(pred_deg.shape[0], gt_deg.shape[0])
#     if T < 2:
#         z = pred_deg.new_zeros(1, pred_deg.shape[1])
#         return z, z
#     lo1 = torch.deg2rad(gt_deg[:T-1, :, 0])
#     la1 = torch.deg2rad(gt_deg[:T-1, :, 1])
#     lo2 = torch.deg2rad(gt_deg[1:T,  :, 0])
#     la2 = torch.deg2rad(gt_deg[1:T,  :, 1])
#     lo3 = torch.deg2rad(pred_deg[1:T, :, 0])
#     la3 = torch.deg2rad(pred_deg[1:T, :, 1])
#     ya  = torch.sin(lo2 - lo1) * torch.cos(la2)
#     xa  = (torch.cos(la1) * torch.sin(la2)
#            - torch.sin(la1) * torch.cos(la2) * torch.cos(lo2 - lo1))
#     be  = torch.atan2(ya, xa)
#     ye  = torch.sin(lo3 - lo2) * torch.cos(la3)
#     xe  = (torch.cos(la2) * torch.sin(la3)
#            - torch.sin(la2) * torch.cos(la3) * torch.cos(lo3 - lo2))
#     bee = torch.atan2(ye, xe)
#     tot = _haversine_deg(pred_deg[1:T], gt_deg[1:T])
#     ang = bee - be
#     return tot * torch.cos(ang), tot * torch.sin(ang)


# # ─────────────────────────────────────────────────────────────────────────────
# #  Scheduler: linear warmup + cosine annealing
# # ─────────────────────────────────────────────────────────────────────────────

# class WarmupCosineScheduler:
#     def __init__(self, optimizer, warmup_epochs: int, total_epochs: int,
#                  lr_init: float, lr_min: float = 1e-6):
#         self.opt     = optimizer
#         self.warmup  = warmup_epochs
#         self.total   = total_epochs
#         self.lr_init = lr_init
#         self.lr_min  = lr_min
#         self.epoch   = 0

#     def step(self) -> float:
#         e = self.epoch
#         if e < self.warmup:
#             lr = self.lr_init * (0.1 + 0.9 * e / max(self.warmup - 1, 1))
#         else:
#             t  = e - self.warmup
#             T  = self.total - self.warmup
#             lr = self.lr_min + 0.5 * (self.lr_init - self.lr_min) * (
#                 1 + math.cos(math.pi * t / max(T, 1)))
#         for pg in self.opt.param_groups:
#             pg["lr"] = lr
#         self.epoch += 1
#         return lr


# # ─────────────────────────────────────────────────────────────────────────────
# #  Evaluation
# # ─────────────────────────────────────────────────────────────────────────────

# @torch.no_grad()
# def evaluate(model, loader, device, tag: str = "",
#              ddim_steps: int = 8, ema: Optional[EMAModel] = None) -> Dict:
#     bk = None
#     if ema is not None:
#         try: bk = ema.apply_to(model)
#         except: pass

#     model.eval()
#     all_ade, all_ate, all_cte = [], [], []
#     step_ade = defaultdict(list)

#     for batch in loader:
#         bl   = move(list(batch), device)
#         gt   = bl[1]
#         pred, _, _ = model.sample(bl, ddim_steps=ddim_steps)

#         T        = min(pred.shape[0], gt.shape[0])
#         pred_deg = _norm_to_deg(pred[:T])
#         gt_deg   = _norm_to_deg(gt[:T])
#         dist     = _haversine_deg(pred_deg, gt_deg)  # [T, B]
#         ate, cte = _ate_cte(pred_deg, gt_deg)

#         all_ade.extend(dist.mean(0).tolist())
#         if ate.shape[0] > 0:
#             all_ate.extend(ate.abs().mean(0).tolist())
#             all_cte.extend(cte.abs().mean(0).tolist())

#         for h, s in HORIZON_STEPS.items():
#             if s < T:
#                 step_ade[h].extend(dist[s].tolist())

#     def _m(lst): return float(np.mean(lst)) if lst else float("nan")

#     result = {
#         "ADE": _m(all_ade),
#         "ATE": _m(all_ate),
#         "CTE": _m(all_cte),
#         "n":   len(all_ade),
#     }
#     for h in HORIZON_STEPS:
#         result[f"{h}h"] = _m(step_ade[h])

#     if bk is not None:
#         try: ema.restore(model, bk)
#         except: pass

#     # Print
#     def _v(k): return result.get(k, float("nan"))
#     def _ok(k, ref): return "✓" if np.isfinite(_v(k)) and _v(k) < ref else "✗"

#     print(f"\n  {'='*60}")
#     print(f"  [{tag}]")
#     print(f"  ADE={_v('ADE'):.1f} {_ok('ADE', ST_TRANS_TARGETS['ADE'])}"
#           f"  (target < {ST_TRANS_TARGETS['ADE']})")
#     print(f"  12h={_v('12h'):.0f}  24h={_v('24h'):.0f}"
#           f"  48h={_v('48h'):.0f}  72h={_v('72h'):.0f}"
#           f"  {_ok('72h', ST_TRANS_TARGETS['72h'])}")
#     print(f"  ATE={_v('ATE'):.1f}  CTE={_v('CTE'):.1f}")

#     beat = []
#     for k, ref in ST_TRANS_TARGETS.items():
#         v = _v(k)
#         if np.isfinite(v) and v < ref:
#             beat.append(f"{k}:{v:.1f}<{ref:.1f}")
#     if beat:
#         print(f"  *** BEAT ST-TRANS: {' | '.join(beat)} ***")
#     print(f"  {'='*60}\n")

#     return result


# # ─────────────────────────────────────────────────────────────────────────────
# #  Checkpoint
# # ─────────────────────────────────────────────────────────────────────────────

# def _save(path, epoch, model, opt, sched, best_ade, ema=None):
#     m  = _unwrap(model)
#     esd = None
#     if ema is not None:
#         try: esd = {k: v.cpu().clone() for k, v in ema.shadow.items()}
#         except: pass
#     torch.save({
#         "epoch":      epoch,
#         "model":      m.state_dict(),
#         "optimizer":  opt.state_dict(),
#         "scheduler":  sched.epoch,
#         "best_ade":   best_ade,
#         "ema":        esd,
#     }, path)


# # ─────────────────────────────────────────────────────────────────────────────
# #  Args
# # ─────────────────────────────────────────────────────────────────────────────

# def get_args():
#     p = argparse.ArgumentParser(
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     # Data
#     p.add_argument("--dataset_root",  default="TCND_vn")
#     p.add_argument("--obs_len",       default=8,     type=int)
#     p.add_argument("--pred_len",      default=12,    type=int)
#     p.add_argument("--num_workers",   default=2,     type=int)
#     p.add_argument("--other_modal",   default="gph")
#     p.add_argument("--delim",         default=" ")
#     p.add_argument("--skip",          default=1,     type=int)
#     p.add_argument("--min_ped",       default=1,     type=int)
#     p.add_argument("--threshold",     default=0.002, type=float)
#     # Model
#     p.add_argument("--d_cond",        default=256,   type=int)
#     p.add_argument("--d_model",       default=256,   type=int)
#     p.add_argument("--nhead",         default=8,     type=int)
#     p.add_argument("--num_dec_layers",default=4,     type=int)
#     p.add_argument("--dim_ff",        default=512,   type=int)
#     p.add_argument("--dropout",       default=0.1,   type=float)
#     p.add_argument("--unet_in_ch",    default=13,    type=int)
#     p.add_argument("--sigma_min",     default=0.04,  type=float,
#                    help="FM sigma min (~22km, 15%% ADE). Phải < ADE_norm=0.32")
#     p.add_argument("--sigma_max",     default=0.08,  type=float,
#                    help="FM sigma max (~43km, 25%% ADE). Phải << ADE_norm")
#     p.add_argument("--lambda_reg",    default=0.2,   type=float,
#                    help="ADE signal weight (FIX-FATAL-1)")
#     p.add_argument("--use_ot",        default=True,  action="store_true")
#     p.add_argument("--no_ot",         dest="use_ot", action="store_false")
#     p.add_argument("--n_ensemble",    default=20,    type=int)
#     p.add_argument("--sigma_inference",default=0.079, type=float,
#                    help="noise std lúc inference (~43km → diversity ~60km std)")
#     # Training
#     p.add_argument("--num_epochs",    default=150,   type=int)
#     p.add_argument("--batch_size",    default=64,    type=int)
#     p.add_argument("--lr",            default=2e-4,  type=float)
#     p.add_argument("--lr_min",        default=1e-6,  type=float)
#     p.add_argument("--warmup_epochs", default=5,     type=int)
#     p.add_argument("--weight_decay",  default=1e-4,  type=float)
#     p.add_argument("--grad_clip",     default=1.0,   type=float)
#     p.add_argument("--use_amp",       action="store_true", default=False)
#     p.add_argument("--use_ema",       default=True,  action="store_true")
#     p.add_argument("--no_ema",        dest="use_ema",action="store_false")
#     p.add_argument("--ema_decay",     default=0.995, type=float)
#     # Eval
#     p.add_argument("--val_freq",      default=5,     type=int)
#     p.add_argument("--val_subset",    default=500,   type=int)
#     p.add_argument("--ddim_steps",    default=8,     type=int)
#     p.add_argument("--patience",      default=30,    type=int)
#     p.add_argument("--min_ep",        default=20,    type=int)
#     # IO
#     p.add_argument("--output_dir",    default="runs/fm_v2")
#     p.add_argument("--gpu_num",       default="0")
#     p.add_argument("--resume",        default=None)
#     p.add_argument("--test_at_end",   action="store_true", default=True)
#     return p.parse_args()


# # ─────────────────────────────────────────────────────────────────────────────
# #  Main
# # ─────────────────────────────────────────────────────────────────────────────

# def main(args):
#     if torch.cuda.is_available():
#         os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     os.makedirs(args.output_dir, exist_ok=True)
#     best_ckpt = os.path.join(args.output_dir, "best_model.pth")

#     print("=" * 65)
#     print("  TC-FlowMatching v2  |  FM đúng cách")
#     print(f"  sigma: {args.sigma_max}→{args.sigma_min}  "
#           f"lambda_reg={args.lambda_reg}  K={args.n_ensemble}")
#     print(f"  N_steps={args.ddim_steps}  sigma_inf={args.sigma_inference}")
#     print(f"  FIX-FATAL-1: L_reg tại t=0 (gradient không dilute)")
#     print(f"  FIX-FATAL-2: 2 loss terms (l_cfm + lambda_reg*l_reg)")
#     print(f"  FIX-FATAL-3: Best-of-K thay SelectorNet")
#     print(f"  FIX-MAJOR-4: sigma_max={args.sigma_max}, "
#           f"sigma_inf={args.sigma_inference}")
#     print(f"  FIX-MAJOR-5: FM trên 2D (không 4D)")
#     print("=" * 65)

#     # ── Data ──────────────────────────────────────────────────────────────
#     trd, trl = data_loader(
#         args, {"root": args.dataset_root, "type": "train"}, test=False)
#     vd, _    = data_loader(
#         args, {"root": args.dataset_root, "type": "val"},   test=True)
#     val_sub  = make_subset_loader(vd, args.val_subset, args.batch_size, seq_collate)
#     print(f"  train: {len(trd)}  val: {len(vd)}")

#     # ── Model ─────────────────────────────────────────────────────────────
#     model = TCFlowMatching(
#         pred_len=args.pred_len, obs_len=args.obs_len,
#         unet_in_ch=args.unet_in_ch,
#         d_cond=args.d_cond, d_model=args.d_model,
#         nhead=args.nhead, num_dec_layers=args.num_dec_layers,
#         dim_ff=args.dim_ff, dropout=args.dropout,
#         sigma_min=args.sigma_min, sigma_max=args.sigma_max,
#         lambda_reg=args.lambda_reg,
#         use_ot=args.use_ot,
#         use_ema=args.use_ema,
#         n_ensemble=args.n_ensemble,
#         sigma_inference=args.sigma_inference,
#     ).to(device)

#     model.init_ema()
#     ema = getattr(_unwrap(model), "_ema", None)

#     n_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     n_enc   = sum(p.numel() for p in model.encoder.parameters())
#     n_vel   = sum(p.numel() for p in model.velocity.parameters())
#     print(f"\n  Params total  : {n_total:,}")
#     print(f"  Encoder       : {n_enc:,}  (FNO3D+Mamba, giữ nguyên)")
#     print(f"  VelocityTrans : {n_vel:,}  (4-layer transformer, 2D only)")

#     # ── Optimizer & scheduler ─────────────────────────────────────────────
#     opt    = optim.AdamW(model.parameters(), lr=args.lr,
#                           weight_decay=args.weight_decay)
#     scaler = GradScaler("cuda", enabled=args.use_amp)
#     sched  = WarmupCosineScheduler(
#         opt, warmup_epochs=args.warmup_epochs,
#         total_epochs=args.num_epochs,
#         lr_init=args.lr, lr_min=args.lr_min)

#     # ── Resume ────────────────────────────────────────────────────────────
#     start_ep = 0
#     best_ade = float("inf")
#     patience_cnt = 0

#     if args.resume and os.path.exists(args.resume):
#         ck = torch.load(args.resume, map_location=device)
#         _unwrap(model).load_state_dict(ck["model"], strict=False)
#         try: opt.load_state_dict(ck["optimizer"])
#         except: pass
#         sched.epoch = ck.get("scheduler", 0)
#         start_ep    = ck.get("epoch", 0) + 1
#         best_ade    = ck.get("best_ade", float("inf"))
#         if ema and ck.get("ema"):
#             for k, v in ck["ema"].items():
#                 if k in ema.shadow:
#                     ema.shadow[k].copy_(v.to(device))
#         print(f"  ↩ Resumed ep{start_ep}  best_ADE={best_ade:.1f} km")

#     try:
#         model = torch.compile(model, mode="reduce-overhead")
#         print("  torch.compile: ok")
#     except:
#         pass

#     print()
#     print("=" * 65)
#     print(f"  TRAINING  ({len(trl)} steps/epoch, {args.num_epochs} epochs)")
#     print(f"  Warmup: {args.warmup_epochs} ep  lr: {args.lr}→{args.lr_min}")
#     print(f"  L_reg ramp: ep10→ep30 (0→{args.lambda_reg})")
#     print("=" * 65)

#     nstep = len(trl)

#     for ep in range(start_ep, args.num_epochs):
#         model.train()
#         sum_loss, sum_cfm, sum_reg, sum_ade1 = 0.0, 0.0, 0.0, 0.0
#         t0 = time.perf_counter()

#         for i, batch in enumerate(trl):
#             bl = move(list(batch), device)

#             opt.zero_grad()
#             with autocast(device_type="cuda", enabled=args.use_amp):
#                 bd = model.get_loss_breakdown(bl, epoch=ep)

#             scaler.scale(bd["total"]).backward()
#             scaler.unscale_(opt)
#             torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
#             scaler.step(opt)
#             scaler.update()
#             model.ema_update()

#             sum_loss += bd["total"].item()
#             sum_cfm  += bd["l_cfm"]
#             sum_reg  += bd["l_reg"]
#             sum_ade1 += bd["ade_1step"]

#             if i % 30 == 0:
#                 lr = opt.param_groups[0]["lr"]
#                 print(f"  [{ep:>3}][{i:>3}/{nstep}]"
#                       f"  total={bd['total'].item():.4f}"
#                       f"  cfm={bd['l_cfm']:.4f}"
#                       f"  reg={bd['l_reg']:.4f}"
#                       f"  lam={bd['lam_reg']:.3f}"
#                       f"  ade_1step={bd['ade_1step']:.0f}km"
#                       f"  sigma={bd['sigma']:.3f}"
#                       f"  hs_mean={bd.get('hard_score_mean',0):.2f}"
#                       f"  lr={lr:.2e}")

#         avg_loss = sum_loss / nstep
#         avg_cfm  = sum_cfm  / nstep
#         avg_reg  = sum_reg  / nstep
#         avg_ade1 = sum_ade1 / nstep
#         cur_lr   = sched.step()

#         print(f"  Epoch {ep:>3}"
#               f"  loss={avg_loss:.4f}"
#               f"  cfm={avg_cfm:.4f}"
#               f"  reg={avg_reg:.4f}"
#               f"  ade_1step={avg_ade1:.0f}km"
#               f"  lr={cur_lr:.2e}"
#               f"  t={time.perf_counter()-t0:.0f}s")

#         # ── Val ───────────────────────────────────────────────────────────
#         if ep % args.val_freq == 0:
#             r = evaluate(model, val_sub, device,
#                          tag=f"val ep{ep}",
#                          ddim_steps=args.ddim_steps,
#                          ema=ema)
#             ade = r["ADE"]

#             if ade < best_ade and ep >= args.min_ep:
#                 best_ade     = ade
#                 patience_cnt = 0
#                 _save(best_ckpt, ep, model, opt, sched, best_ade, ema)
#                 print(f"  ✅ Best ADE = {best_ade:.2f} km  (ep {ep})")
#             else:
#                 patience_cnt += args.val_freq
#                 print(f"  No improve {patience_cnt}/{args.patience}"
#                       f"  (best={best_ade:.1f})")
#                 if ep >= args.min_ep and patience_cnt >= args.patience:
#                     print(f"  ⛔ Early stop @ ep{ep}")
#                     break

#         # Periodic ckpt mỗi 10 epoch
#         if ep % 10 == 0:
#             _save(os.path.join(args.output_dir, f"ckpt_ep{ep:03d}.pth"),
#                   ep, model, opt, sched, best_ade, ema)

#     print("=" * 65)
#     print(f"  Best ADE: {best_ade:.2f} km")
#     print(f"  Target:   {ST_TRANS_TARGETS['ADE']:.2f} km (ST-Trans)")
#     print(f"  Gap:      {best_ade - ST_TRANS_TARGETS['ADE']:+.2f} km")
#     print("=" * 65)

#     # ── Test ──────────────────────────────────────────────────────────────
#     if args.test_at_end and os.path.exists(best_ckpt):
#         print("\n  Test set evaluation...")
#         try:
#             _, test_loader = data_loader(
#                 args, {"root": args.dataset_root, "type": "test"}, test=True)
#         except:
#             print("  No test set → using val")
#             _, test_loader = data_loader(
#                 args, {"root": args.dataset_root, "type": "val"}, test=True)

#         ck = torch.load(best_ckpt, map_location=device)
#         _unwrap(model).load_state_dict(ck["model"], strict=False)
#         if ema and ck.get("ema"):
#             for k, v in ck["ema"].items():
#                 if k in ema.shadow:
#                     ema.shadow[k].copy_(v.to(device))

#         evaluate(model, test_loader, device,
#                  tag="TEST (EMA best)",
#                  ddim_steps=args.ddim_steps,
#                  ema=ema)


# if __name__ == "__main__":
#     args = get_args()
#     np.random.seed(42)
#     torch.manual_seed(42)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(42)
#     main(args)
"""
scripts/train_fm.py  ——  TC-FlowMatching v2 Training
═══════════════════════════════════════════════════════

Tương ứng với flow_matching_model.py (FM đúng cách).

THIẾT KẾ TRAINING ĐƠN GIẢN:
  - 1 optimizer, 1 scheduler (Cosine + warmup)
  - Không GradNorm (chỉ có 2 loss terms)
  - Không phase transitions phức tạp
  - Không alpha_hard, không selector training
  - EMA + early stopping trên val ADE

HYPERPARAMETERS MẶC ĐỊNH:
  lr = 2e-4   (nhỏ hơn bản cũ vì FM transformer deeper)
  batch_size = 64
  num_epochs = 150
  warmup = 5 epoch
  sigma_max = 0.3, sigma_min = 0.1 (schedule tự động)
  lambda_reg = 0.2 (ramp từ epoch 10-30)
  n_ensemble = 20, ddim_steps = 8

CHẠY:
  python scripts/train_fm.py \
    --dataset_root /kaggle/input/.../tc-ofm \
    --output_dir   runs/fm_v2 \
    --num_epochs   150 \
    --batch_size   64

RESUME:
  python scripts/train_fm.py \
    --resume runs/fm_v2/best_model.pth \
    --dataset_root ...
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse, math, random, time
from collections import defaultdict
from typing import Dict, Optional

import numpy as np
import torch
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset

from Model.data.loader_training import data_loader
from Model.data.trajectoriesWithMe_unet_training import seq_collate
from Model.flow_matching_model import (
    TCFlowMatching, _norm_to_deg, _haversine_deg, EMAModel,
)

# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────
R_EARTH = 6371.0
HORIZON_STEPS = {12: 1, 24: 3, 48: 7, 72: 11}

ST_TRANS_TARGETS = {
    "ADE": 172.68, "72h": 321.39, "ATE": 142.21, "CTE": 42.04,
    "12h": 65.42,  "24h": 104.67, "48h": 205.10,
}


# ─────────────────────────────────────────────────────────────────────────────
#  Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _unwrap(m):
    return m._orig_mod if hasattr(m, "_orig_mod") else m

def move(batch, device):
    out = list(batch)
    for i, x in enumerate(out):
        if torch.is_tensor(x):
            out[i] = x.to(device)
        elif isinstance(x, dict):
            out[i] = {k: v.to(device) if torch.is_tensor(v) else v
                      for k, v in x.items()}
    return out

def make_subset_loader(dataset, n, batch_size, collate_fn):
    idx = random.Random(42).sample(range(len(dataset)), min(n, len(dataset)))
    return DataLoader(Subset(dataset, idx), batch_size=batch_size,
                      shuffle=False, collate_fn=collate_fn,
                      num_workers=0, drop_last=False)

def _ate_cte(pred_deg, gt_deg):
    """ATE, CTE per step"""
    T = min(pred_deg.shape[0], gt_deg.shape[0])
    if T < 2:
        z = pred_deg.new_zeros(1, pred_deg.shape[1])
        return z, z
    lo1 = torch.deg2rad(gt_deg[:T-1, :, 0])
    la1 = torch.deg2rad(gt_deg[:T-1, :, 1])
    lo2 = torch.deg2rad(gt_deg[1:T,  :, 0])
    la2 = torch.deg2rad(gt_deg[1:T,  :, 1])
    lo3 = torch.deg2rad(pred_deg[1:T, :, 0])
    la3 = torch.deg2rad(pred_deg[1:T, :, 1])
    ya  = torch.sin(lo2 - lo1) * torch.cos(la2)
    xa  = (torch.cos(la1) * torch.sin(la2)
           - torch.sin(la1) * torch.cos(la2) * torch.cos(lo2 - lo1))
    be  = torch.atan2(ya, xa)
    ye  = torch.sin(lo3 - lo2) * torch.cos(la3)
    xe  = (torch.cos(la2) * torch.sin(la3)
           - torch.sin(la2) * torch.cos(la3) * torch.cos(lo3 - lo2))
    bee = torch.atan2(ye, xe)
    tot = _haversine_deg(pred_deg[1:T], gt_deg[1:T])
    ang = bee - be
    return tot * torch.cos(ang), tot * torch.sin(ang)


# ─────────────────────────────────────────────────────────────────────────────
#  Scheduler: linear warmup + cosine annealing
# ─────────────────────────────────────────────────────────────────────────────

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int,
                 lr_init: float, lr_min: float = 1e-6):
        self.opt     = optimizer
        self.warmup  = warmup_epochs
        self.total   = total_epochs
        self.lr_init = lr_init
        self.lr_min  = lr_min
        self.epoch   = 0

    def step(self) -> float:
        e = self.epoch
        if e < self.warmup:
            lr = self.lr_init * (0.1 + 0.9 * e / max(self.warmup - 1, 1))
        else:
            t  = e - self.warmup
            T  = self.total - self.warmup
            lr = self.lr_min + 0.5 * (self.lr_init - self.lr_min) * (
                1 + math.cos(math.pi * t / max(T, 1)))
        for pg in self.opt.param_groups:
            pg["lr"] = lr
        self.epoch += 1
        return lr


# ─────────────────────────────────────────────────────────────────────────────
#  Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, tag: str = "",
             ddim_steps: int = 8, ema: Optional[EMAModel] = None) -> Dict:
    bk = None
    if ema is not None:
        try: bk = ema.apply_to(model)
        except: pass

    model.eval()
    all_ade, all_ate, all_cte = [], [], []
    step_ade = defaultdict(list)

    for batch in loader:
        bl   = move(list(batch), device)
        gt   = bl[1]
        pred, _, _ = model.sample(bl, ddim_steps=ddim_steps)

        T        = min(pred.shape[0], gt.shape[0])
        pred_deg = _norm_to_deg(pred[:T])
        gt_deg   = _norm_to_deg(gt[:T])
        dist     = _haversine_deg(pred_deg, gt_deg)  # [T, B]
        ate, cte = _ate_cte(pred_deg, gt_deg)

        all_ade.extend(dist.mean(0).tolist())
        if ate.shape[0] > 0:
            all_ate.extend(ate.abs().mean(0).tolist())
            all_cte.extend(cte.abs().mean(0).tolist())

        for h, s in HORIZON_STEPS.items():
            if s < T:
                step_ade[h].extend(dist[s].tolist())

    def _m(lst): return float(np.mean(lst)) if lst else float("nan")

    result = {
        "ADE": _m(all_ade),
        "ATE": _m(all_ate),
        "CTE": _m(all_cte),
        "n":   len(all_ade),
    }
    for h in HORIZON_STEPS:
        result[f"{h}h"] = _m(step_ade[h])

    if bk is not None:
        try: ema.restore(model, bk)
        except: pass

    # Print
    def _v(k): return result.get(k, float("nan"))
    def _ok(k, ref): return "✓" if np.isfinite(_v(k)) and _v(k) < ref else "✗"

    print(f"\n  {'='*60}")
    print(f"  [{tag}]")
    print(f"  ADE={_v('ADE'):.1f} {_ok('ADE', ST_TRANS_TARGETS['ADE'])}"
          f"  (target < {ST_TRANS_TARGETS['ADE']})")
    print(f"  12h={_v('12h'):.0f}  24h={_v('24h'):.0f}"
          f"  48h={_v('48h'):.0f}  72h={_v('72h'):.0f}"
          f"  {_ok('72h', ST_TRANS_TARGETS['72h'])}")
    print(f"  ATE={_v('ATE'):.1f}  CTE={_v('CTE'):.1f}")

    beat = []
    for k, ref in ST_TRANS_TARGETS.items():
        v = _v(k)
        if np.isfinite(v) and v < ref:
            beat.append(f"{k}:{v:.1f}<{ref:.1f}")
    if beat:
        print(f"  *** BEAT ST-TRANS: {' | '.join(beat)} ***")
    print(f"  {'='*60}\n")

    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def _save(path, epoch, model, opt, sched, best_ade, ema=None,
          patience_cnt=0, scaler=None, args=None):
    m   = _unwrap(model)
    esd = None
    if ema is not None:
        try: esd = {k: v.cpu().clone() for k, v in ema.shadow.items()}
        except: pass
    torch.save({
        # ── Core weights ──────────────────────────────────────────────
        "epoch":        epoch,
        "model":        m.state_dict(),
        "ema":          esd,
        # ── Optimizer / scheduler / scaler ────────────────────────────
        # Lưu đủ để resume tiếp tục chính xác như chưa bị interrupt:
        #   optimizer : Adam momentum + variance -> không mất đà
        #   scheduler : sched.epoch -> WarmupCosine biết đang ở đâu
        #   scaler    : AMP loss scale đã calibrate -> không bị reset
        "optimizer":    opt.state_dict(),
        "scheduler":    sched.epoch,
        "sched_total":  sched.total,    # giữ cosine tail đúng khi resume với num_epochs khác
        "scaler":       scaler.state_dict() if scaler is not None else None,
        # ── Training state ────────────────────────────────────────────
        "best_ade":     best_ade,
        "patience_cnt": patience_cnt,
        # ── Hyperparams snapshot ──────────────────────────────────────
        "args":         vars(args) if args is not None else None,
    }, path)


# ─────────────────────────────────────────────────────────────────────────────
#  Args
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Data
    p.add_argument("--dataset_root",  default="TCND_vn")
    p.add_argument("--obs_len",       default=8,     type=int)
    p.add_argument("--pred_len",      default=12,    type=int)
    p.add_argument("--num_workers",   default=2,     type=int)
    p.add_argument("--other_modal",   default="gph")
    p.add_argument("--delim",         default=" ")
    p.add_argument("--skip",          default=1,     type=int)
    p.add_argument("--min_ped",       default=1,     type=int)
    p.add_argument("--threshold",     default=0.002, type=float)
    # Model
    p.add_argument("--d_cond",        default=256,   type=int)
    p.add_argument("--d_model",       default=256,   type=int)
    p.add_argument("--nhead",         default=8,     type=int)
    p.add_argument("--num_dec_layers",default=4,     type=int)
    p.add_argument("--dim_ff",        default=512,   type=int)
    p.add_argument("--dropout",       default=0.1,   type=float)
    p.add_argument("--unet_in_ch",    default=13,    type=int)
    p.add_argument("--sigma_min",     default=0.04,  type=float,
                   help="FM sigma min (~22km, 15%% ADE). Phải < ADE_norm=0.32")
    p.add_argument("--sigma_max",     default=0.08,  type=float,
                   help="FM sigma max (~43km, 25%% ADE). Phải << ADE_norm")
    p.add_argument("--lambda_reg",    default=0.2,   type=float,
                   help="ADE signal weight (FIX-FATAL-1)")
    p.add_argument("--use_ot",        default=True,  action="store_true")
    p.add_argument("--ot_epsilon",    default=0.05,  type=float,
                   help="Sinkhorn epsilon cho OT matching")
    p.add_argument("--no_ot",         dest="use_ot", action="store_false")
    p.add_argument("--n_ensemble",    default=20,    type=int)
    p.add_argument("--sigma_inference",default=0.079, type=float,
                   help="noise std lúc inference (~43km → diversity ~60km std)")
    # Training
    p.add_argument("--num_epochs",    default=150,   type=int)
    p.add_argument("--batch_size",    default=64,    type=int)
    p.add_argument("--lr",            default=2e-4,  type=float)
    p.add_argument("--lr_min",        default=1e-6,  type=float)
    p.add_argument("--warmup_epochs", default=5,     type=int)
    p.add_argument("--weight_decay",  default=1e-4,  type=float)
    p.add_argument("--grad_clip",     default=1.0,   type=float)
    p.add_argument("--use_amp",       action="store_true", default=False)
    p.add_argument("--use_ema",       default=True,  action="store_true")
    p.add_argument("--no_ema",        dest="use_ema",action="store_false")
    p.add_argument("--ema_decay",     default=0.995, type=float)
    # Eval
    p.add_argument("--val_freq",      default=5,     type=int)
    p.add_argument("--val_subset",    default=1000,  type=int,
                   help="Số samples val mỗi lần eval (default 1000 ≈ 30% val set)")
    p.add_argument("--ddim_steps",    default=8,     type=int)
    p.add_argument("--patience",      default=30,    type=int)
    p.add_argument("--min_ep",        default=20,    type=int)
    # IO
    p.add_argument("--output_dir",    default="runs/fm_v2")
    p.add_argument("--gpu_num",       default="0")
    p.add_argument("--resume",        default=None)
    p.add_argument("--test_at_end",   action="store_true", default=True)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    best_ckpt = os.path.join(args.output_dir, "best_model.pth")

    print("=" * 65)
    print("  TC-FlowMatching v2  |  FM đúng cách")
    print(f"  sigma: {args.sigma_max}→{args.sigma_min}  "
          f"lambda_reg={args.lambda_reg}  K={args.n_ensemble}")
    print(f"  N_steps={args.ddim_steps}  sigma_inf={args.sigma_inference}")
    print(f"  FIX-FATAL-1: L_reg tại t=0 (gradient không dilute)")
    print(f"  FIX-FATAL-2: 2 loss terms (l_cfm + lambda_reg*l_reg)")
    print(f"  FIX-FATAL-3: Best-of-K thay SelectorNet")
    print(f"  FIX-MAJOR-4: sigma_max={args.sigma_max}, "
          f"sigma_inf={args.sigma_inference}")
    print(f"  FIX-MAJOR-5: FM trên 2D (không 4D)")
    print("=" * 65)

    # ── Data ──────────────────────────────────────────────────────────────
    trd, trl = data_loader(
        args, {"root": args.dataset_root, "type": "train"}, test=False)
    vd, _    = data_loader(
        args, {"root": args.dataset_root, "type": "val"},   test=True)
    val_sub  = make_subset_loader(vd, args.val_subset, args.batch_size, seq_collate)
    print(f"  train: {len(trd)}  val: {len(vd)}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = TCFlowMatching(
        pred_len=args.pred_len, obs_len=args.obs_len,
        unet_in_ch=args.unet_in_ch,
        d_cond=args.d_cond, d_model=args.d_model,
        nhead=args.nhead, num_dec_layers=args.num_dec_layers,
        dim_ff=args.dim_ff, dropout=args.dropout,
        sigma_min=args.sigma_min, sigma_max=args.sigma_max,
        lambda_reg=args.lambda_reg,
        use_ot=args.use_ot,
        ot_epsilon=args.ot_epsilon,
        use_ema=args.use_ema,
        n_ensemble=args.n_ensemble,
        sigma_inference=args.sigma_inference,
    ).to(device)

    model.init_ema()
    ema = getattr(_unwrap(model), "_ema", None)

    n_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_enc   = sum(p.numel() for p in model.encoder.parameters())
    n_vel   = sum(p.numel() for p in model.velocity.parameters())
    print(f"\n  Params total  : {n_total:,}")
    print(f"  Encoder       : {n_enc:,}  (FNO3D+Mamba, giữ nguyên)")
    print(f"  VelocityTrans : {n_vel:,}  (4-layer transformer, 2D only)")

    # ── Optimizer & scheduler ─────────────────────────────────────────────
    opt    = optim.AdamW(model.parameters(), lr=args.lr,
                          weight_decay=args.weight_decay)
    scaler = GradScaler("cuda", enabled=args.use_amp)
    sched  = WarmupCosineScheduler(
        opt, warmup_epochs=args.warmup_epochs,
        total_epochs=args.num_epochs,
        lr_init=args.lr, lr_min=args.lr_min)

    # ── Resume ────────────────────────────────────────────────────────────
    start_ep = 0
    best_ade = float("inf")
    patience_cnt = 0

    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=device)
        _unwrap(model).load_state_dict(ck["model"], strict=False)
        try: opt.load_state_dict(ck["optimizer"])
        except: pass
        sched.epoch  = ck.get("scheduler", 0)
        if "sched_total" in ck:
            sched.total = ck["sched_total"]   # giữ đúng cosine curve khi resume
        start_ep     = ck.get("epoch", 0) + 1
        best_ade     = ck.get("best_ade", float("inf"))
        patience_cnt = ck.get("patience_cnt", 0)   # ← restore patience
        if scaler is not None and ck.get("scaler") is not None:
            try: scaler.load_state_dict(ck["scaler"])   # ← restore AMP scale
            except: pass
        if ema and ck.get("ema"):
            for k, v in ck["ema"].items():
                if k in ema.shadow:
                    ema.shadow[k].copy_(v.to(device))
        print(f"  ↩ Resumed ep{start_ep}  best_ADE={best_ade:.1f} km"
              f"  patience={patience_cnt}/{args.patience}")

    try:
        model = torch.compile(model, mode="reduce-overhead")
        print("  torch.compile: ok")
    except:
        pass

    print()
    print("=" * 65)
    print(f"  TRAINING  ({len(trl)} steps/epoch, {args.num_epochs} epochs)")
    print(f"  Warmup: {args.warmup_epochs} ep  lr: {args.lr}→{args.lr_min}")
    print(f"  L_reg ramp: ep10→ep30 (0→{args.lambda_reg})")
    print("=" * 65)

    nstep = len(trl)

    for ep in range(start_ep, args.num_epochs):
        model.train()
        sum_loss, sum_cfm, sum_reg, sum_ade1 = 0.0, 0.0, 0.0, 0.0
        t0 = time.perf_counter()

        for i, batch in enumerate(trl):
            bl = move(list(batch), device)

            opt.zero_grad()
            with autocast(device_type="cuda", enabled=args.use_amp):
                bd = model.get_loss_breakdown(bl, epoch=ep)

            scaler.scale(bd["total"]).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            model.ema_update()

            sum_loss += bd["total"].item()
            sum_cfm  += bd["l_cfm"]
            sum_reg  += bd["l_reg"]
            sum_ade1 += bd["ade_1step"]

            if i % 30 == 0:
                lr = opt.param_groups[0]["lr"]
                print(f"  [{ep:>3}][{i:>3}/{nstep}]"
                      f"  total={bd['total'].item():.4f}"
                      f"  cfm={bd['l_cfm']:.4f}"
                      f"  reg={bd['l_reg']:.4f}"
                      f"  lam={bd['lam_reg']:.3f}"
                      f"  ade_1step={bd['ade_1step']:.0f}km"
                      f"  sigma={bd['sigma']:.3f}"
                      f"  hs_mean={bd.get('hard_score_mean',0):.2f}"
                      f"  lr={lr:.2e}")

        avg_loss = sum_loss / nstep
        avg_cfm  = sum_cfm  / nstep
        avg_reg  = sum_reg  / nstep
        avg_ade1 = sum_ade1 / nstep
        # Lấy lr hiện tại (đang dùng trong epoch này) TRƯỚC khi sched.step() thay đổi
        cur_lr = opt.param_groups[0]["lr"]
        sched.step()   # advance lr cho epoch tiếp theo

        print(f"  Epoch {ep:>3}"
              f"  loss={avg_loss:.4f}"
              f"  cfm={avg_cfm:.4f}"
              f"  reg={avg_reg:.4f}"
              f"  ade_1step={avg_ade1:.0f}km"
              f"  lr={cur_lr:.2e}"      # lr epoch này (không phải epoch sau)
              f"  t={time.perf_counter()-t0:.0f}s")

        # ── Periodic ckpt mỗi 10 epoch ──────────────────────────────────────
        # Đặt TRƯỚC val block để không bị skip khi early stop break
        if ep % 10 == 0:
            _save(os.path.join(args.output_dir, f"ckpt_ep{ep:03d}.pth"),
                  ep, model, opt, sched, best_ade, ema,
                  patience_cnt=patience_cnt, scaler=scaler, args=args)

        # ── Val ───────────────────────────────────────────────────────────
        if ep % args.val_freq == 0:
            r = evaluate(model, val_sub, device,
                         tag=f"val ep{ep}",
                         ddim_steps=args.ddim_steps,
                         ema=ema)
            ade = r["ADE"]

            if ade < best_ade:
                best_ade     = ade
                patience_cnt = 0
                # Lưu ngay từ epoch 0, không chờ min_ep
                _save(best_ckpt, ep, model, opt, sched, best_ade, ema,
                      patience_cnt=patience_cnt, scaler=scaler, args=args)
                print(f"  ✅ Best ADE = {best_ade:.2f} km  (ep {ep})")
            else:
                # Patience chỉ đếm sau min_ep — tránh early stop quá sớm
                if ep >= args.min_ep:
                    patience_cnt += args.val_freq
                print(f"  No improve {patience_cnt}/{args.patience}"
                      f"  (best={best_ade:.1f})")
                if ep >= args.min_ep and patience_cnt >= args.patience:
                    print(f"  ⛔ Early stop @ ep{ep}")
                    break

    print("=" * 65)
    print(f"  Best ADE: {best_ade:.2f} km")
    print(f"  Target:   {ST_TRANS_TARGETS['ADE']:.2f} km (ST-Trans)")
    print(f"  Gap:      {best_ade - ST_TRANS_TARGETS['ADE']:+.2f} km")
    print("=" * 65)

    # ── Test ──────────────────────────────────────────────────────────────
    if args.test_at_end and os.path.exists(best_ckpt):
        print("\n  Test set evaluation...")
        try:
            _, test_loader = data_loader(
                args, {"root": args.dataset_root, "type": "test"}, test=True)
        except:
            print("  No test set → using val")
            _, test_loader = data_loader(
                args, {"root": args.dataset_root, "type": "val"}, test=True)

        ck = torch.load(best_ckpt, map_location=device)
        _unwrap(model).load_state_dict(ck["model"], strict=False)
        if ema and ck.get("ema"):
            for k, v in ck["ema"].items():
                if k in ema.shadow:
                    ema.shadow[k].copy_(v.to(device))

        evaluate(model, test_loader, device,
                 tag="TEST (EMA best)",
                 ddim_steps=args.ddim_steps,
                 ema=ema)


if __name__ == "__main__":
    args = get_args()
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    main(args)