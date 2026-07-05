# TC-FlowMatching v2.4 — README

## Cấu trúc thư mục (Kaggle)

```
/kaggle/working/TC_OFM/
├── Model/
│   ├── data/
│   │   └── loader_training.py
│   ├── flow_matching_model.py        ← copy flow_matching_model_v24_final.py
│   ├── FNO3D_encoder.py
│   ├── mamba_encoder.py
│   ├── env_net_transformer_gphsplit.py
│   └── utils.py
├── scripts/
│   └── train_flowmatching.py         ← copy train_fm_v24.py (GHI ĐÈ file cũ)
├── ablation_runner.py
├── evaluate_full.py
├── run_all.py
└── statistical_tests.py

Dataset:
/kaggle/input/datasets/kaggle1234uitvn/tc-ofm/

Output:
/kaggle/working/runs/
```

---

## Thay đổi so với lệnh cũ

Lệnh cũ của bạn:
```bash
!python /kaggle/working/TC_OFM/scripts/train_flowmatching.py \
    --dataset_root /kaggle/input/datasets/kaggle1234uitvn/tc-ofm \
    --output_dir   /kaggle/working/runs/fm_v21_strategy \
    --num_epochs 200 --gpu_num 0
```

Lệnh mới v2.4 (tương đương, 1 seed):
```bash
!python /kaggle/working/TC_OFM/scripts/train_flowmatching.py \
    --dataset_root /kaggle/input/datasets/kaggle1234uitvn/tc-ofm \
    --output_dir   /kaggle/working/runs/fm_v24 \
    --num_epochs   150 \
    --seed         42 \
    --gpu_num      0
```

> **Giải thích thay đổi:**
> - `--num_epochs 150` thay vì 200: model plateau sau ~ep75, 150 đủ
> - `--seed 42`: giữ nguyên seed cũ để kết quả comparable
> - Không cần thêm flag nào khác: mọi fix (Kendall, heading uniform, ramp_calib) đã được hardcode trong model mới

---

## Lệnh chạy chi tiết

### 1. Train đơn giản nhất (giống bữa giờ, 1 seed)

```bash
!python /kaggle/working/TC_OFM/scripts/train_flowmatching.py \
    --dataset_root /kaggle/input/datasets/kaggle1234uitvn/tc-ofm \
    --output_dir   /kaggle/working/runs/fm_v24 \
    --num_epochs   150 \
    --gpu_num      0
```

### 2. Resume từ checkpoint bị ngắt giữa chừng

```bash
!python /kaggle/working/TC_OFM/scripts/train_flowmatching.py \
    --dataset_root /kaggle/input/datasets/kaggle1234uitvn/tc-ofm \
    --output_dir   /kaggle/working/runs/fm_v24 \
    --resume       /kaggle/working/runs/fm_v24/last_model.pth \
    --num_epochs   150 \
    --gpu_num      0
```

> Resume từ `last_model.pth` (lưu mỗi epoch) hoặc `best_model.pth`.
> Tự động khôi phục: model weights, optimizer, scheduler, EMA, epoch, best_score, patience.

### 3. Train với pretrain (như --pretrain cũ)

v2.4 dùng `--resume` thay cho `--pretrain`:

```bash
!python /kaggle/working/TC_OFM/scripts/train_flowmatching.py \
    --dataset_root /kaggle/input/datasets/kaggle1234uitvn/tc-ofm \
    --output_dir   /kaggle/working/runs/fm_v24_finetune \
    --resume       /kaggle/working/runs/fm_v24/best_model.pth \
    --num_epochs   150 \
    --gpu_num      0
```

### 4. Train 3 seeds (ESWA mean±std)

Chạy lần lượt 3 cell trên Kaggle (mỗi cell = 1 session):

**Seed 0:**
```bash
!python /kaggle/working/TC_OFM/scripts/train_flowmatching.py \
    --dataset_root /kaggle/input/datasets/kaggle1234uitvn/tc-ofm \
    --output_dir   /kaggle/working/runs/fm_v24_seed0 \
    --num_epochs   150 \
    --seed         0 \
    --gpu_num      0
```

**Seed 1:**
```bash
!python /kaggle/working/TC_OFM/scripts/train_flowmatching.py \
    --dataset_root /kaggle/input/datasets/kaggle1234uitvn/tc-ofm \
    --output_dir   /kaggle/working/runs/fm_v24_seed1 \
    --num_epochs   150 \
    --seed         1 \
    --gpu_num      0
```

**Seed 2:**
```bash
!python /kaggle/working/TC_OFM/scripts/train_flowmatching.py \
    --dataset_root /kaggle/input/datasets/kaggle1234uitvn/tc-ofm \
    --output_dir   /kaggle/working/runs/fm_v24_seed2 \
    --num_epochs   150 \
    --seed         2 \
    --gpu_num      0
```

### 5. Run tất cả 1 lệnh (dùng run_all.py)

```bash
!python /kaggle/working/TC_OFM/run_all.py \
    --dataset_root /kaggle/input/datasets/kaggle1234uitvn/tc-ofm \
    --output_dir   /kaggle/working/runs/eswa_full \
    --num_epochs   150 \
    --seeds        0 1 2 \
    --gpu          0
```

> Thực hiện tuần tự: train 3 seeds → ODE sweep → eval → statistical tests → tổng hợp bảng paper.

---

## Eval sau khi train xong

### Eval test set từ best checkpoint

```bash
!python /kaggle/working/TC_OFM/evaluate_full.py \
    --checkpoint   /kaggle/working/runs/fm_v24/best_model.pth \
    --dataset_root /kaggle/input/datasets/kaggle1234uitvn/tc-ofm \
    --split        test \
    --output_dir   /kaggle/working/results \
    --n_ensemble   20 \
    --gpu          0
```

### Eval val set

```bash
!python /kaggle/working/TC_OFM/evaluate_full.py \
    --checkpoint   /kaggle/working/runs/fm_v24/best_model.pth \
    --dataset_root /kaggle/input/datasets/kaggle1234uitvn/tc-ofm \
    --split        val \
    --output_dir   /kaggle/working/results \
    --gpu          0
```

### Statistical tests (FM vs ST-Trans)

```bash
!python /kaggle/working/TC_OFM/statistical_tests.py \
    --fm_results       /kaggle/working/results/eval_test_ep149.json \
    --use_st_trans_ref \
    --fm_n_storms      420 \
    --baseline_name    ST-Trans \
    --output_dir       /kaggle/working/results/stats \
    --n_bootstrap      10000
```

---

## Ablation studies

### Ablation: no L_heading

```bash
!python /kaggle/working/TC_OFM/scripts/train_flowmatching.py \
    --dataset_root      /kaggle/input/datasets/kaggle1234uitvn/tc-ofm \
    --output_dir        /kaggle/working/runs/ablation_no_heading \
    --num_epochs        150 \
    --disable_l_heading \
    --ablation_name     no_L_heading \
    --gpu_num           0
```

### Ablation: no L_calib (no speed correction training)

```bash
!python /kaggle/working/TC_OFM/scripts/train_flowmatching.py \
    --dataset_root    /kaggle/input/datasets/kaggle1234uitvn/tc-ofm \
    --output_dir      /kaggle/working/runs/ablation_no_calib \
    --num_epochs      150 \
    --disable_l_calib \
    --ablation_name   no_L_calib \
    --gpu_num         0
```

### Ablation: CFM only (bỏ tất cả auxiliary loss)

```bash
!python /kaggle/working/TC_OFM/scripts/train_flowmatching.py \
    --dataset_root      /kaggle/input/datasets/kaggle1234uitvn/tc-ofm \
    --output_dir        /kaggle/working/runs/ablation_cfm_only \
    --num_epochs        150 \
    --disable_l_heading \
    --disable_l_calib   \
    --disable_l_reg     \
    --ablation_name     cfm_only \
    --gpu_num           0
```

### Ablation: no AUG-C (không có recurvature augmentation)

```bash
!python /kaggle/working/TC_OFM/scripts/train_flowmatching.py \
    --dataset_root   /kaggle/input/datasets/kaggle1234uitvn/tc-ofm \
    --output_dir     /kaggle/working/runs/ablation_no_augc \
    --num_epochs     150 \
    --disable_aug_c  \
    --ablation_name  no_aug_c \
    --gpu_num        0
```

### ODE steps sweep (chỉ cần 1 checkpoint, không train lại)

```bash
!python /kaggle/working/TC_OFM/ablation_runner.py \
    --mode         ode_steps \
    --checkpoint   /kaggle/working/runs/fm_v24/best_model.pth \
    --dataset_root /kaggle/input/datasets/kaggle1234uitvn/tc-ofm \
    --output_dir   /kaggle/working/ablations \
    --gpu          0
```

### Tổng hợp kết quả ablation

```bash
!python /kaggle/working/TC_OFM/ablation_runner.py \
    --mode         summarize \
    --ablation_dir /kaggle/working/ablations
```

---

## Theo dõi quá trình train

Các thư mục checkpoint được lưu tự động:

```
/kaggle/working/runs/fm_v24/
├── best_model.pth       ← best val ADE (dùng cho eval và paper)
├── hard_best_model.pth  ← best trên hard storms (test distribution)
├── last_model.pth       ← epoch cuối (dùng để resume)
├── swa_model.pth        ← SWA weights (nếu SWA bật)
└── footprint.json       ← n_params, memory, wall-clock (dùng cho Table E paper)
```

Log mỗi epoch hiển thị:
```
[ep][step]  tot=...  cfm=...  reg=...  h4s=...  lam_d=...  ade1=...km  enc=active  lr=2.00e-04

[VAL ep50]  ADE=185.3km  ATE=176.1km  CTE=48.2km
[LEARN] speed_corr(12h-24h)=[0.92,0.95,0.97,0.98]
[LEARN] log_sigma: reg=0.621  heading=1.124  calib=0.912
[LEARN] eff_lambda: reg=0.152  heading=0.042  calib=0.082
```

---

## Tất cả flags của train_flowmatching.py

| Flag | Default | Mô tả |
|------|---------|-------|
| `--dataset_root` | `TCND_vn` | Đường dẫn dataset |
| `--output_dir` | `runs/fm_v26` | Thư mục lưu checkpoint |
| `--num_epochs` | `150` | Số epoch train |
| `--seed` | `42` | Random seed |
| `--gpu_num` | `0` | GPU index |
| `--resume` | `None` | Path checkpoint để resume |
| `--batch_size` | `64` | Batch size |
| `--lr` | `2e-4` | Learning rate velocity |
| `--lr_min` | `1e-6` | LR tối thiểu (cosine decay) |
| `--freeze_encoder_epochs` | `10` | Số epoch freeze encoder |
| `--val_freq` | `5` | Eval val mỗi N epoch |
| `--patience` | `20` | Early stopping patience |
| `--n_ensemble` | `20` | K samples khi inference |
| `--use_amp` | `False` | Mixed precision (tiết kiệm VRAM) |
| `--no_ema` | — | Tắt EMA |
| `--no_test` | — | Không eval test sau khi train |
| `--disable_l_heading` | `False` | Ablation: tắt L_heading |
| `--disable_l_calib` | `False` | Ablation: tắt L_calib |
| `--disable_l_reg` | `False` | Ablation: tắt L_reg |
| `--disable_aug_c` | `False` | Ablation: tắt AUG-C recurvature |
| `--ablation_name` | `""` | Tag tên ablation vào output_dir |

---

## Ghi chú Kaggle

**Session timeout:** Kaggle P100 session giới hạn ~9h. Nếu bị ngắt:
```bash
# Kiểm tra epoch cuối đã lưu
!ls -la /kaggle/working/runs/fm_v24/

# Resume từ last_model.pth
!python /kaggle/working/TC_OFM/scripts/train_flowmatching.py \
    --dataset_root /kaggle/input/datasets/kaggle1234uitvn/tc-ofm \
    --output_dir   /kaggle/working/runs/fm_v24 \
    --resume       /kaggle/working/runs/fm_v24/last_model.pth \
    --num_epochs   150 \
    --gpu_num      0
```

**Lưu output sang dataset** (tránh mất khi session hết):
```python
# Trong Kaggle notebook cell
import shutil, os
os.makedirs("/kaggle/working/save", exist_ok=True)
shutil.copy("/kaggle/working/runs/fm_v24/best_model.pth",
            "/kaggle/working/save/best_model_v24.pth")
```

**Mixed precision** (nếu GPU bị OOM):
```bash
# Thêm --use_amp vào lệnh train
!python /kaggle/working/TC_OFM/scripts/train_flowmatching.py \
    --dataset_root /kaggle/input/datasets/kaggle1234uitvn/tc-ofm \
    --output_dir   /kaggle/working/runs/fm_v24 \
    --num_epochs   150 \
    --use_amp \
    --gpu_num      0
```