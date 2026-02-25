# scripts/distill 使用说明

## 脚本
- `scripts/distill/train_distill_mvp.py`
- `scripts/distill/train_distill_mlp_mvp.py`
- `scripts/distill/infer_distill_mlp_mvp.py`
- `scripts/distill/benchmark_infer_mvp.py`
- `scripts/distill/train_distill_real_mvp.py`
- `scripts/distill/benchmark_real_mvp.py`

## 作用
执行 Teacher-Student 蒸馏 MVP：
- 从 `cases_mvp.csv` 构造 teacher 目标；
- 训练线性 student；
- 输出模型参数和评估指标。

## 示例
```bash
python3 scripts/distill/train_distill_mvp.py \
  --cases data/meta/cases_mvp.csv \
  --output-dir distill/artifacts/mvp_distill
```

## 输出
- `distill/artifacts/mvp_distill/teacher_targets_mvp.csv`
- `distill/artifacts/mvp_distill/student_preds_mvp.csv`
- `distill/artifacts/mvp_distill/student_model_mvp.json`
- `distill/artifacts/mvp_distill/metrics_mvp.json`

## MLP 蒸馏示例
```bash
python3 scripts/distill/train_distill_mlp_mvp.py \
  --cases data/meta/cases_mvp.csv \
  --output-dir distill/artifacts/mvp_distill_mlp \
  --epochs 200 \
  --hidden-dim 32
```

输出：
- `distill/artifacts/mvp_distill_mlp/student_mlp_model.pt`
- `distill/artifacts/mvp_distill_mlp/metrics_train.json`
- `distill/artifacts/mvp_distill_mlp/metrics_val.json`
- `distill/artifacts/mvp_distill_mlp/train_history.csv`

## 推理与基准
```bash
python3 scripts/distill/infer_distill_mlp_mvp.py \
  --cases data/meta/cases_mvp.csv \
  --model distill/artifacts/mvp_distill_mlp/student_mlp_model.pt \
  --output distill/artifacts/mvp_distill_mlp/infer_preds.csv

python3 scripts/distill/benchmark_infer_mvp.py \
  --cases data/meta/cases_mvp.csv \
  --model distill/artifacts/mvp_distill_mlp/student_mlp_model.pt \
  --num-repeats 50 \
  --output distill/artifacts/mvp_distill_mlp/infer_benchmark.json
```

## 真实 FDS 标签蒸馏示例
```bash
python3 scripts/fds/extract_real_labels_mvp.py \
  --outputs-dir data/fds_outputs \
  --output data/meta/fds_labels_real_mvp.csv

python3 scripts/distill/train_distill_real_mvp.py \
  --cases data/meta/cases_mvp.csv \
  --labels data/meta/fds_labels_real_mvp.csv \
  --output-dir distill/artifacts/real_mvp \
  --hidden-dim-1 64 \
  --hidden-dim-2 64 \
  --select-model mlp
```

建议将 `mlp(64x64)` 作为默认主模型，`linear` 作为对照基线。  
如需自动按验证误差选优，可用：`--select-model auto`。

真实标签模型推理基准：
```bash
python3 scripts/distill/benchmark_real_mvp.py \
  --cases data/meta/cases_mvp.csv \
  --model distill/artifacts/real_mvp/student_real_mvp.pt \
  --num-repeats 5000 \
  --output distill/artifacts/real_mvp/infer_benchmark.json
```
