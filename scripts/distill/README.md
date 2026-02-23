# scripts/distill 使用说明

## 脚本
- `scripts/distill/train_distill_mvp.py`

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
