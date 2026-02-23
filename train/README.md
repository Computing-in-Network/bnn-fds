# train 目录说明

用于存放训练入口与训练辅助模块：
- `train_mvp.py`：MVP 训练入口（占位可运行）。
- `models/`（后续）：模型定义。
- `losses/`（后续）：损失函数与物理约束项。

训练目标请遵循：
- `Accuracy >= 85%`
- `P95 <= 1.0s`（在部署模型上验证）

## 运行示例
```bash
python3 train/train_mvp.py --config configs/train/train_mvp.json
```

运行后会输出：
- `train/artifacts/mvp_run/model_meta.json`
- `train/artifacts/mvp_run/train_log.json`
