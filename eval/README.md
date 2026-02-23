# eval 目录说明

用于存放模型评估和验收相关脚本：
- 指标计算（准确率、MAE、RMSE、峰值误差）。
- 性能评估（P95 推理时延、吞吐、显存占用）。
- OOD 稳定性评估。

验收以 `docs/spec.md` 中阈值为准。

## MVP 评估脚本
- 脚本：`eval/metrics_mvp.py`
- 输入：
  - `y_true.csv`
  - `y_pred.csv`
  - `latency_ms.txt`（每行一个毫秒值）
- 输出：
  - `eval/reports/metrics_mvp.json`

示例：
```bash
python3 eval/metrics_mvp.py \
  --y-true eval/samples/y_true.csv \
  --y-pred eval/samples/y_pred.csv \
  --latency eval/samples/latency_ms.txt \
  --output eval/reports/metrics_mvp.json
```
