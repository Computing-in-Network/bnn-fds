# scripts/data 使用说明

## 目标
生成数据引擎 MVP 工况清单（CSV），用于后续批量 FDS 仿真任务编排。

## 脚本
- `scripts/data/generate_cases.py`

## 示例命令
```bash
python3 scripts/data/generate_cases.py --n-cases 100
python3 scripts/data/generate_cases.py --n-cases 1000 --output data/meta/cases_mvp_1000.csv
```

## 输出字段
- `case_id`
- `fire_x`
- `fire_y`
- `hrr_peak_kw`
- `vent_open_ratio`
- `duration_s`

## 配置文件
- 默认配置：`configs/data/case_space_mvp.json`
- 可配置项：
  - 随机种子 `seed`
  - 各字段取值范围 `ranges`
