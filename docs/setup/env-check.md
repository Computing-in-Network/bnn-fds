# 环境安装与验证记录（ISSUE-0006）

## 执行日期
- 2026-02-23

## 系统信息
- OS: Ubuntu 22.04.5 LTS
- Kernel: 6.8.0-94-generic

## 安装内容
- FDS / Smokeview:
  - 来源：`firemodels/fds` 官方 release
  - 版本：`FDS-6.10.1_SMV-6.10.1`
  - 安装路径：`/home/zyren/FDS/FDS6`
  - 安装脚本：`/tmp/fds_installer_lnx.sh`

## 验证命令与结果
- 命令可用性（已通过）：
  - `command -v fds` -> `/home/zyren/FDS/FDS6/bin/fds`
  - `command -v smokeview` -> `/home/zyren/FDS/FDS6/smvbin/smokeview`
- Python 运行时（已通过）：
  - `numpy`
  - `torch`
  - `tensorflow`

## 一键复查脚本
- 脚本：`scripts/setup/check_runtime.sh`
- 用法：
```bash
bash scripts/setup/check_runtime.sh
```

## 注意事项
- 新 shell 若未加载 FDS 环境，请先执行：
```bash
source /home/zyren/FDS/FDS6/bin/FDS6VARS.sh
source /home/zyren/FDS/FDS6/bin/SMV6VARS.sh
```
- 如需长期生效，可将上述两行写入 `~/.bashrc`。
