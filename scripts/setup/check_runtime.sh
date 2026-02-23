#!/usr/bin/env bash
set -euo pipefail

FDS_ROOT="${FDS_ROOT:-$HOME/FDS/FDS6}"

if [[ -f "$FDS_ROOT/bin/FDS6VARS.sh" ]]; then
  # shellcheck disable=SC1090
  source "$FDS_ROOT/bin/FDS6VARS.sh"
fi
if [[ -f "$FDS_ROOT/bin/SMV6VARS.sh" ]]; then
  # shellcheck disable=SC1090
  source "$FDS_ROOT/bin/SMV6VARS.sh"
fi

echo "== Command Check =="
command -v fds >/dev/null 2>&1 && echo "[OK] fds: $(command -v fds)" || echo "[FAIL] fds"
command -v smokeview >/dev/null 2>&1 && echo "[OK] smokeview: $(command -v smokeview)" || echo "[FAIL] smokeview"

echo
echo "== Python Runtime Check =="
python3 - <<'PY'
import importlib.util as u
for name in ["numpy", "torch", "tensorflow"]:
    print(f"[{'OK' if u.find_spec(name) else 'FAIL'}] {name}")
PY
