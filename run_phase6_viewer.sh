#!/bin/bash
# ==============================================================================
# MyRobot Phase 6: 连续两次跳跃 3D 成果窗口与交互控制启动脚本
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

source "${SCRIPT_DIR}/activate_jump.sh"

if [ -z "$DISPLAY" ]; then
    export DISPLAY=:1
fi

echo "=================================================================="
echo " 正在启动 Isaac Gym Phase 6 连续跳跃 3D 可视化窗口 (纯手动遥控模式)..."
echo " 显示环境 : DISPLAY=${DISPLAY}"
echo " 控制模式 : 100% 纯手动控制 (原地待命，等待用户按键控制移动、高度与跳跃)"
echo " 键盘控制 : [W/S] 加速/减速 | [A/D] 转向 | [Q/E] 升降机身 | [Space] 跳跃 | [X] 刹车 | [R] 复位 | [ESC] 退出"
echo " 遥测导出 : 50 Hz 逐帧数据写入 evaluation_assets/phase6/viewer_telemetry.csv"
echo "=================================================================="

python "${SCRIPT_DIR}/legged_gym/legged_gym/scripts/teleop_dual.py" "$@"
