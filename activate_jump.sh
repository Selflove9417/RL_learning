#!/bin/bash
# ==============================================================================
# IsaacTu_jump: 独立跳跃工作区环境激活脚本
# 保证优先加载本工作区的 legged_gym 与 rsl_rl，完全隔离原 IsaacTu 项目
# ==============================================================================

# 1. 加载底层 CUDA/PyTorch 与基础运行时
source /home/xy/Documents/Codex/2026-10-05/chatgpt-conversation-6ac307b2-cac0-83ee-98c3/outputs/activate_rl.sh

# 2. 获取当前工作区绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 3. 优先注入本工作区的 Python 模块路径
export PYTHONPATH="${SCRIPT_DIR}/legged_gym:${SCRIPT_DIR}/rsl_rl:${PYTHONPATH}"

echo "============================================================"
echo " [IsaacTu_jump] 独立跳跃研发环境已激活"
echo " 工作区路径 : ${SCRIPT_DIR}"
echo " Python 解释器: $(which python)"
echo " Legged Gym   : ${SCRIPT_DIR}/legged_gym"
echo " RSL_RL       : ${SCRIPT_DIR}/rsl_rl"
echo "============================================================"
