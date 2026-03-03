#!/usr/bin/env bash
set -euo pipefail

# usage:
#   ./scripts/sim_ros.sh
#   ./scripts/sim_ros.sh --headless
#   ./scripts/sim_ros.sh --gui
#
# optional env overrides:
#   TASK=standing SIM_STEPS=3000 ACTION_SOURCE=sine USD_PATH=/abs/path/to/scene.usd ./scripts/sim_ros.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IRL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ISAACLAB_SH="${IRL_ROOT}/dep/IsaacLab/isaaclab.sh"
ISAACLAB_RL_DIR="${IRL_ROOT}/isaaclab_rl"

TASK="${TASK:-standing}"
SIM_STEPS="${SIM_STEPS:-3000}"
ACTION_SOURCE="${ACTION_SOURCE:-sine}"
CMD_TOPIC="${CMD_TOPIC:-/sim/joint_cmd}"
FB_TOPIC="${FB_TOPIC:-/sim/joint_state_fb}"
USD_PATH="${USD_PATH:-${IRL_ROOT}/robots/gaoda_jiyuan/scene.usd}"
LOG_EVERY="${LOG_EVERY:-200}"
TMPDIR_DEFAULT="${TMPDIR:-${HOME}/workspace/tmp/isaaclab_tmp}"

RUN_MODE="auto"
if [[ "${1:-}" == "--headless" ]]; then
  RUN_MODE="headless"
  shift
elif [[ "${1:-}" == "--gui" ]]; then
  RUN_MODE="gui"
  shift
fi

if [[ ! -f "${ISAACLAB_SH}" ]]; then
  echo "[ERROR] isaaclab.sh 不存在: ${ISAACLAB_SH}" >&2
  exit 2
fi

if [[ ! -f "${USD_PATH}" ]]; then
  echo "[ERROR] USD 文件不存在: ${USD_PATH}" >&2
  exit 2
fi

export TMPDIR="${TMPDIR_DEFAULT}"
mkdir -p "${TMPDIR}"

export ROS_DISTRO=jazzy
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
export ROS_AUTOMATIC_DISCOVERY_RANGE="${ROS_AUTOMATIC_DISCOVERY_RANGE:-SUBNET}"
export FASTDDS_BUILTIN_TRANSPORTS="${FASTDDS_BUILTIN_TRANSPORTS:-UDPv4}"
# 默认清理静态 peers，避免旧环境变量导致发现/传输异常；如需保留可设置 SIM_ROS_KEEP_STATIC_PEERS=1。
if [[ "${SIM_ROS_KEEP_STATIC_PEERS:-0}" != "1" ]]; then
  unset ROS_STATIC_PEERS || true
fi

if [[ -z "${ISAAC_ROS2_LIB:-}" ]]; then
  ISAAC_ROS2_LIB="$(ls -d "${IRL_ROOT}"/.venv/lib/python3.*/site-packages/isaacsim/exts/isaacsim.ros2.bridge/jazzy/lib 2>/dev/null | head -n 1 || true)"
fi
if [[ -z "${ISAAC_ROS2_LIB:-}" ]]; then
  echo "[ERROR] 找不到 ISAAC_ROS2_LIB，请手动设置环境变量 ISAAC_ROS2_LIB" >&2
  exit 2
fi
export ISAAC_ROS2_LIB
export LD_LIBRARY_PATH="${ISAAC_ROS2_LIB}:/opt/ros/jazzy/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

APP_ARGS=()
if [[ "${RUN_MODE}" == "headless" ]]; then
  APP_ARGS+=(--headless)
elif [[ "${RUN_MODE}" == "auto" && -z "${DISPLAY:-}" ]]; then
  APP_ARGS+=(--headless)
fi

cd "${ISAACLAB_RL_DIR}"
echo "[INFO] DISPLAY=${DISPLAY:-<empty>} RUN_MODE=${RUN_MODE} APP_ARGS=${APP_ARGS[*]:-<none>}"
echo "[INFO] USD_PATH=${USD_PATH}"
echo "[INFO] ROS_DOMAIN_ID=${ROS_DOMAIN_ID} RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION} FASTDDS_BUILTIN_TRANSPORTS=${FASTDDS_BUILTIN_TRANSPORTS}"
echo "[INFO] ROS_LOCALHOST_ONLY=${ROS_LOCALHOST_ONLY} ROS_AUTOMATIC_DISCOVERY_RANGE=${ROS_AUTOMATIC_DISCOVERY_RANGE} ROS_STATIC_PEERS=${ROS_STATIC_PEERS:-<unset>}"
echo "[INFO] Logs: ${ISAACLAB_RL_DIR}/logs_run_sim_ros / logs_err_sim_ros"

set +e
"${ISAACLAB_SH}" -p scripts/sim_ros.py \
  "${APP_ARGS[@]}" \
  --task "${TASK}" \
  --usd_path "${USD_PATH}" \
  --sim_steps "${SIM_STEPS}" \
  --action_source "${ACTION_SOURCE}" \
  --cmd_topic "${CMD_TOPIC}" \
  --fb_topic "${FB_TOPIC}" \
  --ros_domain_id "${ROS_DOMAIN_ID}" \
  --log_every "${LOG_EVERY}" \
  "$@" \
  1>logs_run_sim_ros 2>logs_err_sim_ros
RC=$?
set -e

echo "===== STDERR ====="
tail -n 120 logs_err_sim_ros || true
echo "===== STDOUT ====="
tail -n 80 logs_run_sim_ros || true

exit "${RC}"
