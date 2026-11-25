#!/usr/bin/env bash
# P1 data collection: pi0* (--method vanilla) rollouts on the 9 SIMPLER tasks, with a
# per-episode dump (npz + mp4) for the FlashVLA bootstrap dataset.
# Plan: docs/sdn_flashvla/flashvla_simpler/3.flashvla_simpler.md §3.
#
# Usage (from anywhere): bash policy/pizero/scripts/run_episode_dump.sh [num_gpus] [result_root]
#
# Environment overrides:
#   N_TRAJS      episodes per task; overrides the per-task quota table below
#   START_SEED   1000 (default) -> disjoint from seeds 0-49 (R1 eval set) and
#                100-149 (planned G3 holdout), see §1.2 constraint 3
#   DUMP_IMAGES  mp4 (default) | npz | none
#   TASKS        space-separated subset, e.g. TASKS="widowx_stack_cube"
#   RELOAD_EVERY policy rebuild period in episodes (default 50, 0 disables). Every rebuild
#                costs ~50 s (env init + ckpt load + torch.compile), so the R1 default of 8
#                would waste ~5 h over the full 2300-episode plan. Measured over 20 episodes
#                with 0: GPU memory stayed flat at 8.1 GB, thanks to the per-episode
#                gc.collect() + empty_cache() in evaluate.py.
#   CHECKPOINT   pretrained/open-pi-zero (default)
#
# Why --method vanilla: dumping with --method sdn bakes the candidate selection into the
# weights, which would make the FlashVLA baseline optimistic (§1.2 constraint 1).
# Why the per-task quotas: 50 episodes/task only yields ~300 successful trajectories and
# ~11k frames in total, i.e. 1/145 of the LIBERO recipe's sample budget (§3.2).
#
# Re-running resumes: a task whose result dir already holds a complete dump is skipped.
set -euo pipefail
cd "$(dirname "$0")/../../.."

NUM_GPUS=${1:-1}
RESULT_ROOT=${2:-./eval_logs/p1_dump}
START_SEED=${START_SEED:-1000}
DUMP_IMAGES=${DUMP_IMAGES:-mp4}
RELOAD_EVERY=${RELOAD_EVERY:-50}
CHECKPOINT=${CHECKPOINT:-pretrained/open-pi-zero}

# Episodes per task (success rate from docs/sdn_flashvla/sdn/results.md §2 -> expected
# number of successful trajectories in the comment).
TASK_ORDER=(
    google_robot_place_apple_in_closed_top_drawer   # 20% -> ~100
    google_robot_open_drawer                        # 48% -> ~240
    widowx_carrot_on_plate                          # 56% -> ~170
    google_robot_move_near                          # 72% -> ~145
    google_robot_close_drawer                       # 78% -> ~156
    google_robot_pick_coke_can                      # 86% -> ~130
    widowx_put_eggplant_in_basket                   # 88% -> ~130
    widowx_spoon_on_towel                           # 82% -> ~125
    widowx_stack_cube                               # 76% -> ~115
)
declare -A QUOTA=(
    [google_robot_place_apple_in_closed_top_drawer]=500
    [google_robot_open_drawer]=500
    [widowx_carrot_on_plate]=300
    [google_robot_move_near]=200
    [google_robot_close_drawer]=200
    [google_robot_pick_coke_can]=150
    [widowx_put_eggplant_in_basket]=150
    [widowx_spoon_on_towel]=150
    [widowx_stack_cube]=150
)

if [[ -n "${TASKS:-}" ]]; then
    read -r -a TASKS_ARR <<< "$TASKS"
else
    TASKS_ARR=("${TASK_ORDER[@]}")
fi

echo "=== P1 dump | gpus=$NUM_GPUS root=$RESULT_ROOT start_seed=$START_SEED images=$DUMP_IMAGES"
echo "=== tasks: ${TASKS_ARR[*]}"

FAILED=()
for task in "${TASKS_ARR[@]}"; do
    n=${N_TRAJS:-${QUOTA[$task]:-150}}
    echo "=== vanilla | $task | n_trajs=$n seeds=$START_SEED-$((START_SEED + n - 1)) @ $(date '+%F %T')"
    # a failure of one task must not kill the remaining 10+ hours of collection;
    # re-running the script resumes (finished tasks are skipped)
    if ! python -m policy.pizero.evaluate \
        --method vanilla \
        --task "$task" \
        --checkpoint "$CHECKPOINT" \
        --num-gpus "$NUM_GPUS" \
        --n-trajs "$n" \
        --start-seed "$START_SEED" \
        --reload-every "$RELOAD_EVERY" \
        --result-root "$RESULT_ROOT" \
        --dump-episodes \
        --dump-images "$DUMP_IMAGES" \
        --no-video; then
        FAILED+=("$task")
        echo "=== FAILED: $task @ $(date '+%F %T') (re-run to resume)"
    fi
    echo "=== $task done @ $(date '+%F %T')"
done

echo "=== done. dumps under $RESULT_ROOT/vanilla/seeds=${START_SEED}-*/*/dump/"
echo "=== health check: python scripts/p1/check_raw.py --root $RESULT_ROOT"
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "=== FAILED TASKS: ${FAILED[*]}"
    exit 1
fi
