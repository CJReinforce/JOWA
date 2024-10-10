#!/bin/bash

# checkpoints of JOWA
ckpt_path="checkpoints/JOWA"
wm_name="JOWA_150M"

# others
game=TimePilot
max_steps=108000
num_eval_episodes=4
total_gpus=2

# hyperparameters of JOWA's planning
use_planning=True
beam_width=3
horizon=2

buffer_size=(4)
# buffer_size=(1 2 3 4 5 6 7 8)


k=0

for i in "${buffer_size[@]}"; do
    for j in $(seq 1 "$num_eval_episodes"); do
        log_name="$wm_name"_play_"$game"_buffer_size_"$i"_plan_"$use_planning"_bw_"$beam_width"_h_"$horizon"_date_$(date +%m%d_%H%M%S)_$RANDOM.log
        > $log_name  # clear log

        gpu_id=$((k % "$total_gpus"))
        device="cuda:$gpu_id" 
        k=$((k + 1))

        /root/miniconda3/envs/lwm38/bin/python src/play.py \
        initialization.path_to_checkpoint=$ckpt_path \
        initialization.world_model_name=$wm_name \
        common.num_given_steps=$i \
        common.game_name="$game" \
        common.max_steps="$max_steps" \
        common.use_planning="$use_planning" \
        common.beam_width="$beam_width" \
        common.horizon="$horizon" \
        common.device="$device" \
        common.num_eval_episodes=1 \
        hydra/job_logging=disabled hydra/hydra_logging=disabled \
        >> $log_name 2>&1 &
    done
done
