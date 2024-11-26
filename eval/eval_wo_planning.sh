#!/bin/bash

# checkpoints of JOWA
ckpt_path="checkpoints/JOWA"
model_name="JOWA_150M"  # in [JOWA_150M, JOWA_70M, JOWA_40M]

game=Assault
num_rollouts=16
num_gpus=8
buffer_size=(1 2 3 4 5 6 7 8)

# no planning
use_planning=False
beam_width=1
horizon=0


echo model: "$ckpt_path"/"$model_name".pt
echo game: "$game"
echo buffer_size: "${buffer_size[*]}"
echo num_rollouts: "$num_rollouts"
echo num_gpus: "$num_gpus"


k=0
max_steps=108000


for i in "${buffer_size[@]}"; do
    for j in $(seq 1 "$num_rollouts"); do
        log_name="$model_name"_play_"$game"_buffer_size_"$i"_date_$(date +%m%d_%H%M%S)_$RANDOM.log
        > $log_name  # clear log

        gpu_id=$((k % "$num_gpus"))
        device="cuda:$gpu_id" 
        k=$((k + 1))

        python src/play.py \
        transformer=$model_name \
        critic_head=$model_name \
        initialization.path_to_checkpoint=$ckpt_path \
        initialization.jowa_model_name=$model_name \
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