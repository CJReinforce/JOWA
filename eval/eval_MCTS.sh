#!/bin/bash

# checkpoints of JOWA
ckpt_path="checkpoints/JOWA"
model_name="JOWA_150M"  # in [JOWA_150M, JOWA_70M, JOWA_40M]

# hyperparameters of JOWA's planning
use_planning=MCTS
horizon=6  # max depth of the tree, H in [1, 7]
temperature=0.9
num_simulations=20  # for the same expansion state budget as beam search, num_simulations = beam_width^2 * (beam_search_horizon - 1) + beam_width
use_mean=False  # False means V=Q.max(), otherwise V=Q.mean()
use_count=False  # False means argmax(root.children.value), otherwise argmax(root.children.visit_count)
buffer_size=(8)

# config of Atari
game=SpaceInvaders  # 1 of the 15 pretrained games
max_steps=108000

# how many rollouts (episodes) during evaluation
num_rollouts=16
# how many gpus can be used. will uniformly assign eval tasks to each gpu
num_gpus=8



echo model: "$ckpt_path"/"$model_name".pt
echo game: "$game"
echo horizon: "$horizon"
echo temperature: "$temperature"
echo num_simulations: "$num_simulations"
echo num_rollouts: "$num_rollouts"
echo num_gpus: "$num_gpus"


k=0

for i in "${buffer_size[@]}"; do
    for j in $(seq 1 "$num_rollouts"); do
        log_name="$model_name"_play_"$game"_buffer_size_"$i"_plan_"$use_planning"_h_"$horizon"_T_"$temperature"_ns_"$num_simulations"_use_mean_"$use_mean"_use_count_"$use_count"_date_$(date +%m%d_%H%M%S)_$RANDOM.log
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
        common.horizon="$horizon" \
        common.temperature="$temperature" \
        common.num_simulations="$num_simulations" \
        common.use_mean="$use_mean" \
        common.use_count="$use_count" \
        common.device="$device" \
        common.num_eval_episodes=1 \
        hydra/job_logging=disabled hydra/hydra_logging=disabled \
        >> $log_name 2>&1 &
    done
done
