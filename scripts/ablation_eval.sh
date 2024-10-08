#!/bin/bash

ckpt_path="outputs_finetune_Robotank/tokenizer_plus_world_model_plus_critic/2024-09-30/20-23-53/checkpoints/epoch_44_step_10000"
wm_name="world_model_200M"
name=Robotank
use_planning=False
beam_width=4
horizon=2
num_eval_episodes=6
ngss=(1 2 3 4 5 6 7 8)
num_params=200
total_gpus=8


declare -A gpu_ids=(
    [1]=0
    [2]=1
    [3]=2
    [4]=3
    [5]=4
    [6]=5
    [7]=7
    [8]=1
    [9]=2
    [10]=3
    [11]=4
    [12]=5

    [13]=4
    [14]=5
    [15]=5
    [16]=6
    [17]=6
    [18]=7
    [19]=7
    [20]=1
)


k=0
max_steps=108000

# num_given_steps iterates from 4 to 20
for i in "${ngss[@]}"; do
    for j in $(seq 1 "$num_eval_episodes"); do
        log_name=wm_name_"$wm_name"_params_"$num_params"M_play_"$name"_ngs_"$i"_plan_"$use_planning"_bw_"$beam_width"_h_"$horizon"_ep_"$num_eval_episodes"_$(date +%m%d_%H%M%S)_$RANDOM.log
        > $log_name  # clear log

        gpu_id=$((k % "$total_gpus"))
        # if [ $gpu_id -ge 3 ]; then
        #     gpu_id=$((gpu_id + 1))
        # fi
        # gpu_id=$((gpu_id + 4))
        device="cuda:$gpu_id" 
        k=$((k + 1))
        # gpu_id=${gpu_ids[$k]}
        # device="cuda:$gpu_id"
        # echo "Using GPU: $device"
        
        # actor_critic.td_loss=mse \
        # actor_critic.use_task_embed=False \
        # actor_critic.num_q=1 \
        # world_model.max_blocks=20 \

        python src/play.py \
        initialization.path_to_checkpoint=$ckpt_path \
        initialization.world_model_name=$wm_name \
        common.num_given_steps=$i \
        common.game_name="$name" \
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
