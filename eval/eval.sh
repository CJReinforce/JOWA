#!/bin/bash

# checkpoints of JOWA
ckpt_path="checkpoints/JOWA"
model_name="JOWA_150M"  # in [JOWA_150M, JOWA_70M, JOWA_40M]

# config of Atari
game=Atlantis  # 1 of the 15 pretrained games
max_steps=108000

# how many rollouts (episodes) during evaluation
num_rollouts=16
# how many gpus can be used. will uniformly assign eval tasks to each gpu
num_gpus=8



# hyperparameters of JOWA's planning
use_planning=True

# find hyperparameters in config json
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
config_file="${script_dir}/${model_name}_raw_scores.json"
find_flag=true

if [ ! -f "$config_file" ]; then
    echo "Warning: Configuration file $config_file not found. Use default planning hyperparameters."
    find_flag=false
fi

if ! command -v jq &> /dev/null; then
    echo "Warning: jq is not installed. Please install jq to parse JSON files: apt install jq. Use default planning hyperparameters."
    find_flag=false
fi

if $find_flag; then
    beam_width=$(jq -r ".[\"$game\"].beam_width" "$config_file")
    horizon=$(jq -r ".[\"$game\"].planning_horizon" "$config_file")
    buffer_size=$(jq -r ".[\"$game\"].buffer_size" "$config_file")
    buffer_size=($(echo $buffer_size | jq -r '.[]'))
fi

# if not find hyperparameters in config json, use default hyperparameters
if ! $find_flag; then
    beam_width=1
    horizon=0

    buffer_size=(8)
    # buffer_size=(1 2 3 4 5 6 7 8)
    # recommand searching over possible buffer size for the best performance
    # we employ this hyperparameter search for all algorithms that can handle variable-length input，
    # i.e., JOWA, MTBC, MGDT, and EDT
fi
# buffer_size=(1 2 3 4 5 6)

echo model: "$ckpt_path"/"$model_name".pt
echo game: "$game"
echo beam_width: "$beam_width"
echo horizon: "$horizon"
echo buffer_size: "${buffer_size[*]}"
echo num_rollouts: "$num_rollouts"
echo num_gpus: "$num_gpus"


k=0

for i in "${buffer_size[@]}"; do
    for j in $(seq 1 "$num_rollouts"); do
        log_name="$model_name"_play_"$game"_buffer_size_"$i"_plan_"$use_planning"_bw_"$beam_width"_h_"$horizon"_date_$(date +%m%d_%H%M%S)_$RANDOM.log
        > $log_name  # clear log

        gpu_id=$((k % "$num_gpus"))
        device="cuda:$gpu_id" 
        k=$((k + 1))

        python src/play.py \
        transformer=$model_name \
        critic_head=$model_name \
        initialization.path_to_checkpoint=$ckpt_path \
        initialization.world_model_name=$model_name \
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
