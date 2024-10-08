#!/bin/bash

name=Robotank
use_planning=False
beam_width=4
horizon=2
num_eval_episodes=1

if [ "$name" = "Phoenix" ]; then
    max_steps=5000
elif [ "$name" = "NameThisGame" ]; then
    max_steps=10000
else
    max_steps=10000000000
fi

# max_steps=10000

declare -A gpu_ids=(
    [1]=0
    [2]=1
    [3]=2
    [4]=3
    [5]=4
    [6]=5
    [7]=6
    [8]=7
    [9]=2
    [10]=3
    [11]=3
    [12]=4
    [13]=4
    [14]=5
    [15]=5
    [16]=6
    [17]=6
    [18]=7
    [19]=7
    [20]=1
)

# num_given_steps iterates from 4 to 20
for i in $(seq 1 8); do
    log_name=play_"$name"_ngs_"$i"_plan_"$use_planning"_bw_"$beam_width"_h_"$horizon"_ep_"$num_eval_episodes".log
    > $log_name  # clear log

    gpu_id=${gpu_ids[$i]}
    device="cuda:$gpu_id" 

    python src/play.py \
    common.num_given_steps=$i \
    common.game_name="$name" \
    common.max_steps="$max_steps" \
    common.use_planning="$use_planning" \
    common.beam_width="$beam_width" \
    common.horizon="$horizon" \
    common.device="$device" \
    common.num_eval_episodes="$num_eval_episodes" \
    hydra/job_logging=disabled hydra/hydra_logging=disabled \
    >> $log_name 2>&1 &
done
