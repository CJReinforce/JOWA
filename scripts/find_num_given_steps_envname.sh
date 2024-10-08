#!/bin/bash
name=Assault
device="cuda:0"

# clear log
> play_"$name".log

# Iterate from 4 to 20
for i in $(seq 4 20); do
  echo "Running with num_given_steps=$i" >> play_"$name".log
  python src/play.py common.num_given_steps=$i common.game_name="$name" common.use_planning=False common.device="$device" | grep 'return' >> play_"$name".log
  echo "" >> play_"$name".log
done
