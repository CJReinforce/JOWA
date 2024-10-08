#!/bin/bash

# clear log
> play.log

# Iterate from 4 to 20
for i in $(seq 4 20); do
  echo "Running with num_given_steps=$i" >> play.log
  python src/play.py common.num_given_steps=$i | grep 'return' >> play.log
  echo "" >> play.log
done
