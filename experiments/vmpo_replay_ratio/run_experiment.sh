#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

echo "script dir is " $DIR

python3 vmpo_replay_ratio.py --epochs=1 --log_dir=$DIR/../logs/vmpo_replay_ratio/1
python3 vmpo_replay_ratio.py --epochs=1 --log_dir=$DIR/../logs/vmpo_replay_ratio/1
python3 vmpo_replay_ratio.py --epochs=1 --log_dir=$DIR/../logs/vmpo_replay_ratio/1

python3 vmpo_replay_ratio.py --epochs=2 --log_dir=$DIR/../logs/vmpo_replay_ratio/2
python3 vmpo_replay_ratio.py --epochs=2 --log_dir=$DIR/../logs/vmpo_replay_ratio/2
python3 vmpo_replay_ratio.py --epochs=2 --log_dir=$DIR/../logs/vmpo_replay_ratio/2


python3 vmpo_replay_ratio.py --epochs=3 --log_dir=$DIR/../logs/vmpo_replay_ratio/3
python3 vmpo_replay_ratio.py --epochs=3 --log_dir=$DIR/../logs/vmpo_replay_ratio/3
python3 vmpo_replay_ratio.py --epochs=3 --log_dir=$DIR/../logs/vmpo_replay_ratio/3

python3 vmpo_replay_ratio.py --epochs=4 --log_dir=$DIR/../logs/vmpo_replay_ratio/4
python3 vmpo_replay_ratio.py --epochs=4 --log_dir=$DIR/../logs/vmpo_replay_ratio/4
python3 vmpo_replay_ratio.py --epochs=4 --log_dir=$DIR/../logs/vmpo_replay_ratio/4
