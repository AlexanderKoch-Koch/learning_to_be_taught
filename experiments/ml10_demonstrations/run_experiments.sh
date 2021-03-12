#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

echo "script dir is " $DIR

python3 ml10_language_instructions.py --log_dir=$DIR/../logs/ml10/language_instructions
python3 ml10_language_instructions.py --log_dir=$DIR/../logs/ml10/language_instructions
python3 ml10_language_instructions.py --log_dir=$DIR/../logs/ml10/language_instructions

python3 ml10_from_rewards.py --log_dir=$DIR/../logs/ml10/dense_rewards
python3 ml10_from_rewards.py --log_dir=$DIR/../logs/ml10/dense_rewards
python3 ml10_from_rewards.py --log_dir=$DIR/../logs/ml10/dense_rewards

python3 ml10_demonstrations_experiment.py --log_dir=$DIR/../logs/ml10/demonstrations
python3 ml10_demonstrations_experiment.py --log_dir=$DIR/../logs/ml10/demonstrations
python3 ml10_demonstrations_experiment.py --log_dir=$DIR/../logs/ml10/demonstrations
