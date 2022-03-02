# Learning to be Taught
[Thesis PDF](thesis_final.pdf)

## Abstract
Deep reinforcement learning (RL) has recently been very successful at learning complex behaviors.
But it requires a lot of interactions with the environment to learn a new task.
This makes it difficult to apply it to real-world robotic tasks. Meta-reinforcement learning (meta-RL)
aims to learn an efficient RL algorithm that can quickly learn anew  task. 
However,  in  meta-RL  the  new  task  is  communicated  to  the  agent  only with rewards.
We think it might be advantageous to give the agent some additional information about the task. 
In this thesis, we show that meta-RL can be improved by inserting task instructions directly into the observations
of the agent. We evaluate our algorithm on the challenging Meta-World benchmarks for robotic manipulation tasks.
We demonstrate similar results for two different kinds of task instructions:
task demonstrations and language instructions.

## Requirements
1. Python 3.8.5 (older versions might work too)
2. rlpyt (see instructions at https://github.com/astooke/rlpyt)
3. Meta-World (https://github.com/rlworkgroup/metaworld)

## Installation
```bash
git clone https://github.com/AlexanderKoch-Koch/learning_to_be_taught
cd learning_to_be_taught
pip install -e . 
```
In order to use the Meta-World environments with language instructions the word vector representations have to be saved first.
This is done by executing learning_to_be_taught/environments/meta_world/save_used_word_embeddings.py.

## Experiments

### Meta-World ML10 training
This experiment is in the experiments/ml10_demonstrations directory.
run_experiments.sh starts all the training runs serially. To optimize the training for different hardware the files
ml10_language_instructions.py, ml10_from_rewards, ml10_demonstrations_experiment have to be modified.
The default setting is to run the sampler and learner asynchronously which only works on machines with a GPU.
The training progress figure can be obtained with make_training_progress_figure.py.
The ablations can be done by executing ml10_demonstrations_experiment.py after modifying the environment at
learning_to_be_taught/environments/meta_world/generalized_meta_world.py. For the Pop-Art ablation only the 
ml10_demonstrations_experiment.py has to be modified by adding pop_art_reward_normalization=False to the async_vmpo_kwargs.

### Meta-World ML45 training
The code for this experiment is in experiments/ml45. ml45_demonstrations_training.py and ml45_language_training.py
start the corresponding training runs. make_figure.py creates the training progress figure. The directories in the script might have to be changed. The trained policies are also avaible in experiments/logs/ml45.

### Sample efficient V-MPO experiment
The scripts for this experiment are in the experiments/vmpo_replay_ratio folder.
The run_experiment.sh script starts all the training runs for this experiment in series.
It is currently optimized to run on a machine with two GPUs and a 12 core CPU with 24 hardware threads.
In order to run it on a different hardware configuration the vmpo_replay_ratio.py file
might have to be changed. The result figure can be produced with make_figure.py.
