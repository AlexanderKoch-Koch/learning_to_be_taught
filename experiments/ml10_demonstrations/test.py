import argparse
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import gym
import torch
import time
import numpy as np
from rlpyt.envs.gym import GymEnvWrapper, EnvInfoWrapper
from rlpyt.envs.gym_schema import GymEnvWrapper, EnvInfoWrapper
from rlpyt.utils.buffer import torchify_buffer, buffer_from_example, numpify_buffer
from rlpyt.agents.pg.categorical import CategoricalPgAgent
from learning_to_be_taught.vmpo.models import FfModel, CategoricalFfModel, TransformerModel, FfSharedModel
from rlpyt.agents.pg.mujoco import  MujocoLstmAgent, MujocoFfAgent
from learning_to_be_taught.environments.meta_world.meta_world import MetaWorld
from learning_to_be_taught.environments.meta_world.generalized_meta_world import GeneralizedMetaWorld
from learning_to_be_taught.environments.meta_world.language_meta_world import LanguageMetaWorld
from rlpyt.agents.qpg.sac_agent import SacAgent
from learning_to_be_taught.recurrent_sac.efficient_recurrent_sac_agent import EfficientRecurrentSacAgent
from learning_to_be_taught.environments.pendulum import Pendulum
from learning_to_be_taught.recurrent_sac.recurrent_sac_agent import RecurrentSacAgent
from learning_to_be_taught.recurrent_sac.transformer_model import PiTransformerModel, QTransformerModel
from learning_to_be_taught.behavioral_cloning.behavioral_cloning_agent import BehavioralCloningAgent
from rlpyt.agents.pg.mujoco import MujocoLstmAgent, MujocoFfAgent
from learning_to_be_taught.vmpo.gaussian_vmpo_agent import MujocoVmpoAgent
from learning_to_be_taught.vmpo.models import TransformerModel, GeneralizedTransformerModel
from learning_to_be_taught.vmpo.compressive_transformer import CompressiveTransformer
from learning_to_be_taught.vmpo.models import FfModel, CategoricalFfModel
from pathlib import Path
from os import listdir
from make_training_progress_figure import plot_with_std
from make_language_env import make_language_env


def simulate_policy(env, agent, render, num_episodes=10, trials_per_episode=3, oracle_policy=False):
    obs = env.reset()
    observation = buffer_from_example(obs, 1)
    agent.to_device(0)
    loop_time = 0.01
    returns = []
    successes = []
    trial_success = [[0 for _ in range(trials_per_episode)] for _ in range(num_episodes)]
    trial_returns = [[0 for _ in range(trials_per_episode)] for _ in range(num_episodes)]
    trial_steps_to_solve = [[0 for _ in range(trials_per_episode)] for _ in range(num_episodes)]
    for episode in range(num_episodes):
        observation[0] = env.reset()
        action = buffer_from_example(env.action_space.null_value(), 1)
        reward = np.zeros(1, dtype="float32")
        obs_pyt, act_pyt, rew_pyt = torchify_buffer((observation, action, reward))
        agent.reset()
        print(f'env name {env.env_name}')
        done = False
        print(f'env max path length {env.env.env.env.max_path_length}')
        step = 0
        reward_sum = 0
        trial_return = 0
        trial = 0
        steps_until_success = env.env.env.env.max_path_length
        while not done:
            loop_start = time.time()
            step += 1
            if oracle_policy:
                action = env.oracle_policy.get_action(obs.state[:12])
            else:
                act_pyt, agent_info = agent.step(obs_pyt, act_pyt, rew_pyt)
                action = numpify_buffer(act_pyt)[0]
            obs, reward, done, info = env.step(action)
            if env.trial_in_episode > trial:
                trial_return = 0 # next trial
                trial = min(env.trial_in_episode, trials_per_episode - 1)
                steps_until_success = 200
                print(f'next trial; previous steps to solve {trial_steps_to_solve[episode][trial -1]} action_repeat {env.action_repeat}')

            reward_sum += reward
            observation[0] = obs
            rew_pyt[0] = float(reward)
            trial_success[episode][trial] = env.env.env.trial_success
            trial_returns[episode][trial] = trial_return
            
            time_passed = env.env.env.step_in_trial
            steps_until_success = min(time_passed, steps_until_success) if env.env.env.trial_success else steps_until_success
            # print(f'steps to solve {steps_until_success} trial success {env.env.env.trial_success}')
            trial_steps_to_solve[episode][trial] = steps_until_success

            trial_return += reward
            sleep_time = loop_time - (time.time() - loop_start)
            sleep_time = 0 if (sleep_time < 0) else sleep_time
            if render:
                time.sleep(sleep_time)
                env.render(mode='human')

        # if info.demonstration_success > 0:
        successes.append(info.episode_success)
        print('episode success: ' + str(info.episode_success) + 'avg success: ' + str(sum(successes)/ len(successes)))
        returns.append(reward_sum)
        print('avg return: ' + str(sum(returns) / len(returns)) + ' return: ' + str(reward_sum) + '  num_steps: ' + str(
            step))
        # print(f'trial returns {trial_returns}')
        # print(f'trial success {trial_success}')
        # print(f'trial steps to solve {trial_steps_to_solve}')
    return trial_steps_to_solve


def make_envs(language=False, **kwargs):
    info_example = {'timeout': 0}
    language_env = LanguageMetaWorld(benchmark='ml10', action_repeat=2, demonstration_action_repeat=5,
                           max_trials_per_episode=3, **kwargs)
    demonstration_env = GeneralizedMetaWorld(benchmark='ml10', action_repeat=2, demonstration_action_repeat=5,
                           max_trials_per_episode=3, **kwargs)
    demonstration_env = GymEnvWrapper(EnvInfoWrapper(demonstration_env, info_example))
    language_env = GymEnvWrapper(EnvInfoWrapper(language_env, info_example))
    return demonstration_env, language_env

def make_figure(dense_rewards_returns, demonstrations_returns, language_returns, oracle_returns, name):
    # returns in shape num_episodes x num_trials_per_episode
    x = [1, 2, 3]
    with PdfPages(r'./' + name) as export_pdf:
        dense_rewards_returns = np.array(dense_rewards_returns)
        demonstrations_returns = np.array(demonstrations_returns)
        oracle_returns = np.array(oracle_returns)
        plot_with_std(x, np.mean(dense_rewards_returns, axis=1), 'Meta-RL with Dense Rewards', 'blue')
        plot_with_std(x, np.mean(demonstrations_returns, axis=1), 'Meta-RL with Demonstrations', 'red')
        plot_with_std(x, np.mean(language_returns, axis=1), 'Meta-RL with Language Instructions', 'green')
        plt.plot(x, np.mean(oracle_returns).repeat(3), label='Demonstration', color='orange')
        plt.legend()
        plt.xticks([1, 2, 3])
        plt.ylim(bottom=30)
        plt.grid(True)
        plt.xlabel('Trial', fontsize=14)
        plt.ylabel('Time Steps until Solved', fontsize=14)

        # plt.show()
        export_pdf.savefig()
        # plt.close()
    
def evaluate_runs_in_dir(agent, path, env, num_episodes=200):
    log_dirs = listdir(path)
    results = []
    for log_dir in log_dirs:
        run_path = path / log_dir
        print(f'evaluating new agent############################################################')
        print(f'loading agent from {run_path}')
        agent_state_dict = torch.load(run_path / 'params.pkl', map_location='cpu')['agent_state_dict']
        agent.load_state_dict(agent_state_dict)
        agent.eval_mode(1)
        results.append(simulate_policy(env, agent, render=False, num_episodes=num_episodes))

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', help='path to params.pkl',
                        default='/home/alex/learning_to_be_taught/logs/run_1/params.pkl')
    parser.add_argument('--mode', help='either "meta_testing" or "meta_training"',
                        default='meta_training')
    args = parser.parse_args()
    env, language_env = make_envs(mode=args.mode, sample_num_classes=50)

    agent_rewards = MujocoVmpoAgent(ModelCls=CompressiveTransformer, model_kwargs=dict(linear_value_output=False,
                                                                               size='medium', sequence_length=64,
                                                                               seperate_value_network=False))

    agent_demonstrations = MujocoVmpoAgent(ModelCls=CompressiveTransformer, model_kwargs=dict(linear_value_output=False,
                                                                                       size='medium', sequence_length=75,
                                                                                       seperate_value_network=False))
    agent_rewards.initialize(env_spaces=env.spaces)
    agent_demonstrations.initialize(env_spaces=env.spaces)


    path = Path(__file__).parent / '..' / 'logs' / 'ml10'
    env.env.env.demonstrations = True
    env.env.env.dense_rewards = False
    demonstrations_results = evaluate_runs_in_dir(agent_demonstrations, path / 'demonstrations', env)
    env.env.env.demonstrations = False
    env.env.env.dense_rewards = True
    dense_rewards_results = evaluate_runs_in_dir(agent_rewards, path / 'dense_rewards', env)
    

    env.env.env.partially_observable = False
    env.env.env.action_repeat = 1
    env.action_repeat = 1
    oracle_returns = simulate_policy(env, agent_demonstrations, render=False, oracle_policy=True, num_episodes=100)
    
    agent_language = MujocoVmpoAgent(ModelCls=CompressiveTransformer, model_kwargs=dict(linear_value_output=False,
                                                                               size='medium', sequence_length=64,
                                                                               seperate_value_network=False))
    agent_language.initialize(env_spaces=language_env.spaces)
    language_results = evaluate_runs_in_dir(agent_language, path / 'language_instructions', language_env, num_episodes=200)
    make_figure(dense_rewards_results, demonstrations_results, language_results, oracle_returns, args.mode + '.pdf')
    # make_figure([[1, 1, 1]], [[1, 1, 1]], oracle_returns, 'debug')

