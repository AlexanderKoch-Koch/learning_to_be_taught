import argparse
import gym
import GPUtil
import multiprocessing
import torch
from typing import Dict
from learning_to_be_taught.environments.fixed_length_env_wrapper import FixedLengthEnvWrapper
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.samplers.async_.cpu_sampler import AsyncCpuSampler
from rlpyt.samplers.async_.gpu_sampler import AsyncGpuSampler
from rlpyt.samplers.async_.alternating_sampler import AsyncAlternatingSampler
from rlpyt.runners.async_rl import AsyncRlEval
from rlpyt.utils.launching.affinity import affinity_from_code
from rlpyt.utils.launching.variant import load_variant, update_config
from rlpyt.utils.launching.affinity import make_affinity
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.envs.gym import GymEnvWrapper
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from learning_to_be_taught.traj_info import EnvInfoTrajInfo
from learning_to_be_taught.logger_context import config_logger
from learning_to_be_taught.environments.meta_world.meta_world import MetaWorld
from learning_to_be_taught.environments.classic_control.acrobot import Acrobot
from learning_to_be_taught.environments.env_info_wrapper import EnvInfoWrapper
from rlpyt.envs.gym import TimeLimit
from learning_to_be_taught.vmpo.v_mpo import VMPO
from learning_to_be_taught.vmpo.async_vmpo import AsyncVMPO
from rlpyt.agents.pg.mujoco import MujocoLstmAgent, MujocoFfAgent, AlternatingMujocoLstmAgent
from learning_to_be_taught.vmpo.agent import VmpoAgent
from rlpyt.agents.pg.categorical import CategoricalPgAgent, RecurrentCategoricalPgAgent
from learning_to_be_taught.vmpo.models import FfModel, TransformerModel, FfSharedModel
from learning_to_be_taught.vmpo.lstm_model import LstmModel
from rlpyt.samplers.async_.alternating_sampler import AsyncAlternatingSampler
from rlpyt.algos.pg.ppo import PPO
from learning_to_be_taught.vmpo.gaussian_vmpo_agent import MujocoVmpoAgent, AlternatingVmpoAgent
from learning_to_be_taught.vmpo.categorical_vmpo_agent import CategoricalVmpoAgent
from learning_to_be_taught.vmpo.categorical_models import CategorialFfModel



def build_and_train(slot_affinity_code=None, log_dir='./data', run_ID=0,
                    serial_mode=True,
                    alternating_sampler=False,
                    snapshot: Dict = None,
                    config_update: Dict = None,
                    name='vmpo_replay_ratio',
                    epochs=1):
    # default configuration
    episode_length = 40
    config = dict(
        vmpo_kwargs=dict(epochs=epochs, minibatches=100, discrete_actions=False, pop_art_reward_normalization=True,
                         T_target_steps=100),
        ppo_kwargs=dict(epochs=4, minibatches=1, learning_rate=1e-4),
        async_vmpo_kwargs=dict(epochs=epochs, discrete_actions=False, T_target_steps=100, batch_B=64,
                               batch_T=episode_length, pop_art_reward_normalization=True),
        sampler_kwargs=dict(batch_T=episode_length, batch_B=64 * 100,
                            env_kwargs=dict(id='Ant-v3'),
                            eval_env_kwargs=dict(id='Ant-v3', fixed_episode_length=1000),
                            eval_n_envs=22,
                            eval_max_steps=1e5,
                            eval_max_trajectories=22,
                            TrajInfoCls=EnvInfoTrajInfo),
        # agent_kwargs=dict(ModelCls=CategorialFfModel, model_kwargs=dict(linear_value_output=False,
        #                                                                 seperate_value_network=True)),
        # agent_kwargs=dict(ModelCls=CategoricalFfModel, model_kwargs=dict(observation_shape=(6,), action_size=3)),
        agent_kwargs=dict(ModelCls=FfSharedModel, model_kwargs=dict(linear_value_output=False, layer_norm=False)),
        # agent_kwargs=dict(ModelCls=FfSharedModel, model_kwargs=dict(linear_value_output=False, layer_norm=False)),
        # agent_kwargs=dict(ModelCls=LstmModel, model_kwargs=dict(linear_value_output=False)),
        runner_kwargs=dict(n_steps=4e8, log_interval_steps=5e6),
        snapshot=snapshot,
        # algo='vmpo',
        algo='async_vmpo'
    )

    # update default config if available in log_dir or was provided as parameter
    serial_mode = False
    try:
        variant = load_variant(log_dir)
        config = update_config(config, variant)
    except FileNotFoundError:
        if config_update is not None:
            config = update_config(config, config_update)

    if config['algo'] == 'vmpo':
        AlgoClass = VMPO
        SamplerClass = CpuSampler
        # SamplerClass = SerialSampler
        RunnerClass = MinibatchRlEval
        algo_kwargs = config['vmpo_kwargs']
        _async = False
    elif config['algo'] == 'async_vmpo':
        AlgoClass = AsyncVMPO
        if alternating_sampler:
            SamplerClass = AsyncAlternatingSampler
        else:
            SamplerClass = AsyncCpuSampler
        RunnerClass = AsyncRlEval
        algo_kwargs = config['async_vmpo_kwargs']
        _async = True

    if serial_mode:
        SamplerClass = SerialSampler
        config['sampler_kwargs']['batch_B']= 4
        algo_kwargs['minibatches'] =1


    affinity = choose_affinity(slot_affinity_code, serial_mode, alternating_sampler, _async,
                               config['sampler_kwargs']['batch_B'])
    # continue training from saved state_dict if provided
    agent_state_dict = optimizer_state_dict = None
    if config['snapshot'] is not None:
        agent_state_dict = config['snapshot']['agent_state_dict']
        optimizer_state_dict = config['snapshot']['optimizer_state_dict']

    if alternating_sampler:
        AgentClass = AlternatingVmpoAgent
    else:
        AgentClass = MujocoVmpoAgent

    sampler = SamplerClass(
        **config['sampler_kwargs'],
        EnvCls=make_gym_env,
        max_decorrelation_steps=100,
    )
    algo = AlgoClass(**algo_kwargs, initial_optim_state_dict=optimizer_state_dict)
    agent = AgentClass(initial_model_state_dict=agent_state_dict, **config['agent_kwargs'])
    runner = RunnerClass(
        **config['runner_kwargs'],
        algo=algo,
        agent=agent,
        sampler=sampler,
        affinity=affinity
    )
    config_logger(log_dir, name=name, snapshot_mode='last', log_params=config)
    runner.train()

def make_gym_env(**kwargs):
    info_example = {'timeout': 0}
    if 'fixed_episode_length' in kwargs.keys():
        fixed_episode_length = kwargs['fixed_episode_length']
        kwargs.pop('fixed_episode_length')
    else:
        fixed_episode_length = None
    env = TimeLimit(gym.make(**kwargs)) # use env spec timelimit
    env = FixedLengthEnvWrapper(env, fixed_episode_length=fixed_episode_length)

    return GymEnvWrapper(EnvInfoWrapper(env, info_example))


def make_metaworld_env(**kwargs):
    info_example = {'timeout': 0}
    env = MetaWorld(**kwargs)
    # env = Pendulum(**kwargs)
    # env = EasyReacher(**kwargs)
    # env = CartPole(**kwargs)
    # env = Acrobot(**kwargs)
    # env = gym.make('Pendulum-v0')
    return GymEnvWrapper(EnvInfoWrapper(env, info_example))


def choose_affinity(slot_affinity_code, serial_mode, alternating_sampler, async_mode, sampler_batch_B):
    if slot_affinity_code is None:
        num_cpus = 12 # multiprocessing.cpu_count()
        num_gpus = len(GPUtil.getGPUs())
        if serial_mode:
            affinity = make_affinity(n_cpu_core=num_cpus // 2, n_gpu=0, set_affinity=False)
        elif alternating_sampler:
            affinity = make_affinity(n_cpu_core=num_cpus // 2, n_gpu=num_gpus, async_sample=True, optim_sample_share_gpu=True,
                                     alternating=True, set_affinity=True)
            affinity['sampler'][0]['alternating'] = True
            affinity['sampler'].append(affinity['sampler'][0])
        elif async_mode:
            affinity = make_affinity(n_cpu_core=num_cpus, n_gpu=1, async_sample=True, set_affinity=True)
            affinity['optimizer'][0]['cuda_idx'] = 1
        else:
            affinity = make_affinity(n_cpu_core=num_cpus, cpu_per_run=num_cpus, n_gpu=num_gpus, async_sample=False, set_affinity=False)
            affinity['workers_cpus'] = tuple((x,) for x in range(24))
            affinity['cuda_idx'] = 1

    else:
        affinity = affinity_from_code(slot_affinity_code)
    return affinity


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
    parser.add_argument('slot_affinity_code', nargs='?', default=None,
                        help='using all possible resources when not specified')
    parser.add_argument('log_dir_positional', nargs='?', help='required for automatic launching')
    parser.add_argument('run_id', nargs='?', help='required for automatic launching')
    parser.add_argument('--serial_mode', dest='serial_mode', action='store_true',
                        help='flag to run in serial mode is easier for debugging')
    parser.add_argument('--no_serial_mode', dest='serial_mode', action='store_false',
                        help='flag to run in serial mode is easier for debugging')
    parser.add_argument('--log_dir', required=False,
                        help='path to directory where log folder will be; Overwrites log_dir_positional')
    parser.add_argument('--snapshot_file', help='path to snapshot params.pkl containing state_dicts',
                        default=None)
    parser.add_argument('--name', help='path to snapshot params.pkl containing state_dicts',
                        default='run')
    parser.add_argument('--epochs', help='epochs for vmpo', type=int)

    args = parser.parse_args()
    log_dir = args.log_dir or args.log_dir_positional or './logs'
    print("training started with parameters: " + str(args))
    snapshot = None
    if args.snapshot_file is not None:
        snapshot = torch.load(args.snapshot_file, map_location=torch.device('cpu'))

    config_update = dict()

    # for epochs in range(1, 9):
    build_and_train(slot_affinity_code=args.slot_affinity_code,
                    log_dir=log_dir,
                    run_ID=args.run_id,
                    snapshot=snapshot,
                    config_update=config_update,
                    serial_mode=args.serial_mode,
                    name=args.name,
                    epochs=args.epochs)
