import argparse
import gym
import GPUtil
import multiprocessing
import torch
from typing import Dict
from rlpyt.samplers.async_.cpu_sampler import AsyncCpuSampler
from rlpyt.samplers.async_.gpu_sampler import AsyncGpuSampler
from rlpyt.runners.async_rl import AsyncRlEval
from rlpyt.utils.launching.affinity import affinity_from_code
from rlpyt.utils.launching.variant import load_variant, update_config
from learning_to_be_taught.recurrent_dqn.dqn_agent_env_spaces import DqnAgentEnvSpaces
from rlpyt.utils.launching.affinity import make_affinity
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.envs.gym import GymEnvWrapper
from rlpyt.replays.non_sequence.uniform import AsyncUniformReplayBuffer
from rlpyt.algos.dqn.dqn import DQN
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from traj_info import EnvInfoTrajInfo
from logger_context import config_logger
from rlpyt.algos.qpg.sac import SAC
from rlpyt.agents.qpg.sac_agent import SacAgent
from learning_to_be_taught.recurrent_dqn.meta_imitation_agent import MetaImitationAgent
from learning_to_be_taught.recurrent_sac.recurrent_sac import RecurrentSac
from learning_to_be_taught.recurrent_sac.recurrent_sac_agent import RecurrentSacAgent, AlternatingRecurrentSacAgent
from learning_to_be_taught.behavioral_cloning.behavioral_cloning_algo import BehavioralCloning
from learning_to_be_taught.behavioral_cloning.behavioral_cloning_agent import BehavioralCloningAgent
from learning_to_be_taught.models.models import DemonstrationQModel
from learning_to_be_taught.recurrent_dqn.recurrent_dqn_algo import RecurrentDqn
from learning_to_be_taught.environments.meta_world.meta_world import MetaWorld
from learning_to_be_taught.environments.pendulum import Pendulum
from learning_to_be_taught.environments.meta_world.easy_reacher import EasyReacher
from learning_to_be_taught.environments.classic_control.cart_pole_continuous import CartPole
from learning_to_be_taught.environments.classic_control.acrobot import Acrobot
from learning_to_be_taught.environments.env_info_wrapper import EnvInfoWrapper
from learning_to_be_taught.recurrent_sac.transformer_model import PiTransformerModel, QTransformerModel
from learning_to_be_taught.models.transformer_models import DemonstrationTransformerModel
from learning_to_be_taught.models.recurrent_models import FakeRecurrentDemonstrationQModel
from learning_to_be_taught.recurrent_sac.models import FakeRecurrentPiModel, FakeRecurrentQModel
from learning_to_be_taught.recurrent_sac.lstm_model import PiLstmDemonstrationModel, QLstmDemonstrationModel
from learning_to_be_taught.recurrent_sac.efficient_recurrent_sac import EfficientRecurrentSac
from learning_to_be_taught.recurrent_sac.efficient_recurrent_sac_agent import EfficientRecurrentSacAgent
from rlpyt.algos.pg.ppo import PPO
from rlpyt.agents.pg.mujoco import MujocoLstmAgent
from learning_to_be_taught.models.ppo_transformer_model import PpoTransformerModel
from rlpyt.samplers.parallel.gpu.alternating_sampler import AlternatingSampler
from rlpyt.samplers.async_.alternating_sampler import AsyncAlternatingSampler
from rlpyt.models.qpg.mlp import PiMlpModel, QofMuMlpModel


def build_and_train(slot_affinity_code=None, log_dir='./data', run_ID=0,
                    serial_mode=True,
                    alternating_sampler=False,
                    snapshot: Dict = None,
                    config_update: Dict = None,
                    name='run'):
    # default configuration
    episode_length = 50
    config = dict(
        dqn_kwargs=dict(learning_rate=2.5e-4, double_dqn=True, ReplayBufferCls=AsyncUniformReplayBuffer,
                        delta_clip=None, batch_size=64, min_steps_learn=1000, replay_ratio=50),
        recurrent_dqn_kwargs=dict(learning_rate=1e-4, store_rnn_state_interval=149, batch_T=149, batch_B=8,
                                  replay_ratio=20, loss_type='bc'),
        recurrent_sac_kwargs=dict(warmup_T=0, bootstrap_timelimit=False, batch_T=episode_length, batch_size=256,
                                  replay_ratio=2, store_rnn_state_interval=episode_length, min_steps_learn=int(1e4),
                                  max_learning_rate=3e-4, mixed_precision=False),
        sac_kwargs=dict(bootstrap_timelimit=False, replay_ratio=100),
        ppo_kwargs=dict(),
        behavioral_cloning_kwargs=dict(warmup_T=10, batch_T=20, store_rnn_state_interval=10, min_steps_learn=int(1e5),
                                       batch_B=500),
        sampler_kwargs=dict(batch_T=1, batch_B=10,
                            env_kwargs=dict(demonstrations_flag=True, benchmark='ml1',
                                            action_repeat=150 // episode_length),
                            eval_n_envs=4,
                            eval_max_steps=1e5,
                            eval_max_trajectories=4,
                            TrajInfoCls=EnvInfoTrajInfo),
        dqn_agent_kwargs=dict(ModelCls=DemonstrationQModel, eps_eval=0),
        recurrent_dqn_agent_kwargs=dict(ModelCls=DemonstrationTransformerModel),
        recurrent_sac_agent_kwargs=dict(ModelCls=PiTransformerModel, QModelCls=QTransformerModel,
                                        model_kwargs=dict(size='tiny'),
                                        q_model_kwargs=dict(size='tiny', state_action_input=True)),
        # recurrent_sac_agent_kwargs=dict(ModelCls=FakeRecurrentPiModel, QModelCls=FakeRecurrentQModel),
        efficient_recurrent_sac_agent_kwargs=dict(),
        behavioral_cloning_agent_kwargs=dict(ModelCls=FakeRecurrentDemonstrationQModel),
        sac_agent_kwargs=dict(),
        ppo_agent_kwargs=dict(ModelCls=PpoTransformerModel),
        runner_kwargs=dict(n_steps=1e9, log_interval_steps=3e4),
        snapshot=snapshot,
        algo='recurrent_sac'
    )
    # added alternative  pi loss in sac rlpyt

    # update default config if available in log_dir or was provided as parameter
    try:
        variant = load_variant(log_dir)
        config = update_config(config, variant)
    except FileNotFoundError:
        if config_update is not None:
            config = update_config(config, config_update)

    affinity = choose_affinity(slot_affinity_code, serial_mode, alternating_sampler,
                               config['sampler_kwargs']['batch_B'])
    # continue training from saved state_dict if provided
    agent_state_dict = optimizer_state_dict = None
    if config['snapshot'] is not None:
        agent_state_dict = config['snapshot']['agent_state_dict']
        optimizer_state_dict = config['snapshot']['optimizer_state_dict']

    if config['algo'] == 'dqn':
        AgentClass = DqnAgentEnvSpaces
        AlgoClass = DQN
        algo_kwargs = config['dqn_kwargs']
        agent_kwargs = config['dqn_agent_kwargs']
    elif config['algo'] == 'recurrent_dqn':
        AgentClass = MetaImitationAgent
        AlgoClass = RecurrentDqn
        algo_kwargs = config['recurrent_dqn_kwargs']
        agent_kwargs = config['recurrent_dqn_agent_kwargs']
    elif config['algo'] == 'recurrent_sac':
        if alternating_sampler:
            AgentClass = AlternatingRecurrentSacAgent
        else:
            AgentClass = RecurrentSacAgent
        AlgoClass = RecurrentSac
        algo_kwargs = config['recurrent_sac_kwargs']
        agent_kwargs = config['recurrent_sac_agent_kwargs']
    elif config['algo'] == 'efficient_recurrent_sac':
        AgentClass = EfficientRecurrentSacAgent
        AlgoClass = EfficientRecurrentSac
        algo_kwargs = config['recurrent_sac_kwargs']
        agent_kwargs = config['efficient_recurrent_sac_agent_kwargs']
    elif config['algo'] == 'sac':
        AgentClass = SacAgent
        AlgoClass = SAC
        algo_kwargs = config['sac_kwargs']
        agent_kwargs = config['sac_agent_kwargs']
    elif config['algo'] == 'behavioral_cloning':
        AgentClass = BehavioralCloningAgent
        AlgoClass = BehavioralCloning
        algo_kwargs = config['behavioral_cloning_kwargs']
        agent_kwargs = config['behavioral_cloning_agent_kwargs']
    elif config['algo'] == 'ppo':
        AgentClass = MujocoLstmAgent
        # AgentClass = PpoTransformerModel
        AlgoClass = PPO
        algo_kwargs = config['ppo_kwargs']
        agent_kwargs = config['ppo_agent_kwargs']

    if serial_mode:
        SamplerClass = SerialSampler
        RunnerClass = MinibatchRlEval
    elif alternating_sampler:
        SamplerClass = AsyncAlternatingSampler
        RunnerClass = AsyncRlEval
    else:
        SamplerClass = AsyncCpuSampler
        RunnerClass = AsyncRlEval
        affinity['cuda_idx'] = 0

    # make debugging easier in serial mode
    if serial_mode:
        config['runner_kwargs']['log_interval_steps'] = 3e2
        # algo_kwargs['min_steps_learn'] = 300
        # config['sampler_kwargs']['batch_B'] = 1
        config['sampler_kwargs']['eval_n_envs'] = 1
        # algo_kwargs['batch_size'] = 4
    config['sampler_kwargs']['max_decorrelation_steps'] = 0

    sampler = SamplerClass(
        **config['sampler_kwargs'],
        EnvCls=make_env,
        eval_env_kwargs=config['sampler_kwargs']['env_kwargs']
    )
    algo = AlgoClass(**algo_kwargs, initial_optim_state_dict=optimizer_state_dict)
    agent = AgentClass(initial_model_state_dict=agent_state_dict, **agent_kwargs)
    runner = RunnerClass(
        **config['runner_kwargs'],
        algo=algo,
        agent=agent,
        sampler=sampler,
        affinity=affinity
    )
    config_logger(log_dir, name=name, snapshot_mode='last', log_params=config)
    # start training
    runner.train()


def make_env(**kwargs):
    info_example = {'timeout': 0}
    env = MetaWorld(**kwargs)
    # env = Pendulum(**kwargs)
    # env = EasyReacher(**kwargs)
    # env = CartPole(**kwargs)
    # env = Acrobot(**kwargs)
    # env = gym.make('Pendulum-v0')
    return GymEnvWrapper(EnvInfoWrapper(env, info_example))


def choose_affinity(slot_affinity_code, serial_mode, alternating_sampler, sampler_batch_B):
    if slot_affinity_code is None:
        num_cpus = multiprocessing.cpu_count()  # divide by two due to hyperthreading
        num_gpus = len(GPUtil.getGPUs())
        if serial_mode:
            affinity = make_affinity(n_cpu_core=num_cpus // 2, n_gpu=0, set_affinity=False)
        elif alternating_sampler:
            # affinity = dict(n_gpu=num_gpus, workers_cpus=list(range(sampler_batch_B // 2)))
            affinity = make_affinity(n_cpu_core=sampler_batch_B // 2, n_gpu=num_gpus, async_sample=True,
                                     alternating=True, set_affinity=True)
            # affinity['worker_cpus'] = list(range(sampler_batch_B // 2))
            # affinity["workers_cpus"] += affinity["workers_cpus"]  # (Double list)
            # affinity["alternating"] = True  # Sampler will check for this.
            # affinity['async_sample'] = True
        else:
            # affinity = make_affinity(n_cpu_core=num_cpus, n_gpu=num_gpus, async_sample=True, set_affinity=False, optim_sample_share_gpu=True)
            affinity = make_affinity(n_cpu_core=num_cpus, n_gpu=num_gpus, async_sample=True, set_affinity=False)
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

    args = parser.parse_args()
    log_dir = args.log_dir or args.log_dir_positional or './logs'
    print("training started with parameters: " + str(args))
    snapshot = None
    if args.snapshot_file is not None:
        snapshot = torch.load(args.snapshot_file, map_location=torch.device('cpu'))

    config_update = None  # dict(sampler_kwargs=dict(env_kwargs=dict(side_length=4)))

    build_and_train(slot_affinity_code=args.slot_affinity_code,
                    log_dir=log_dir,
                    run_ID=args.run_id,
                    snapshot=snapshot,
                    config_update=config_update,
                    serial_mode=args.serial_mode)
