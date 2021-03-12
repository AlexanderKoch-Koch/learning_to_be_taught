import argparse
import torch
import time
import numpy as np
from rlpyt.envs.gym import GymEnvWrapper, EnvInfoWrapper
from rlpyt.utils.buffer import torchify_buffer, buffer_from_example, numpify_buffer
from learning_to_be_taught.environments.meta_world.language_meta_world import LanguageMetaWorld
from learning_to_be_taught.vmpo.gaussian_vmpo_agent import MujocoVmpoAgent
from learning_to_be_taught.vmpo.compressive_transformer import CompressiveTransformer
from torchtext.vocab import GloVe
from collections import namedtuple

EnvStep = namedtuple("EnvStep",
                     ["observation", "reward", "done", "env_info"])



def interactive_language_control(env, agent):
    embed_dim = LanguageMetaWorld.EMBED_DIM
    word_embedding = GloVe(name='6B', dim=embed_dim)
    while True:
        command = input('enter instruction for robot')
        if command in ['exit', 'quit', 'q']:
            break

        print(f'instruction {command}')
        instruction = []
        for word in command.split():
            instruction.append(word_embedding.get_vecs_by_tokens(word) if word != 'goal_pos'
                               else torch.cat((torch.tensor(env.env.env.env._get_pos_goal()), torch.zeros(embed_dim - 3))))

        print(instruction)
        # new_instruction  = instruction
        simulate_episode(env, agent, command)

def simulate_episode(env, agent, instruction):
    # agent.to_device(1)
    obs = env.reset()
    print(f'setting env isntruction to {instruction}')
    observation = buffer_from_example(obs, 1)
    loop_time = 0.01
    instruction_index = 0
    observation[0] = env.reset()
    env.env.env.instruction = instruction.split()
    action = buffer_from_example(env.action_space.null_value(), 1)
    reward = np.zeros(1, dtype="float32")
    # if instruction_index < len(instruction):
        # observation.state[0, :] = instruction[instruction_index]
        # instruction_index += 1
    obs_pyt, act_pyt, rew_pyt = torchify_buffer((observation, action, reward))
    agent.reset()
    done = False
    step = 0
    reward_sum = 0
    while not done:
        loop_start = time.time()
        step += 1
        # print(obs_pyt)
        act_pyt, agent_info = agent.step(obs_pyt, act_pyt, rew_pyt)
        action = numpify_buffer(act_pyt)[0]
        obs, reward, done, info = env.step(action)
        reward_sum += reward
        observation[0] = obs
        rew_pyt[0] = float(reward)

        sleep_time = loop_time - (time.time() - loop_start)
        sleep_time = 0 if (sleep_time < 0) else sleep_time
        time.sleep(sleep_time)
        env.render(mode='human')

def make_env(**kwargs):
    info_example = {'timeout': 0}
    env = LanguageMetaWorld(benchmark='reach-v1', action_repeat=2, mode='meta_training', max_trials_per_episode=1,
                            sample_num_classes=1)
    env = GymEnvWrapper(EnvInfoWrapper(env, info_example))
    return env


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', help='path to params.pkl',
                        default='/home/alex/learning_to_be_taught/logs/ml45_language_small_obs_new/params.pkl')
    args = parser.parse_args()

    snapshot = torch.load(args.path, map_location=torch.device('cpu'))
    agent_state_dict = snapshot['agent_state_dict']
    env = make_env()
    agent = MujocoVmpoAgent(ModelCls=CompressiveTransformer, model_kwargs=dict(linear_value_output=False,
                                                                               size='medium', sequence_length=64,
                                                                               observation_normalization=False))
    agent.initialize(env_spaces=env.spaces)
    agent.load_state_dict(agent_state_dict)
    agent.eval_mode(1)
    interactive_language_control(env, agent)
