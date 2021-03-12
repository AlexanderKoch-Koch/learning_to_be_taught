import numpy as np
import time
from scipy.spatial.distance import cdist
from gym import utils
from PIL import Image
import gc
import glob
import os
from natsort import natsorted
from gym.envs.mujoco import mujoco_env
import gym



if __name__ == '__main__':
    env = gym.make('ReacherMILTest-v1')
    # env = ReacherMILEnv()
    while True:
        done = False
        env.env.next()
        obs = env.reset()
        while not done:
            action = env.action_space.sample()
            action = [-0.01, -0.1]
            obs, reward, done, _ = env.step(action)
            print('reward: ' + str(reward))
            time.sleep(0.2)
            env.render()
