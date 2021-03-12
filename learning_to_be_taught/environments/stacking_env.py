from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig, CameraConfig
from pyrep.const import RenderMode
from rlbench.tasks import ReachTarget, PutMoneyInSafe, TakeToiletRollOffStand, InsertUsbInComputer, PourFromCupToCup, EmptyDishwasher, StackBlocks
import numpy as np


# To use 'saved' demos, set the path below, and set live_demos=False
live_demos = True
DATASET = '' if live_demos else 'PATH/TO/YOUR/DATASET'

# obs_config.set_all(True)

action_mode = ActionMode(ArmActionMode.ABS_JOINT_VELOCITY)
observation_config = ObservationConfig(left_shoulder_camera=CameraConfig(rgb=False, depth=False, mask=False),
                                       right_shoulder_camera=CameraConfig(rgb=False, depth=False, mask=False),
                                       front_camera=CameraConfig(rgb=True, depth=False, mask=False,
                                                                 image_size=(64, 64), render_mode=RenderMode.OPENGL),
                                       wrist_camera=CameraConfig(rgb=False, depth=False, mask=False))
env = Environment(
    action_mode, DATASET, observation_config, False)
env.launch()

task = env.get_task(StackBlocks)

while True:
    demo = task.get_demos(1, live_demos=live_demos)[0]  # -> List[List[Observation]]
    task.reset_to_demo(demo)
    done = False
    step = 0
    while not done:
        # action = np.random.normal(size=env.action_size)
        action = demo._observations[step].joint_velocities
        print('gripper open: ' + str(demo._observations[step].gripper_open))
        action= list(action) + [demo._observations[step].gripper_open]
        obs, reward, done = task.step(action)
        print('reward: '+ str(reward))
        step += 1

# # An example of using the demos to 'train' using behaviour cloning loss.
# for i in range(100):
#     print("'training' iteration %d" % i)
#     batch = np.random.choice(demos, replace=False)
#     batch_images = [obs.left_shoulder_rgb for obs in batch]
#     predicted_actions = il.predict_action(batch_images)
#     ground_truth_actions = [obs.joint_velocities for obs in batch]
#     loss = il.behaviour_cloning_loss(ground_truth_actions, predicted_actions)

print('Done')
env.shutdown()
