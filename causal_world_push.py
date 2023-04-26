import gym
import gym.wrappers
import numpy as np
from causal_world.envs import CausalWorld
from causal_world.task_generators import generate_task, PushingTaskGenerator
import cv2


class CausalWorldPush(gym.Env):
    def __init__(self, image_size=96, variables_space='space_a'):
        task = PushingTaskGenerator(variables_space=variables_space)
        self.env = CausalWorld(
            task=task,
            observation_mode='pixel',
            camera_indicies=[0],
            normalize_observations=False
        )
        self.image_size = image_size
        c,h,w = self.reset()['image'].shape
        self.observation_space = gym.spaces.MultiBinary((c,h,w))
        
    @property
    def action_space(self):
        return self.env.action_space

    def reset(self):
        image = self.env.reset()[0]
        image = cv2.resize(image, dsize=(self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
        obs = {"image": image.transpose(2, 0, 1)}
        return obs
    
    def step(self, action):
        image, reward, done, info = self.env.step(action)
        image = cv2.resize(image[0], dsize=(self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
        obs = {"image": image.transpose(2, 0, 1)}
        return obs, reward, done, info
    