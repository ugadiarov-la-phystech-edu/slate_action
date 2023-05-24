import gym
import gym.wrappers
import numpy as np
import cv2


class AtariEnv(gym.Env):
    def __init__(self, env_id, image_size=96, frame_skip=None, **kwargs):
        self.env_id = env_id
        self.env = gym.make(env_id, frameskip=8)
        self.crop = self._init_crop()
        self.image_size = image_size
        c,h,w = self.reset()['image'].shape
        self.observation_space = gym.spaces.MultiBinary((c,h,w))
        
    
    def _init_crop(self):
        env_id = self.env_id.lower()
        if 'pong' in env_id:
            return (35, 190)
        elif 'spaceinvaders' in env_id:
            return (30, 200)
    
    def _process_image(self, image):
        if self.crop is not None:
            image = image[self.crop[0]:self.crop[1]]
        image = cv2.resize(image, dsize=(self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
            
        return image
        
        
    @property
    def action_space(self):
        return self.env.action_space

    def reset(self):
        image = self.env.reset()
        image = self._process_image(image)
        obs = {"image": image.transpose(2, 0, 1)}
        return obs
    
    def step(self, action):
        image, reward, done, info = self.env.step(action)
        image = self._process_image(image)
        obs = {"image": image.transpose(2, 0, 1)}
        return obs, reward, done, info
    