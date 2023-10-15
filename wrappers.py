import os
from typing import Callable

import cv2
import gym
import numpy as np
from gym.wrappers.monitoring import video_recorder


class CurrentStateWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        observation_space = self.env.observation_space
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=observation_space.shape[1:], dtype=np.uint8)

    def reset(self):
        return super().reset()[0]

    def step(self, action):
        observation, reward, done, info = super().step(action)
        return observation[0], reward, done, info

    def render(self, mode="human", **kwargs):
        current_images = self.env._robot.get_current_camera_observations()[0]
        goal_images = self.env._stage.get_current_goal_image()[0]
        image = np.concatenate((current_images, goal_images), axis=1)
        return image


class ResizeWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, width=96, height=96):
        super().__init__(env)
        self.width = width
        self.height = height
        observation_space = self.env.observation_space
        self.observation_space = gym.spaces.Box(low=np.min(observation_space.low), high=np.max(observation_space.high),
                                                shape=(self.width, self.height, observation_space.shape[2]),
                                                dtype=observation_space.dtype)

    def _resize(self, observation):
        return cv2.resize(observation, dsize=(self.width, self.height), interpolation=cv2.INTER_CUBIC)

    def reset(self):
        observation = super().reset()
        return self._resize(observation)

    def step(self, action):
        observation, reward, done, info = super().step(action)
        return self._resize(observation), reward, done, info


class EpisodeRecorder(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        video_folder: str,
        record_video_trigger: Callable[[int], bool],
        video_length: int = 200,
        name_prefix: str = "rl-video",
    ):

        super().__init__(env)

        self.env = env

        self.record_video_trigger = record_video_trigger
        self.video_recorder = None

        self.video_folder = os.path.abspath(video_folder)
        # Create output folder if needed
        os.makedirs(self.video_folder, exist_ok=True)

        self.name_prefix = name_prefix
        self.episode_id = 0
        self.video_length = video_length

        self.recording = False
        self.recorded_frames = 0

    def reset(self):
        obs = self.env.reset()
        if self._video_enabled():
            self.start_video_recorder()
        return obs

    def start_video_recorder(self) -> None:
        self.close_video_recorder()

        video_name = f"{self.name_prefix}-ep-{self.episode_id}"
        base_path = os.path.join(self.video_folder, video_name)
        self.video_recorder = video_recorder.VideoRecorder(
            env=self.env, base_path=base_path, metadata={"episode_id": self.episode_id}
        )

        self.video_recorder.capture_frame()
        self.recorded_frames = 1
        self.recording = True

    def _video_enabled(self) -> bool:
        return self.record_video_trigger(self.episode_id)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)

        if self.recording:
            self.video_recorder.capture_frame()
            self.recorded_frames += 1
            if done or self.recorded_frames > self.video_length:
                print(f"Saving video to {self.video_recorder.path}")
                self.close_video_recorder()

        if done:
            self.episode_id += 1

        return obs, rew, done, info

    def close_video_recorder(self) -> None:
        if self.recording:
            self.video_recorder.close()
        self.recording = False
        self.recorded_frames = 1

    def close(self) -> None:
        super().close()
        self.close_video_recorder()

    def __del__(self):
        self.close()


class SuccessWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

    def step(self, action):
        observation, reward, done, info = super().step(action)
        success = info['success']
        info['is_success'] = success
        if success:
            done = True

        return observation, reward, done, info
