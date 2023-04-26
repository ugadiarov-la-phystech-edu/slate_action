import torch
from torch.utils.data import Dataset
import numpy as np


class EnvDataset(Dataset):
    def __init__(self, env, num_episodes, steps_per_episode, data_path=None):
        self.env = env
        self.num_episodes = num_episodes
        self.steps_per_episode = steps_per_episode
        self.data_path = data_path

        if self.data_path is not None and self.load_data():
            print("Data loaded from files.")
        else:
            self.observations, self.actions = self.collect_data()
            if self.data_path is not None:
                self.save_data()
                print("Data collected and saved.")

    def __len__(self):
        return len(self.observations) * (self.steps_per_episode - 1)

    def __getitem__(self, idx):
        idx_episode = idx // (self.steps_per_episode - 1)
        idx_step = idx % (self.steps_per_episode - 1)

        obs_before = self.observations[idx_episode, idx_step]
        obs_after = self.observations[idx_episode, idx_step + 1]
        action = self.actions[idx_episode, idx_step]
        return obs_before, action, obs_after

    def collect_data(self):
        all_observations = []
        all_actions = []

        for _ in range(self.num_episodes):
            observations = []
            actions = []
            obs = self.env.reset()
            for _ in range(self.steps_per_episode):
                action = self.env.action_space.sample()
                observations.append(obs['image'])
                actions.append(action)
                obs, _, done, _ = self.env.step(action)
                if done:
                    break

            if len(observations) == self.steps_per_episode:
                all_observations.append(observations)
                all_actions.append(actions)

        return (torch.tensor(all_observations, dtype=torch.float32),
                torch.tensor(all_actions, dtype=torch.float32))

    def save_data(self):
        np.save(f"{self.data_path}/observations.npy", self.observations.numpy())
        np.save(f"{self.data_path}/actions.npy", self.actions.numpy())

    def load_data(self):
        try:
            self.observations = torch.tensor(np.load(f"{self.data_path}/observations.npy"), dtype=torch.float32)
            self.actions = torch.tensor(np.load(f"{self.data_path}/actions.npy"), dtype=torch.float32)
            return True
        except FileNotFoundError:
            return False
