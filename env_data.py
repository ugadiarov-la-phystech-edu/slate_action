import torch
from torch.utils.data import Dataset
import numpy as np

from utils import create_dirs, to_one_hot


class EnvDataset(Dataset):
    def __init__(self, env, num_episodes, steps_per_episode, warmup_min=0, warmup_max=0, data_path=None, one_hot_actions=False):
        self.env = env
        self.num_episodes = num_episodes
        self.steps_per_episode = steps_per_episode
        self.data_path = data_path
        self.one_hot_actions = one_hot_actions
        assert warmup_min <= warmup_max
        self.warmup_min = warmup_min
        self.warmup_max = warmup_max
        if self.data_path is not None and self.load_data():
            print("Data loaded from files.")
        else:
            print('Collecting Dataset')
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
        return obs_before / 255., action, obs_after / 255.

    def collect_data(self):
        all_observations = []
        all_actions = []
        
        warmup = self.warmup_max
            
        if self.warmup_max != self.warmup_min:
            warmup = np.random.randint(self.warmup_min, self.warmup_max)
                
        if warmup > 0:
            for _ in range(warmup):
                action = self.env.action_space.sample()
                obs, *_ = self.env.step(action)

        for ep in range(self.num_episodes):
            observations = []
            actions = []
            obs = self.env.reset()
            for _ in range(self.steps_per_episode):
                action = self.env.action_space.sample()
                observations.append(obs['image'])
                obs, _, done, _ = self.env.step(action)
                if self.one_hot_actions:
                    action = to_one_hot(action, self.env.action_space.n)
                actions.append(action)
                if done:
                    break
            

            if len(observations) == self.steps_per_episode:
                all_observations.append(observations)
                all_actions.append(actions)
                
                if ep % 10 == 0:
                    print(f'Collected {ep} episodes')

        return (torch.tensor(all_observations, dtype=torch.float32),
                torch.tensor(all_actions, dtype=torch.float32))

    def save_data(self):
        create_dirs(self.data_path)
        np.save(f"{self.data_path}/observations.npy", self.observations.numpy())
        np.save(f"{self.data_path}/actions.npy", self.actions.numpy())

    def load_data(self):
        try:
            self.observations = torch.tensor(np.load(f"{self.data_path}/observations.npy"), dtype=torch.float32)
            self.actions = torch.tensor(np.load(f"{self.data_path}/actions.npy"), dtype=torch.float32)
            return True
        except FileNotFoundError:
            return False


class EnvTestDataset(EnvDataset):
    def __init__(self, env, num_episodes, steps_per_episode, warmup_min=0, warmup_max=0, data_path=None, one_hot_actions=False):
        super().__init__(env, num_episodes, steps_per_episode, warmup_min, warmup_max, data_path=data_path, one_hot_actions=one_hot_actions)
        self.steps_fractions = self.steps_per_episode // (self.steps_per_episode-1)

    def __len__(self):
        return len(self.observations) * self.steps_fractions

    def __getitem__(self, idx):
        idx_episode = idx // self.steps_fractions
        idx_step = idx % self.steps_fractions
        start_step = idx_step * self.steps_per_episode
        end_step = start_step + self.steps_per_episode
        obs = self.observations[idx_episode][start_step:end_step] / 255.
        action = self.actions[idx_episode][start_step:end_step]
        return obs.float(), action