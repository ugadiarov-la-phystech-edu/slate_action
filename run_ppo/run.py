import argparse

from gym.wrappers import TimeLimit

import wandb
from causal_world.envs import CausalWorld
from causal_world.task_generators import PushingTaskGenerator, StackedBlocksGeneratorTask
from omegaconf import OmegaConf
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage, DummyVecEnv
from wandb.integration.sb3 import WandbCallback

from callbacks import CustomEvalCallback
from wrappers import CurrentStateWrapper, ResizeWrapper, EpisodeRecorder, SuccessWrapper

task2class = {'push':  PushingTaskGenerator, 'stack': StackedBlocksGeneratorTask}


def make_env(config, rank):
    def _init():
        seed = config['seed'] + rank
        timelimit = config['timelimit']
        task = task2class[config['task']](variables_space='space_a')
        env = CausalWorld(
            task=task,
            observation_mode='pixel',
            camera_indicies=[0],
            normalize_observations=False,
            max_episode_length=timelimit,
            seed=seed,
        )
        if timelimit is None:
            timelimit = env._max_episode_length

        env.action_space.seed(seed)
        env = CurrentStateWrapper(env)
        env = ResizeWrapper(env, width=config['width'], height=config['height'])
        env = SuccessWrapper(env)
        env = TimeLimit(env, timelimit)
        env = Monitor(env)

        return env

    return _init


def make_eval_env(config, rank, video_folder, do_record_video):
    def init():
        env = make_env(config, rank)()
        return EpisodeRecorder(env, video_folder, record_video_trigger=lambda x: do_record_video)

    return init


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True)
    args = parser.parse_args()
    config = OmegaConf.load(args.config_path)

    env = SubprocVecEnv([make_env(config, i) for i in range(config['n_envs'])])
    run = wandb.init(
        project=config['project'],
        config=OmegaConf.to_container(config, resolve=True),
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
        name=f'run-{config["seed"]}',
    )

    model = PPO(
        policy='CnnPolicy',
        env=env,
        learning_rate=config['learning_rate'], #3e-4,
        n_steps=config['n_steps'], #2048,
        batch_size=config['batch_size'], #64,
        n_epochs=config['n_epochs'], #10,
        gamma=config['gamma'], #0.99,
        gae_lambda=config['gae_lambda'], #0.95,
        clip_range=config['clip_range'], #0.2,
        clip_range_vf=config['clip_range_vf'], #None,
        normalize_advantage=config['normalize_advantage'], #True,
        ent_coef=config['ent_coef'], #0.0,
        vf_coef=config['vf_coef'], #0.5,
        max_grad_norm=config['max_grad_norm'], #0.5,
        use_sde=config['use_sde'], #False,
        sde_sample_freq=config['sde_sample_freq'], #-1,
        target_kl=config['target_kl'], #None,
        tensorboard_log=wandb.run.dir,
        create_eval_env=False,
        policy_kwargs=config['policy_kwargs'], #None,
        verbose=config['verbose'], #0,
        seed=config['seed'], #None,
        device=config['device'], #"auto",
    )

    eval_env = VecTransposeImage(SubprocVecEnv([make_eval_env(
        config,
        config['n_envs'] + i,
        video_folder=f"{wandb.run.dir}/videos/",
        do_record_video=i == 0
    ) for i in range(config['n_eval_episodes'])]))

    model.learn(
        total_timesteps=config['total_timesteps'],
        callback=[
            WandbCallback(
                gradient_save_freq=0,
                verbose=2,
            ),
            CustomEvalCallback(
                eval_env,
                eval_freq=config['eval_freq'],
                n_eval_episodes=config['n_eval_episodes'],
                best_model_save_path=f"{wandb.run.dir}/models/",
                log_path=f"{wandb.run.dir}/eval_logs/",
                deterministic=False,
            ),
        ]
    )
    run.finish()
