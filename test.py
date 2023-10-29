from collections import defaultdict
import os.path
import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from datetime import datetime

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from causal_world_push import CausalWorldPush
from compas import COMPAS
from utils import get_env, slots_distance_matrix

from env_data import EnvTestDataset

parser = argparse.ArgumentParser()

parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=64)

parser.add_argument('--clip', type=float, default=1.0)
parser.add_argument('--image_size', type=int, default=96)

parser.add_argument('--action_size', type=int, default=9)
parser.add_argument('--num_episodes', type=int, default=1000)
parser.add_argument('--steps_per_episode', type=int, default=101)
parser.add_argument('--one_hot_actions', action='store_true')

parser.add_argument('--checkpoint_path', default='checkpoint.pt.tar')
parser.add_argument('--log_path', default='logs')
parser.add_argument('--data_path', default=None)
parser.add_argument('--env_name', default='cw_push')
parser.add_argument('--name', default='default')

parser.add_argument('--num_dec_blocks', type=int, default=6)
parser.add_argument('--vocab_size', type=int, default=512)
parser.add_argument('--d_model', type=int, default=192)
parser.add_argument('--num_heads', type=int, default=4)
parser.add_argument('--dropout', type=float, default=0.1)

parser.add_argument('--num_iterations', type=int, default=3)
parser.add_argument('--num_slots', type=int, default=3)
parser.add_argument('--num_slot_heads', type=int, default=1)
parser.add_argument('--slot_size', type=int, default=192)
parser.add_argument('--mlp_hidden_size', type=int, default=192)
parser.add_argument('--img_channels', type=int, default=3)
parser.add_argument('--pos_channels', type=int, default=4)

parser.add_argument('--weights', required=True)
parser.add_argument('--test_steps', default='1,5,10,50')
parser.add_argument('--max_steps_per_batch', default=20, type=int)
parser.add_argument('--max_batches', default=20, type=int)

args = parser.parse_args()

# torch.manual_seed(args.seed)
#
# arg_str_list = ['{}={}'.format(k, v) for k, v in vars(args).items()]
# arg_str = '__'.join(arg_str_list)
# log_dir = os.path.join(args.log_path, datetime.today().isoformat())
# writer = SummaryWriter(log_dir)
# writer.add_text('hparams', arg_str)
#
# loader_kwargs = {
#     'batch_size': 1,
#     'shuffle': True,
#     'num_workers': args.num_workers,
#     'pin_memory': True,
#     'drop_last': True,
# }
#
# env = get_env(args.env_name, dict(image_size=args.image_size))

model = COMPAS(args)
state = torch.load(args.weights, map_location='cpu')
model.load_state_dict(state)
model.cuda()
model.zero_action.cuda()
model.eval()

env = CausalWorldPush()
obs = env.env.reset()[0]
obs = cv2.resize(obs, dsize=(96, 96), interpolation=cv2.INTER_CUBIC).transpose(2, 0, 1)
obs_torch = torch.as_tensor(obs, dtype=torch.float32, device='cuda') / 255.
obs_torch = obs_torch.unsqueeze(0)

reconstruction, attn = model.reconstruct_autoregressive(obs_torch, action=None, eval=True)
obs = obs.transpose(1, 2, 0)

reconstruction = reconstruction[0].detach().cpu().numpy()
reconstruction = reconstruction.transpose(1, 2, 0)

attn = attn[0].detach().cpu().numpy()
attn = attn.transpose(0, 2, 3, 1)


fig, axes = plt.subplots(nrows=1, ncols=8, figsize=(4, 4))
for i in range(8):
    axes[i].axis('off')
    if i == 0:
        axes[0].imshow(obs)
    elif i == 1:
        axes[1].imshow(reconstruction)
    else:
        axes[i].imshow(attn[i - 2])

plt.show()


test_steps = map(int, args.test_steps.split(','))
test_sampler = None


@torch.no_grad()
def calc_metrics(model, dataloader, num_steps, hits_at_seq=(1,), max_batches=30):
    model.eval()

    pred_states = []
    next_states = []

    hits_at = defaultdict(int)
    num_samples = 0
    rr_sum = 0
    steps = 0
    res = dict()

    for batch_idx, data_batch in enumerate(dataloader):
        if batch_idx == max_batches:
            break
        observations, actions = data_batch
        observations = observations.cuda()
        actions = actions.cuda()
        _, T, *_ = observations.shape

        start_step = np.random.randint(0, T - num_steps - 1)
        obs = observations[:, start_step]
        next_obs = observations[:, start_step + num_steps]

        state = model.extract_state(obs)
        next_state = model.extract_state(next_obs)

        pred_state = state
        for i in range(num_steps):
            pred_state = model.next_state(pred_state, actions[:, i])

        pred_state = pred_state[0]
        next_state = next_state[0]

        pred_states.append(pred_state.cpu())
        next_states.append(next_state.cpu())
        steps += 1

    print('Calculating metrics')

    pred_state_cat = torch.cat(pred_states, dim=0)
    next_state_cat = torch.cat(next_states, dim=0)

    full_size = pred_state_cat.size(0)

    # Flatten object/feature dimensions    writer.add_h
    dist_matrix = slots_distance_matrix(next_state_cat, pred_state_cat)

    dist_matrix_diag = torch.diag(dist_matrix).unsqueeze(-1)
    dist_matrix_augmented = torch.cat(
        [dist_matrix_diag, dist_matrix], dim=1)

    # Workaround to get a stable sort in numpy.
    dist_np = dist_matrix_augmented.numpy()
    indices = []
    for row in dist_np:
        keys = (np.arange(len(row)), row)
        indices.append(np.lexsort(keys))
    indices = np.stack(indices, axis=0)
    indices = torch.from_numpy(indices).long()
    labels = torch.zeros(
        indices.size(0), device=indices.device,
        dtype=torch.int64).unsqueeze(-1)
    num_samples += full_size
    for k in hits_at_seq:
        match = (indices[:, :k] == labels).any(dim=1)
        num_matches = match.sum()
        hits_at[k] += num_matches.item()

    match = indices == labels
    _, ranks = match.max(1)

    reciprocal_ranks = torch.reciprocal(ranks.double() + 1)
    rr_sum += reciprocal_ranks.sum().item()

    for k in hits_at_seq:
        result_hits = hits_at[k] / float(num_samples)
        res[f"HITS_at_{k}"] = result_hits

    result_mrr = rr_sum / float(num_samples)
    res['MRR'] = result_mrr
    return res


test_dataset = EnvTestDataset(env, args.num_episodes, args.steps_per_episode,
                              one_hot_actions=args.one_hot_actions, data_path=os.path.join(args.data_path, 'test'))
test_loader = DataLoader(test_dataset, sampler=test_sampler, **loader_kwargs)

results = {}

with torch.no_grad():
    for k in test_steps:

        print(f'Evaluation {k} steps...')
        metrics = calc_metrics(model, test_loader, k)
        print(f"Eval results at {k} steps")
        for key, val in metrics.items():
            print(f'{key}: {val}')
            writer.add_text(f'test_{k}_step', f'{key}: {val}')
