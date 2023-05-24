import math
import os
import numpy as np
from scipy import optimize
import torch
import torch.nn as nn
import torch.nn.functional as F
from atari_env import AtariEnv

from causal_world_push import CausalWorldPush


def gumbel_max(logits, dim=-1):
    
    eps = torch.finfo(logits.dtype).tiny
    
    gumbels = -(torch.empty_like(logits).exponential_() + eps).log()
    gumbels = logits + gumbels
    
    return gumbels.argmax(dim)


def gumbel_softmax(logits, tau=1., hard=False, dim=-1):
    
    eps = torch.finfo(logits.dtype).tiny
    
    gumbels = -(torch.empty_like(logits).exponential_() + eps).log()
    gumbels = (logits + gumbels) / tau
    
    y_soft = F.softmax(gumbels, dim)
    
    if hard:
        index = y_soft.argmax(dim, keepdim=True)
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.)
        return y_hard - y_soft.detach() + y_soft
    else:
        return y_soft


def log_prob_gaussian(value, mean, std):
    
    var = std ** 2
    if isinstance(var, float):
        return -0.5 * (((value - mean) ** 2) / var + math.log(var) + math.log(2 * math.pi))
    else:
        return -0.5 * (((value - mean) ** 2) / var + var.log() + math.log(2 * math.pi))


def conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0,
           dilation=1, groups=1, bias=True, padding_mode='zeros',
           weight_init='xavier'):
    
    m = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                  dilation, groups, bias, padding_mode)
    
    if weight_init == 'kaiming':
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
    else:
        nn.init.xavier_uniform_(m.weight)
    
    if bias:
        nn.init.zeros_(m.bias)
    
    return m


class Conv2dBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        
        self.m = conv2d(in_channels, out_channels, kernel_size, stride, padding,
                        bias=False, weight_init='kaiming')
        self.weight = nn.Parameter(torch.ones(out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels))
    
    
    def forward(self, x):
        
        x = self.m(x)
        return F.relu(F.group_norm(x, 1, self.weight, self.bias))


def linear(in_features, out_features, bias=True, weight_init='xavier', gain=1.):
    
    m = nn.Linear(in_features, out_features, bias)
    
    if weight_init == 'kaiming':
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
    else:
        nn.init.xavier_uniform_(m.weight, gain)
    
    if bias:
        nn.init.zeros_(m.bias)
    
    return m


def gru_cell(input_size, hidden_size, bias=True):
    
    m = nn.GRUCell(input_size, hidden_size, bias)
    
    nn.init.xavier_uniform_(m.weight_ih)
    nn.init.orthogonal_(m.weight_hh)
    
    if bias:
        nn.init.zeros_(m.bias_ih)
        nn.init.zeros_(m.bias_hh)
    
    return m


def get_env(env_name, env_config = dict()):
    if env_name == 'cw-push':
        return CausalWorldPush(**env_config)
    elif env_name == 'pong':
        return AtariEnv('PongDeterministic-v4', **env_config)
    elif env_name == 'spinv':
        return AtariEnv('SpaceInvadersDeterministic-v4', **env_config)
    else:
        ValueError(f"Env {env_name} is not supported")
        

def create_dirs(dir_path):
    if len(dir_path) == 0:
        return

    if not os.path.isdir(dir_path):
        if os.path.isfile(dir_path):
            return
        os.makedirs(dir_path)
        
def to_one_hot(val, max_val):
    result = np.zeros(max_val)
    result[val] = 1
    return result

def hungarian_l2_loss(x, y):
    n_objs = x.shape[1]
    pairwise_cost = torch.pow(torch.unsqueeze(y, -2).expand(-1, -1, n_objs, -1) - torch.unsqueeze(x, -3).expand(-1, n_objs, -1, -1), 2).mean(dim=-1)
    indices = np.array(list(map(optimize.linear_sum_assignment, pairwise_cost.detach().cpu().numpy())))
    transposed_indices = np.transpose(indices, axes=(0, 2, 1))
    final_costs = torch.gather(pairwise_cost, dim=-1, index=torch.LongTensor(transposed_indices).to(pairwise_cost.device))[:, :, 1]
    return final_costs.sum(dim=1)

def hungarian_huber_loss(x, y):
    n_objs = x.shape[1]
    pairwise_cost = F.smooth_l1_loss(torch.unsqueeze(y, -2).expand(-1, -1, n_objs, -1), torch.unsqueeze(x, -3).expand(-1, n_objs, -1, -1), reduction='none').mean(dim=-1)
    indices = np.array(list(map(optimize.linear_sum_assignment, pairwise_cost.detach().cpu().numpy())))
    transposed_indices = np.transpose(indices, axes=(0, 2, 1))
    final_costs = torch.gather(pairwise_cost, dim=-1, index=torch.LongTensor(transposed_indices).to(pairwise_cost.device))[:, :, 1]
    return final_costs.sum(dim=1)

def slots_distance_matrix(x, y):
    num_samples, num_objects, dim = x.shape
    

    x = x.unsqueeze(1).expand(num_samples, num_samples, num_objects, dim)
    y = y.unsqueeze(0).expand(num_samples, num_samples, num_objects, dim)
    
    loss_matrix = hungarian_l2_loss(x.reshape(-1, num_objects, dim), y.reshape(-1, num_objects, dim))
    return loss_matrix.reshape(num_samples, num_samples)
    
    

def pairwise_distance_matrix(x, y):

    num_samples = x.size(0)
    dim = x.size(1)

    x = x.unsqueeze(1).expand(num_samples, num_samples, dim)
    y = y.unsqueeze(0).expand(num_samples, num_samples, dim)

    return torch.pow(x - y, 2).sum(2)
