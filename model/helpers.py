import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops.layers.torch import Rearrange
import pdb
from Simulations.Lorenz_Atractor.parameters import m1x_0, m2x_0, m, n,\
f, f_batch, h, hRotate, H_Rotate, H_Rotate_inv, Q_structure, R_structure

# import utils as utils

#-----------------------------------------------------------------------------#
#---------------------------------- modules ----------------------------------#
#-----------------------------------------------------------------------------#

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, mish=True, n_groups=8):
        super().__init__()

        if mish:
            act_fn = nn.Mish()
        else:
            act_fn = nn.SiLU()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange('batch channels 1 horizon -> batch channels horizon'),
            act_fn,
        )

    def forward(self, x):
        return self.block(x)


#-----------------------------------------------------------------------------#
#---------------------------------- sampling ---------------------------------#
#-----------------------------------------------------------------------------#

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)


def apply_conditioning(x, conditions, action_dim):

    mask = conditions != 0  # [batch_size, horizon, transition]

    x = torch.where(mask, conditions, x)

    return x
def crop_tensor(tensor, target_tensor):
    target_size = target_tensor.size()[2]
    tensor_size = tensor.size()[2]
    delta = tensor_size - target_size
    return tensor[:, :, 0:tensor_size - delta]
#-----------------------------------------------------------------------------#
#---------------------------------- losses -----------------------------------#
#-----------------------------------------------------------------------------#

class WeightedLoss(nn.Module):

    def __init__(self, weights, action_dim):
        super().__init__()
        self.register_buffer('weights', weights)
        self.action_dim = action_dim

    def forward(self, pred, targ):
        '''
            pred, targ : tensor
                [ batch_size x horizon x transition_dim ]
        '''
        loss = self._loss(pred, targ)
        weighted_loss = (loss * self.weights).mean()
        a0_loss = (loss[:, 0, :self.action_dim] / self.weights[0, :self.action_dim]).mean()
        return weighted_loss, {'a0_loss': a0_loss}

class WeightedStateLoss(nn.Module):

    def __init__(self, weights):
        super().__init__()
        self.register_buffer('weights', weights)
    def forward(self, pred, targ):
        '''
            pred, targ : tensor
                [ batch_size x horizon x transition_dim ]
        '''
        loss = self._loss(pred, targ)
        weighted_loss = loss.mean()
        # weighted_loss = (loss * self.weights).mean()
        return weighted_loss, {'a0_loss': weighted_loss}

class WeightedStateLoss1(nn.Module):

    def __init__(self, weights):
        super().__init__()
        self.register_buffer('weights', weights)
    def forward(self, pred, targ): #2
        '''
            pred, targ : tensor
                [ batch_size x horizon x transition_dim ]

        '''
        mask = (targ != 0).float()

        loss = (pred - targ) ** 2  # [batch_size x horizon x transition_dim]

        masked_loss = loss * mask  # [batch_size x horizon x transition_dim]

        num_valid_steps_per_sample = mask.sum(dim=[1, 2])  # [batch_size]

        total_loss_per_sample = masked_loss.sum(dim=[1, 2])  # [batch_size]
        avg_loss_per_sample = total_loss_per_sample / num_valid_steps_per_sample  # [batch_size]

        weighted_loss = avg_loss_per_sample.mean()

        return weighted_loss, {'a0_loss': weighted_loss}
class WeightedStateLoss2(nn.Module):

    def __init__(self, weights):
        super().__init__()
        self.register_buffer('weights', weights)
    def forward(self, pred, targ):
        '''
            pred, targ : tensor
                [ batch_size x horizon x transition_dim ]
        '''

        mask = (targ != 0).float()

        loss = (pred - targ) ** 2  # [batch_size x horizon x transition_dim]

        masked_loss = loss * mask  # [batch_size x horizon x transition_dim]

        eps = 1e-8

        mask_horizon = mask.sum(dim=[0, 2])  # [horizon]
        loss_horizon = masked_loss.sum(dim=[0, 2])  # [horizon]

        masked_loss_horizon = (loss_horizon != 0).float()
        sum_masked_loss_horizon = sum(masked_loss_horizon)

        masked_mask_horizon = (mask_horizon != 0).float()
        sum_masked_mask_horizon= sum(masked_mask_horizon)

        mask_horizon = mask_horizon + eps
        weighted_loss_per_horizon = loss_horizon / mask_horizon

        weighted_loss = sum(weighted_loss_per_horizon)/sum_masked_loss_horizon
        # weighted_loss = (loss * self.weights).mean()
        return weighted_loss, {'a0_loss': weighted_loss}

class WeightedStateLoss3(nn.Module):

    def __init__(self, weights):
        super().__init__()
        self.register_buffer('weights', weights)
    def forward(self, pred, targ):
        '''
            pred, targ : tensor
                [ batch_size x horizon x transition_dim ]
        '''
        mask = (targ != 0).float()
        loss = (pred - targ) ** 2  # [batch_size x horizon x transition_dim]

        masked_loss = loss * mask  # [batch_size x horizon x transition_dim]

        weighted_loss = masked_loss.sum()
        weighted_loss = torch.mean(weighted_loss)
        # weighted_loss = (loss * self.weights).mean()
        return weighted_loss, {'a0_loss': weighted_loss}

def w_linear(t, T, K):

    t_normalized = t / (T - 1)

    f_t = 1 + (K - 1) * t_normalized

    return f_t
def w(t, T, K, b=2.5):

    t_normalized = t / (T - 1)

    f_t = 1 + (K - 1) * (t_normalized ** b)

    return f_t
class WeightedStateLoss4(nn.Module):

    def __init__(self, weights):
        super().__init__()
        self.register_buffer('weights', weights)
    def forward(self, pred, targ):
        '''
            pred, targ : tensor
                [ batch_size x horizon x transition_dim ]
        '''
        masked = torch.zeros_like(targ)
        mask = (targ != 0).float()
        loss = (pred - targ) ** 2  # [batch_size x horizon x transition_dim]

        loss_first_dim = loss[:, :, :1]

        non_zero_counts = mask.sum(dim=1)  # [batch_size x transition_dim]
        weight = torch.zeros_like(non_zero_counts)

        for i in range(mask.shape[0]):
            t = non_zero_counts[i, 1].item()
            # weight[i] = 1
            weight[i] = w(t, mask.shape[1],1.7)

            masked[i, int(t) - 1, :] = weight[i]

        masked_loss = loss_first_dim * masked  # [batch_size x horizon x transition_dim]
        weighted_loss = masked_loss.sum(dim=[1, 2])
        weighted_loss = torch.mean(weighted_loss)
        weighted_loss = weighted_loss
        # weighted_loss = (loss * self.weights).mean()
        return weighted_loss, {'a0_loss': weighted_loss}
class WeightedStateLoss4NCLT(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.register_buffer('weights', weights)
    def forward(self, pred, targ):
        '''
            pred, targ : tensor
                [ batch_size x horizon x transition_dim ]
        '''
        masked = torch.zeros_like(targ)
        mask = (targ != 0).float()

        loss = (pred - targ) ** 2  # [batch_size x horizon x transition_dim]

        non_zero_counts = mask.sum(dim=1)  # [batch_size x transition_dim]
        weight = torch.zeros_like(non_zero_counts)
        for i in range(mask.shape[0]):
            t = non_zero_counts[i, 1].item()
            masked[i, int(t) - 1, :] = 1

        masked_loss = loss * masked  # [batch_size x horizon x transition_dim]
        weighted_loss = masked_loss.sum()
        weighted_loss = torch.mean(weighted_loss)
        # weighted_loss = (loss * self.weights).mean()
        return weighted_loss, {'a0_loss': weighted_loss}
class WeightedStateLossModel4(nn.Module):

    def __init__(self, weights):
        super().__init__()
        self.register_buffer('weights', weights)
    def forward(self, pred, targ):
        '''
            pred, targ : tensor
                [ batch_size x horizon x transition_dim ]
        '''
        batch_size = pred.size(0)

        masked = torch.zeros_like(targ)
        masked1 = torch.zeros_like(targ)
        mask = (targ != 0).float()

        loss = (pred - targ) ** 2  # [batch_size x horizon x transition_dim]
        loss1 = torch.zeros_like(loss)
        loss2 = (pred - targ)

        loss3 = torch.zeros_like(loss2)

        loss3[:, 1:] = loss2[:, :-1]
        loss1 = torch.abs(loss2-loss3)

        non_zero_counts = mask.sum(dim=1)  # [batch_size x transition_dim]
        weight = torch.zeros_like(non_zero_counts)

        for i in range(mask.shape[0]):
            t = non_zero_counts[i, 1].item()
            weight[i] = w(t, mask.shape[1],1.5)

            masked[i, int(t) - 1, :] = weight[i]
            if t > 1:
                masked1[i, int(t) - 1, :] = 1

        # 使用掩码仅保留非零部分计算损失
        masked_loss = loss * masked  # [batch_size x horizon x transition_dim]
        masked_loss1 = loss1 * masked
        weighted_loss = masked_loss.sum(dim=[1, 2])
        weighted_loss1 = masked_loss1.sum(dim=[1, 2])
        weighted_loss = weighted_loss + weighted_loss1
        weighted_loss = torch.mean(weighted_loss)
        # weighted_loss = (loss * self.weights).mean()
        return weighted_loss, {'a0_loss': weighted_loss}


class WeightedStateLoss5(nn.Module):

    def __init__(self, weights):
        super().__init__()
        self.register_buffer('weights', weights)

    def forward(self, pred, targ):
        '''
            pred, targ : tensor
                [ batch_size x horizon x transition_dim ]
        '''
        masked = torch.zeros_like(targ)
        mask = (targ != 0).float()
        loss = (pred - targ) ** 2  # [batch_size x horizon x transition_dim]

        non_zero_counts = mask.sum(dim=1)  # [batch_size]

        for i in range(mask.shape[0]):
            t = non_zero_counts[i, 1].item()

            masked[i, int(t) - 1, :] = 1

        # 使用掩码仅保留非零部分计算损失
        masked_loss = loss * mask  # [batch_size x horizon x transition_dim]
        weighted_loss = masked_loss.sum(dim=[1, 2])
        weighted_loss = torch.mean(weighted_loss)
        # weighted_loss = (loss * self.weights).mean()
        return weighted_loss, {'a0_loss': weighted_loss}
class WeightedStateLoss7(nn.Module):#

    def __init__(self, weights):
        super().__init__()
        self.register_buffer('weights', weights)
    def forward(self, pred, targ, num_time):
        '''
            pred, targ : tensor
                [ batch_size x 1 x transition_dim ]
        '''

        batch_size, horizon, dim = targ.shape
        masked = torch.zeros_like(targ)
        loss = (pred - targ) ** 2  # [batch_size x horizon x transition_dim]
        T_max = 20
        sum_loss = loss.sum(dim=[1, 2]).unsqueeze(-1)
        weight = w(num_time, T=T_max, K=1.1)
        weighted_loss = sum_loss * weight
        weighted_loss = 10 * torch.mean(weighted_loss)
        # weighted_loss = (loss * self.weights).mean()
        return weighted_loss, {'a0_loss': weighted_loss}
class WeightedStateLoss6(nn.Module):

    def __init__(self, weights):
        super().__init__()
        self.register_buffer('weights', weights)
    def forward(self, pred, targ):
        '''
            pred, targ : tensor
                [ batch_size x horizon x transition_dim ]
        '''
        batch_size, horizon, dim = targ.shape
        masked = torch.zeros_like(targ)
        predF = torch.zeros_like(pred)
        targF = torch.zeros_like(targ)
        predF = f_batch(pred.unsqueeze(-1)).squeeze()
        targF = f_batch(targ.unsqueeze(-1)).squeeze()
        # for j in range(horizon):
        #
        #     predF[:, j, :] = f(pred[:, j, :].unsqueeze(-1)).squeeze() #f需要的输入是batchsize,dim,1
        #     targF[:, j, :] = f(targ[:, j, :].unsqueeze(-1)).squeeze()

        mask = (targ != 0).float()

        lossF = (predF - targF) ** 2
        loss = (pred - targ) ** 2  # [batch_size x horizon x transition_dim]

        non_zero_counts = mask.sum(dim=1)  # [batch_size x transition_dim]
        weight = torch.zeros_like(non_zero_counts)

        for i in range(mask.shape[0]):
            t = non_zero_counts[i, 1].item()
            weight[i] = w(t, mask.shape[1],2)

            masked[i, int(t) - 1, :] = weight[i]


        masked_loss = loss * masked  # [batch_size x horizon x transition_dim]
        masked_lossF = lossF * masked
        masked_loss_plus = masked_loss + 5 * masked_lossF

        weighted_loss = masked_loss_plus.sum(dim=[1, 2])
        weighted_loss = torch.mean(weighted_loss)
        # weighted_loss = (loss * self.weights).mean()
        return weighted_loss, {'a0_loss': weighted_loss}
class ValueLoss(nn.Module):
    def __init__(self, *args):
        super().__init__()
        pass

    def forward(self, pred, targ):
        loss = self._loss(pred, targ).mean()

        if len(pred) > 1:
            corr = np.corrcoef(
                utils.to_np(pred).squeeze(),
                utils.to_np(targ).squeeze()
            )[0,1]
        else:
            corr = np.NaN

        info = {
            'mean_pred': pred.mean(), 'mean_targ': targ.mean(),
            'min_pred': pred.min(), 'min_targ': targ.min(),
            'max_pred': pred.max(), 'max_targ': targ.max(),
            'corr': corr,
        }

        return loss, info

class WeightedL1(WeightedLoss):

    def _loss(self, pred, targ):
        return torch.abs(pred - targ)

class WeightedL2(WeightedLoss):

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')

class WeightedStateL2(WeightedStateLoss):

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')

class WeightedStateL21(WeightedStateLoss1):

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')

class WeightedStateL22(WeightedStateLoss2):

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')
class ValueL1(ValueLoss):

    def _loss(self, pred, targ):
        return torch.abs(pred - targ)
class WeightedStateL23(WeightedStateLoss3):

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')

class WeightedStateL24(WeightedStateLoss4):

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')

class WeightedStateL24NCLT(WeightedStateLoss4NCLT):

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')

class WeightedStateL2model4(WeightedStateLossModel4):
    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')

class WeightedStateL25(WeightedStateLoss5):

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')

class WeightedStateL26(WeightedStateLoss6):

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')

class WeightedStateL27(WeightedStateLoss7):

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')
class ValueL2(ValueLoss):

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')

Losses = {
    'l1': WeightedL1,
    'l2': WeightedL2,
    'state_l2': WeightedStateL2,
    'state_l21': WeightedStateL21, #1+1/2+1/3+...+1/n unbalance
    'state_l22': WeightedStateL22, #balanced
    'state_l23': WeightedStateL23,
    'state_l24': WeightedStateL24,
    'state_l24NCLT': WeightedStateL24NCLT,
    'state_l2_model4': WeightedStateL2model4, #model4 loss
    'state_l25': WeightedStateL25,
    'state_l26': WeightedStateL26,
    'state_l27': WeightedStateL27,
    'value_l1': ValueL1,
    'value_l2': ValueL2,
}
