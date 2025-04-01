import os
import copy
import numpy as np
import torch
import pdb
from copy import deepcopy
import random
import time
from torch.optim.lr_scheduler import CosineAnnealingLR

def expand_train_data(train_target_long, train_input_long, train_init):
    batch_size = train_target_long.size(0)
    horizon = train_target_long.size(1)
    transition = train_target_long.size(2)

    expanded_target = torch.zeros(batch_size * horizon, horizon, transition, device=train_target_long.device)
    expanded_input = torch.zeros(batch_size * horizon, horizon, transition, device=train_input_long.device)
    expanded_init = torch.zeros(batch_size * horizon, 1, transition, device=train_input_long.device)

    for i in range(batch_size):
        for t in range(1, horizon + 1):
            expanded_target[i * horizon + (t - 1), :t, :] = train_target_long[i, :t, :]
            expanded_input[i * horizon + (t - 1), :t, :] = train_input_long[i, :t, :]
            expanded_init[i * horizon + (t - 1), :, :] = train_init[i, :, :]

    return expanded_target, expanded_input, expanded_init
def count_params(model):
    """
    Counts two types of parameters:

    - Total no. of parameters in the model (including trainable parameters)
    - Number of trainable parameters (i.e. parameters whose gradients will be computed)

    """
    total_num_params = sum(p.numel() for p in model.parameters())
    total_num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad == True)
    return total_num_params, total_num_trainable_params
def data_iter(batch_size, dataset):
    num_examples = len(dataset)
    indices = list(range(num_examples))
    random.shuffle(indices)

    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])

        batch = [torch.stack([x[idx] for idx in batch_indices]) for x in zip(*dataset)]
        yield batch

def cycle(dl):
    while True:
        for data in dl:
            yield data

class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new
class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        train_lr=8e-5,
        ema_decay=0.995,
        train_batch_size=128,
        total_step=12000,
        gradient_accumulate_every=10,
        step_start_ema=600, #
        update_ema_every=10,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,#
        label_freq=100000,
        save_parallel=False,
        n_reference=8,
        bucket=None,
        train_device='cuda',
        save_checkpoints=True,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every
        self.save_checkpoints = save_checkpoints

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel
        self.total_step = total_step
        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset
        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=3000, gamma=0.85)

        self.bucket = bucket
        self.n_reference = n_reference

        self.reset_parameters()
        self.step = 0

        self.device = train_device
    
    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())
    
    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#
    def train(self, n_train_steps):

        total_num_params, total_num_trainable_params = count_params(self.model)
        print("No. of trainable parameters: {}\n".format(total_num_trainable_params))
        start_time = time.time()
        tmp_loss = 0
        loss_history = []
        for step in range(n_train_steps):

            for batch in data_iter(self.batch_size, self.dataset):
                x = batch[0].clone().detach().to(device='cuda', dtype=torch.float32)
                cond = batch[1].clone().detach().to(device='cuda', dtype=torch.float32)
                returns = batch[2].clone().detach().to(device='cuda', dtype=torch.float32)

                loss, infos = self.model.loss(x, cond, returns)
                loss = loss / self.gradient_accumulate_every
                loss.backward()
                tmp_loss += loss.detach().item()

                loss_history.append(loss.detach().item())
            self.optimizer.step()
            self.scheduler.step()

            self.optimizer.zero_grad()
            if self.step % 200 ==0:
                print("loss:", loss)
                print(f"Step {self.step}, Learning rate: {self.optimizer.param_groups[0]['lr']}")

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step % self.save_freq == 0 : #and self.step >= 4000
                print('Saving checkpoint...')

                end_time = time.time()
                step_time = end_time - start_time
                print(f"Step {step} | Loss: {tmp_loss:.4f} | Time: {step_time:.4f} seconds")
                self.save()

            self.step += 1

        return loss_history

    def save(self):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        savepath = os.path.join(self.bucket, 'checkpoint')  # Removed logger.prefix
        os.makedirs(savepath, exist_ok=True)

        if self.save_checkpoints:
            savepath = os.path.join(savepath, f'state_kitchen_partial_test2_{self.step}.pt')
        else:
            savepath = os.path.join(savepath, 'state_kitchen_partial_test2.pt')

        torch.save(data, savepath)
        print(f'[ utils/training ] Saved model to {savepath}')  # Use print instead of logger.print