import pickle
import numpy as np
from model.diffusion import GaussianDiffusion
from model.temporal import TemporalUnet
from model.helpers import apply_conditioning
from trainer_stepLR_nonlinearH_gaussian import Trainer
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import time
import argparse
from mpl_toolkits.mplot3d import Axes3D

DTYPE = torch.float
DEVICE = 'cuda'

import torch
import torch.nn as nn
def expand_train_data(train_target_long, train_input_long, train_init):
    K = 2
    batch_size = train_target_long.size(0)
    horizon = train_target_long.size(1)
    transition = train_target_long.size(2)

    expanded_target = torch.zeros(batch_size * horizon, horizon, transition, device=train_target_long.device)
    expanded_input = torch.zeros(batch_size * horizon, horizon, transition, device=train_input_long.device)
    expanded_init = torch.zeros(batch_size * horizon, 1, transition, device=train_input_long.device)
    cond = torch.zeros_like(expanded_target, dtype=torch.float32, device='cuda')
    returns = torch.zeros(expanded_input.size(0), K, expanded_input.size(2))

    for i in range(batch_size):
        for t in range(1, horizon + 1):

            expanded_target[i * horizon + (t - 1), :t, :] = train_target_long[i, :t, :]
            expanded_input[i * horizon + (t - 1), :t, :] = train_input_long[i, :t, :]
            expanded_init[i * horizon + (t - 1), :, :] = train_init[i, :, :]

    mask = (expanded_input != 0).float()
    non_zero_counts = mask.sum(dim=1)
    for i in range(mask.shape[0]):
        t = non_zero_counts[i, 1]
        mask[i, int(t) - 1, :] = 0
        if t > 1:
            returns[i, 0, :] = expanded_input[i, int(t) - 2, :]
            returns[i, 1, :] = expanded_input[i, int(t) - 1, :]
        else:
            returns[i, 1, :] = expanded_input[i, int(t) - 1, :]
    cond = expanded_input * mask + cond
    returns = returns.view(returns.size(0), -1)
    return expanded_target, returns, cond

def expand_test_input(test_input):

    batch_size = test_input.size(0)
    horizon = test_input.size(1)
    transition = test_input.size(2)

    expanded_input = torch.zeros(batch_size * horizon, horizon, transition, device=test_input.device)

    for i in range(batch_size):
        for t in range(1, horizon + 1):
            expanded_input[i * horizon + (t - 1), :t, :] = test_input[i, :t, :]

    return expanded_input


def expand_test_init(test_init, horizon):

    batch_size = test_init.size(0)
    transition = test_init.size(2)

    expanded_init = torch.zeros(batch_size * horizon, 1, transition, device=test_init.device)

    for i in range(batch_size):
        for t in range(horizon):
            expanded_init[i * horizon + t, :, :] = test_init[i, :, :]

    return expanded_init


def reverse_operation(predicted_samples, expanded_input, horizon):
    """
    Reverse the operation on predicted_samples according to expanded_input, restoring each sample from [batch_size*horizon, horizon, transition]
    to its original [batch_size, horizon, transition].

    Parameters:
    - predicted_samples: shape [batch_size * horizon, horizon, transition]
    - expanded_input: shape [batch_size * horizon, horizon, transition]
    - horizon: horizon value in the original data

    Returns:
    - reversed_predicted_samples: shape [batch_size, horizon, transition]
    """
    batch_size = expanded_input.size(0) // horizon
    reversed_predicted_samples = torch.zeros(batch_size, horizon, expanded_input.size(2),
                                             device=predicted_samples.device)

    for i in range(batch_size):
        for t in range(horizon):
            reversed_predicted_samples[i, t, :] = predicted_samples[i * horizon + t, t, :]

    return reversed_predicted_samples
def evaluate_model(trainer, test_input, test_target, test_init):
    """
    Evaluate model performance using Mean Squared Error (MSE), and compute the mean, standard deviation, and confidence interval.
    # Supports observation inputs of arbitrary length to produce real-time estimates xk

    Parameters:
    - trainer: trainer object containing the trained model
    - test_input: test set input with shape [batch_size, horizon, transition]
    - test_target: test set ground truth with shape [batch_size, horizon, transition]
    - test_init: initial conditions for test set with shape [batch_size, 1, transition]

    Returns:
    - MSE_dB_avg: average MSE of test set in dB units
    - MSE_dB_std: standard deviation of MSE in test set in dB units
    - MSE_linear_arr: array storing linear MSE values for each batch
    """

    batch_size = test_input.size(0)
    horizon = test_input.size(1)
    K = 2 #L=2

    expanded_input = expand_test_input(test_input).to(trainer.device)

    returns = expanded_input

    cond = torch.zeros_like(expanded_input, dtype=torch.float32, device='cuda')
    returns = torch.zeros(expanded_input.size(0), K, expanded_input.size(2)).to(trainer.device)

    mask = (expanded_input != 0).float()
    non_zero_counts = mask.sum(dim=1)  # [batch_size]
    for i in range(mask.shape[0]):
        t = non_zero_counts[i, 1]
        mask[i, int(t) - 1, :] = 0
        if t > 1:
            returns[i, 0, :] = expanded_input[i, int(t) - 2, :] #zt-1
            returns[i, 1, :] = expanded_input[i, int(t) - 1, :] #zt
        else:
            returns[i, 1, :] = expanded_input[i, int(t) - 1, :]
    cond = expanded_input * mask + cond

    returns = returns.view(returns.size(0), -1)
    predicted_samples = trainer.ema_model.conditional_sample(cond, returns, expanded_input)

    x_t_estiametd = reverse_operation(predicted_samples, expanded_input, horizon)

    combined_out = {
        'predicted_out': predicted_out,
        'test_target_out': test_target_out,
        'test_input_out': test_input_out
    }

    loss_fn = nn.MSELoss(reduction='none')
    MSE_linear_arr = loss_fn(x_t_estiametd, test_target)  # [batch, horizon, transition]
    MSE_linear_arr_meas = loss_fn(test_input, test_target)  # [batch, horizon, transition]

    MSE_linear_array = torch.mean(MSE_linear_arr, dim=[1, 2])
    MSE_linear_array_meas = torch.mean(MSE_linear_arr_meas, dim=[1, 2])

    MSE_linear_avg = torch.mean(MSE_linear_array)
    MSE_linear_std = torch.std(MSE_linear_array)

    print("MSE_linear_arr.shape",MSE_linear_arr.shape)
    MSE_linear_std = torch.std(MSE_linear_array, unbiased=True)


    MSE_dB_avg = 10 * torch.log10(MSE_linear_avg)
    MSE_dB_array = 10 * torch.log10(MSE_linear_array)
    MSE_dB_avg_meas = 10 * torch.log10(MSE_linear_array_meas)
    MSE_dB_std = 10 * torch.log10(MSE_linear_std + MSE_linear_avg) - MSE_dB_avg

    return MSE_dB_avg, MSE_dB_std, MSE_dB_array, combined_out, MSE_linear_arr


def evaluate(trainer, checkpointfolderName, checkpoint_names, test_input1, test_target1, test_init1):
    for checkpoint_name in checkpoint_names:
        checkpoint_path = os.path.join(checkpointfolderName, checkpoint_name)
        print("checkpoint_path", checkpoint_path)
        trainer = load_model(trainer, checkpoint_path, device)
        trainer.model.eval()
        with torch.no_grad():

            MSE_dB_avg, MSE_dB_std, MSE_dB_array, combine_out, MSE_linear_arr = evaluate_model(trainer, test_input1, test_target1, test_init1)
            torch.save(combine_out, combine_out_path)
            print(f"Average MSE (dB): {MSE_dB_avg:.4f}")
            print(f"Standard Deviation (dB): {MSE_dB_std:.4f}")
            print("MSE_dB_array=", MSE_dB_array)

        trainer.model.train()

def plot_trajectories(odom_trajectories, truth_trajectories, init_positions, estimated_trajectories, checkpointfolderName):

    odom_trajectories = odom_trajectories.cpu().numpy()
    truth_trajectories = truth_trajectories.cpu().numpy()
    init_positions = init_positions.cpu().numpy()
    estimated_trajectories = estimated_trajectories.cpu().numpy()

    x_odom = odom_trajectories[:, 0]
    y_odom = odom_trajectories[:, 1]
    z_odom = odom_trajectories[:, 2]
    x_truth = truth_trajectories[:, 0]
    y_truth = truth_trajectories[:, 1]
    z_truth = truth_trajectories[:, 2]
    x_est = estimated_trajectories[:, 0]
    y_est = estimated_trajectories[:, 1]
    z_est = estimated_trajectories[:, 2]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(x_odom, y_odom, z_odom, label='Odometry', color='blue', marker='o', markersize=2, linestyle='-')

    ax.plot(x_truth, y_truth, z_truth, label='Ground Truth', color='green', marker='o', markersize=2, linestyle='-')

    ax.plot(x_est, y_est, z_est, label='Estimated Trajectories', color='red', marker='o', markersize=2, linestyle='-')

    ax.set_title("3D Trajectories Visualization", fontsize=16)
    ax.set_xlabel("X Position", fontsize=12)
    ax.set_ylabel("Y Position", fontsize=12)
    ax.set_zlabel("Z Position", fontsize=12)
    ax.legend(loc='best')

    ax.view_init(elev=20, azim=30)
    os.makedirs(checkpointfolderName, exist_ok=True)
    save_path = os.path.join(checkpointfolderName, 'trajectories.png')
    plt.savefig(save_path, dpi=300)
    print(f"Trajectory plot saved to: {save_path}")

    plt.show()

def load_model(trainer, checkpoint_path, device):

    checkpoint = torch.load(checkpoint_path, map_location=device)

    trainer.model.load_state_dict(checkpoint['model'])
    print("Model loaded successfully.")

    if 'ema' in checkpoint:
        trainer.ema_model.load_state_dict(checkpoint['ema'])
        print("EMA model loaded successfully.")
    else:
        print("No EMA model found in the checkpoint.")

    return trainer

def parse_args():
    parser = argparse.ArgumentParser(description="Choose model and data configuration")
    parser.add_argument('r_text', type=str, help="Model configuration text")
    parser.add_argument('lr', type=float, help="Model configuration learning rate")
    return parser.parse_args()

if __name__ == '__main__':
    # args = parse_args()
    # text = args.r_text
    # lr = args.lr
    batch_size = 128
    text = 'r0.001'
    lr = 12e-5

    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    mode_Train = False #False True 12
    mode_continueTrain = False
    plot_loss = False
    mode_Test = True
    plot_trajectory = False
    horizon=20
    K = 2
    n_train_steps = 18000 #
    n_train_steps_continue = 12000
    n_steps_per_epoch = 1000 #
    max_n_episodes=10000
    max_path_length=300
    termination_penalty=0
    discount=0.99
    observation_dim=3
    action_dim=0

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using GPU")
    else:
        raise Exception("No GPU found, please set args.use_cuda = False")

    print("lr=", lr)
    #dataset
    DatafolderName = f'results/LOR/model3/nonlineargaussian/dataset_expF/'
    ModelfolderName = f'results/LOR/model3/nonlineargaussian/{text}'
    dataFileName = [f'data_lor_v-20_{text}_T20_100_nonlinear_gaussian_nonlinearF.pt']
    print("datasetName", dataFileName)
    [train_input_long, train_target_long, cv_input, cv_target, test_input, test_target, train_init, cv_init,
     test_init] = torch.load(DatafolderName + dataFileName[0], map_location=device)

    train_target_long = train_target_long.permute(0, 2, 1)
    train_input_long = train_input_long.permute(0, 2, 1)
    train_init = train_init.permute(0, 2, 1)

    test_input = test_input.permute(0, 2, 1) #[x,1,3]
    test_init = test_init.permute(0, 2, 1)#[x,horizon,obserdim]
    test_target = test_target.permute(0, 2, 1)#[x,horizon,obserdim]

    train_target_long1, train_input_long1, train_init1 = expand_train_data(train_target_long, train_input_long, train_init)
    dataset_train = list(zip(train_target_long1, train_init1, train_input_long1))
    if not os.path.exists(ModelfolderName):
        os.makedirs(ModelfolderName)

    checkpointfolderName = os.path.join(ModelfolderName, 'checkpoint/')
    if not os.path.exists(checkpointfolderName):
        os.makedirs(checkpointfolderName)

    loss_history_path = os.path.join(checkpointfolderName, 'loss_history.pt')
    loss_history_path_continue = os.path.join(checkpointfolderName, 'loss_history_continue.pt')
    train_loss_path = os.path.join(checkpointfolderName, 'train_loss.png')
    combine_out_path = os.path.join(checkpointfolderName, 'combine_out.pt')

    #Unet model
    model = TemporalUnet(
        horizon=horizon,
        transition_dim=observation_dim,
        cond_dim=observation_dim*K).to(device=device)
    # diffusion model
    diffusion = GaussianDiffusion(
        model=model,
        horizon=horizon,
        loss_type='state_l24',
        observation_dim=observation_dim,
        action_dim=action_dim).to(device=device)
    n_epochs = int(n_train_steps // n_steps_per_epoch)
    trainer = Trainer(
        diffusion_model=diffusion,
        dataset=dataset_train,
        bucket=ModelfolderName,
        train_lr=lr,
        train_batch_size = batch_size)#No. of trainable parameters: 40236035

    # -----------------------------------------------------------------------------#
    # --------------------------------- main loop ---------------------------------#
    # -----------------------------------------------------------------------------#

    loss_over_time = []

    if mode_Train:
        print("start training...")
        for i in range(n_epochs):

            start_time = time.time()
            print(f"Epoch {i + 1}/{n_epochs}")
            loss_history = trainer.train(n_train_steps=n_steps_per_epoch)
            end_time = time.time()
            epoch_time = end_time - start_time
            print(f"Epoch time: {epoch_time}")
            loss_over_time.extend(loss_history)

        loss_tensor = torch.tensor(loss_over_time)


        torch.save(loss_tensor, loss_history_path)
        print("end training...")

    if mode_continueTrain:
        print("Continuing training from saved model...")

        checkpointfolderName = os.path.join(ModelfolderName, 'checkpoint/')
        checkpoint_path = os.path.join(checkpointfolderName, 'state_kitchen_partial_test2_3000.pt')
        print("checkpoint_path ", checkpoint_path )


        trainer = load_model(trainer, checkpoint_path, device)
        n_epochs_continue = int(n_train_steps_continue // n_steps_per_epoch)

        for i in range(n_epochs_continue):
            start_time = time.time()
            print(f"Epoch {i + 1}/{n_epochs_continue} (Continuing Training)")
            loss_history = trainer.train(n_train_steps=n_steps_per_epoch)
            end_time = time.time()
            epoch_time = end_time - start_time
            print(f"Epoch time: {epoch_time}")
            loss_over_time.extend(loss_history)

        loss_tensor = torch.tensor(loss_over_time)
        torch.save(loss_tensor, loss_history_path_continue)
        print("end continued training...")

    if mode_Test:
        print("start evaluate()")

        checkpoint_names = [
            # 'state_kitchen_partial_test2_continue.pt',
            'state_kitchen_partial_test2_best.pt',
            # 'state_kitchen_partial_test2_16000.pt',
            # 'state_kitchen_partial_test2_17000.pt'

        ]

        checkpointfolderName = os.path.join(ModelfolderName, 'checkpoint/')  # 1
        test_target1 = test_target
        test_input1 = test_input
        test_init1 = test_target
        print("test_test")
        start_time = time.time()
        evaluate(trainer, checkpointfolderName, checkpoint_names, test_input1, test_target1, test_init1)
        end_time = time.time()
        duration = end_time - start_time
        print(f"inference time: {duration:.2f}s")

        if plot_trajectory:
            print("plot_trajectory")
            dataFileName_combined = [f'combine_out.pt']
            combined_out = torch.load(checkpointfolderName + dataFileName_combined[0], map_location=device)

            estimated_trajectories = combined_out['predicted_out']
            truth_trajectories = combined_out['test_target_out']
            odom_trajectories = combined_out['test_input_out']
            init_positions = combined_out['test_init_out']

            plot_trajectories(odom_trajectories, truth_trajectories, init_positions, estimated_trajectories, checkpointfolderName)