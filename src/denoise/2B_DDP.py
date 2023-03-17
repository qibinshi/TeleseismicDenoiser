"""
Transfer learning from WaveDecompNet
with extended 2-branch U-net layers,
Data being augmented on the fly.

@auther: Qibin Shi (qibins@uw.edu)
"""
import os
import h5py
import time
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from matplotlib import pyplot as plt
from numpy.random import default_rng
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from autoencoder_1D_models_torch import T_model
from denoiser_util import mkdir, write_progress
from sklearn.model_selection import train_test_split
from torch_tools import WaveformDataset, try_gpu, CCMSELoss
from torch_tools import training_loop_branches_augmentation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int, help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int, help='ranking within the nodes')
    parser.add_argument('--epochs', default=2, type=int, metavar='N', help='number of total epochs to run')
    args = parser.parse_args()
    #########################################################
    args.world_size = args.gpus * args.nodes
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    mp.spawn(train, nprocs=args.gpus, args=(args,))
    #########################################################

    print("Plotting the loss history.")
    bottleneck_name = 'LSTM'
    model_structure = "Branch_Encoder_Decoder"
    model_name = model_structure + "_" + bottleneck_name
    model_dir = 'Release_Middle_augmentation_P4Hz_DDP'
    with h5py.File(model_dir + f'/{model_name}_Training_history_gpu0.hdf5', 'r') as f:
        loss = f['loss'][:]
        val_loss = f['val_loss'][:]
        test_loss = np.array(f['test_loss'])
        noise_loss = f['noise_loss'][:]
        noise_val_loss = f['noise_val_loss'][:]
        earthquake_loss = f['earthquake_loss'][:]
        earthquake_val_loss = f['earthquake_val_loss'][:]
    print(test_loss, type(test_loss))

    plt.close('all')
    plt.figure()
    plt.plot(loss, 'o', label='loss')
    plt.plot(val_loss, '-', label='Validation loss')
    plt.plot(noise_loss, 'o', label='noise train loss')
    plt.plot(noise_val_loss, '-', label='noise valid loss')
    plt.plot(earthquake_loss, 'o', label='quake train loss')
    plt.plot(earthquake_val_loss, '-', label='quake valid loss')
    plt.plot(len(loss), test_loss, 'r*', label=f'Test loss = {test_loss:.2f}', ms=10, lw=2, zorder=10)

    plt.legend()
    plt.title(model_name)
    plt.savefig(model_dir + f'/{model_name}_Training_history.png')


def train(gpu, args):
    since = time.time()
    #####################################################
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    #####################################################

    npts = 3000
    strmax = 6
    mid_pt = 25000
    test_size = 0.5  # (1-60%) x 50%
    train_size = 0.6  # 60%
    rand_seed1 = 43
    rand_seed2 = 11
    random.seed(0)
    np.random.seed(20)
    torch.manual_seed(99)
    weighted_loss = False

    devc = try_gpu(i=gpu)
    bottleneck_name = 'LSTM'
    model_structure = "Branch_Encoder_Decoder"
    model_name = model_structure + "_" + bottleneck_name
    model_dir = 'Release_Middle_augmentation_P4Hz_DDP'
    datadir = '/fd1/QibinShi_data/matfiles_for_denoiser/'
    wave_preP = datadir + 'Alldepths_snr25_2000_21_sample10_lpass4_P_preP_MP_both_BH_HH.hdf5'
    if gpu == 0:
        mkdir(model_dir)

    # %% Neural Net
    model = torch.load('Model_and_datasets_1D_all_snr_40' + f'/{model_name}/{model_name}_Model.pth')
    model = T_model(model)
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    # %% Hyper-parameters for training
    batch_size, epochs, lr, minimum_epochs, patience = 128, args.epochs, 1e-3, 30, args.epochs
    loss_fn = CCMSELoss(use_weight=weighted_loss)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    ###############################################################
    # Wrap the model
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=True)
    ###############################################################

    # %% Read the pre-processed datasets
    with h5py.File(wave_preP, 'r') as f:
        X_train = f['pwave'][:]
        Y_train = f['noise'][:, (0 - npts):, :]
    X_sum = np.sum(np.square(X_train), axis=1)
    indX = np.where(X_sum == 0)[0]
    X_train = np.delete(X_train, indX, 0)
    Y_train = np.delete(Y_train, indX, 0)
    X_train = (X_train - np.mean(X_train, axis=1, keepdims=True)) / (np.std(X_train, axis=1, keepdims=True) + 1e-12)
    Y_train = (Y_train - np.mean(Y_train, axis=1, keepdims=True)) / (np.std(Y_train, axis=1, keepdims=True) + 1e-12)
    X_training,X_test,Y_training,Y_test=train_test_split(X_train,Y_train,train_size=train_size,random_state=rand_seed1)
    X_validate,X_test,Y_validate,Y_test=train_test_split(X_test, Y_test,  test_size=test_size, random_state=rand_seed2)
    # %% Convert to torch class.
    training_data = WaveformDataset(X_training, Y_training)
    validate_data = WaveformDataset(X_validate, Y_validate)
    test_data = WaveformDataset(X_test, Y_test)
    ################################################################
    train_sampler = DistributedSampler(training_data, num_replicas=args.world_size, rank=rank)
    valid_sampler = DistributedSampler(validate_data, num_replicas=args.world_size, rank=rank)
    test_sampler = DistributedSampler(test_data, num_replicas=args.world_size, rank=rank)
    ################################################################

    training_iter = DataLoader(dataset=training_data, batch_size=batch_size, shuffle=False, sampler=train_sampler)
    validate_iter = DataLoader(dataset=validate_data, batch_size=batch_size, shuffle=False, sampler=valid_sampler)
    test_iter = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, sampler=test_sampler)

    # %% training loop
    print("start training on GPU: ", gpu)
    model, loss, val_loss, partial_loss = training_loop_branches_augmentation(training_iter, validate_iter,
                                                                         model, loss_fn, optimizer,
                                                                         epochs=epochs, patience=patience,
                                                                         device=devc, minimum_epochs=minimum_epochs,
                                                                         npts=npts, mid_pt=mid_pt, strmax=strmax)

    # Calculate the test loss
    print("calculating test loss ...")
    test_loss = 0.0
    model.eval()
    for X0, y0 in test_iter:
        nbatch = X0.size(0)
        std_wgt = torch.ones(nbatch, dtype=torch.float64)
        quak2 = torch.zeros(nbatch, y0.size(1), npts * 2, dtype=torch.float64)
        quake = torch.zeros(y0.size(), dtype=torch.float64)
        stack = torch.zeros(y0.size(), dtype=torch.float64)

        rng = default_rng(17)
        rng_snr = default_rng(23)
        rng_sqz = default_rng(11)
        start_pt = rng.choice(npts - int(npts * 0.2), nbatch) + int(npts * 0.1)
        snr = 10 ** rng_snr.uniform(-0.3, 1, nbatch)
        sqz = rng_sqz.choice(strmax, nbatch) + 1
        pt1 = mid_pt - sqz * npts
        pt2 = mid_pt + sqz * npts

        for i in np.arange(nbatch):
            # %% squeeze earthquake signal
            quak2[i] = X0[i, :, pt1[i]:pt2[i]:sqz[i]]
            # %% shift earthquake signal
            tmp = quak2[i, :, start_pt[i]:start_pt[i] + npts]
            for j in np.arange(X0.size(1)):
                quake[i, j] = torch.div(torch.sub(tmp[j], torch.mean(tmp[j], dim=-1)),
                                        torch.std(tmp[j], dim=-1) + 1e-12) * snr[i]
            # %% stack signal and noise
            stack[i] = quake[i] + y0[i]
            # %% normalize
            scale_mean = torch.mean(stack[i], dim=1)
            scale_std = torch.std(stack[i], dim=1) + 1e-12
            std_wgt[i] = torch.nanmean(scale_std)
            for j in np.arange(X0.size(1)):
                stack[i, j] = torch.div(torch.sub(stack[i, j], scale_mean[j]), scale_std[j])
                quake[i, j] = torch.div(torch.sub(quake[i, j], scale_mean[j]), scale_std[j])

        X, y = stack.to(devc), quake.to(devc)
        snr = torch.from_numpy(snr).to(devc)
        std_wgt = std_wgt.to(devc)

        if len(y.data) != batch_size:
            break
        output1, output2 = model(X)
        loss_pred = loss_fn(output1, y, snr**2) + loss_fn(output2, X-y) + loss_fn(output1 + output2, X, std_wgt ** 2)
        test_loss += loss_pred.item() * X.size(0)

    test_loss = test_loss / len(test_iter.dataset)

    ###############################################################
    # Save in process 0
    if gpu == 0:
        elapse = time.time() - since
        print("Training is done! Time elapsed: %.2f s" % elapse)
        print("Saving model ...")
        torch.save(model, model_dir + f'/{model_name}_Model.pth')

        # Save split info for testing
        with h5py.File(model_dir + f'/{model_name}_Dataset_split.hdf5', 'w') as f:
            f.attrs['rand_seed1'] = rand_seed1
            f.attrs['rand_seed2'] = rand_seed2
            f.attrs['model_name'] = model_name
            f.attrs['train_size'] = train_size
            f.attrs['test_size'] = test_size

    print("Saving history ...")
    with h5py.File(model_dir + f'/{model_name}_Training_history_gpu' + str(gpu) + '.hdf5', 'w') as f:
        f.create_dataset("loss", data=loss)
        f.create_dataset("val_loss", data=val_loss)
        if model_structure == "Branch_Encoder_Decoder":
            f.create_dataset("test_loss", data=test_loss)
            f.create_dataset("noise_loss", data=partial_loss[2])
            f.create_dataset("noise_val_loss", data=partial_loss[3])
            f.create_dataset("earthquake_loss", data=partial_loss[0])
            f.create_dataset("earthquake_val_loss", data=partial_loss[1])
    print("Done! Exit process ", gpu)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
