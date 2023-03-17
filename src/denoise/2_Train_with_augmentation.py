"""
Stack WaveDecompNet kernel with
2xCNN and 1xFCNN layers and
perform transfer learning with
data augmented on the fly.

@auther: Qibin Shi (qibins@uw.edu)
"""
import h5py
import torch
import random
import configparser
import numpy as np
import torch.nn as nn
from matplotlib import pyplot as plt
from numpy.random import default_rng
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch_tools import WaveformDataset, try_gpu, CCMSELoss
from torch_tools import training_loop_branches_augmentation
from denoiser_util import mkdir, write_progress
from autoencoder_1D_models_torch import T_model


def train(configure_file='config.ini'):
    config = configparser.ConfigParser()
    config.read(configure_file)

    transfer = config.getint('training', 'transfer')
    gpu = config.getint('training', 'gpu')
    storage_home = config.get('directories', 'storage_home')
    data_dir = storage_home + config.get('directories', 'data_dir')
    model_dir = storage_home + config.get('directories', 'save_model_dir')
    data_file = config.get('data', 'data_file')
    half_length = config.getint('data', 'half_length')
    train_size = config.getfloat('training', 'train_size')
    test_size = config.getfloat('training', 'test_size')
    rand_seed1 = config.getint('training', 'rand_seed1')
    rand_seed2 = config.getint('training', 'rand_seed2')
    batch_size = config.getint('training', 'batch_size')
    lr = config.getfloat('training', 'learning_rate')
    epochs = config.getint('training', 'epochs')
    minimum_epochs = config.getint('training', 'minimum_epochs')
    patience = config.getint('training', 'patience')
    pre_trained_denote = config.get('directories', 'pre_trained_denote')
    pre_trained_WaveDecompNet = config.get('directories', 'pre_trained_WaveDecompNet')

    print('transfer', transfer)
    print('gpu', gpu)
    print('data_dir', data_dir)
    print('model_dir', model_dir)
    print('half_length', half_length)
    print('train_size', train_size)
    print('test_size', test_size*(1-train_size))
    print('rand_seed', rand_seed1, rand_seed2)
    print('batch_size', batch_size)
    print('learning rate', lr)
    print('epochs', epochs)
    print('minimum_epochs', minimum_epochs)
    print('patience', patience)

    wave_raw = data_dir + data_file
    mid_pt = half_length
    npts = 1500
    strmax = 6

    progress_file = model_dir + '/Running_progress.txt'
    mkdir(model_dir)
    frac = 0.1  # starting fraction not included in shifting window
    weighted_loss = False
    bottleneck_name = 'LSTM'
    model_structure = "Branch_Encoder_Decoder"
    model_name = model_structure + "_" + bottleneck_name

    # %% Read the pre-processed datasets
    print("#" * 12 + " Loading quake signals and pre-signal noises " + "#" * 12)
    with h5py.File(wave_raw, 'r') as f:
        X_train = f['pwave'][:]
        Y_train = f['noise'][:, (0 - npts):, :]

    X_sum = np.sum(np.sum(np.square(X_train), axis=1), axis=1)
    ind_X = np.where(X_sum == np.nan)[0]
    X_train = np.delete(X_train, ind_X, 0)
    Y_train = np.delete(Y_train, ind_X, 0)
    X_sum = np.sum(np.square(X_train), axis=1)
    ind_X = np.unique(np.where(X_sum == 0)[0])
    X_train = np.delete(X_train, ind_X, 0)
    Y_train = np.delete(Y_train, ind_X, 0)

    print("#" * 12 + " Normalizing signal and noises " + "#" * 12)
    X_train = (X_train - np.mean(X_train, axis=1, keepdims=True)) / (np.std(X_train, axis=1, keepdims=True) + 1e-12)
    Y_train = (Y_train - np.mean(Y_train, axis=1, keepdims=True)) / (np.std(Y_train, axis=1, keepdims=True) + 1e-12)

    # %% Split datasets into train, valid and test sets
    X_training, X_test, Y_training, Y_test = train_test_split(X_train, Y_train,
                                                              train_size=train_size,
                                                              random_state=rand_seed1)
    X_validate, X_test, Y_validate, Y_test = train_test_split(X_test, Y_test,
                                                              test_size=test_size,
                                                              random_state=rand_seed2)

    # %% Convert to torch class. Or WaveformDataset_h5 for limited memory
    training_data = WaveformDataset(X_training, Y_training)
    validate_data = WaveformDataset(X_validate, Y_validate)
    test_data = WaveformDataset(X_test, Y_test)

    # %% Fix seed for model initialization
    random.seed(0)
    np.random.seed(20)
    torch.manual_seed(99)
    torch.backends.cudnn.benchmark = False

    # %% Neural Net structure
    print("#" * 12 + " Loading model " + model_name + " " + "#" * 12)
    devc = try_gpu(i=gpu)
    if transfer:
        # %% transfer learning
        if torch.cuda.device_count() > gpu:
            model = torch.load(pre_trained_denote, map_location=devc)
            model.to(devc)
        else:
            model = torch.load(pre_trained_denote, map_location=try_gpu(i=1))
            model = model.module.to(try_gpu(i=10))
    else:
        if torch.cuda.device_count() > gpu:
            model = torch.load(pre_trained_WaveDecompNet, map_location=devc)

            # for param in model.parameters():
            #     param.requires_grad = False

            # %% Data parallelism for multiple GPUs,
            # %! model=model.module.to(device) for application on CPU
            model = T_model(model)
            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                model = nn.DataParallel(model, device_ids=[gpu])
                # model = nn.DataParallel(model)
                model.to(devc)
        else:
            model = torch.load(pre_trained_WaveDecompNet, map_location=try_gpu(i=1))
            model = model.module.to(devc)


    n_para = 0
    for idx, param in enumerate(model.parameters()):
        if not param.requires_grad:
            print(idx, param.shape)
        else:
            n_para += np.prod(param.shape)
    print(f'Number of parameters to be trained: {n_para}\n')

    # %% Hyper-parameters for training
    loss_fn = CCMSELoss(use_weight=weighted_loss)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    train_iter = DataLoader(training_data, batch_size=batch_size, shuffle=False)
    validate_iter = DataLoader(validate_data, batch_size=batch_size, shuffle=False)
    test_iter = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # %% Loop for training
    print("#" * 12 + " training model " + model_name + " " + "#" * 12)

    model, avg_train_losses, avg_valid_losses, partial_loss = training_loop_branches_augmentation(train_iter,
                                                                                                  validate_iter,
                                                                                                  model,
                                                                                                  loss_fn,
                                                                                                  optimizer,
                                                                                                  epochs=epochs,
                                                                                                patience=patience,
                                                                                                  device=devc,
                                                                                          minimum_epochs=minimum_epochs,
                                                                                                    npts=npts,
                                                                                                  mid_pt=mid_pt,
                                                                                                  strmax=strmax)
    print("Training is done!")
    write_progress(progress_file, text_contents="Training is done!" + '\n')

    # %% Save the model
    torch.save(model.state_dict(), model_dir + f'/{model_name}_weights.pth')

    # %% Save the training history
    loss = avg_train_losses
    val_loss = avg_valid_losses
    with h5py.File(model_dir + f'/{model_name}_Training_history.hdf5', 'w') as f:
        f.create_dataset("loss", data=loss)
        f.create_dataset("val_loss", data=val_loss)
        if model_structure == "Branch_Encoder_Decoder":
            f.create_dataset("earthquake_loss", data=partial_loss[0])
            f.create_dataset("earthquake_val_loss", data=partial_loss[1])
            f.create_dataset("noise_loss", data=partial_loss[2])
            f.create_dataset("noise_val_loss", data=partial_loss[3])

    # %% Save the training info
    with h5py.File(model_dir + f'/{model_name}_Dataset_split.hdf5', 'w') as f:
        f.attrs['model_name'] = model_name
        f.attrs['train_size'] = train_size
        f.attrs['test_size'] = test_size
        f.attrs['rand_seed1'] = rand_seed1
        f.attrs['rand_seed2'] = rand_seed2

    # Calculate the test loss
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
        start_pt = rng.choice(npts - int(npts * frac * 2.0), nbatch) + int(npts * frac)
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
        loss_pred = loss_fn(output1, y, snr**2) + loss_fn(output2, X - y) + loss_fn(output1 + output2, X, std_wgt**2)
        test_loss += loss_pred.item() * X.size(0)

    test_loss = test_loss/len(test_iter.dataset)

    # %% Show loss evolution when training is done
    plt.close('all')
    plt.figure()
    plt.plot(loss, 'o', label='loss')
    plt.plot(val_loss, '-', label='Validation loss')
    plt.plot([len(loss)], [test_loss], 'r*', label=f'Test loss = {test_loss:.4f}', markersize=10, linewidth=2, zorder=10)

    loss_name_list = ['earthquake train loss', 'earthquake valid loss', 'noise train loss', 'noise valid loss']
    loss_plot_list = ['o', '', 'o', '']
    for ii in range(4):
        plt.plot(partial_loss[ii], marker=loss_plot_list[ii], label=loss_name_list[ii])

    plt.legend()
    plt.title(model_name)
    plt.savefig(model_dir + f'/{model_name}_Training_history.png')


if __name__ == '__main__':
    train()
