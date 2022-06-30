"""
Transfer learning for WaveDecompNet
with extended 2-branch U-net layers,
Data being augmented on the fly.

@auther: Qibin Shi (qibins@uw.edu)
"""
import h5py
import torch
import random
import numpy as np
from matplotlib import pyplot as plt
from numpy.random import default_rng
from torch.utils.data import DataLoader
from denoiser_util import mkdir, write_progress
from sklearn.model_selection import train_test_split
from torch_tools import WaveformDataset, try_gpu, CCMSELoss, MSELossOnly
from torch_tools import training_loop_branches_augmentation
from autoencoder_1D_models_torch import T_model, W_model

# %%
weighted_loss = False
pttp = 10000
npts = 3000
gpu_num = 0
devc = try_gpu(i=gpu_num)
bottleneck_name = 'LSTM'
model_dir = 'Freeze_Middle_augmentation'
model_structure = "Branch_Encoder_Decoder"
progress_file = model_dir + '/Running_progress.txt'
model_name = model_structure + "_" + bottleneck_name
datadir = '/mnt/DATA0/qibin_data/matfiles_for_denoiser/'
wave_preP = datadir + 'Alldepths_snr25_2000_21_sample10_lpass2_P_preP_MP1.hdf5'
wave_stead = datadir + 'Alldepths_snr25_2000_21_sample10_lpass2_P_STEAD_MP1.hdf5'
mkdir(model_dir)

# %% Read the pre-processed datasets
print("#" * 12 + " Loading P wave and pre-P noises " + "#" * 12)

with h5py.File(wave_preP, 'r') as f:
    X_train = f['pwave'][:]
    Y_train = f['noise'][:, (0 - npts):, :]

X_sum = np.sum(np.square(X_train), axis=1)
indX = np.where(X_sum == 0)[0]
X_train = np.delete(X_train, indX, 0)
Y_train = np.delete(Y_train, indX, 0)

print("#" * 12 + " Loading P wave and STEAD noises " + "#" * 12)
with h5py.File(wave_stead, 'r') as f:
    X_stead = f['pwave'][:]
    Y_stead = f['noise'][:, (0 - npts):, :]
X_train = np.append(X_train, X_stead, axis=0)
Y_train = np.append(Y_train, Y_stead, axis=0)

print("#" * 12 + " Normalizing P wave and noises " + "#" * 12)
X_train = (X_train - np.mean(X_train, axis=1, keepdims=True)) / (np.std(X_train, axis=1, keepdims=True) + 1e-12)
Y_train = (Y_train - np.mean(Y_train, axis=1, keepdims=True)) / (np.std(Y_train, axis=1, keepdims=True) + 1e-12)

train_size = 0.6  # 60% for training
test_size = 0.5  # (1-80%) x 50% for testing
# rand_seed1 = 13
# rand_seed2 = 20
rand_seed1 = 43
rand_seed2 = 11
X_training,X_test,Y_training,Y_test=train_test_split(X_train,Y_train,train_size=train_size,random_state=rand_seed1)
X_validate,X_test,Y_validate,Y_test=train_test_split(X_test, Y_test,  test_size=test_size, random_state=rand_seed2)
# %% Convert to torch class. Or WaveformDataset_h5 for limited memory
training_data = WaveformDataset(X_training, Y_training)
validate_data = WaveformDataset(X_validate, Y_validate)
test_data     = WaveformDataset(X_test, Y_test)

# %% Give a fixed seed for model initialization
random.seed(0)
np.random.seed(20)
torch.manual_seed(99)
torch.backends.cudnn.benchmark = False

# %% Set up Neural Net structure
print("#" * 12 + " Loading model " + model_name + " " + "#" * 12)

model = torch.load('Model_and_datasets_1D_all_snr_40' + f'/{model_name}/{model_name}_Model.pth', map_location=devc)

for param in model.parameters():
    param.requires_grad = False

model = T_model(model).to(devc)

n_para = 0
for idx, param in enumerate(model.parameters()):
    if not param.requires_grad:
        print(idx, param.shape)
    else:
        n_para += np.prod(param.shape)
print(f'Number of parameters to be trained: {n_para}\n')

# %% Hyper-parameters for training
batch_size, epochs, lr = 512, 200, 1e-3
minimum_epochs, patience = 30, 20  # patience for early stopping
loss_fn = CCMSELoss(use_weight=weighted_loss)
# loss_fn = MSELossOnly(use_weight=weighted_loss)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
train_iter = DataLoader(training_data, batch_size=batch_size, shuffle=False)
validate_iter = DataLoader(validate_data, batch_size=batch_size, shuffle=False)
test_iter = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# %% Loop for training
print("#" * 12 + " training model " + model_name + " " + "#" * 12)

model, avg_train_losses, avg_valid_losses, partial_loss = training_loop_branches_augmentation(train_iter, validate_iter,
                                                                                 model, loss_fn, optimizer,
                                                                                 epochs=epochs, patience=patience,
                                                                                 device=devc,
                                                                                 minimum_epochs=minimum_epochs,
                                                                                 npts=npts, pttp=pttp)
print("Training is done!")
write_progress(progress_file, text_contents="Training is done!" + '\n')

# %% Save the model
torch.save(model, model_dir + f'/{model_name}_Model.pth')

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
    start_pt = rng.choice(npts - int(npts * 0.05), nbatch) + int(npts * 0.05)
    snr = 10 ** rng_snr.uniform(-1, 1, nbatch)
    sqz = rng_sqz.choice(2, nbatch) + 1
    pt1 = pttp - sqz * npts
    pt2 = pttp + sqz * npts

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
