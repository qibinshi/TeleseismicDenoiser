# %%
import os
import h5py
import copy
import torch
import random
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import savemat, loadmat
from torch.utils.data import DataLoader
from utilities import mkdir, write_progress
from sklearn.model_selection import train_test_split
from torch_tools import WaveformDataset, EarlyStopping, try_gpu, training_loop, training_loop_branches
from autoencoder_1D_models_torch import Autoencoder_Conv1D, Autoencoder_Conv2D, Attention_bottleneck, \
    Attention_bottleneck_LSTM, SeismogramEncoder, SeismogramDecoder, SeisSeparator

gpu_num=1
devc = try_gpu(i=gpu_num)
wave_mat = './data_stacked_M6_plus_POHA.mat'
model_dir = 'Finetune_Model'
mkdir(model_dir)
progress_file = model_dir + '/Running_progress.txt'
model_structure = "Branch_Encoder_Decoder"
bottleneck_name = 'LSTM'
model_name = model_structure + "_" + bottleneck_name

# %% Recording progress to a text file
write_progress(progress_file, text_contents = "=" * 5 + " Working on " + bottleneck_name + "=" * 5  + '\n')

# %% Read the pre-processed datasets
print("#" * 12 + " Loading data " + "#" * 12)
X_train = loadmat(wave_mat)["stack_waves"]
Y_train = loadmat(wave_mat)["quake_waves"]

train_size = 0.6 # 60% for training
test_size  = 0.5 # (1-60%) x 50% for testing
rand_seed1 = 13
rand_seed2 = 20
X_training,X_test,Y_training,Y_test=train_test_split(X_train,Y_train,train_size=train_size,random_state=rand_seed1)
X_validate,X_test,Y_validate,Y_test=train_test_split(X_test, Y_test, test_size = test_size,random_state=rand_seed2)
# %% Convert to the torch class, use WaveformDataset_h5 for small memory
training_data = WaveformDataset(X_training, Y_training)
validate_data = WaveformDataset(X_validate, Y_validate)

# %% Give a fixed seed for model initialization
# torch.manual_seed(99)
# random.seed(0)
# np.random.seed(20)
# torch.backends.cudnn.benchmark = False

# %% Set up model network
print("#" * 12 + " Building model " + model_name + " " + "#" * 12)
# encoder = SeismogramEncoder()
# bottleneck = torch.nn.LSTM(64, 32, 2, bidirectional=True, batch_first=True, dtype=torch.float64)
# bottleneck_quake = copy.deepcopy(bottleneck)
# bottleneck_noise = copy.deepcopy(bottleneck)
# decoder_quake = SeismogramDecoder(bottleneck=bottleneck_quake)
# decoder_noise = SeismogramDecoder(bottleneck=bottleneck_noise)
# model = SeisSeparator(model_name, encoder, decoder_quake, decoder_noise).to(devc)

model = torch.load('Model_and_datasets_1D_all_snr_40'+f'/{model_name}/{model_name}_Model.pth', map_location=devc)
# %% Tune training parameters
batch_size, epochs, lr = 32, 200, 1e-4
minimum_epochs = 30
patience = 5  # for early stopping
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
train_iter    = DataLoader(training_data, batch_size=batch_size, shuffle=False)
validate_iter = DataLoader(validate_data, batch_size=batch_size, shuffle=False)

print("#" * 12 + " training model " + model_name + " " + "#" * 12)

model, avg_train_losses, avg_valid_losses, partial_loss = training_loop_branches(train_iter, validate_iter,
                                                                                     model, loss_fn, optimizer,
                                                                                     epochs=epochs, patience=patience,
                                                                                     device=devc,
                                                                                     minimum_epochs=minimum_epochs)
print("Training is done!")
write_progress(progress_file, text_contents="Training is done!" + '\n')

# %% Save the model
torch.save(model, model_dir + f'/{model_name}_Model.pth')

loss = avg_train_losses
val_loss = avg_valid_losses
# %% Save the training history
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

# %% Show loss evolution when training is done
plt.close('all')
plt.figure()
plt.plot(loss, 'o', label='loss')
plt.plot(val_loss, '-', label='Validation loss')

loss_name_list = ['earthquake train loss', 'earthquake valid loss', 'noise train loss', 'noise valid loss']
loss_plot_list = ['o', '', 'o', '']
for ii in range(4):
    plt.plot(partial_loss[ii], marker=loss_plot_list[ii], label=loss_name_list[ii])

plt.legend()
plt.title(model_name)
plt.savefig(model_dir + f'/{model_name}_Training_history.png')
