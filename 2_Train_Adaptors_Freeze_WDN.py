"""
Transfer learning by adding input and output layers
to the pre-trained WaveDecompNet in the middle

@auther: Qibin Shi (qibins@uw.edu)
"""
import h5py
import torch
import random
import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from utilities import mkdir, write_progress
from sklearn.model_selection import train_test_split
from torch_tools import WaveformDataset, try_gpu, training_loop_branches, CCMSELoss
from autoencoder_1D_models_torch import InputLinear, InputConv, OutputLinear, OutputDconv

# %%
gpu_num = 0
devc = try_gpu(i=gpu_num)
bottleneck_name = 'LSTM'
model_dir = 'Freeze_Middle'
model_structure = "Branch_Encoder_Decoder"
progress_file = model_dir + '/Running_progress.txt'
model_name = model_structure + "_" + bottleneck_name
wave_mat = './data_stacked_POHA_Ponly_2004_18_shallow_snr_25_sample10Hz_lowpass2Hz.mat'
mkdir(model_dir)
# %% Read the pre-processed datasets
print("#" * 12 + " Loading data " + "#" * 12)
X_train = loadmat(wave_mat)["stack_waves"]
Y_train = loadmat(wave_mat)["quake_waves"]

train_size = 0.6  # 60% for training
test_size = 0.5  # (1-80%) x 50% for testing
rand_seed1 = 13
rand_seed2 = 20
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

Linear_pre0 = InputLinear(600, 600)
Linear_pre1 = InputLinear(2400, 1200)
Linear_pre2 = InputLinear(1200, 600)
Conv0 = InputConv(3, 3, 9, 4, 4)
Conv1 = InputConv(3, 8, 9, 1, 'same')
Conv2 = InputConv(8, 8, 9, 2, 4)
Conv3 = InputConv(8, 8, 9, 1, 'same')
Conv4 = InputConv(8, 3, 9, 1, 'same')
Dconv4 = OutputDconv(3, 8, 9, 1, 4, 0)
Dconv3 = OutputDconv(8, 8, 9, 1, 4, 0)
Dconv2 = OutputDconv(8, 8, 9, 2, 4, 1)
Dconv1 = OutputDconv(8, 3, 9, 1, 4, 0)
Dconv0 = OutputDconv(3, 3, 9, 4, 3, 1)
Linear_post2 = OutputLinear(600, 1200)
Linear_post1 = OutputLinear(1200, 2400)
Linear_post0 = OutputLinear(600, 600)

for param in model.parameters():
    param.requires_grad = False

model = torch.nn.Sequential(Conv1, Conv2, Conv4, Linear_pre2, model, Linear_post2, Dconv4, Dconv2, Dconv1).to(devc)

n_para = 0
for idx, param in enumerate(model.parameters()):
    if not param.requires_grad:
        print(idx, param.shape)
    else:
        n_para += np.prod(param.shape)
print(f'Number of parameters to be trained: {n_para}\n')

# %% Hyper-parameters for training
batch_size, epochs, lr = 256, 200, 1e-3
minimum_epochs, patience = 30, 8  # patience for early stopping
#loss_fn = torch.nn.MSELoss()
loss_fn = CCMSELoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
train_iter = DataLoader(training_data, batch_size=batch_size, shuffle=False)
validate_iter = DataLoader(validate_data, batch_size=batch_size, shuffle=False)
test_iter = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# %% Loop for training
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
for X, y in test_iter:
    if len(y.data) != batch_size:
        break
    output1, output2 = model(X)
    loss = loss_fn(output1, y) + loss_fn(output2, X - y)
    test_loss += loss.item() * X.size(0)

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
