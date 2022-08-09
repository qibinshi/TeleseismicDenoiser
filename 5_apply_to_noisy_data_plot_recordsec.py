"""
Apply the model to noise data
e.g. deep earthquakes

author: Qibin Shi
"""
import os
import time
import h5py
import torch
import matplotlib
import numpy as np
import pandas as pd
from functools import partial
from multiprocessing import Pool
from torch_tools import try_gpu
from denoiser_util import mkdir, plot_record_section

matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 18})

# %%
dt = 0.1
npts = 3000  #P
npts_trim = 1000
# npts = 7500  #S
# npts_trim = 1000
devc = try_gpu(i=10)  # for parallel plotting, let's only use CPUs
datadir = '/fd1/QibinShi_data/matfiles_for_denoiser/'
csv_file = datadir + "metadata_M6_deep250km_SNRmax10_P.csv"
wave_mat = datadir + 'M6_deep250km_SNRmax10_P.hdf5'
model_dir = 'Release_Middle_augmentation'
fig_dir = model_dir + '/Apply_releaseWDN_M6_deep250km_SNRmax10'
mkdir(fig_dir)

# %% Read the noisy waveforms of multiple events
with h5py.File(wave_mat, 'r') as f:
    X_train = f['pwave'][:]

# %% Load Denoiser
model = torch.load(model_dir + '/Branch_Encoder_Decoder_LSTM_Model.pth')
model = model.module.to(devc)
model.eval()

# %% Read the metadata
meta_all = pd.read_csv(csv_file, low_memory=False)
evids = meta_all.source_id.unique()
since = time.time()

####################### Save the denoised traces to disk #########################
partial_func = partial(plot_record_section, meta_all=meta_all, X_train=X_train, model=model, fig_dir=fig_dir, dt=dt, npts=npts, npts_trim=npts_trim, normalized_stack=True)
num_proc = min(os.cpu_count(), 10)
print('---- %d threads for plotting %d record sections\n' % (num_proc, len(evids)))
with Pool(processes=num_proc) as pool:
    pool.map(partial_func, evids)

print('---- All are plotted, using %.2f s\n' % (time.time() - since))
