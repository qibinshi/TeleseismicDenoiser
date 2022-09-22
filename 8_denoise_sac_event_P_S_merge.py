import glob
import torch
import matplotlib
import numpy as np
from obspy import read, UTCDateTime
from torch_tools import WaveformDataset, try_gpu
from torch.utils.data import DataLoader
from denoiser_util import mkdir
matplotlib.use('Agg')
import matplotlib.pyplot as plt

nptP = 3000
nptS = 7500
preP = 150.0
preS = 300.0
aftP = nptP * 0.1 - preP
aftS = nptS * 0.1 - preS
devc = try_gpu(i=10)
workdir = '/Users/qibin/Projects/MX22/inv/'
# sac_dir = 'Petrolia_tel'
# sac_dir = 'groun_vel_aceh'
sac_dir = 'MX_rtz'
mkdir(workdir + 'denoised_mergedPS_' + sac_dir)
P_model = 'Release_Middle_augmentation/Branch_Encoder_Decoder_LSTM_Model.pth'
S_model = 'Release_Middle_augmentation_S_batchsize512/Branch_Encoder_Decoder_LSTM_Model.pth'

all_P = np.zeros((0, nptP, 3), dtype=np.double)
all_S = np.zeros((0, nptS, 3), dtype=np.double)
scalingP = np.zeros((0, 1, 3), dtype=np.double)
scalingS = np.zeros((0, 1, 3), dtype=np.double)
sta_idx = []


# %% Prepare P and S separately. Need to mark tp as a and ts as t0
stalist = glob.glob(workdir + sac_dir + '/*.z')
print('Number of stations: ', len(stalist))
for j in range(len(stalist)):
    sta = stalist[j]
    try:
        st0 = read(sta[:-1]+'?')
        stp = st0.copy()
        sts = st0.copy()
    except:
        continue

    if len(stp) >= 3:
        tp = UTCDateTime(stp[2].stats.starttime - stp[2].stats.sac.b + stp[2].stats.sac.a)
        ts = UTCDateTime(sts[2].stats.starttime - sts[2].stats.sac.b + sts[2].stats.sac.t0)
        stp.trim(tp - preP, tp + aftP)
        sts.trim(ts - preS, ts + aftS)

        one_P = np.zeros((nptP, 3), dtype=np.double)
        one_S = np.zeros((nptS, 3), dtype=np.double)
        for i in range(3):
            one_P[:, i] = np.array(stp[i].data)[0:nptP]
            one_S[:, i] = np.array(sts[i].data)[0:nptS]
        one_P[np.isnan(one_P)] = 0
        one_S[np.isnan(one_S)] = 0

        scale_mean = np.mean(one_P, axis=0, keepdims=True)
        scale_stdv = np.std(one_P, axis=0, keepdims=True) + 1e-12
        one_P = (one_P - scale_mean) / scale_stdv
        all_P = np.append(all_P, one_P[np.newaxis, :, :], axis=0)
        scalingP = np.append(scalingP, scale_stdv[np.newaxis, :, :], axis=0)

        scale_mean = np.mean(one_S, axis=0, keepdims=True)
        scale_stdv = np.std(one_S, axis=0, keepdims=True) + 1e-12
        one_S = (one_S - scale_mean) / scale_stdv
        all_S = np.append(all_S, one_S[np.newaxis, :, :], axis=0)
        scalingS = np.append(scalingS, scale_stdv[np.newaxis, :, :], axis=0)

        sta_idx.append(j)
    else:
        print('Failed to convert station: ', sta[:-4])

# %% after all stations are converted to numpy
batch_size = all_P.shape[0]
test_data = WaveformDataset(all_P, all_S)
test_iter = DataLoader(test_data, batch_size=batch_size, shuffle=False)
data_iter = iter(test_iter)
X, Y = data_iter.next()

# %% Load Denoisers
model_P = torch.load(P_model, map_location=devc)
model_P = model_P.module.to(devc)
model_P.eval()
with torch.no_grad():
    P_denoised, noise_P = model_P(X)

model_S = torch.load(S_model, map_location=devc)
model_S = model_S.module.to(devc)
model_S.eval()
with torch.no_grad():
    S_denoised, noise_S = model_S(Y)

denoised_P = P_denoised.numpy()
denoised_S = S_denoised.numpy()

# %% Merge denoised P and S waves, and rewrite
print('Number of denoised stations: ', denoised_P.shape[0])
merge = np.zeros((denoised_P.shape[0], 15000, 3), dtype=np.double)

for j in range(denoised_P.shape[0]):
    idx = sta_idx[j]
    sta = stalist[idx]

    try:
        st0 = read(sta[:-1]+'?')
        st = st0.copy()
    except:
        continue

    for i in range(3):

        pt_diff = int((st[i].stats.sac.t0 - preS - st[i].stats.sac.a + preP) / st[i].stats.sac.delta)
        merge[j, :nptP, i] = denoised_P[j, i, :] * scalingP[j, 0, i]
        merge[j, pt_diff:pt_diff+nptS, i] = denoised_S[j, i, :] * scalingS[j, 0, i]

        st[i].data = merge[j, :, i]
        nt = st[i].stats.network
        nm = st[i].stats.station
        lc = st[i].stats.location
        ch = st[i].stats.channel

        st[i].stats.starttime = UTCDateTime(st[i].stats.starttime - st[i].stats.sac.b + st[i].stats.sac.a - preP)

        st[i].write(workdir + 'denoised_mergedPS_' + sac_dir + '/' + nt + '.' + nm + '.' + lc + '.' + ch, format='SAC')




