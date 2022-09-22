"""
Apply the model to noise data
e.g. deep earthquakes

author: Qibin Shi
"""

import time
import matplotlib
import numpy as np
import pandas as pd
from denoiser_util import get_vp_vs
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 28})
since = time.time()

# %% Read the file of event durations
model_dir = 'Release_Middle_augmentation'
fig_dir = model_dir + '/Apply_releaseWDN_M5.5-8_deep100km_allSNR'
# fig_dir = model_dir
csv_file = fig_dir + '/source_measurements.csv'
meta_result = pd.read_csv(csv_file, low_memory=False)

mb = meta_result.source_magnitude.to_numpy()
dp = meta_result.source_depth_km.to_numpy().astype(np.float32)
dura_deno = meta_result.duration_denoised.to_numpy()
ctr_speed_deno = meta_result.centroid_speed_denoised.to_numpy()
ctr_dir_deno = meta_result.centroid_direction_denoised.to_numpy()
dura_noisy = meta_result.duration_noisy.to_numpy()
ctr_speed_noisy = meta_result.centroid_speed_noisy.to_numpy()
ctr_dir_noisy = meta_result.centroid_direction_noisy.to_numpy()
Es_denoised = meta_result.Es_denoised.to_numpy()
Es_noisy = meta_result.Es_noisy.to_numpy()
falloff_denoised = meta_result.falloff_denoised.to_numpy()
falloff_noisy = meta_result.falloff_noisy.to_numpy()
corner_freq_denoised = meta_result.corner_freq_denoised.to_numpy()
corner_freq_noisy = meta_result.corner_freq_noisy.to_numpy()
nsta = meta_result.num_station.to_numpy()

minsta = 10
mb = mb[nsta>minsta]
dp = dp[nsta>minsta]
dura_deno = dura_deno[nsta>minsta]
dura_noisy = dura_noisy[nsta>minsta]
ctr_speed_deno = ctr_speed_deno[nsta>minsta]
ctr_speed_noisy = ctr_speed_noisy[nsta>minsta]
ctr_dir_deno = ctr_dir_deno[nsta>minsta]
ctr_dir_noisy = ctr_dir_noisy[nsta>minsta]
Es_denoised = Es_denoised[nsta>minsta]
Es_noisy = Es_noisy[nsta>minsta]
falloff_denoised = falloff_denoised[nsta>minsta]
falloff_noisy = falloff_noisy[nsta>minsta]
corner_freq_denoised = corner_freq_denoised[nsta>minsta]
corner_freq_noisy = corner_freq_noisy[nsta>minsta]




mw = np.exp(0.741+0.21*mb)-0.785
lg_moment = (mw + 6.07) * 1.5
moment = 10 ** lg_moment

vp = np.ones(len(dp), dtype=np.float64)
vs = np.ones(len(dp), dtype=np.float64)
for i in range(len(dp)):
    vp[i], vs[i] = get_vp_vs(dp[i])

# %% Plot scaling relations
plt.close('all')
fig, ax = plt.subplots(6, 4, figsize=(40, 60), constrained_layout=True)
cmap = matplotlib.cm.jet.reversed()

# %% duration--mw
ax[0, 0].scatter(moment, dura_noisy, marker='o', s=200, c=dp, edgecolors='k', cmap=cmap)
im=ax[0, 1].scatter(moment, dura_deno, marker='o', s=200, c=dp, edgecolors='k', cmap=cmap)
plt.colorbar(im, ax=ax[0, 1])
ax[0, 0].set_yscale("log")
ax[0, 1].set_yscale("log")
ax[0, 0].set_xscale("log")
ax[0, 1].set_xscale("log")

# %% duration--depth
ts_clean = dura_deno * np.power(10, (19-lg_moment)/4) * vs / 4.5
ts_noisy = dura_noisy * np.power(10, (19-lg_moment)/4) * vs / 4.5
ax[0, 2].scatter(dp, ts_noisy, marker='o', s=200, c=lg_moment, edgecolors='k', cmap=cmap)
im=ax[0, 3].scatter(dp, ts_clean, marker='o', s=200, c=lg_moment, edgecolors='k', cmap=cmap)
plt.colorbar(im, ax=ax[0, 3])

# %% corner frequency
ax[1, 1].scatter(moment, 1/corner_freq_denoised, marker='o', s=200, c=dp, edgecolors='k', cmap=cmap)
im=ax[1, 0].scatter(moment, 1/corner_freq_noisy, marker='o', s=200, c=dp, edgecolors='k', cmap=cmap)
plt.colorbar(im, ax=ax[1, 1])
ax[1, 3].scatter(dp, 1/corner_freq_denoised, marker='o', s=200, c=lg_moment, edgecolors='k', cmap=cmap)
im=ax[1, 2].scatter(dp, 1/corner_freq_noisy, marker='o', s=200, c=lg_moment, edgecolors='k', cmap=cmap)
plt.colorbar(im, ax=ax[1, 3])
ax[1, 0].set_yscale("log")
ax[1, 1].set_yscale("log")
ax[1, 2].set_yscale("log")
ax[1, 3].set_yscale("log")
ax[1, 0].set_xscale("log")
ax[1, 1].set_xscale("log")

# %% centroid speed--mw
app_vr_clean = (ctr_speed_deno * 1)
app_vr_noisy = (ctr_speed_noisy * 1)
ax[2, 0].scatter(moment, app_vr_noisy, marker='o', s=200, c=dp, edgecolors='k', cmap=cmap)
im=ax[2, 1].scatter(moment, app_vr_clean, marker='o', s=200, c=dp, edgecolors='k', cmap=cmap)
plt.colorbar(im, ax=ax[2, 1])
ax[2, 2].scatter(dp, app_vr_noisy, marker='o', s=200, c=lg_moment, edgecolors='k', cmap=cmap)
im=ax[2, 3].scatter(dp, app_vr_clean, marker='o', s=200, c=lg_moment, edgecolors='k', cmap=cmap)
plt.colorbar(im, ax=ax[2, 3])
ax[2, 0].set_xscale("log")
ax[2, 1].set_xscale("log")

# %% Energy
ax[3, 1].scatter(moment, Es_denoised*moment*moment, marker='o', s=200, c=dp, edgecolors='k', cmap=cmap)
im=ax[3, 0].scatter(moment, Es_noisy*moment*moment, marker='o', s=200, c=dp, edgecolors='k', cmap=cmap)
plt.colorbar(im, ax=ax[3, 1])
ax[3, 3].scatter(dp, Es_denoised*moment*moment, marker='o', s=200, c=lg_moment, edgecolors='k', cmap=cmap)
im=ax[3, 2].scatter(dp, Es_noisy*moment*moment, marker='o', s=200, c=lg_moment, edgecolors='k', cmap=cmap)
plt.colorbar(im, ax=ax[3, 3])
ax[3, 0].set_yscale("log")
ax[3, 1].set_yscale("log")
ax[3, 2].set_yscale("log")
ax[3, 3].set_yscale("log")
ax[3, 0].set_xscale("log")
ax[3, 1].set_xscale("log")

ax[4, 1].scatter(moment, Es_denoised*moment, marker='o', s=200, c=dp, edgecolors='k', cmap=cmap)
im=ax[4, 0].scatter(moment, Es_noisy*moment, marker='o', s=200, c=dp, edgecolors='k', cmap=cmap)
plt.colorbar(im, ax=ax[4, 1])
ax[4, 3].scatter(dp, Es_denoised*moment, marker='o', s=200, c=lg_moment, edgecolors='k', cmap=cmap)
im=ax[4, 2].scatter(dp, Es_noisy*moment, marker='o', s=200, c=lg_moment, edgecolors='k', cmap=cmap)
plt.colorbar(im, ax=ax[4, 3])
ax[4, 0].set_yscale("log")
ax[4, 1].set_yscale("log")
ax[4, 2].set_yscale("log")
ax[4, 3].set_yscale("log")
ax[4, 0].set_xscale("log")
ax[4, 1].set_xscale("log")

# %% falloff power
ax[5, 1].scatter(moment, falloff_denoised, marker='o', s=200, c=dp, edgecolors='k', cmap=cmap)
im=ax[5, 0].scatter(moment, falloff_noisy, marker='o', s=200, c=dp, edgecolors='k', cmap=cmap)
plt.colorbar(im, ax=ax[5, 1])
ax[5, 3].scatter(dp, falloff_denoised, marker='o', s=200, c=lg_moment, edgecolors='k', cmap=cmap)
im=ax[5, 2].scatter(dp, falloff_noisy, marker='o', s=200, c=lg_moment, edgecolors='k', cmap=cmap)
plt.colorbar(im, ax=ax[5, 3])
ax[5, 0].set_xscale("log")
ax[5, 1].set_xscale("log")


# %% Fitting lines
lg_mg = np.arange(17.5, 23, 0.5, dtype=np.float64)
for k in range(2):

    lg_t = lg_mg / 4 - 4.2
    ax[0, k].plot(10**lg_mg, 10**lg_t, '-k', linewidth=5, alpha=1, label='M~T^4')
    ax[1, k].plot(10 ** lg_mg, 10 ** lg_t, '-k', linewidth=5, alpha=1, label='M~T^4')
    ax[0, k].legend(loc=1)
    lg_E = lg_mg * 1.17 - 7 - 0*k
    ax[3, k].plot(10**lg_mg, 10**lg_E, ':r', linewidth=5, alpha=1, label='Er~M^1.17 (Poli&Prieto)')
    lg_E = lg_mg + 2 -3*k
    ax[3, k].plot(10**lg_mg, 10**lg_E, ':k', linewidth=5, alpha=0.3, label='Er~M')
    lg_E = lg_mg - 6 +1*k
    ax[3, k].plot(10**lg_mg, 10**lg_E, ':k', linewidth=5, alpha=0.3)
    ax[3, k].legend(loc=0)
    lg_E = lg_mg / 4 - 6
    ax[4, k].plot(10**lg_mg, 10**lg_E, '-k', linewidth=5, alpha=1, label='Er~M^1/4')
    lg_E = lg_mg / 2 - 12
    ax[4, k].plot(10 ** lg_mg, 10 ** lg_E, '-k', linewidth=5, alpha=1, label='Er~M^1/2')
    ax[4, k].legend(loc=0)

ax[0, 0].set_ylim(1e-1, 1e2)
ax[0, 1].set_ylim(1e-1, 1e2)
ax[0, 2].set_ylim(0, 50)
ax[0, 3].set_ylim(0, 50)
ax[1, 0].set_ylim(1e-1, 1e3)
ax[1, 1].set_ylim(1e-1, 1e3)
ax[1, 2].set_ylim(1e-1, 1e3)
ax[1, 3].set_ylim(1e-1, 1e3)
ax[2, 0].set_ylim(-0.05, 0.6)
ax[2, 1].set_ylim(-0.05, 0.6)
ax[2, 2].set_ylim(-0.05, 0.6)
ax[2, 3].set_ylim(-0.05, 0.6)
ax[3, 0].set_ylim(1e12, 5e24)
ax[3, 1].set_ylim(1e12, 5e24)
ax[3, 2].set_ylim(1e12, 5e24)
ax[3, 3].set_ylim(1e12, 5e24)
ax[4, 0].set_ylim(1e-6, 1e1)
ax[4, 1].set_ylim(1e-6, 1e1)
ax[4, 2].set_ylim(1e-6, 1e1)
ax[4, 3].set_ylim(1e-6, 1e1)
ax[5, 0].set_ylim(0, 3)
ax[5, 1].set_ylim(0, 3)
ax[5, 2].set_ylim(0, 3)
ax[5, 3].set_ylim(0, 3)

ax[0, 0].set_ylabel('Duration', fontsize=30)
ax[2, 0].set_ylabel('Apparent speed/ Vp', fontsize=30)
ax[3, 0].set_ylabel('Radiated energy', fontsize=30)
ax[4, 0].set_ylabel('Radiated energy/moment', fontsize=30)
ax[5, 0].set_ylabel('Falloff power', fontsize=30)
ax[1, 0].set_ylabel('1/Corner frequency (Hz)', fontsize=30)
ax[5, 0].set_xlabel('Moment (Nm)', fontsize=30)
ax[5, 1].set_xlabel('Moment (Nm)', fontsize=30)

ax[0, 2].set_ylabel('Scaled duration', fontsize=30)
ax[5, 2].set_xlabel('Depth (km)', fontsize=30)
ax[5, 3].set_xlabel('Depth (km)', fontsize=30)

ax[0, 0].set_title('noisy data', fontsize=30)
ax[0, 1].set_title('denoised data', fontsize=30)
ax[0, 2].set_title('noisy data', fontsize=30)
ax[0, 3].set_title('denoised data', fontsize=30)

plt.savefig(fig_dir + '/scaling_duration_magnitude_depth.png')
