"""
Apply the model to noise data
e.g. deep earthquakes

author: Qibin Shi
"""

import time
import glob
import h5py
import matplotlib
import numpy as np
import pandas as pd
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['axes.axisbelow'] = True
matplotlib.rcParams.update({'font.size': 24})
from scipy.interpolate import interp1d
from numpy.random import default_rng


# %% grid search the best lg_fc and falloff
def model_spec(logspec, f_mod, dn=0.1, df=0.01):
    l2_min = 10000.0
    n_best = 2
    fc_best = -2
    for n in np.arange(1.0, 3.1, dn, dtype=np.float64):
        for fc in np.arange(-2.0, 0.4, df, dtype=np.float64):
            model_func = 10**logspec[0] / (1.0 + 10 ** ((f_mod - fc) * n))
            l2 = np.sum(np.square(logspec[:len(f_mod)] - np.log10(model_func)))
            l2 = np.sum(np.fabs(logspec[:len(f_mod)] - np.log10(model_func)))
            if l2 < l2_min:
                l2_min = l2
                n_best = n
                fc_best = fc

    return n_best, fc_best


minsta = 10
since = time.time()
# %% Read the file of event durations
model_dir = 'Release_Middle_augmentation_S4Hz_150s_removeCoda_SoverCoda25'
fig_dir = model_dir + '/Apply_releaseWDN_M5.5-8_deep100km_SNR2_azimuthBined_fixedWindow'
csv_file = fig_dir + '/source_measurements.csv'
meta_result = pd.read_csv(csv_file, low_memory=False)

evid = meta_result.source_id.to_numpy()
mb = meta_result.source_magnitude.to_numpy()
nsta = meta_result.num_station.to_numpy()
Es_deno = meta_result.Es_denoised.to_numpy()
Es_noisy = meta_result.Es_noisy.to_numpy()
dura_deno = meta_result.duration_denoised.to_numpy()
dura_noisy = meta_result.duration_noisy.to_numpy()
falloff_deno = meta_result.falloff_denoised.to_numpy()
falloff_noisy = meta_result.falloff_noisy.to_numpy()
corner_freq_deno = meta_result.corner_freq_denoised.to_numpy()
corner_freq_noisy = meta_result.corner_freq_noisy.to_numpy()
ctr_speed_noisy = meta_result.centroid_speed_noisy.to_numpy()
ctr_speed_deno = meta_result.centroid_speed_denoised.to_numpy()
ctr_dir_noisy = meta_result.centroid_direction_noisy.to_numpy()
ctr_dir_deno = meta_result.centroid_direction_denoised.to_numpy()
depth = meta_result.source_depth_km.to_numpy().astype(np.float32)

ind_select = np.where(np.logical_and(depth > 300.0, nsta > minsta))[0]

mb = mb[ind_select]
evid = evid[ind_select]
depth = depth[ind_select]
Es_deno = Es_deno[ind_select]
Es_noisy = Es_noisy[ind_select]
dura_deno = dura_deno[ind_select]
dura_noisy = dura_noisy[ind_select]
falloff_deno = falloff_deno[ind_select]
falloff_noisy = falloff_noisy[ind_select]
ctr_dir_deno = ctr_dir_deno[ind_select]
ctr_dir_noisy = ctr_dir_noisy[ind_select]
ctr_speed_deno = ctr_speed_deno[ind_select]
ctr_speed_noisy = ctr_speed_noisy[ind_select]
corner_freq_deno = corner_freq_deno[ind_select]
corner_freq_noisy = corner_freq_noisy[ind_select]

mw = np.exp(0.741+0.21*mb)-0.785
# mw = mb

N_bootstrap = 50
mw_group = ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '']
event_group = ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '']
fc_deno_group = ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '']

for j in range(6):
    mw_group[j] = mw[np.logical_and(mw <= 8.0-j*0.4, mw > 7.6-j*0.4)]
    event_group[j] = evid[np.logical_and(mw <= 8.0-j*0.4, mw > 7.6-j*0.4)]
    fc_deno_group[j] = corner_freq_deno[np.logical_and(mw <= 8.0-j*0.4, mw > 7.6-j*0.4)]

# %% Plot scaling relations
plt.close('all')
fig, ax = plt.subplots(1, 2, figsize=(20, 10), constrained_layout=True)

f_new = np.arange(np.log10(0.02), np.log10(5.0), 0.01)
f_mod = np.arange(np.log10(0.02), np.log10(0.7), 0.01)

for j in np.arange(0, 6):
    # %% bootstrapping
    bootstrap_sum_deno = np.zeros(len(f_new), dtype=np.float64)
    bootstrap_sum_noisy = np.zeros(len(f_new), dtype=np.float64)
    bootstrap_sum_m0 = 0.0
    N_events = int(len(mw_group[j]) * 0.5)
    for k in range(N_bootstrap):

        sum_deno = np.zeros(len(f_new), dtype=np.float64)
        sum_noisy = np.zeros(len(f_new), dtype=np.float64)
        sum_m0 = 0.0
        print(len(mw_group[j]))
        rng = default_rng((j+1)*(k+1)+1)
        eve_ind = rng.choice(len(mw_group[j]), N_events, replace=False)
        mw_list = mw_group[j][eve_ind]
        eve_list = event_group[j][eve_ind]
        fc_list = fc_deno_group[j][eve_ind]


        for i in range(N_events):

            event = int(eve_list[i])

            # %% Read the spectra
            fname = fig_dir + '/' + str(event) + '_timeDenoised_and_specStack.hdf5'
            with h5py.File(fname, 'r') as f:
                freq = f['stack_spec_freq'][:]
                deno = f['stack_spec_deno'][:]
                noisy = f['stack_spec_noisy'][:]

            # %% resample log_spec
            log_freq = np.squeeze(np.log10(freq[1:]))
            log_deno = np.squeeze(np.log10(deno[1:]))
            log_noisy = np.squeeze(np.log10(noisy[1:]))

            interp_deno = interp1d(log_freq, log_deno, bounds_error=False, fill_value=0.)
            log_deno_new = interp_deno(f_new)
            interp_noisy = interp1d(log_freq, log_noisy, bounds_error=False, fill_value=0.)
            log_noisy_new = interp_noisy(f_new)

            # ax[0].loglog(np.power(10, f_new), 10**(log_noisy_new) *np.power(10, (mw_list[i] + 6.07) * 1.5), ls='-', color='gray', lw=0.5)
            # ax[1].loglog(np.power(10, f_new), 10**(log_deno_new) * np.power(10, (mw_list[i] + 6.07) * 1.5), ls='-', color='gray', lw=0.5)
            # ax[1].loglog(fc_list[i], np.power(10, (mw_list[i] + 6.07) * 1.5) / 2.0, 'o', markersize=5, mec='k', mfc='g')

            # %% average log_spec of each group
            mw_here = mw_list[i]
            # mw_here = 7.8-j*0.3
            sum_deno  = sum_deno  + 10**(log_deno_new) * np.power(10, mw_here * 1.5 + 9.1)
            sum_noisy = sum_noisy + 10**(log_noisy_new)* np.power(10, mw_here * 1.5 + 9.1)
            # sum_m0 += np.power(10, mw_here * 1.5 + 9.1)
        ave_deno  = sum_deno / N_events
        ave_noisy = sum_noisy/ N_events

        n_noisy, fc_noisy = model_spec(np.log10(ave_noisy), f_mod)
        n_deno,  fc_deno  = model_spec(np.log10(ave_deno), f_mod)

        ax[0].loglog(np.power(10, f_new), ave_noisy, ls='-', color='gray', lw=1)
        ax[1].loglog(np.power(10, f_new), ave_deno, ls='-', color='gray', lw=1)

        bootstrap_sum_noisy = bootstrap_sum_noisy + ave_noisy
        bootstrap_sum_deno = bootstrap_sum_deno + ave_deno

    bootstrap_ave_noisy = bootstrap_sum_noisy / N_bootstrap
    bootstrap_ave_deno = bootstrap_sum_deno / N_bootstrap

    n_noisy, fc_noisy = model_spec(np.log10(bootstrap_ave_noisy), f_mod)
    n_deno, fc_deno = model_spec(np.log10(bootstrap_ave_deno), f_mod)

    ax[0].loglog(np.power(10, f_new), bootstrap_ave_noisy, ls='-', color='r', lw=2)
    ax[0].loglog(np.power(10, f_mod), bootstrap_ave_noisy[0] / (1.0 + 10 ** ((f_mod - fc_noisy) * n_noisy)), ls='--', color='b', lw=5)
    ax[0].loglog(np.power(10, fc_noisy), bootstrap_ave_noisy[0] / 2.0, 'o', markersize=15, mec='k', mfc='r')

    ax[1].loglog(np.power(10, f_new), bootstrap_ave_deno, ls='-', color='r', lw=2)
    ax[1].loglog(np.power(10, f_mod), bootstrap_ave_deno[0] / (1.0 + 10 ** ((f_mod - fc_deno) * n_deno)), ls='--', color='b', lw=5)
    ax[1].loglog(np.power(10, fc_deno), bootstrap_ave_deno[0] / 2.0, 'o', markersize=15, mec='k', mfc='r')

    font = {'family': 'serif',
            'color': 'darkred',
            'weight': 'normal',
            'size': 24}

    for i in range(2):
        ax[i].text(2.2e-2, 1.1 * bootstrap_ave_deno[0], fr'$M_W${7.6 - j*0.4: .1f}-{8.0 - j*0.4: .1f}', fontdict=font)
        ax[i].set_xlabel('frequency (Hz)', fontsize=24)
        ax[i].set_ylabel('amplitude spectrum', fontsize=24)

# %% Fitting lines
lg_mg = np.arange(17, 22, 0.5, dtype=np.float64)
lg_f = 4.3 - lg_mg / 4
ax[0].plot(10 ** lg_f, 10 ** lg_mg, '-g', linewidth=5, alpha=1, label=r'$M_0$~$f_c^\frac{1}{4}$')
ax[1].plot(10 ** lg_f, 10 ** lg_mg, '-g', linewidth=5, alpha=1, label=r'$M_0$~$f_c^\frac{1}{4}$')
lg_f = 6 - lg_mg / 3
ax[0].plot(10 ** lg_f, 10 ** lg_mg, '-', color='limegreen', lw=5, label=r'$M_0$~$f_c^\frac{1}{3}$')
ax[1].plot(10 ** lg_f, 10 ** lg_mg, '-', color='limegreen', lw=5, label=r'$M_0$~$f_c^\frac{1}{3}$')
ax[0].legend(loc='upper right')
ax[1].legend(loc='upper right')

for i in range(2):
    ax[i].grid(which='major', color='#DDDDDD', linewidth=3)
    ax[i].grid(which='minor', color='#EEEEEE', linestyle='--', linewidth=2)
    ax[i].minorticks_on()
    ax[i].set_ylim(1e16, 5e22)
    ax[i].set_xlim(2e-2, 5e0)

plt.savefig(fig_dir + '/stacked_spec_depth300_SNR2.png')
