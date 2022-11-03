"""
Apply the model to noise data
e.g. deep earthquakes

author: Qibin Shi
"""

import time
import matplotlib
import numpy as np
import pandas as pd
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from denoiser_util import get_vp_vs
plt.rcParams['axes.axisbelow'] = True
matplotlib.rcParams.update({'font.size': 35})


# %% grid search the best a and b
def line_fit(x, y, da=0.01, db=0.1):
    l2_min = 10000.0
    a_best = 2.00
    b_best = 0.00
    for a in np.arange(0.0, 2.0, da, dtype=np.float64):
        for b in np.arange(0, 20, db, dtype=np.float64):
            line_func = x * a - b
            l2 = np.sum(np.square(y - line_func))
            if l2 < l2_min:
                l2_min = l2
                a_best = a
                b_best = b

    return a_best, b_best


minsta = 10
since = time.time()
# %% Read the file of event durations
model_dir = 'Release_Middle_augmentation_S4Hz_150s_removeCoda_SoverCoda25'
fig_dir = model_dir + '/Apply_releaseWDN_M5.5-8_deep100km_SNR2_azimuthBined_fixedWindow'
csv_file = fig_dir + '/source_measurements.csv'
meta_result = pd.read_csv(csv_file, low_memory=False)

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

# ind_select = np.where(nsta > minsta)[0]
ind_select = np.where(np.logical_and(depth > 00.0, nsta > minsta))[0]

mb = mb[ind_select]
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
lg_moment = (mw + 6.07) * 1.5
moment = 10 ** lg_moment

vp = np.ones(len(depth), dtype=np.float64)
vs = np.ones(len(depth), dtype=np.float64)
den= np.ones(len(depth), dtype=np.float64)
for i in range(len(depth)):
    vp[i], vs[i], den[i] = get_vp_vs(depth[i])

# %% Plot scaling relations
plt.close('all')
cmap = matplotlib.cm.bwr.reversed()
cmap1 = matplotlib.cm.hot.reversed()
fig, ax = plt.subplots(7, 4, figsize=(50, 70), constrained_layout=True)

v1 = 0
v2 = 600
v3 = 5.0
v4 = 8.0
# %% duration--mw
ino = ax[0, 0].scatter(moment, dura_noisy, marker='o', s=200, c=depth, edgecolors='k', cmap=cmap, vmin=v1, vmax=v2)
ide = ax[0, 1].scatter(moment, dura_deno, marker='o', s=200, c=depth, edgecolors='k', cmap=cmap, vmin=v1, vmax=v2)
cbr = plt.colorbar(ide, ax=ax[0, 1])
cbr.set_label('Depth (km)')

# %% duration--depth
ind_m0_clean = np.where(np.logical_and(10**(lg_moment * 0.25 - 3.7) > dura_deno, 10**(lg_moment * 0.25 - 4.8) < dura_deno))[0]
ind_m0_noisy = np.where(np.logical_and(10**(lg_moment * 0.25 - 3.7) > dura_noisy, 10**(lg_moment * 0.25 - 4.8) < dura_noisy))[0]
ts_clean = dura_deno[ind_m0_clean] * np.power(10, (19-lg_moment[ind_m0_clean])/4) * vs[ind_m0_clean] / 4.5
ts_noisy = dura_noisy[ind_m0_noisy] * np.power(10, (19-lg_moment[ind_m0_noisy])/4) * vs[ind_m0_noisy] / 4.5
dp_clean = depth[ind_m0_clean]
dp_noisy = depth[ind_m0_noisy]
m0_clean = 10 ** lg_moment[ind_m0_clean]
m0_noisy = 10 ** lg_moment[ind_m0_noisy]
ino = ax[0, 2].scatter(dp_noisy, ts_noisy, marker='o', s=200, c=mw[ind_m0_noisy], edgecolors='k', cmap=cmap1, vmin=v3, vmax=v4)
ide = ax[0, 3].scatter(dp_clean, ts_clean, marker='o', s=200, c=mw[ind_m0_clean], edgecolors='k', cmap=cmap1, vmin=v3, vmax=v4)
cbr = plt.colorbar(ide, ax=ax[0, 3])
cbr.set_label(r'$M_W$')

try:
    ind_150_clean = np.where(np.logical_and(dp_clean < 200, dp_clean > 100))[0]
    ind_150_noisy = np.where(np.logical_and(dp_noisy < 200, dp_noisy > 100))[0]
    ind_250_clean = np.where(np.logical_and(dp_clean < 300, dp_clean > 200))[0]
    ind_250_noisy = np.where(np.logical_and(dp_noisy < 300, dp_noisy > 200))[0]
    ind_350_clean = np.where(np.logical_and(dp_clean < 400, dp_clean > 300))[0]
    ind_350_noisy = np.where(np.logical_and(dp_noisy < 400, dp_noisy > 300))[0]
    ind_450_clean = np.where(np.logical_and(dp_clean < 500, dp_clean > 400))[0]
    ind_450_noisy = np.where(np.logical_and(dp_noisy < 500, dp_noisy > 400))[0]
    ind_550_clean = np.where(np.logical_and(dp_clean < 600, dp_clean > 500))[0]
    ind_550_noisy = np.where(np.logical_and(dp_noisy < 600, dp_noisy > 500))[0]
    ind_650_clean = np.where(np.logical_and(dp_clean < 800, dp_clean > 600))[0]
    ind_650_noisy = np.where(np.logical_and(dp_noisy < 800, dp_noisy > 600))[0]

    ax[0, 2].errorbar(150, np.mean(ts_noisy[ind_150_noisy]), yerr=np.std(ts_noisy[ind_150_noisy]),fmt='o',ms=30,mec='g',mfc='g',ecolor='g',elinewidth=9,capsize=15,capthick=9)
    ax[0, 3].errorbar(150, np.mean(ts_clean[ind_150_clean]), yerr=np.std(ts_clean[ind_150_clean]),fmt='o',ms=30,mec='g',mfc='g',ecolor='g',elinewidth=9,capsize=15,capthick=9)
    ax[0, 2].errorbar(250, np.mean(ts_noisy[ind_250_noisy]), yerr=np.std(ts_noisy[ind_250_noisy]),fmt='o',ms=30,mec='g',mfc='g',ecolor='g',elinewidth=9,capsize=15,capthick=9)
    ax[0, 3].errorbar(250, np.mean(ts_clean[ind_250_clean]), yerr=np.std(ts_clean[ind_250_clean]),fmt='o',ms=30,mec='g',mfc='g',ecolor='g',elinewidth=9,capsize=15,capthick=9)
    ax[0, 2].errorbar(350, np.mean(ts_noisy[ind_350_noisy]), yerr=np.std(ts_noisy[ind_350_noisy]),fmt='o',ms=30,mec='g',mfc='g',ecolor='g',elinewidth=9,capsize=15,capthick=9)
    ax[0, 3].errorbar(350, np.mean(ts_clean[ind_350_clean]), yerr=np.std(ts_clean[ind_350_clean]),fmt='o',ms=30,mec='g',mfc='g',ecolor='g',elinewidth=9,capsize=15,capthick=9)
    ax[0, 2].errorbar(450, np.mean(ts_noisy[ind_450_noisy]), yerr=np.std(ts_noisy[ind_450_noisy]),fmt='o',ms=30,mec='g',mfc='g',ecolor='g',elinewidth=9,capsize=15,capthick=9)
    ax[0, 3].errorbar(450, np.mean(ts_clean[ind_450_clean]), yerr=np.std(ts_clean[ind_450_clean]),fmt='o',ms=30,mec='g',mfc='g',ecolor='g',elinewidth=9,capsize=15,capthick=9)
    ax[0, 2].errorbar(550, np.mean(ts_noisy[ind_550_noisy]), yerr=np.std(ts_noisy[ind_550_noisy]),fmt='o',ms=30,mec='g',mfc='g',ecolor='g',elinewidth=9,capsize=15,capthick=9)
    ax[0, 3].errorbar(550, np.mean(ts_clean[ind_550_clean]), yerr=np.std(ts_clean[ind_550_clean]),fmt='o',ms=30,mec='g',mfc='g',ecolor='g',elinewidth=9,capsize=15,capthick=9)
    ax[0, 2].errorbar(650, np.mean(ts_noisy[ind_650_noisy]), yerr=np.std(ts_noisy[ind_650_noisy]),fmt='o',ms=30,mec='g',mfc='g',ecolor='g',elinewidth=9,capsize=15,capthick=9)
    ax[0, 3].errorbar(650, np.mean(ts_clean[ind_650_clean]), yerr=np.std(ts_clean[ind_650_clean]),fmt='o',ms=30,mec='g',mfc='g',ecolor='g',elinewidth=9,capsize=15,capthick=9)
except:
    pass

# %% corner frequency
ax[1, 0].scatter(moment, 1 / corner_freq_noisy, marker='o', s=200, c=depth, edgecolors='k', cmap=cmap, vmin=v1, vmax=v2)
ax[1, 1].scatter(moment, 1 / corner_freq_deno, marker='o', s=200, c=depth, edgecolors='k', cmap=cmap, vmin=v1, vmax=v2)

# %% fc - T
ax[6, 0].scatter(dura_noisy, 1 / corner_freq_noisy, marker='o', s=200, c=depth, edgecolors='k', cmap=cmap, vmin=v1, vmax=v2)
ax[6, 1].scatter(dura_deno, 1 / corner_freq_deno, marker='o', s=200, c=depth, edgecolors='k', cmap=cmap, vmin=v1, vmax=v2)
ax[6, 2].scatter(dura_noisy, 1 / corner_freq_noisy, marker='o', s=200, c=mw, edgecolors='k', cmap=cmap1, vmin=v3, vmax=v4)
ax[6, 3].scatter(dura_deno, 1 / corner_freq_deno, marker='o', s=200, c=mw, edgecolors='k', cmap=cmap1, vmin=v3, vmax=v4)

# %% 1/fc--depth
ind_m0_clean = np.where(np.logical_and(10**(lg_moment * 0.25 - 3.7) > 1 / corner_freq_deno, 10**(lg_moment * 0.25 - 4.8) < 1 / corner_freq_deno))[0]
ind_m0_noisy = np.where(np.logical_and(10**(lg_moment * 0.25 - 3.7) > 1 / corner_freq_noisy, 10**(lg_moment * 0.25 - 4.8) < 1 / corner_freq_noisy))[0]
ts_clean = 1 / corner_freq_deno[ind_m0_clean] * np.power(10, (19-lg_moment[ind_m0_clean])/4) * vs[ind_m0_clean] / 4.5
ts_noisy = 1 / corner_freq_noisy[ind_m0_noisy] * np.power(10, (19-lg_moment[ind_m0_noisy])/4) * vs[ind_m0_noisy] / 4.5
dp_clean = depth[ind_m0_clean]
dp_noisy = depth[ind_m0_noisy]
m0_clean = 10 ** lg_moment[ind_m0_clean]
m0_noisy = 10 ** lg_moment[ind_m0_noisy]
ino = ax[1, 2].scatter(dp_noisy, ts_noisy, marker='o', s=200, c=mw[ind_m0_noisy], edgecolors='k', cmap=cmap1, vmin=v3, vmax=v4)
ide = ax[1, 3].scatter(dp_clean, ts_clean, marker='o', s=200, c=mw[ind_m0_clean], edgecolors='k', cmap=cmap1, vmin=v3, vmax=v4)

try:
    ind_150_clean = np.where(np.logical_and(dp_clean < 200, dp_clean > 100))[0]
    ind_150_noisy = np.where(np.logical_and(dp_noisy < 200, dp_noisy > 100))[0]
    ind_250_clean = np.where(np.logical_and(dp_clean < 300, dp_clean > 200))[0]
    ind_250_noisy = np.where(np.logical_and(dp_noisy < 300, dp_noisy > 200))[0]
    ind_350_clean = np.where(np.logical_and(dp_clean < 400, dp_clean > 300))[0]
    ind_350_noisy = np.where(np.logical_and(dp_noisy < 400, dp_noisy > 300))[0]
    ind_450_clean = np.where(np.logical_and(dp_clean < 500, dp_clean > 400))[0]
    ind_450_noisy = np.where(np.logical_and(dp_noisy < 500, dp_noisy > 400))[0]
    ind_550_clean = np.where(np.logical_and(dp_clean < 600, dp_clean > 500))[0]
    ind_550_noisy = np.where(np.logical_and(dp_noisy < 600, dp_noisy > 500))[0]
    ind_650_clean = np.where(np.logical_and(dp_clean < 800, dp_clean > 600))[0]
    ind_650_noisy = np.where(np.logical_and(dp_noisy < 800, dp_noisy > 600))[0]

    ax[1, 2].errorbar(150, np.mean(ts_noisy[ind_150_noisy]), yerr=np.std(ts_noisy[ind_150_noisy]),fmt='o',ms=30,mec='g',mfc='g',ecolor='g',elinewidth=9,capsize=15,capthick=9)
    ax[1, 3].errorbar(150, np.mean(ts_clean[ind_150_clean]), yerr=np.std(ts_clean[ind_150_clean]),fmt='o',ms=30,mec='g',mfc='g',ecolor='g',elinewidth=9,capsize=15,capthick=9)
    ax[1, 2].errorbar(250, np.mean(ts_noisy[ind_250_noisy]), yerr=np.std(ts_noisy[ind_250_noisy]),fmt='o',ms=30,mec='g',mfc='g',ecolor='g',elinewidth=9,capsize=15,capthick=9)
    ax[1, 3].errorbar(250, np.mean(ts_clean[ind_250_clean]), yerr=np.std(ts_clean[ind_250_clean]),fmt='o',ms=30,mec='g',mfc='g',ecolor='g',elinewidth=9,capsize=15,capthick=9)
    ax[1, 2].errorbar(350, np.mean(ts_noisy[ind_350_noisy]), yerr=np.std(ts_noisy[ind_350_noisy]),fmt='o',ms=30,mec='g',mfc='g',ecolor='g',elinewidth=9,capsize=15,capthick=9)
    ax[1, 3].errorbar(350, np.mean(ts_clean[ind_350_clean]), yerr=np.std(ts_clean[ind_350_clean]),fmt='o',ms=30,mec='g',mfc='g',ecolor='g',elinewidth=9,capsize=15,capthick=9)
    ax[1, 2].errorbar(450, np.mean(ts_noisy[ind_450_noisy]), yerr=np.std(ts_noisy[ind_450_noisy]),fmt='o',ms=30,mec='g',mfc='g',ecolor='g',elinewidth=9,capsize=15,capthick=9)
    ax[1, 3].errorbar(450, np.mean(ts_clean[ind_450_clean]), yerr=np.std(ts_clean[ind_450_clean]),fmt='o',ms=30,mec='g',mfc='g',ecolor='g',elinewidth=9,capsize=15,capthick=9)
    ax[1, 2].errorbar(550, np.mean(ts_noisy[ind_550_noisy]), yerr=np.std(ts_noisy[ind_550_noisy]),fmt='o',ms=30,mec='g',mfc='g',ecolor='g',elinewidth=9,capsize=15,capthick=9)
    ax[1, 3].errorbar(550, np.mean(ts_clean[ind_550_clean]), yerr=np.std(ts_clean[ind_550_clean]),fmt='o',ms=30,mec='g',mfc='g',ecolor='g',elinewidth=9,capsize=15,capthick=9)
    ax[1, 2].errorbar(650, np.mean(ts_noisy[ind_650_noisy]), yerr=np.std(ts_noisy[ind_650_noisy]),fmt='o',ms=30,mec='g',mfc='g',ecolor='g',elinewidth=9,capsize=15,capthick=9)
    ax[1, 3].errorbar(650, np.mean(ts_clean[ind_650_clean]), yerr=np.std(ts_clean[ind_650_clean]),fmt='o',ms=30,mec='g',mfc='g',ecolor='g',elinewidth=9,capsize=15,capthick=9)

except:
    pass

# %% centroid speed--mw
app_vr_clean = (ctr_speed_deno * 1)
app_vr_noisy = (ctr_speed_noisy * 1)
ax[2, 0].scatter(moment, app_vr_noisy, marker='o', s=200, c=depth, edgecolors='k', cmap=cmap, vmin=v1, vmax=v2)
ax[2, 1].scatter(moment, app_vr_clean, marker='o', s=200, c=depth, edgecolors='k', cmap=cmap, vmin=v1, vmax=v2)
ax[2, 2].scatter(depth, app_vr_noisy, marker='o', s=200, c=mw, edgecolors='k', cmap=cmap1, vmin=v3, vmax=v4)
ax[2, 3].scatter(depth, app_vr_clean, marker='o', s=200, c=mw, edgecolors='k', cmap=cmap1, vmin=v3, vmax=v4)

# %% Energy
ax[3, 0].scatter(moment, Es_noisy * moment * moment, marker='o', s=200, c=depth, edgecolors='k', cmap=cmap, vmin=v1, vmax=v2)
ax[3, 1].scatter(moment, Es_deno * moment * moment, marker='o', s=200, c=depth, edgecolors='k', cmap=cmap, vmin=v1, vmax=v2)
ax[3, 2].scatter(depth, Es_noisy * moment * moment, marker='o', s=200, c=mw, edgecolors='k', cmap=cmap1, vmin=v3, vmax=v4)
ax[3, 3].scatter(depth, Es_deno * moment * moment, marker='o', s=200, c=mw, edgecolors='k', cmap=cmap1, vmin=v3, vmax=v4)
ax[4, 0].scatter(moment, Es_noisy * moment* 30000, marker='o', s=200, c=depth, edgecolors='k', cmap=cmap, vmin=v1, vmax=v2)
ax[4, 1].scatter(moment, Es_deno * moment* 30000, marker='o', s=200, c=depth, edgecolors='k', cmap=cmap, vmin=v1, vmax=v2)
ax[4, 2].scatter(depth, Es_noisy * moment* 30000, marker='o', s=200, c=mw, edgecolors='k', cmap=cmap1, vmin=v3, vmax=v4)
ax[4, 3].scatter(depth, Es_deno * moment* 30000, marker='o', s=200, c=mw, edgecolors='k', cmap=cmap1, vmin=v3, vmax=v4)


# %% falloff power
ax[5, 0].scatter(moment, falloff_noisy, marker='o', s=200, c=depth, edgecolors='k', cmap=cmap, vmin=v1, vmax=v2)
ax[5, 1].scatter(moment, falloff_deno, marker='o', s=200, c=depth, edgecolors='k', cmap=cmap, vmin=v1, vmax=v2)
ax[5, 2].scatter(depth, falloff_noisy, marker='o', s=200, c=mw, edgecolors='k', cmap=cmap1, vmin=v3, vmax=v4)
ax[5, 3].scatter(depth, falloff_deno, marker='o', s=200, c=mw, edgecolors='k', cmap=cmap1, vmin=v3, vmax=v4)




# %% Fitting lines
lg_mg = np.arange(17, 23, 0.5, dtype=np.float64)
for k in range(2):

    lg_t = lg_mg * 0.25 - 4.25
    ax[0, k].plot(10 ** lg_mg, 10 ** lg_t, '-k', linewidth=5, alpha=1, label=r'T~$M_0^\frac{1}{4}$')
    ax[1, k].plot(10 ** lg_mg, 10 ** lg_t, '-k', linewidth=5, alpha=1, label=r'T~$M_0^\frac{1}{4}$')
    lg_t = lg_mg * 0.25 - 3.7
    ax[0, k].plot(10 ** lg_mg, 10 ** lg_t, '--k', linewidth=5, alpha=1)
    ax[1, k].plot(10 ** lg_mg, 10 ** lg_t, '--k', linewidth=5, alpha=1)
    lg_t = lg_mg * 0.25 - 4.8
    ax[0, k].plot(10 ** lg_mg, 10 ** lg_t, '--k', linewidth=5, alpha=1)
    ax[1, k].plot(10 ** lg_mg, 10 ** lg_t, '--k', linewidth=5, alpha=1)
    ax[0, k].legend(loc='lower right')
    ax[1, k].legend(loc='upper left')

    lg_E = lg_mg * 1.17 - 7.0
    ax[3, k].plot(10 ** lg_mg, 10 ** lg_E, '-k', linewidth=5, alpha=1.0, label=r'$E_R$~$M_0^{1.17}$')
    lg_E = lg_mg - 2.5
    ax[3, k].plot(10 ** lg_mg, 10 ** lg_E, '--k', linewidth=5, alpha=0.8, label=r'$E_R$~$M_0$')
    lg_E = lg_mg - 5.5
    ax[3, k].plot(10 ** lg_mg, 10 ** lg_E, '--k', linewidth=5, alpha=0.8)
    a, b = line_fit(np.log10(moment), np.log10(Es_deno * moment * moment))
    print('E-M', a, b)
    lg_E = lg_mg * a - b
    ax[3, k].plot(10 ** lg_mg, 10 ** lg_E, '-', color='g', linewidth=5, alpha=1.0, label=r'$E_R$~$M_0^{1.77}$')
    ax[3, k].legend(loc='lower right')

    lg_E = lg_mg * 0.17 - 5.7 +4.5
    ax[4, k].plot(10 ** lg_mg, 10 ** lg_E, '-k', linewidth=5, alpha=1, label=r'$\sigma_A$~$M_0^{0.17}}$')
    lg_E = lg_mg * (a-1) - b + 4.5
    ax[4, k].plot(10 ** lg_mg, 10 ** lg_E, '-', color='g', linewidth=5, alpha=1, label=r'$\sigma_A$~$M_0^{0.77}$')
    ax[4, k].legend(loc='lower right')


for i in range(4):
    ax[1, i].set_ylim(1e-1, 1e2)
    ax[2, i].set_ylim(-0.05, 0.6)
    ax[3, i].set_ylim(1e6, 5e24)
    ax[4, i].set_ylim(1e-4, 1e5)
    ax[5, i].set_ylim(0, 3)
    ax[6, i].set_ylim(0, 20)
    ax[6, i].set_xlim(0, 20)
for i in range(6):
    ax[i, 0].set_xlim(1e16, 1e23)
    ax[i, 1].set_xlim(1e16, 1e23)
    ax[i, 2].set_xlim(50, 750)
    ax[i, 3].set_xlim(50, 750)

ax[0, 0].set_ylim(1e-1, 1e2)
ax[0, 1].set_ylim(1e-1, 1e2)
ax[0, 2].set_ylim(0, 15)
ax[0, 3].set_ylim(0, 15)
ax[1, 2].set_ylim(0, 15)
ax[1, 3].set_ylim(0, 15)


for i in range(2):
    ax[0, i].set_yscale("log")
    ax[4, i].set_xscale("log")
    ax[5, i].set_xscale("log")
    ax[1, i].set_yscale("log")
for i in range(4):
    # ax[1, i].set_yscale("log")
    ax[3, i].set_yscale("log")
    ax[4, i].set_yscale("log")
    for j in range(2):
        ax[i, j].set_xscale("log")

for i in range(7):
    for j in range(4):
        ax[i, j].grid(which='major', color='#DDDDDD', linewidth=3)
        ax[i, j].grid(which='minor', color='#EEEEEE', linestyle='--', linewidth=2)
        ax[i, j].minorticks_on()
        ax[i, j].axvline(300, color='b', linewidth=2)


ax[0, 0].set_ylabel('Duration', fontsize=40)
ax[2, 0].set_ylabel('Apparent speed/ Vp', fontsize=40)
ax[3, 0].set_ylabel('Radiated energy', fontsize=40)
ax[4, 0].set_ylabel('Apparent stress', fontsize=40)
ax[5, 0].set_ylabel('Falloff power', fontsize=40)
ax[1, 0].set_ylabel(r'1/Corner frequency ($Hz^{-1}$)', fontsize=40)
ax[5, 0].set_xlabel('Moment (Nm)', fontsize=40)
ax[5, 1].set_xlabel('Moment (Nm)', fontsize=40)
ax[6, 0].set_ylabel(r'1/Corner frequency ($Hz^{-1}$)', fontsize=40)
for i in range(4):
    ax[6, i].set_xlabel('Duration', fontsize=40)

ax[0, 2].set_ylabel('Scaled duration', fontsize=40)
ax[1, 2].set_ylabel('Scaled duration', fontsize=40)
ax[5, 2].set_xlabel('Depth (km)', fontsize=40)
ax[5, 3].set_xlabel('Depth (km)', fontsize=40)

ax[0, 0].set_title('noisy data', fontsize=40)
ax[0, 1].set_title('denoised data', fontsize=40)
ax[0, 2].set_title('noisy data', fontsize=40)
ax[0, 3].set_title('denoised data', fontsize=40)

plt.savefig(fig_dir + '/scaling.png')
