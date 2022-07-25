
from denoiser_util import shift2maxcc, dura_cc
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')
timex = np.linspace(-8, 8, 1000)
dt = timex[1] - timex[0]
wide = np.sinc(timex)
thin = np.sinc(5 * timex-3)

thin_shifted, shift, flip = shift2maxcc(wide, thin, maxshift=500, flip_thre=-0.3)
time_new, thin_widen, ratio, cc, flip = dura_cc(wide, thin, timex, max_ratio=5)
plt.plot(timex, wide, '-b', linewidth=5)
plt.plot(timex, thin, '-r')
plt.plot(time_new, thin_widen, '-g')

# wide_shifted, shift, flip = shift2maxcc(thin, wide, maxshift=500, flip_thre=-0.3)
# time_new, wide_squee, ratio, cc, flip = dura_cc(thin, wide, timex, max_ratio=5)
# plt.plot(timex, wide, '-b', timex, thin, '-r', time_new, wide_squee, '-g')

plt.grid()
plt.show()
