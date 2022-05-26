from math import sin, cos, pi
import matplotlib.pyplot as plt
from obspy.imaging.beachball import beach
import obspy
import numpy as np
import tensorflow as tf

# convert strike/dip/rake to moment tensor #
def fm2mt(strike, dip, rake):
    epsilon = 1e-13
    deg2rad = pi / 180.
    S = strike * deg2rad
    D = dip * deg2rad
    R = rake * deg2rad

    for ang in S, D, R:
        if abs(ang) < epsilon:
            ang = 0.

    M11 = -(sin(D) * cos(R) * sin(2 * S) + sin(2 * D) * sin(R) * sin(S) ** 2)
    M22 = +(sin(D) * cos(R) * sin(2 * S) - sin(2 * D) * sin(R) * cos(S) ** 2)
    M33 = sin(2 * D) * sin(R)
    M12 = +(sin(D) * cos(R) * cos(2 * S) + sin(2 * D) * sin(R) * sin(S) * cos(S))
    M13 = -(cos(D) * cos(R) * cos(S) + cos(2 * D) * sin(R) * sin(S))
    M23 = -(cos(D) * cos(R) * sin(S) - cos(2 * D) * sin(R) * cos(S))

    #Moments = [M11, M22, M33, M12, M13, M23]
    Moments = [M33, M11, M22, M13, 0-M23, 0-M12]
    return tuple(Moments)


# plot learning curve #
def plot_curve(history, nm, swth):
    epoch = [i+1 for i in history.epoch]
    plt.figure()
    plt.xlabel('epoch')
    plt.ylabel('mean squared error')
    plt.ylim([0, 1])
    plt.plot(epoch, np.array(history.history['loss']), label='training loss')
    plt.plot(epoch, np.array(history.history['val_loss']), label='validation loss')
    if swth == 1:
        plt.plot(epoch, np.array(history.history['mse']), label='mse')
        plt.plot(epoch, np.array(history.history['val_mse']), label='val_mse')
    plt.legend()
    plt.savefig(nm+'.pdf')


# plot predicted beachballs together with ground truths #
def plot_ball(mtlabel,mtpred,nm):
    if len(mtlabel.shape) ==1:
        mtlabel = mtlabel[None,:]
    if len(mtpred.shape) ==1:
        mtpred = mtpred[None,:]
    nballs = len(mtlabel)
    plt.ylim([0,4])
    plt.xlim([-1,nballs])
    plt.text(nballs-0.3, 0.5, 'label')
    plt.text(nballs-0.3, 2.5, 'prediction')
    ax = plt.gca()
    for i in range(nballs):
        yl = 1
        yp = 3
        xl = i
        xp = i
        bl = beach(mtlabel[i], xy=(xl, yl), linewidth=1, width=0.8)
        bp = beach(mtpred[i], xy=(xp, yp), linewidth=1, width=0.8)
        ax.add_collection(bl)
        ax.add_collection(bp)
    ax.set_aspect("equal")
    plt.savefig(nm+'.pdf')


# Define new loss with penalty on cheating
def custom_mse(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true-y_pred))
    penalty = tf.reduce_mean(tf.square(tf.math.reduce_std(y_true, axis=0) 
                                  - tf.math.reduce_std(y_pred, axis=0)))
    return mse + penalty

def custom_mae(y_true, y_pred):
    mae = tf.reduce_mean(tf.abs(y_true-y_pred))
    penalty = tf.reduce_mean(tf.abs(tf.math.reduce_std(y_true, axis=0)
                                  - tf.math.reduce_std(y_pred, axis=0)))
    return mae + penalty


# Define shuff_split instead of using validation_split
def shuff_split(x_raw, y_raw, batch_size, vali_ratio):
    nevt = x_raw.shape[0]
    ind = np.arange(nevt)
    np.random.shuffle(ind)
    x_raw = x_raw[ind]
    y_raw = y_raw[ind]
    splitter1 = int(nevt / batch_size * vali_ratio) * batch_size
    splitter2 = int(nevt / batch_size ) * batch_size
    x_vali = x_raw[0:splitter1]
    y_vali = y_raw[0:splitter1]
    x_train = x_raw[splitter1:splitter2]
    y_train = y_raw[splitter1:splitter2]

    return x_train, y_train, x_vali, y_vali

# Trim downloaded three components for same time range
def trim_align(trace):
    startt=trace[0].stats.starttime
    if startt <trace[1].stats.starttime:
        startt=trace[1].stats.starttime
    if startt <trace[2].stats.starttime:
        startt=trace[2].stats.starttime

    endt=trace[0].stats.endtime
    if endt >trace[1].stats.endtime:
        endt=trace[1].stats.endtime
    if endt >trace[2].stats.endtime:
        endt=trace[2].stats.endtime

    return trace.trim(startt, endt)
