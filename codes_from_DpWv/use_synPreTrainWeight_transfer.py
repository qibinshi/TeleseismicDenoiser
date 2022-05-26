"""
ConvNet training for moment tensor prediction
"""
import os
import glob
import time
import numpy as np
import argparse as ap
import obspy.core as oc
import tensorflow as tf
import matplotlib.pyplot as plt
from obspy import UTCDateTime, read_events, read_inventory
from obspy.taup import TauPyModel
from obspy.clients.iris import Client
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation, Dropout
from tensorflow.keras.layers import Conv3D, MaxPool3D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from source_util import fm2mt, plot_curve, custom_mse, custom_mae, shuff_split
from scipy.io import savemat, loadmat

if __name__ == "__main__":
    parser = ap.ArgumentParser(prog='use_synPreTrainWeight_transfer.py', description='Learn to guess FM')
    parser.add_argument('-T', type=str, default=None, help='Training set')
    parser.add_argument('-W', type=str, default=None, help='where to load pretrained Weights')
    parser.add_argument('-S', type=str, default=None, help='directory to Save models')
    parser.add_argument('-N', type=str, default=None, help='figure Name')
    parser.add_argument('-B', type=int, default=None, help='Batch size')
    args = parser.parse_args()

    if not os.path.exists(args.S):
        os.makedirs(args.S)
        print("Directory ", args.S, " created")
    else:
        print("Directory ", args.S, " exists")
    
    if not os.path.exists("wave_mt.mat"):
### Read the catalog and extract labels
        cat = read_events(args.T+"/*.xml")
        nevt = len(cat)
        wv = np.zeros((nevt, 6, 6, 1001, 3))
        mt = np.zeros((nevt, 6))
        for i in range(nevt):
            ev = cat[i]
            evnm = str(ev.resource_id)[13:]
            strike = ev.focal_mechanisms[0].nodal_planes.nodal_plane_1.strike
            rake = ev.focal_mechanisms[0].nodal_planes.nodal_plane_1.rake
            dip = ev.focal_mechanisms[0].nodal_planes.nodal_plane_1.dip
            mt[i,:] = fm2mt(strike, dip, rake)
            print("Event ",i,mt[i,:])

### Read the stations
            for j in range(6):
                for k in range(6):
                    sta = args.T+"/"+evnm+"selectTRZ/"+str(j)+"_"+str(k)+"_"
                    if len(glob.glob(sta+"*BHZ.mseed")) == 1:
                        cpz0 = oc.read(sta+"*BHZ.mseed")
                        cpz = cpz0.copy()
                        cpz.resample(1.0)
                        tp = UTCDateTime(cpz[0].stats.starttime+100.0)
                        cpz[0].trim(tp-50, tp+950)
                        cpz.filter(type='bandpass', freqmin=0.02, freqmax=0.48)
                        wv[i,j,k,:,0] = cpz[0].data
                    if len(glob.glob(sta+"*BHR.mseed")) == 1:
                        cpr0 = oc.read(sta+"*BHR.mseed")
                        cpr = cpr0.copy()
                        cpr.resample(1.0)
                        tp = UTCDateTime(cpr[0].stats.starttime+100.0)
                        cpr[0].trim(tp-50, tp+950)
                        cpr.filter(type='bandpass', freqmin=0.02, freqmax=0.48)
                        wv[i,j,k,:,1] = cpr[0].data
                    if len(glob.glob(sta+"*BHT.mseed")) == 1:
                        cpt0 = oc.read(sta+"*BHT.mseed")
                        cpt = cpt0.copy()
                        cpt.resample(1.0)
                        tp = UTCDateTime(cpt[0].stats.starttime+100.0)
                        cpt[0].trim(tp-50, tp+950)
                        cpt.filter(type='bandpass', freqmin=0.02, freqmax=0.48)
                        wv[i,j,k,:,2] = cpt[0].data
    
        wv = wv / np.max(np.abs(wv), axis=(1,2,3,4))[:,None,None,None,None]
        savemat("wave_mt.mat", {"wv":wv, "mt":mt})
    else:
        startT=time.time()
        print("Loading data from wave_mt.mat .......")
        wv = loadmat("wave_mt.mat")["wv"]
        mt = loadmat("wave_mt.mat")["mt"]
        endT=time.time()-startT
        print("OK !! taken %f s" % endT)
    x_train, y_train, x_vali, y_vali = shuff_split(wv, mt, args.B, 0.3)

### Build the network
    strategy = tf.distribute.MirroredStrategy()
    #strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3"])
    print('Number of devices: %d' % strategy.num_replicas_in_sync)
    with strategy.scope():
        model = Sequential()
        model.add(Conv3D(8, (3, 3, 3), padding='same', activation=None, input_shape=(6, 6, 1001, 3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv3D(16, (3, 3, 3), padding='same', activation=None))
        model.add(MaxPool3D(pool_size=(1, 1, 2)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv3D(32, (3, 3, 3), padding='same', activation=None))
        model.add(MaxPool3D(pool_size=(1, 1, 2)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv3D(64, (3, 3, 3), padding='same', activation=None))
        model.add(MaxPool3D(pool_size=(1, 1, 2)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv3D(128, (3, 3, 3), padding='same', activation=None))
        model.add(MaxPool3D(pool_size=(1, 1, 2)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv3D(128, (3, 3, 3), padding='same', activation=None))
        model.add(MaxPool3D(pool_size=(1, 1, 2)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv3D(128, (3, 3, 3), padding='same', activation=None))
        model.add(MaxPool3D(pool_size=(1, 1, 2)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(128))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(32))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(32))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(32))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(6))

        adm = Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model.compile(loss=custom_mae, optimizer=adm, metrics=['mse'])

### load trained model weights
    list = os.listdir(args.W)
    list.sort(key=lambda fn:os.path.getmtime(args.W+'/'+fn))
    wfile = os.path.join(args.W,list[-1])
    print('......Loading',wfile,'......')
    model.load_weights(wfile)

    checkpoint = ModelCheckpoint(filepath=args.S+'/transfer-model-{epoch:03d}-{val_loss:.5f}.h5', 
                                 monitor='val_loss', verbose=1, mode='auto', save_freq='epoch',
                                 save_best_only=True, save_weights_only=True)

    history = model.fit(x_train, y_train, batch_size=args.B, epochs=150, verbose=1, shuffle=True,
                        validation_data=(x_vali, y_vali))

    model.save(args.S+'/transferModel_0.004_0.48hz_PandS.h5')
    plot_curve(history, args.N, 1)
