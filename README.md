# Global Earthquake Denoiser  
Splitting earthquake and noise waveforms using a two-branch U net trained with global teleseismic data.
The neural network includes the pretrained [WaveDecompNet(Yin et al 2022)](https://github.com/yinjiuxun/WaveDecompNet-paper/) for the higher level feature learning.

## Prerequisite: Anaconda environment.
```
conda create -n myenv python=3.9
conda activate myenv
```
## Prerequisite: Essential modules in the recommended order.
```
conda install h5py
conda install pytorch -c pytorch
conda install scipy
conda install obspy -c conda-forge
conda install scikit-learn
conda install pandas
```
## Step 0-1: Generate earthquake and noise waveforms.
Process the event-based waveform and save both P/S wave and pre-P wave signals.
```
0B_event_plus_prenoise_mpi.py
```
Add STEAD and POHA noises to the same earthquake waveforms in previous step to form another 50% of training data.
```
1B_add_STEAD_noise.py
```

## Step 2-3: Train and test the model with data augmentation.
Training with stacked noisy earthquake waveform in order to obtain the denoised earthquake waveform and pure noise. Data is being augmented on the fly by squeezing and shifting P/S waves and stacking noises with variable SNR.
```
2B_Train_with_augmented_P_preP_partial_frozen.py
```
Evaluate the performance of the global earthquake denoiser on the testing data.
```
3B_Test_on_augmented_P_preP.py
```

## Step 4-5: Apply the model to real waveforms
Prepare the raw noisy data.
```
4_new_class_data_for_application.py
```
Denoise and plot the output signal and noise in time and spectrum domain.
```
5_apply_noisy_data.py
```
