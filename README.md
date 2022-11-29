# Teleseismic Denoiser

Separate earthquake signals and noises on teleseismic waveform recordings using a two-branch U-net trained with global teleseismic data.
The Denoiser uses the pretrained kernel [WaveDecompNet(Yin et al 2022)](https://github.com/yinjiuxun/WaveDecompNet-paper/) to learn high-level features.

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
conda install -c gprieto multitaper / pip install multitaper
```
## Step 0-1: Generate earthquake and noise waveforms.
Process the event-based waveform and save both P/S wave and pre-P wave signals.
```
0_quake_plus_prePnoise.py
```
Add STEAD and POHA noises to the same earthquake waveforms in previous step to form another 50% of training data.
```
1_STEAD_POHA_noise.py
```

## Step 2-3: Train and test the model with data augmentation.
Training with stacked noisy earthquake waveform in order to obtain the denoised earthquake waveform and pure noise. Data is being augmented on the fly by squeezing and shifting P/S waves and stacking noises with variable SNR.
```
2_Train_with_augmentation.py
```
Evaluate the performance of the global earthquake denoiser on the testing data.
```
3_Test_with_augmentation.py
```

## Step 4-5: Apply the model to real waveforms
Prepare the raw noisy data.
```
4_new_noisy_data_for_application.py
```
Denoise and plot the output signal and noise in time and spectrum domain.
```
5_apply_to_noisy_data.py
```
