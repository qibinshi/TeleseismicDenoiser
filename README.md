# Global Earthquake Denoiser  
Splitting earthquake and noise waveforms using a two-branch U net trained with global teleseismic data.
The neural network includes the pretrained [WaveDecompNet(Yin et al 2022)](https://github.com/yinjiuxun/WaveDecompNet-paper/) for the higher level feature learning.

## Create a virtual environment in Anaconda.
```
conda create -n myenv python=3.9
conda activate myenv
```
## Install essential modules in our recommended order.
```
conda install h5py
conda install pytorch -c pytorch
conda install scipy
conda install obspy -c conda-forge
conda install scikit-learn
```
## To prepare your own dataset for training, run the scripts. 
#### 0a_buildcatalog.py
- Customize the global earthquake catalog for teleseismic data.   
#### 0b_downloadcatalog.py
- Download all the teleseismic data available from FDSN client.
#### 0c_prepare_event_data_in_mat.py
- Process the earthquake waveform and select high signal-to-noise ratio data for training.
#### 1A_save_with_noise.py
- Prepare noise waveform at global stations.

## Train the neural net and test the model using the following scripts, with data augmentation.
#### 2A_Train_with_augmented_data_partial_frozen.py
- Training with stacked noisy earthquake waveform in order to obtain the denoised earthquake waveform and pure noise. Data is being augmented on the fly.
#### 3A_Test_on_augmented_data.py
- Evaluate the performance of the global earthquake denoiser on the testing data.

### * In the case of sufficient training data or fine-tuning the model, data augmentation can be off by using the following scripts.
#### 1_stack_with_noise.py
#### 2_Train_Adaptors_Freeze_WDN.py
#### 3_test_model_on_teleseismic_wave.py
