# Teleseismic Denoiser
Cite the code [![DOI](https://zenodo.org/badge/496703199.svg)](https://zenodo.org/badge/latestdoi/496703199)

Cite the datasets [![DOI](https://img.shields.io/badge/DOI-10.6069%2F69E7--6667-blue)](https://doi.org/10.6069/69E7-6667)



This package uses auto-encoder neural network to separate teleseismic earthquake and noise waveform recordings.
For earthquake studies, this deep neural network can be considered as a multi-scale and multi-task denoiser.
The denoiser uses the pretrained kernel 
[WaveDecompNet(Yin et al 2022)](https://github.com/yinjiuxun/WaveDecompNet-paper/) 
to learn high-level features.



## Installation

If you like using `conda` to manage virtual environments, 
```
conda create -n codetest python=3.9
conda activate codetest
```
To install, 
```
pip install git+https://github.com/qibinshi/TeleseismicDenoiser.git
```


## Configuration file

1. Download/copy `tests/config.ini` to your work directory. 

2. It allows you to manage the directories (absolute path!) , data shape, model size and training strategies. 
Modified the [directories] section to match your computer. 

3. Specify in section [data] and [prediction] whether you would like to run with the `demo datasets` in the package or
the larger dataset ([download](#data-prep) or prepare your own data). 



## Run executables to test the package

You can run the code with the `demo datasets` and `pretrained models` in the package. 
In the same directory, run the executables for quick tests:

To train the model with demo data
```
denote_train
```
To test the pretrained model with demo data
```
denote_test
```
To predict from the demo noisy data
```
denote_predict
```



## Prepare the full dataset for a complete study <a name="data-prep"></a>
You can either download our training data for testing the performance or 
prepare your own dataset following the data format to do a transfer learning. 
We use H5py for waveform data. 
The training data is in H5py format with the keys `quake` and `noise`. 
The noisy input data for application is in H5py with a signal key `quake`.

Our big training and application datasets can be downloaded with the links provided below.
Please cite our data using the following DOI:10.6069/69E7-6667

We process and assemble the waveform data using the FDSN client. See the [list of DOIs of the original data](http://dasway.ess.washington.edu/qibins/Seismic_network_DOI_list.txt). 


### Training data:
[Big dataset of high-SNR teleseismic waveforms for M6+ earthquakes](http://dasway.ess.washington.edu/qibins/Psnr25_lp4_2000-2021.hdf5)

Key `quake`: A numpy array with shape of (X, 50000, 3). 
It represent X recordings of 3-component earthquake waveforms, 
of which each trace has 50,000 sampling points. Our default sampling rate is 10 Hz.

Key `noise`: A numpy array with shape of (X, 10000, 3). 
It represent X recordings of 3-component noise waveforms preceding the P arrival, 
where each trace has 10,000 points.

Note: 
1. The code split the training data into `train`, `validate` and `test` sets. 
The executables `denote_train` and `denote_test` are both run with the training data.

2. The names of keys can vary, with `config.ini` modified accordingly.

### Noisy application data
[Big dataset of teleseismic waveforms for M6 deep earthquakes including both high- and low-SNR data](http://dasway.ess.washington.edu/qibins/M6_deep100km_P.hdf5)


Key `quake`: A numpy array with shape of (X, 3000, 3). 
It represent X recordings of 3-component earthquake waveforms, 
of which each trace has 3,000 sampling points. Our default sampling rate is 10 Hz. 
Hence the length of each waveform is 300 seconds.
