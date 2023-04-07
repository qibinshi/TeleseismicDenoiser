# Teleseismic Denoiser
Cite the code [![DOI](https://zenodo.org/badge/496703199.svg)](https://zenodo.org/badge/latestdoi/496703199)

Separate earthquake signals and noises on teleseismic waveform recordings using a two-branch U-net trained with global teleseismic data.
The Denoiser uses the pretrained kernel [WaveDecompNet(Yin et al 2022)](https://github.com/yinjiuxun/WaveDecompNet-paper/) to learn high-level features.

## 0. Configuration file
Download/copy `tests/config.ini` to your work directory. 

It allows you to manage the directories, data shape, model size and training strategies. 

Modified the paths on the [directories] section to match your computer.

## 1. Quick run

You can quickly install and run the code with the `demo datasets` and `pretrained models` in the package. 

### Installation

If you like using `conda` to manage virtual environments, 
```
conda create -n codetest python=3.9
conda activate codetest
```
To install, 
```
pip install git+https://github.com/qibinshi/TeleseismicDenoiser.git
```

### Use command line executable to test the package
Modify the paths in `config.ini` that has been saved in your work directory. In the same directory, run the executables for quick tests.

To train the model with demo data
```
denote_train
```
To test the pretrained model with demo data
```
denote_test
```
To predict from the example noisy input
```
denote_predict
```

## 2. Prepare your own data
We use H5py for waveform data. The training data is in H5py format with two keys for the quake and noise signals respectively. The noisy input data for application is in H5py with a signal key.

Our big training and application datasets can be downloaded with the link provided below.

### Training data:
(link to training data)
Key 'quake': Create an numpy array with shape of (X, 50000, 3). It represent X recordings of 3-component earthquake waveforms, of which each trace has 50,000 sampling points. Our default sampling rate is 10 Hz.

Key 'noise': Create an numpy array with shape of (X, 10000, 3). It represent X recordings of 3-component noise waveforms preceding the P arrival, where each trace has 10,000 points.

The names of keys can vary, with `config.ini` modified accordingly.

The code split the training data into train, validate and test sets. The executables `denote_train` and `denote_test` are both run with the training data.

### Noisy application data
(link to application data)
Key 'quake': Create an numpy array with shape of (X, 3000, 3). It represent X recordings of 3-component earthquake waveforms, of which each trace has 3,000 sampling points. Our default sampling rate is 10 Hz. Hence the length of each waveform is 300 seconds.



## (Optional) Download the source code and install all the dependencies
### Prerequisite: Anaconda environment.
```
conda create -n myenv python=3.9
conda activate myenv
```
### Prerequisite: Essential modules in the recommended order.
```
conda install h5py
conda install pytorch -c pytorch
conda install scipy
conda install obspy -c conda-forge
conda install scikit-learn
conda install pandas
conda install -c gprieto multitaper / pip install multitaper
```

