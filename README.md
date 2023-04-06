# Teleseismic Denoiser

Separate earthquake signals and noises on teleseismic waveform recordings using a two-branch U-net trained with global teleseismic data.
The Denoiser uses the pretrained kernel [WaveDecompNet(Yin et al 2022)](https://github.com/yinjiuxun/WaveDecompNet-paper/) to learn high-level features.

## 0. Configuration file
Download/copy `tests/config.ini` to your work directory. Modified the paths to match your computer.

## 1. Quick run
### Use pip3 to install code
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

## 3. Download the source code and install all the dependencies
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

