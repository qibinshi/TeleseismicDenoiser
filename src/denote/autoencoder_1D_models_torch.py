"""
Model architectures

@author: Jiuxun Yin
modified by Qibin Shi
"""
import math
import torch
import torch.nn.functional as F
from torch import nn


# Define the Encoder to extract figures from seismograms
class SeismogramEncoder(nn.Module):
    """The encoder for 3-component seismograms to extract features"""

    def __init__(self):
        super(SeismogramEncoder, self).__init__()
        # convolutional layers
        self.enc1 = nn.Conv1d(3, 8, 9, padding='same', dtype=torch.float64)
        self.enc2 = nn.Conv1d(8, 8, 9, stride=2, padding=4, dtype=torch.float64)
        self.enc3c = nn.Conv1d(8, 16, 7, padding='same', dtype=torch.float64)
        self.enc4 = nn.Conv1d(16, 16, 7, stride=2, padding=3, dtype=torch.float64)
        self.enc5c = nn.Conv1d(16, 32, 5, padding='same', dtype=torch.float64)
        self.enc6 = nn.Conv1d(32, 32, 5, stride=2, padding=2, dtype=torch.float64)
        self.enc7c = nn.Conv1d(32, 64, 3, padding='same', dtype=torch.float64)
        # batch-normalization layers
        self.bn1 = nn.BatchNorm1d(8, dtype=torch.float64)
        self.bn2 = nn.BatchNorm1d(8, dtype=torch.float64)
        self.bn3 = nn.BatchNorm1d(16, dtype=torch.float64)
        self.bn4 = nn.BatchNorm1d(16, dtype=torch.float64)
        self.bn5 = nn.BatchNorm1d(32, dtype=torch.float64)
        self.bn6 = nn.BatchNorm1d(32, dtype=torch.float64)
        self.bn7 = nn.BatchNorm1d(64, dtype=torch.float64)

    def forward(self, x):
        x = F.relu(self.bn1(self.enc1(x)))
        x = F.relu(self.bn2(self.enc2(x)))
        x1 = self.enc3c(x)
        x = F.relu(self.bn3(x1))
        x = F.relu(self.bn4(self.enc4(x)))
        x2 = self.enc5c(x)
        x = F.relu(self.bn5(x2))
        x = F.relu(self.bn6(self.enc6(x)))
        x3 = self.enc7c(x)
        x = F.relu(self.bn7(x3))

        return x, x1, x2, x3


# Define the Decoder (including the bottleneck) to reconstruct seismograms
class SeismogramDecoder(nn.Module):
    """The decoder with bottleneck to reconstruct 3-component seismograms"""

    def __init__(self, bottleneck=None):
        super(SeismogramDecoder, self).__init__()
        # bottleneck to map features of input seismograms to earthquake signals or ambient noise
        self.bottleneck = bottleneck
        # transpose convolutional layers
        self.dec1c = nn.ConvTranspose1d(64, 64, 3, stride=3, dtype=torch.float64)
        self.dec2 = nn.ConvTranspose1d(64, 32, 3, padding=1, dtype=torch.float64)
        self.dec3c = nn.ConvTranspose1d(32, 32, 5, stride=2, padding=2, output_padding=1, dtype=torch.float64)
        self.dec4 = nn.ConvTranspose1d(32, 16, 5, padding=2, dtype=torch.float64)
        self.dec5c = nn.ConvTranspose1d(16, 16, 7, stride=2, padding=3, output_padding=1, dtype=torch.float64)
        self.dec6 = nn.ConvTranspose1d(16, 8, 7, padding=3, dtype=torch.float64)
        self.dec7 = nn.ConvTranspose1d(8, 8, 9, stride=2, padding=4, output_padding=1, dtype=torch.float64)
        self.dec8 = nn.ConvTranspose1d(8, 3, 9, padding=4, dtype=torch.float64)
        # batch-normalization layers
        self.bn8 = nn.BatchNorm1d(64, dtype=torch.float64)
        self.bn9 = nn.BatchNorm1d(32, dtype=torch.float64)
        self.bn10 = nn.BatchNorm1d(32, dtype=torch.float64)
        self.bn11 = nn.BatchNorm1d(16, dtype=torch.float64)
        self.bn12 = nn.BatchNorm1d(16, dtype=torch.float64)
        self.bn13 = nn.BatchNorm1d(8, dtype=torch.float64)
        self.bn14 = nn.BatchNorm1d(8, dtype=torch.float64)
        self.bn15 = nn.BatchNorm1d(3, dtype=torch.float64)

    def forward(self, x, x1, x2, x3):
        if self.bottleneck is not None:
            # print(x.shape)
            x = x.permute(0, 2, 1)  # change to (batch_size, num_steps, num_features)
            # print(x.shape)

            if isinstance(self.bottleneck, torch.nn.LSTM):
                x, _ = self.bottleneck(x)  # LSTM will also output state variable
            else:
                x = self.bottleneck(x)

            x = x.permute(0, 2, 1)  # change to (batch_size, num_features, num_steps)
            # print(x.shape)

        # x = self.dec1c(x)
        x = F.relu(self.bn8(x + x3))
        x = F.relu(self.bn9(self.dec2(x)))
        x = self.dec3c(x)
        x = F.relu(self.bn10(x + x2))
        x = F.relu(self.bn11(self.dec4(x)))
        x = self.dec5c(x)
        x = F.relu(self.bn12(x + x1))
        x = F.relu(self.bn13(self.dec6(x)))
        x = F.relu(self.bn14(self.dec7(x)))
        x = self.bn15(self.dec8(x))

        return x


# The Encoder-Decoder model with variable bottleneck, there are two branches for the decoder part
# to handle both earthquake signal and noise
class SeisSeparator(nn.Module):
    """The encoder-decoder architecture to separate earthquake and noise signals"""

    def __init__(self, model_name, encoder, decoder1, decoder2, skip_connection=True):
        super(SeisSeparator, self).__init__()
        self.model_name = model_name
        self.encoder = encoder
        self.earthquake_decoder = decoder1
        self.noise_decoder = decoder2
        self.skip_connection = skip_connection

    def forward(self, x):
        enc_outputs, x1, x2, x3 = self.encoder(x)
        # Added a switch to turn off skip connection
        if self.skip_connection is False:
            x1, x2, x3 = 0., 0., 0.
            
        output1 = self.earthquake_decoder(enc_outputs, x1, x2, x3)
        output2 = self.noise_decoder(enc_outputs, x1, x2, x3)
        return output1, output2


# The Encoder-Decoder model with variable bottleneck
class Autoencoder_Conv1D(nn.Module):
    def __init__(self, model_name, bottleneck=None):
        """Class of the autoencoder, the type of the bottleneck can be specified"""
        super(Autoencoder_Conv1D, self).__init__()
        self.model_name = model_name
        self.enc1 = nn.Conv1d(3, 8, 9, padding='same', dtype=torch.float64)
        self.enc2 = nn.Conv1d(8, 8, 9, stride=2, padding=4, dtype=torch.float64)
        self.enc3c = nn.Conv1d(8, 16, 7, padding='same', dtype=torch.float64)
        self.enc4 = nn.Conv1d(16, 16, 7, stride=2, padding=3, dtype=torch.float64)
        self.enc5c = nn.Conv1d(16, 32, 5, padding='same', dtype=torch.float64)
        self.enc6 = nn.Conv1d(32, 32, 5, stride=2, padding=2, dtype=torch.float64)
        self.enc7c = nn.Conv1d(32, 64, 3, padding='same', dtype=torch.float64)
        self.enc8 = nn.Conv1d(64, 64, 3, stride=3, dtype=torch.float64)

        self.bottleneck = bottleneck

        self.dec1c = nn.ConvTranspose1d(64, 64, 3, stride=3, dtype=torch.float64)
        self.dec2 = nn.ConvTranspose1d(64, 32, 3, padding=1, dtype=torch.float64)
        self.dec3c = nn.ConvTranspose1d(32, 32, 5, stride=2, padding=2, output_padding=1, dtype=torch.float64)
        self.dec4 = nn.ConvTranspose1d(32, 16, 5, padding=2, dtype=torch.float64)
        self.dec5c = nn.ConvTranspose1d(16, 16, 7, stride=2, padding=3, output_padding=1, dtype=torch.float64)
        self.dec6 = nn.ConvTranspose1d(16, 8, 7, padding=3, dtype=torch.float64)
        self.dec7 = nn.ConvTranspose1d(8, 8, 9, stride=2, padding=4, output_padding=1, dtype=torch.float64)
        self.dec8 = nn.ConvTranspose1d(8, 3, 9, padding=4, dtype=torch.float64)

        self.bn1 = nn.BatchNorm1d(8, dtype=torch.float64)
        self.bn2 = nn.BatchNorm1d(8, dtype=torch.float64)
        self.bn13 = nn.BatchNorm1d(8, dtype=torch.float64)
        self.bn14 = nn.BatchNorm1d(8, dtype=torch.float64)

        self.bn3 = nn.BatchNorm1d(16, dtype=torch.float64)
        self.bn4 = nn.BatchNorm1d(16, dtype=torch.float64)
        self.bn11 = nn.BatchNorm1d(16, dtype=torch.float64)
        self.bn12 = nn.BatchNorm1d(16, dtype=torch.float64)

        self.bn5 = nn.BatchNorm1d(32, dtype=torch.float64)
        self.bn6 = nn.BatchNorm1d(32, dtype=torch.float64)
        self.bn9 = nn.BatchNorm1d(32, dtype=torch.float64)
        self.bn10 = nn.BatchNorm1d(32, dtype=torch.float64)

        self.bn7 = nn.BatchNorm1d(64, dtype=torch.float64)
        self.bn8 = nn.BatchNorm1d(64, dtype=torch.float64)

        self.bn15 = nn.BatchNorm1d(3, dtype=torch.float64)

    def forward(self, x):
        x = F.relu(self.bn1(self.enc1(x)))
        x = F.relu(self.bn2(self.enc2(x)))
        x1 = self.enc3c(x)
        x = F.relu(self.bn3(x1))
        x = F.relu(self.bn4(self.enc4(x)))
        x2 = self.enc5c(x)
        x = F.relu(self.bn5(x2))
        x = F.relu(self.bn6(self.enc6(x)))
        x3 = self.enc7c(x)
        x = F.relu(self.bn7(x3))
        # x = F.relu(self.bn7(self.enc8(x3)))

        if self.bottleneck is not None:
            # print(x.shape)
            x = x.permute(0, 2, 1)  # change to (batch_size, num_steps, num_features)
            # print(x.shape)

            if isinstance(self.bottleneck, torch.nn.LSTM):
                x, _ = self.bottleneck(x)  # LSTM will also output state variable
            else:
                x = self.bottleneck(x)

            x = x.permute(0, 2, 1)  # change to (batch_size, num_features, num_steps)
            # print(x.shape)

        # x = self.dec1c(x)
        x = F.relu(self.bn8(x + x3))
        x = F.relu(self.bn9(self.dec2(x)))
        x = self.dec3c(x)
        x = F.relu(self.bn10(x + x2))
        x = F.relu(self.bn11(self.dec4(x)))
        x = self.dec5c(x)
        x = F.relu(self.bn12(x + x1))
        x = F.relu(self.bn13(self.dec6(x)))
        x = F.relu(self.bn14(self.dec7(x)))
        x = self.bn15(self.dec8(x))

        return x


# The Encoder-Decoder model with variable bottleneck, the first layer is a Conv2D layer that tries to combine
# the information of all three channels
class Autoencoder_Conv2D(nn.Module):
    def __init__(self, model_name, bottleneck=None):
        """Class of the autoencoder, the type of the bottleneck can be specified"""
        super(Autoencoder_Conv2D, self).__init__()
        self.model_name = model_name
        self.enc1 = nn.Conv2d(1, 8, (3, 9), padding=(0, 4), dtype=torch.float64)
        self.enc2 = nn.Conv1d(8, 8, 9, stride=2, padding=4, dtype=torch.float64)
        self.enc3c = nn.Conv1d(8, 16, 7, padding='same', dtype=torch.float64)
        self.enc4 = nn.Conv1d(16, 16, 7, stride=2, padding=3, dtype=torch.float64)
        self.enc5c = nn.Conv1d(16, 32, 5, padding='same', dtype=torch.float64)
        self.enc6 = nn.Conv1d(32, 32, 5, stride=2, padding=2, dtype=torch.float64)
        self.enc7c = nn.Conv1d(32, 64, 3, padding='same', dtype=torch.float64)
        self.enc8 = nn.Conv1d(64, 64, 3, stride=3, dtype=torch.float64)

        self.bottleneck = bottleneck

        self.dec1c = nn.ConvTranspose1d(64, 64, 3, stride=3, dtype=torch.float64)
        self.dec2 = nn.ConvTranspose1d(64, 32, 3, padding=1, dtype=torch.float64)
        self.dec3c = nn.ConvTranspose1d(32, 32, 5, stride=2, padding=2, output_padding=1, dtype=torch.float64)
        self.dec4 = nn.ConvTranspose1d(32, 16, 5, padding=2, dtype=torch.float64)
        self.dec5c = nn.ConvTranspose1d(16, 16, 7, stride=2, padding=3, output_padding=1, dtype=torch.float64)
        self.dec6 = nn.ConvTranspose1d(16, 8, 7, padding=3, dtype=torch.float64)
        self.dec7 = nn.ConvTranspose1d(8, 8, 9, stride=2, padding=4, output_padding=1, dtype=torch.float64)
        self.dec8 = nn.ConvTranspose1d(8, 3, 9, padding=4, dtype=torch.float64)

        self.bn1 = nn.BatchNorm1d(8, dtype=torch.float64)
        self.bn2 = nn.BatchNorm1d(8, dtype=torch.float64)
        self.bn13 = nn.BatchNorm1d(8, dtype=torch.float64)
        self.bn14 = nn.BatchNorm1d(8, dtype=torch.float64)

        self.bn3 = nn.BatchNorm1d(16, dtype=torch.float64)
        self.bn4 = nn.BatchNorm1d(16, dtype=torch.float64)
        self.bn11 = nn.BatchNorm1d(16, dtype=torch.float64)
        self.bn12 = nn.BatchNorm1d(16, dtype=torch.float64)

        self.bn5 = nn.BatchNorm1d(32, dtype=torch.float64)
        self.bn6 = nn.BatchNorm1d(32, dtype=torch.float64)
        self.bn9 = nn.BatchNorm1d(32, dtype=torch.float64)
        self.bn10 = nn.BatchNorm1d(32, dtype=torch.float64)

        self.bn7 = nn.BatchNorm1d(64, dtype=torch.float64)
        self.bn8 = nn.BatchNorm1d(64, dtype=torch.float64)

        self.bn15 = nn.BatchNorm1d(3, dtype=torch.float64)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.enc1(x).squeeze()
        x = F.relu(self.bn1(x))
        x = F.relu(self.bn2(self.enc2(x)))
        x1 = self.enc3c(x)
        x = F.relu(self.bn3(x1))
        x = F.relu(self.bn4(self.enc4(x)))
        x2 = self.enc5c(x)
        x = F.relu(self.bn5(x2))
        x = F.relu(self.bn6(self.enc6(x)))
        x3 = self.enc7c(x)
        x = F.relu(self.bn7(x3))
        # x = F.relu(self.bn7(self.enc8(x3)))

        if self.bottleneck is not None:
            # print(x.shape)
            x = x.permute(0, 2, 1)  # change to (batch_size, num_steps, num_features)
            # print(x.shape)

            if isinstance(self.bottleneck, torch.nn.LSTM):
                x, _ = self.bottleneck(x)  # LSTM will also output state variable
            else:
                x = self.bottleneck(x)

            x = x.permute(0, 2, 1)  # change to (batch_size, num_features, num_steps)
            # print(x.shape)

        # x = self.dec1c(x)
        x = F.relu(self.bn8(x + x3))
        x = F.relu(self.bn9(self.dec2(x)))
        x = self.dec3c(x)
        x = F.relu(self.bn10(x + x2))
        x = F.relu(self.bn11(self.dec4(x)))
        x = self.dec5c(x)
        x = F.relu(self.bn12(x + x1))
        x = F.relu(self.bn13(self.dec6(x)))
        x = F.relu(self.bn14(self.dec7(x)))
        x = self.bn15(self.dec8(x))

        return x

    # Below are the classes and functions used to define the self-attention


class DotProductAttention(nn.Module):
    """Scaled dot product attention."""

    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # Shape of `queries`: (`batch_size`, no. of queries, `d`)
    # Shape of `keys`: (`batch_size`, no. of key-value pairs, `d`)
    # Shape of `values`: (`batch_size`, no. of key-value pairs, value
    # dimension)
    # Shape of `valid_lens`: (`batch_size`,) or (`batch_size`, no. of queries)
    def forward(self, queries, keys, values):
        d = queries.shape[-1]
        # Set `transpose_b=True` to swap the last two dimensions of `keys`
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = F.softmax(scores, dim=0)
        return torch.bmm(self.dropout(self.attention_weights), values)


class PositionalEncoding(nn.Module):
    """Positional encoding."""

    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough `P`
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(
            10000,
            torch.arange(0, num_hiddens, 2, dtype=torch.float32) /
            num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)


def transpose_qkv(X, num_heads):
    """Transposition for parallel computation of multiple attention heads."""
    # Shape of input `X`:
    # (`batch_size`, no. of queries or key-value pairs, `num_hiddens`).
    # Shape of output `X`:
    # (`batch_size`, no. of queries or key-value pairs, `num_heads`,
    # `num_hiddens` / `num_heads`)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # Shape of output `X`:
    # (`batch_size`, `num_heads`, no. of queries or key-value pairs,
    # `num_hiddens` / `num_heads`)
    X = X.permute(0, 2, 1, 3)

    # Shape of `output`:
    # (`batch_size` * `num_heads`, no. of queries or key-value pairs,
    # `num_hiddens` / `num_heads`)
    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):
    """Reverse the operation of `transpose_qkv`."""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


class MultiHeadAttention(nn.Module):
    """Multi-head attention."""

    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias, dtype=torch.float64)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias, dtype=torch.float64)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias, dtype=torch.float64)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias, dtype=torch.float64)

    def forward(self, queries, keys, values):
        # Shape of `queries`, `keys`, or `values`:
        # (`batch_size`, no. of queries or key-value pairs, `num_hiddens`)
        # Shape of `valid_lens`:
        # (`batch_size`,) or (`batch_size`, no. of queries)
        # After transposing, shape of output `queries`, `keys`, or `values`:
        # (`batch_size` * `num_heads`, no. of queries or key-value pairs,
        # `num_hiddens` / `num_heads`)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        # Shape of `output`: (`batch_size` * `num_heads`, no. of queries,
        # `num_hiddens` / `num_heads`)
        output = self.attention(queries, keys, values)

        # Shape of `output_concat`:
        # (`batch_size`, no. of queries, `num_hiddens`)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


class Attention_bottleneck(nn.Module):
    def __init__(self, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super(Attention_bottleneck, self).__init__(**kwargs)
        self.pe = PositionalEncoding(num_hiddens, dropout)
        self.attention = MultiHeadAttention(num_hiddens, num_hiddens,
                                            num_hiddens, num_hiddens,
                                            num_heads, dropout)

    def forward(self, X):
        # shape of X: (batch_size, num_steps, num_features)
        X = self.pe(X)  # postional encoding
        X = self.attention(X, X, X)  # self attention
        return X


class Attention_bottleneck_LSTM(nn.Module):
    def __init__(self, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super(Attention_bottleneck_LSTM, self).__init__(**kwargs)
        # self.pe = PositionalEncoding(num_hiddens, dropout)
        self.lstm = torch.nn.LSTM(num_hiddens, int(num_hiddens / 2), 1, bidirectional=True,
                                  batch_first=True, dtype=torch.float64)
        self.attention = MultiHeadAttention(num_hiddens, num_hiddens,
                                            num_hiddens, num_hiddens,
                                            num_heads, dropout)

    def forward(self, X):
        # shape of X: (batch_size, num_steps, num_features)
        X, _ = self.lstm(X)  # postional encoding
        X = self.attention(X, X, X)  # self attention
        return X



### The input adaptors for Teleseismic waves (author: Qibin Shi)
class InputLinear(nn.Module):
    def __init__(self, insize, outsize):
        super().__init__()
        self.fc1 = nn.Linear(insize, outsize, dtype=torch.float64)
        self.bn1 = nn.BatchNorm1d(3, dtype=torch.float64)

    def forward(self, x):
        y = F.relu(self.bn1(self.fc1(x)))
        return y

class InputConv(nn.Module):
    def __init__(self, insize, outsize, kernelsize, dsamp, pad):
        super().__init__()
        self.enc1 = nn.Conv1d(insize, outsize, kernelsize, stride=dsamp, padding=pad, dtype=torch.float64)
        self.bn1 = nn.BatchNorm1d(outsize, dtype=torch.float64)

    def forward(self, x):
        y = F.relu(self.bn1(self.enc1(x)))
        return y

### The output adatptors for teleseismic waves (author: Qibin Shi)
class OutputLinear(nn.Module):
    def __init__(self, insize, outsize):
        super().__init__()
        self.fc1 = nn.Linear(insize, outsize, dtype=torch.float64)
        self.fc2 = nn.Linear(insize, outsize, dtype=torch.float64)
        self.bn1 = nn.BatchNorm1d(3, dtype=torch.float64)
        self.bn2 = nn.BatchNorm1d(3, dtype=torch.float64)

    def forward(self, x0):
        x1 = x0[0]
        x2 = x0[1]
        x = self.bn1(self.fc1(F.relu(x1)))
        y = self.bn2(self.fc2(F.relu(x2)))
        return x, y

class OutputDconv(nn.Module):
    def __init__(self, insize, outsize, kernelsize, usamp, pad, opad):
        super().__init__()
        self.dec1 = nn.ConvTranspose1d(insize, outsize, kernelsize, stride=usamp, padding=pad, output_padding=opad, dtype=torch.float64)
        self.dec2 = nn.ConvTranspose1d(insize, outsize, kernelsize, stride=usamp, padding=pad, output_padding=opad, dtype=torch.float64)
        self.bn1 = nn.BatchNorm1d(outsize, dtype=torch.float64)
        self.bn2 = nn.BatchNorm1d(outsize, dtype=torch.float64)

    def forward(self, x0):
        x1 = x0[0]
        x2 = x0[1]
        x = self.bn1(self.dec1(F.relu(x1)))
        y = self.bn2(self.dec2(F.relu(x2)))
        return x, y


class T_model(nn.Module):
    def __init__(self, model, half_insize=750):
        super(T_model, self).__init__()

        self.enc1 = nn.Conv1d(3, 8, 45, stride=1, padding='same', dtype=torch.float64)
        self.bn1 = nn.BatchNorm1d(8, dtype=torch.float64)
        self.enc2 = nn.Conv1d(8, 8, 21, stride=2, padding=10, dtype=torch.float64)
        self.bn2 = nn.BatchNorm1d(8, dtype=torch.float64)
        self.enc3 = nn.Conv1d(8, 8, 21, stride=2, padding=10, dtype=torch.float64)
        self.bn3 = nn.BatchNorm1d(8, dtype=torch.float64)
        self.enc4 = nn.Conv1d(8, 8, 21, stride=2, padding=10, dtype=torch.float64)
        self.bn4 = nn.BatchNorm1d(8, dtype=torch.float64)
        self.enc5 = nn.Conv1d(8, 3, 9, stride=1, padding='same', dtype=torch.float64)
        self.bn5 = nn.BatchNorm1d(3, dtype=torch.float64)
        self.fc6 = nn.Linear(half_insize, 600, dtype=torch.float64)
        self.bn6 = nn.BatchNorm1d(3, dtype=torch.float64)

        self.model = model

        self.fc61 = nn.Linear(600, half_insize, dtype=torch.float64)
        self.fc62 = nn.Linear(600, half_insize, dtype=torch.float64)
        self.bn61 = nn.BatchNorm1d(3, dtype=torch.float64)
        self.bn62 = nn.BatchNorm1d(3, dtype=torch.float64)
        self.dec51 = nn.ConvTranspose1d(3, 8, 9, stride=1, padding=4, output_padding=0, dtype=torch.float64)
        self.dec52 = nn.ConvTranspose1d(3, 8, 9, stride=1, padding=4, output_padding=0, dtype=torch.float64)
        self.bn51 = nn.BatchNorm1d(8, dtype=torch.float64)
        self.bn52 = nn.BatchNorm1d(8, dtype=torch.float64)
        self.dec41 = nn.ConvTranspose1d(8, 8, 21, stride=2, padding=10, output_padding=1, dtype=torch.float64)
        self.dec42 = nn.ConvTranspose1d(8, 8, 21, stride=2, padding=10, output_padding=1, dtype=torch.float64)
        self.bn41 = nn.BatchNorm1d(8, dtype=torch.float64)
        self.bn42 = nn.BatchNorm1d(8, dtype=torch.float64)
        self.dec31 = nn.ConvTranspose1d(8, 8, 21, stride=2, padding=10, output_padding=1, dtype=torch.float64)
        self.dec32 = nn.ConvTranspose1d(8, 8, 21, stride=2, padding=10, output_padding=1, dtype=torch.float64)
        self.bn31 = nn.BatchNorm1d(8, dtype=torch.float64)
        self.bn32 = nn.BatchNorm1d(8, dtype=torch.float64)
        self.dec21 = nn.ConvTranspose1d(8, 8, 21, stride=2, padding=10, output_padding=1, dtype=torch.float64)
        self.dec22 = nn.ConvTranspose1d(8, 8, 21, stride=2, padding=10, output_padding=1, dtype=torch.float64)
        self.bn21 = nn.BatchNorm1d(8, dtype=torch.float64)
        self.bn22 = nn.BatchNorm1d(8, dtype=torch.float64)
        self.dec11 = nn.ConvTranspose1d(8, 3, 45, stride=1, padding=22, output_padding=0, dtype=torch.float64)
        self.dec12 = nn.ConvTranspose1d(8, 3, 45, stride=1, padding=22, output_padding=0, dtype=torch.float64)
        self.bn11 = nn.BatchNorm1d(3, dtype=torch.float64)
        self.bn12 = nn.BatchNorm1d(3, dtype=torch.float64)

    def forward(self, x):
        x = F.relu(self.bn1(self.enc1(x)))
        #x = F.relu(self.bn2(self.enc2(x)))
        x0= self.enc3(x)
        x = F.relu(self.bn3(x0))
        #x = F.relu(self.bn4(self.enc4(x)))
        x1= self.enc5(x)
        x = F.relu(self.bn5(x1))
        x = F.relu(self.bn6(self.fc6(x)))

        y = self.model(x)
        y1 = y[0]
        y2 = y[1]

        y1 = self.bn61(self.fc61(F.relu(y1)))
        y2 = self.bn62(self.fc62(F.relu(y2)))
        y1 = self.bn51(self.dec51(F.relu(y1)) + x0)
        y2 = self.bn52(self.dec52(F.relu(y2)) + x0)
        # y1 = self.bn41(self.dec41(F.relu(y1)))
        # y2 = self.bn42(self.dec42(F.relu(y2)))
        y1 = self.bn31(self.dec31(F.relu(y1)))
        y2 = self.bn32(self.dec32(F.relu(y2)))
        # y1 = self.bn21(self.dec21(F.relu(y1)))
        # y2 = self.bn22(self.dec22(F.relu(y2)))
        y1 = self.bn11(self.dec11(F.relu(y1)))
        y2 = self.bn12(self.dec12(F.relu(y2)))

        return y1, y2


class T_model_S(nn.Module):
    def __init__(self, model):
        super(T_model_S, self).__init__()

        self.enc1 = nn.Conv1d(3, 8, 45, stride=1, padding='same', dtype=torch.float64)
        self.bn1 = nn.BatchNorm1d(8, dtype=torch.float64)
        self.enc2 = nn.Conv1d(8, 8, 21, stride=2, padding=10, dtype=torch.float64)
        self.bn2 = nn.BatchNorm1d(8, dtype=torch.float64)
        self.enc3 = nn.Conv1d(8, 8, 21, stride=2, padding=10, dtype=torch.float64)
        self.bn3 = nn.BatchNorm1d(8, dtype=torch.float64)
        self.enc4 = nn.Conv1d(8, 8, 21, stride=2, padding=10, dtype=torch.float64)
        self.bn4 = nn.BatchNorm1d(8, dtype=torch.float64)
        self.enc5 = nn.Conv1d(8, 3, 9, stride=1, padding='same', dtype=torch.float64)
        self.bn5 = nn.BatchNorm1d(3, dtype=torch.float64)
        self.fc6 = nn.Linear(2500, 600, dtype=torch.float64)
        self.bn6 = nn.BatchNorm1d(3, dtype=torch.float64)

        self.model = model

        self.fc61 = nn.Linear(600, 2500, dtype=torch.float64)
        self.fc62 = nn.Linear(600, 2500, dtype=torch.float64)
        self.bn61 = nn.BatchNorm1d(3, dtype=torch.float64)
        self.bn62 = nn.BatchNorm1d(3, dtype=torch.float64)
        self.dec51 = nn.ConvTranspose1d(3, 8, 9, stride=1, padding=4, output_padding=0, dtype=torch.float64)
        self.dec52 = nn.ConvTranspose1d(3, 8, 9, stride=1, padding=4, output_padding=0, dtype=torch.float64)
        self.bn51 = nn.BatchNorm1d(8, dtype=torch.float64)
        self.bn52 = nn.BatchNorm1d(8, dtype=torch.float64)
        self.dec41 = nn.ConvTranspose1d(8, 8, 21, stride=2, padding=10, output_padding=1, dtype=torch.float64)
        self.dec42 = nn.ConvTranspose1d(8, 8, 21, stride=2, padding=10, output_padding=1, dtype=torch.float64)
        self.bn41 = nn.BatchNorm1d(8, dtype=torch.float64)
        self.bn42 = nn.BatchNorm1d(8, dtype=torch.float64)
        self.dec31 = nn.ConvTranspose1d(8, 8, 21, stride=2, padding=10, output_padding=1, dtype=torch.float64)
        self.dec32 = nn.ConvTranspose1d(8, 8, 21, stride=2, padding=10, output_padding=1, dtype=torch.float64)
        self.bn31 = nn.BatchNorm1d(8, dtype=torch.float64)
        self.bn32 = nn.BatchNorm1d(8, dtype=torch.float64)
        self.dec21 = nn.ConvTranspose1d(8, 8, 21, stride=2, padding=10, output_padding=1, dtype=torch.float64)
        self.dec22 = nn.ConvTranspose1d(8, 8, 21, stride=2, padding=10, output_padding=1, dtype=torch.float64)
        self.bn21 = nn.BatchNorm1d(8, dtype=torch.float64)
        self.bn22 = nn.BatchNorm1d(8, dtype=torch.float64)
        self.dec11 = nn.ConvTranspose1d(8, 3, 45, stride=1, padding=22, output_padding=0, dtype=torch.float64)
        self.dec12 = nn.ConvTranspose1d(8, 3, 45, stride=1, padding=22, output_padding=0, dtype=torch.float64)
        self.bn11 = nn.BatchNorm1d(3, dtype=torch.float64)
        self.bn12 = nn.BatchNorm1d(3, dtype=torch.float64)

    def forward(self, x):
        x = F.relu(self.bn1(self.enc1(x)))
        #x = F.relu(self.bn2(self.enc2(x)))
        x0= self.enc3(x)
        x = F.relu(self.bn3(x0))
        #x = F.relu(self.bn4(self.enc4(x)))
        x1= self.enc5(x)
        x = F.relu(self.bn5(x1))
        x = F.relu(self.bn6(self.fc6(x)))

        y = self.model(x)
        y1 = y[0]
        y2 = y[1]

        y1 = self.bn61(self.fc61(F.relu(y1)))
        y2 = self.bn62(self.fc62(F.relu(y2)))
        y1 = self.bn51(self.dec51(F.relu(y1)) + x0)
        y2 = self.bn52(self.dec52(F.relu(y2)) + x0)
        # y1 = self.bn41(self.dec41(F.relu(y1)))
        # y2 = self.bn42(self.dec42(F.relu(y2)))
        y1 = self.bn31(self.dec31(F.relu(y1)))
        y2 = self.bn32(self.dec32(F.relu(y2)))
        # y1 = self.bn21(self.dec21(F.relu(y1)))
        # y2 = self.bn22(self.dec22(F.relu(y2)))
        y1 = self.bn11(self.dec11(F.relu(y1)))
        y2 = self.bn12(self.dec12(F.relu(y2)))

        return y1, y2


class W_model(nn.Module):
    def __init__(self, model):
        super(W_model, self).__init__()

        self.enc1 = nn.Conv1d(3, 8, 45, stride=1, padding='same', dtype=torch.float64)
        self.bn1 = nn.BatchNorm1d(8, dtype=torch.float64)
        self.enc2 = nn.Conv1d(8, 8, 21, stride=2, padding=10, dtype=torch.float64)
        self.bn2 = nn.BatchNorm1d(8, dtype=torch.float64)
        self.enc3 = nn.Conv1d(8, 8, 21, stride=2, padding=10, dtype=torch.float64)
        self.bn3 = nn.BatchNorm1d(8, dtype=torch.float64)
        self.enc4 = nn.Conv1d(8, 8, 21, stride=2, padding=10, dtype=torch.float64)
        self.bn4 = nn.BatchNorm1d(8, dtype=torch.float64)
        self.enc5 = nn.Conv1d(8, 3, 9, stride=1, padding='same', dtype=torch.float64)
        self.bn5 = nn.BatchNorm1d(3, dtype=torch.float64)
        self.fc6 = nn.Linear(3000, 600, dtype=torch.float64)
        self.bn6 = nn.BatchNorm1d(3, dtype=torch.float64)

        self.model = model

        self.fc61 = nn.Linear(600, 3000, dtype=torch.float64)
        self.fc62 = nn.Linear(600, 3000, dtype=torch.float64)
        self.bn61 = nn.BatchNorm1d(3, dtype=torch.float64)
        self.bn62 = nn.BatchNorm1d(3, dtype=torch.float64)
        self.dec51 = nn.ConvTranspose1d(3, 8, 9, stride=1, padding=4, output_padding=0, dtype=torch.float64)
        self.dec52 = nn.ConvTranspose1d(3, 8, 9, stride=1, padding=4, output_padding=0, dtype=torch.float64)
        self.bn51 = nn.BatchNorm1d(16, dtype=torch.float64)
        self.bn52 = nn.BatchNorm1d(16, dtype=torch.float64)
        self.dec41 = nn.ConvTranspose1d(8, 8, 21, stride=2, padding=10, output_padding=1, dtype=torch.float64)
        self.dec42 = nn.ConvTranspose1d(8, 8, 21, stride=2, padding=10, output_padding=1, dtype=torch.float64)
        self.bn41 = nn.BatchNorm1d(8, dtype=torch.float64)
        self.bn42 = nn.BatchNorm1d(8, dtype=torch.float64)
        self.dec31 = nn.ConvTranspose1d(16, 8, 21, stride=2, padding=10, output_padding=1, dtype=torch.float64)
        self.dec32 = nn.ConvTranspose1d(16, 8, 21, stride=2, padding=10, output_padding=1, dtype=torch.float64)
        self.bn31 = nn.BatchNorm1d(8, dtype=torch.float64)
        self.bn32 = nn.BatchNorm1d(8, dtype=torch.float64)
        self.dec21 = nn.ConvTranspose1d(8, 8, 21, stride=2, padding=10, output_padding=1, dtype=torch.float64)
        self.dec22 = nn.ConvTranspose1d(8, 8, 21, stride=2, padding=10, output_padding=1, dtype=torch.float64)
        self.bn21 = nn.BatchNorm1d(8, dtype=torch.float64)
        self.bn22 = nn.BatchNorm1d(8, dtype=torch.float64)
        self.dec11 = nn.ConvTranspose1d(8, 3, 45, stride=1, padding=22, output_padding=0, dtype=torch.float64)
        self.dec12 = nn.ConvTranspose1d(8, 3, 45, stride=1, padding=22, output_padding=0, dtype=torch.float64)
        self.bn11 = nn.BatchNorm1d(3, dtype=torch.float64)
        self.bn12 = nn.BatchNorm1d(3, dtype=torch.float64)

    def forward(self, x):
        x = F.relu(self.bn1(self.enc1(x)))
        #x = F.relu(self.bn2(self.enc2(x)))
        x0= self.enc3(x)
        x = F.relu(self.bn3(x0))
        #x = F.relu(self.bn4(self.enc4(x)))
        x1= self.enc5(x)
        x = F.relu(self.bn5(x1))
        x = F.relu(self.bn6(self.fc6(x)))

        y = self.model(x)
        y1 = y[0]
        y2 = y[1]

        y1 = self.bn61(self.fc61(F.relu(y1)))
        y2 = self.bn62(self.fc62(F.relu(y2)))
        y1 = self.dec51(F.relu(y1))
        y2 = self.dec52(F.relu(y2))
        y1 = torch.cat((y1, x0), 1)
        y2 = torch.cat((y2, x0), 1)
        y1 = self.bn51(y1)
        y2 = self.bn52(y2)
        # y1 = self.bn41(self.dec41(F.relu(y1)))
        # y2 = self.bn42(self.dec42(F.relu(y2)))
        y1 = self.bn31(self.dec31(F.relu(y1)))
        y2 = self.bn32(self.dec32(F.relu(y2)))
        # y1 = self.bn21(self.dec21(F.relu(y1)))
        # y2 = self.bn22(self.dec22(F.relu(y2)))
        y1 = self.bn11(self.dec11(F.relu(y1)))
        y2 = self.bn12(self.dec12(F.relu(y2)))

        return y1, y2
