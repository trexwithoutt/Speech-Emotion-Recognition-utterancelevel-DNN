import os
import re
import sys
import json
import requests
import subprocess
from tqdm import tqdm
from contextlib import closing
from multiprocessing import Pool
from collections import namedtuple
from datetime import datetime, timedelta
from shutil import copyfile as copy_file
import math
import numpy as np
import numpy
import tensorflow as tf
from scipy import signal
import librosa
import librosa.filters
import librosa
import numpy as np
import six
import scipy.fftpack as fft
import scipy
from numpy.lib.stride_tricks import as_strided
import scipy.io.wavfile as wav

#based on librosa library
def power_to_db(S, ref=1.0, amin=1e-10, top_db=80.0):
    S = np.asarray(S)

    if np.issubdtype(S.dtype, np.complexfloating):
        magnitude = np.abs(S)
    else:
        magnitude = S

    if six.callable(ref):
        # User supplied a function to calculate reference power
        ref_value = ref(magnitude)
    else:
        ref_value = np.abs(ref)

    log_spec = 10.0 * np.log10(np.maximum(amin, magnitude))
    log_spec -= 10.0 * np.log10(np.maximum(amin, ref_value))

    if top_db is not None:
        if top_db < 0:
            raise print('top_db must be non-negative')
        log_spec = np.maximum(log_spec, log_spec.max() - top_db)

    return log_spec

def db_to_power(S_db, ref=1.0):
    return ref * np.power(10.0, 0.1 * S_db)

def hz_to_mel(frequencies, htk=False):
    frequencies = np.asanyarray(frequencies)

    if htk:
        return 2595.0 * np.log10(1.0 + frequencies / 700.0)

    # Fill in the linear part
    f_min = 0.0
    f_sp = 200.0 / 3

    mels = (frequencies - f_min) / f_sp

    # Fill in the log-scale part

    min_log_hz = 1000.0                         # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp   # same (Mels)
    logstep = np.log(6.4) / 27.0                # step size for log region

    if frequencies.ndim:
        # If we have array data, vectorize
        log_t = (frequencies >= min_log_hz)
        mels[log_t] = min_log_mel + np.log(frequencies[log_t]/min_log_hz) / logstep
    elif frequencies >= min_log_hz:
        # If we have scalar data, heck directly
        mels = min_log_mel + np.log(frequencies / min_log_hz) / logstep

    return mels


def mel_to_hz(mels, htk=False):
    mels = np.asanyarray(mels)

    if htk:
        return 700.0 * (10.0**(mels / 2595.0) - 1.0)

    # Fill in the linear scale
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels

    # And now the nonlinear scale
    min_log_hz = 1000.0                         # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp   # same (Mels)
    logstep = np.log(6.4) / 27.0                # step size for log region

    if mels.ndim:
        # If we have vector data, vectorize
        log_t = (mels >= min_log_mel)
        freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
    elif mels >= min_log_mel:
        # If we have scalar data, check directly
        freqs = min_log_hz * np.exp(logstep * (mels - min_log_mel))

    return freqs


def mel_frequencies(n_mels=128, fmin=0.0, fmax=11025.0, htk=False):
    # 'Center freqs' of mel bands - uniformly spaced between limits
    min_mel = hz_to_mel(fmin, htk=htk)
    max_mel = hz_to_mel(fmax, htk=htk)

    mels = np.linspace(min_mel, max_mel, n_mels)

    return mel_to_hz(mels, htk=htk)

def mel(sr, n_fft, n_mels=128, fmin=0.0, fmax=None, htk=False,
        norm=1):

    if fmax is None:
        fmax = float(sr) / 2

    # Initialize the weights
    n_mels = int(n_mels)
    weights = np.zeros((n_mels, int(1 + n_fft // 2)))

    # Center freqs of each FFT bin
    
    fftfreqs = np.linspace(0,float(sr) / 2, int(1 + n_fft//2), endpoint=True)

    # 'Center freqs' of mel bands - uniformly spaced between limits
    mel_f = mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax, htk=htk)

    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)

    for i in range(n_mels):
        # lower and upper slopes for all bins
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i+2] / fdiff[i+1]

        # .. then intersect them with each other and zero
        weights[i] = np.maximum(0, np.minimum(lower, upper))

    if norm == 1:
        # Slaney-style mel is scaled to be approx constant energy per channel
        enorm = 2.0 / (mel_f[2:n_mels+2] - mel_f[:n_mels])
        weights *= enorm[:, np.newaxis]

    return weights

def dct(n_filters, n_input):
    basis = np.empty((n_filters, n_input))
    basis[0, :] = 1.0 / np.sqrt(n_input)

    samples = np.arange(1, 2*n_input, 2) * np.pi / (2.0 * n_input)

    for i in range(1, n_filters):
        basis[i, :] = np.cos(i*samples) * np.sqrt(2.0/n_input)

    return basis

def preemphasis(x):
    return signal.lfilter([1, -0.97], [1], x)

def inv_preemphasis(x):
    return signal.lfilter([1], [1, -0.97], x)


def framing(y, win_length=400, hop_length=160):
    n_frames = 1 + int((len(y) - win_length) / hop_length)
    y_frames = np.lib.stride_tricks.as_strided(y, shape=(win_length, n_frames), strides=(y.itemsize, hop_length * y.itemsize))
    return y_frames.T

def stft(y, n_fft=512, win_length=400, hop_length=160, window='hann'):
    stft_window = scipy.signal.get_window(window, win_length, fftbins=True)
    frames = framing(preemphasis(y), win_length=win_length, hop_length=hop_length)
    return np.fft.rfft( stft_window * frames , n_fft, axis=1).T

def spectrogram(y=None, n_fft=512, win_length=400, hop_length=160, power=1):
    return np.abs(stft(y, n_fft=n_fft, win_length=win_length, hop_length=hop_length))**power

def melspectrogram(y=None, sr=16000,  n_fft=512, win_length=400, hop_length=160, power=2.0, n_mels=128):
    S = spectrogram(y=y, n_fft=n_fft, hop_length=hop_length, power=power)
    mel_basis = mel(sr, n_fft, n_mels)
    return np.dot(mel_basis, S)

def mfcc(y=None, sr=16000, n_mfcc=13, n_fft=512, win_length=400, hop_length=160):
    S = power_to_db(melspectrogram(y=y, sr=sr, n_fft=n_fft, win_length=win_length, hop_length=hop_length))
    return np.dot(dct(n_mfcc, S.shape[0]), S)

#based on https://github.com/tyiannak/pyAudioAnalysis/blob/master/audioFeatureExtraction.py
def stZCR(frame):
    """Computes zero crossing rate of frame"""
    count = len(frame)
    countZ = numpy.sum(numpy.abs(numpy.diff(numpy.sign(frame)))) / 2
    return (numpy.float64(countZ) / numpy.float64(count-1.0))


def stEnergy(frame):
    """Computes signal energy of frame"""
    return numpy.sum(frame ** 2) / numpy.float64(len(frame))


def stEnergyEntropy(frame, numOfShortBlocks=10):
    """Computes entropy of energy"""
    Eol = numpy.sum(frame ** 2)    # total frame energy
    L = len(frame)
    subWinLength = int(numpy.floor(L / numOfShortBlocks))
    if L != subWinLength * numOfShortBlocks:
            frame = frame[0:subWinLength * numOfShortBlocks]
    # subWindows is of size [numOfShortBlocks x L]
    subWindows = frame.reshape(subWinLength, numOfShortBlocks, order='F').copy()

    # Compute normalized sub-frame energies:
    s = numpy.sum(subWindows ** 2, axis=0) / (Eol + eps)

    # Compute entropy of the normalized sub-frame energies:
    Entropy = -numpy.sum(s * numpy.log2(s + eps))
    return Entropy

def stHarmonic(frame, fs):
    """
    Computes harmonic ratio and pitch
    """
    M = numpy.round(0.016 * fs) - 1
    R = numpy.correlate(frame, frame, mode='full')

    g = R[len(frame)-1]
    R = R[len(frame):-1]

    # estimate m0 (as the first zero crossing of R)
    [a, ] = numpy.nonzero(numpy.diff(numpy.sign(R)))

    if len(a) == 0:
        m0 = len(R)-1
    else:
        m0 = a[0]
    if M > len(R):
        M = len(R) - 1

    Gamma = numpy.zeros((M), dtype=numpy.float64)
    CSum = numpy.cumsum(frame ** 2)
    Gamma[m0:M] = R[m0:M] / (numpy.sqrt((g * CSum[M:m0:-1])) + eps)

    ZCR = stZCR(Gamma)

    if ZCR > 0.15:
        HR = 0.0
        f0 = 0.0
    else:
        if len(Gamma) == 0:
            HR = 1.0
            blag = 0.0
            Gamma = numpy.zeros((M), dtype=numpy.float64)
        else:
            HR = numpy.max(Gamma)
            blag = numpy.argmax(Gamma)

        # Get fundamental frequency:
        f0 = fs / (blag + eps)
        if f0 > 5000:
            f0 = 0.0
        if HR < 0.1:
            f0 = 0.0

    return (HR, f0)

def zero_crossings(y, threshold=1e-10, ref_magnitude=None, pad=True, zero_pos=True, axis=-1):
    # Clip within the threshold
    if threshold is None:
        threshold = 0.0

    if six.callable(ref_magnitude):
        threshold = threshold * ref_magnitude(np.abs(y))

    elif ref_magnitude is not None:
        threshold = threshold * ref_magnitude

    if threshold > 0:
        y = y.copy()
        y[np.abs(y) <= threshold] = 0

    # Extract the sign bit
    if zero_pos:
        y_sign = np.signbit(y)
    else:
        y_sign = np.sign(y)

    # Find the change-points by slicing
    slice_pre = [slice(None)] * y.ndim
    slice_pre[axis] = slice(1, None)

    slice_post = [slice(None)] * y.ndim
    slice_post[axis] = slice(-1)

    # Since we've offset the input by one, pad back onto the front
    padding = [(0, 0)] * y.ndim
    padding[axis] = (1, 0)

    return np.pad((y_sign[slice_post] != y_sign[slice_pre]),padding,mode='constant',constant_values=pad).transpose()

def zero_crossing_rate(y, win_length=400, hop_length=160):
    y_framed = framing(y, win_length=400, hop_length=160)
    crossings = zero_crossings(y_framed)
    return np.mean(crossings, axis=0, keepdims=True)

def _autocorrelation(x):
    x = x-np.mean(x)
    correlation = np.correlate(x,x,mode='ful')/np.sum(x**2)
    return correlation[int(len(correlation)/2):]

def HNR(x,pitch):
    '''this fuction is to caculate harmonics-to-noise ratio,using the autocorrelation function.
    '''
    acf = _autocorrelation(x)
    t0 = acf[0]
    t1 = acf[pitch]
    
    return 10*np.log(np.abs(t1/(t0-t1)))

from scipy.fftpack import fft,ifft

def _cepstrum(y,win_length,hop_length):
    '''
    the process of caculating cepstrum:
        singal->fft->abs->log->ifft->cepstrum
    This script using the fft and ifft of scipy package.
    y: the singal.
    '''
    # windowing,using hanming window.
    window = np.hamming(win_length)
    for i in range(int(np.floor(y.size/hop_length))):
        win_start = i*hop_length
        win_end = (i+1)*hop_length
        if win_end >=y.size:
            win_end = y.size
        for j in range(win_length):
            if j+win_start<win_end:
                y = np.array(y)
                y[win_start+j] *= window[j]

    #fft->abs->ifft
    ceps = ifft(np.log(np.abs(fft(y))))
    ceps = ceps.real
    return ceps
    
def _pitch_period_detection(c,fs):
    '''
        this function using the cepstrum(c) of a singal to get the pitch,
        which is the inverse of pitch period. 
    '''
    ms2=int(np.floor(fs*0.002))
    ms20=int(np.floor(fs*0.02))
    t_c = np.array(c[ms2:ms20])
    t_c = t_c.argsort()
    idx =t_c[-1]
    f0 = fs/(ms2+idx-1)
    return 1.0/f0



###########################################################################################


def pitch_based_features(y, sr, win_length, hop_length):
    y_frames = framing(y, win_length=win_length, hop_length=hop_length)
    pitch_based_features=[]
    for samples_in_window in y_frames:
        ceps = _cepstrum(samples_in_window,win_length,hop_length)
        pitch_period = _pitch_period_detection(ceps,sr)
        t = int(sr*pitch_period)
        HNR_m = HNR(samples_in_window,t)
        feature = [pitch_period,HNR_m]
        pitch_based_features.append(feature)
    return np.array(pitch_based_features).T