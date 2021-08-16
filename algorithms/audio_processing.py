# -*- coding: utf-8 -*-
__author__ = "Lukas Pfeifenberger"


import os
import numpy as np



#----------------------------------------------------------------
# wrapper for python fft
def rfft(Bx, axis=None):

    Fx = np.fft.rfft(Bx, axis=axis)
    return Fx



#----------------------------------------------------------------
def create_hanning_window(wlen):

    t = np.arange(wlen)
    window = 0.5*(1-np.cos(2*np.pi*(t+1)/(wlen+1)))
    window = np.sqrt(window)

    return window



#----------------------------------------------------------------
# perform a multichannel STFT on audio data x
# x.shape = (samples, nchan)
# output Fx.shape = (nbin, nfram)
def mstft(x, wlen=1024):


    x = np.asarray(x, dtype=np.float32)
    samples_x = x.shape[0]
    shift = int(wlen/2)
    nbin = int(wlen/2+1)

    window = create_hanning_window(wlen)

    nfram = int(np.ceil( (samples_x-wlen+shift)/shift ))
    samples = nfram*shift+wlen-shift

    #zero-pad if necessary
    if samples > samples_x:
        pad = np.zeros((samples-samples_x,), dtype=x.dtype)
        x = np.concatenate((x,pad), axis=0)

    Fx = np.zeros((nbin, nfram), dtype=np.complex64)
    for t in range(nfram):
        idx = np.arange(wlen) + t*shift
        Bx = x[idx] * window
        Fx[:,t] = np.fft.rfft(Bx)

    return Fx



#----------------------------------------------------------------
# perform a multichannel inverse STFT on audio data Fx
# Fx.shape = (nbin, nfram)
# output x.shape = (samples)
def mistft(Fx, wlen=1024):

    Fx = np.asarray(Fx, dtype=np.complex64)
    nbin = Fx.shape[0]
    nfram = Fx.shape[1]
    samples = nfram*shift+wlen-shift

    window = create_hanning_window(wlen)

    x = np.zeros((samples,), dtype=np.float32)
    for t in range(nfram):
        Bx = np.real(np.fft.irfft(Fx[:,t]))
        idx = np.arange(wlen) + t*shift
        x[idx] += Bx * window

    return x



#------------------------------------------------------------------------------
def mkdir(path):

    if not os.path.exists(os.path.dirname(path)): 
        os.makedirs(os.path.dirname(path))



#------------------------------------------------------------------------------
def apply_snr(Fx, Fs, target_snr):

    Lx = 20*np.log10(np.abs(Fx).mean()+1e-9)
    Ls = 20*np.log10(np.abs(Fs).mean()+1e-9)

    gain = Lx-Ls+target_snr
    Fz = Fs*np.power(10, gain/20)

    return Fz

    
