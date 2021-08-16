# -*- coding: utf-8 -*-
__author__ = "Lukas Pfeifenberger"


import time
import glob
import os
import numpy as np
import random
from loaders.audio_loader import audio_loader
from algorithms.audio_processing import *
from utils.mat_helper import *



class feature_generator(object):

    # --------------------------------------------------------------------------
    def __init__(self, nband):

        self.fs = int(16e3)
        self.nfram = int(15*self.fs/512)
        self.nband = nband

        self.feat_path = '../data/features/'
        self.prediction_path = '../predictions/'

        self.file_list = glob.glob(self.feat_path+'*.mat')
        self.nfiles = len(self.file_list)
        self.loader = audio_loader(name='WSJ0', path='../data/wsj0/')

        # load mat files
        self.data = {}
        samples = 0
        for filename in self.file_list:
            tmp = load_numpy_from_mat(filename, ['d', 'y'])
            d = tmp['d'][:,0]
            y = tmp['y'][:,0]
            samples = samples + d.shape[0]
            gain = 0.99/(np.amax(np.abs(d)) + 1e-6)
            Fd = mstft(d*gain)
            Fy = mstft(y*gain)
            self.data[filename] = {'Fd':Fd, 'Fy':Fy}

        # load doubletalk
        s = self.loader.load_random_files(samples=samples)
        self.Fs = mstft(s)

        #Fs.shape = (nbin, nfram)
        self.nbin, self.nfram_total = self.Fs.shape

        print('*** loaded %d mat files with a total of %d frames' % (self.nfiles, self.nfram_total) )




    #-------------------------------------------------------------------------
    def generate(self, filename):

        # shape = (nbin, nfram)
        Fd = self.data[filename]['Fd']
        Fy = self.data[filename]['Fy']

        # use Fs as doubletalk with a random offset
        nfram = Fd.shape[1]
        idx = random.randint(0, self.nfram_total-nfram)
        Fs = self.Fs[:,idx:idx+nfram]

        # apply random SIR (with respect to Fd)
        target_snr = np.random.uniform(-30, -20)
        Fs = apply_snr(Fd, Fs, target_snr)

        Fs = Fs.astype(np.complex64).T          # shape = (nfram, nbin)
        Fd = Fd.astype(np.complex64).T          # shape = (nfram, nbin)
        Fy = Fy.astype(np.complex64).T          # shape = (nfram, nbin)

        return Fd, Fy, Fs



    #-------------------------------------------------------------------------
    def generate_batch(self, nbatch=50, shuffle=True):

        Fd = np.zeros((nbatch, self.nfram, self.nbin), dtype=np.complex64)
        Fy = np.zeros((nbatch, self.nfram, self.nbin), dtype=np.complex64)
        Fs = np.zeros((nbatch, self.nfram, self.nbin), dtype=np.complex64)

        if shuffle is True:
            flist = np.random.permutation(self.file_list)
        else:
            flist = self.file_list

        for b in range(nbatch):

            Fd0, Fy0, Fs0 = self.generate(flist[b%self.nfiles])
            nfram0 = Fd0.shape[0]
            tile = int(np.ceil(self.nfram/nfram0))

            if tile > 1:
                Fd0 = np.concatenate([Fd0]*tile, axis=0)
                Fy0 = np.concatenate([Fy0]*tile, axis=0)
                Fs0 = np.concatenate([Fs0]*tile, axis=0)

            Fd[b,:,:] = Fd0[0:self.nfram,:]             # shape = (nbatch, nfram, nbin)
            Fy[b,:,:] = Fy0[0:self.nfram,:]
            Fs[b,:,:] = Fs0[0:self.nfram,:]

        return Fd, Fy, Fs

