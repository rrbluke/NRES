# -*- coding: utf-8 -*-
__author__ = "Lukas Pfeifenberger"


import numpy as np
import glob
import sys
import os
import soundfile as sf




class audio_loader(object):

    #--------------------------------------------------------------------------
    def __init__(self, name, path):

        self.name = name
        self.path = path
        self.fs = 16e3

        self.file_list = glob.glob(self.path+'*.wav')
        self.numof_files = len(self.file_list)

        print('*** loader "%s" found %d files in: %s' % (self.name, len(self.file_list), self.path))



    #-------------------------------------------------------------------------
    # read random files and concatenate them to <samples> length
    def load_random_files(self, samples):

        y = np.zeros((samples,), dtype=np.float32)

        i = 0
        while i<samples:
            
            x, fs = sf.read(np.random.choice(self.file_list))
            i1 = np.minimum(i+x.shape[0], samples)
            i2 = i1-i
            y[i:i1] = x[0:i2]
            i = i1

        return y


