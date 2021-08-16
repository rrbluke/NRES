# -*- coding: utf-8 -*-
__author__ = "Lukas Pfeifenberger"

import argparse
import numpy as np
import json
import os
import sys
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, LSTM, Input, Lambda
import keras.backend as K
import tensorflow as tf

sys.path.append(os.path.abspath('../'))
from loaders.feature_generator import feature_generator
from utils.mat_helper import save_numpy_to_mat, load_numpy_from_mat
from utils.keras_helpers import *
from ops.complex_ops import *


np.set_printoptions(precision=3, threshold=3, edgeitems=3)



#-------------------------------------------------------------------------
#-------------------------------------------------------------------------


class NRES(object):

    def __init__(self, nband):

        self.nband = int(nband)
        self.name = os.path.splitext(os.path.basename(sys.argv[0]))[0]               # filename of this script without extension
        self.file_date = os.path.getmtime(sys.argv[0])                               # timestamp of this script
        self.iterations = 10000

        self.weights_file = '../weights/' + self.name + '_weights.h5'
        self.predictions_file = '../predictions/' + self.name + '.mat'

        self.fgen = feature_generator(nband=self.nband)
        self.logger = Logger(self.name)

        self.nbin = self.fgen.nbin
        self.nfram = self.fgen.nfram
        self.nbatch = 1

        self.create_model()


    #---------------------------------------------------------
    def mix_doubletalk(self, inp):

        Fd = inp[0]                                              # shape = (nbatch, nfram, nbin)
        Fy = inp[1]                                              # shape = (nbatch, nfram, nbin)
        Fs = inp[2]                                              # shape = (nbatch, nfram, nbin)

        Fd = tf.cast(Fd, tf.complex64)
        Fy = tf.cast(Fy, tf.complex64)
        Fs = tf.cast(Fs, tf.complex64)

        Pd = elementwise_abs2(Fd)                                # shape = (nbatch, nfram, nbin)
        Py = elementwise_abs2(Fy)                                # shape = (nbatch, nfram, nbin)
        Pds = elementwise_abs2(Fd+Fs)                            # shape = (nbatch, nfram, nbin)

        Lyd = tf.math.log(Py+1e-6) - tf.math.log(Pd+1e-6)
        Lyds = tf.math.log(Py+1e-6) - tf.math.log(Pds+1e-6)

        return [Lyd, Lyds]



    #---------------------------------------------------------
    def get_erle(self, inp):
        
        Fd = inp[0]                                              # shape = (nbatch, nfram, nbin)
        Fy = inp[1]                                              # shape = (nbatch, nfram, nbin)
        p_erle = inp[2]                                          # shape = (nbatch, nfram, nbin)

        Fd = tf.cast(Fd, tf.complex64)
        Fy = tf.cast(Fy, tf.complex64)

        Fe = Fd-Fy                                               # shape = (nbatch, nbin, nfram)
        Pe = elementwise_abs2(Fe)
        Pz = Pe*p_erle**2
        Le = 10*log10(tf.reduce_sum(Pe, axis=(-2,-1)) + 1e-6)    # shape = (nbatch,)
        Lz = 10*log10(tf.reduce_sum(Pz, axis=(-2,-1)) + 1e-6)    # shape = (nbatch,)
        erle = Le-Lz

        cost = tf.reduce_sum(Pz, axis=(-2,-1))                   # shape = (nbatch,)
        cost /= tf.reduce_sum(Pe, axis=(-2,-1))                  # shape = (nbatch,)
        cost = tf.reduce_mean(cost)

        return [cost, erle]



    #---------------------------------------------------------
    def get_sdr(self, inp):
        
        Fd = inp[0]                                              # shape = (nbatch, nfram, nbin)
        Fy = inp[1]                                              # shape = (nbatch, nfram, nbin)
        Fs = inp[2]                                              # shape = (nbatch, nfram, nbin)
        p_sdr = inp[3]                                           # shape = (nbatch, nfram, nbin)

        Fd = tf.cast(Fd, tf.complex64)
        Fy = tf.cast(Fy, tf.complex64)
        Fs = tf.cast(Fs, tf.complex64)

        Fe = Fd+Fs-Fy                                            # shape = (nbatch, nbin, nfram)
        Ps = elementwise_abs2(Fs)
        Pe = elementwise_abs2(Fe)
        p_opt = Ps/(Pe + 1e-6)
        Pn = Pe*(p_sdr-p_opt)**2
        Ls = 10*log10(tf.reduce_sum(Ps, axis=(-2,-1)) + 1e-6)    # shape = (nbatch,)
        Ln = 10*log10(tf.reduce_sum(Pn, axis=(-2,-1)) + 1e-6)    # shape = (nbatch,)
        sdr = Ls-Ln

        cost = tf.reduce_sum(Pn, axis=(-2,-1))              # shape = (nbatch,)
        cost /= tf.reduce_sum(Pe, axis=(-2,-1))             # shape = (nbatch,)
        cost = tf.reduce_mean(cost)

        return [cost, sdr]



    #---------------------------------------------------------
    def create_model(self):

        print('*** creating model: %s' % self.name)

        nbatch = self.nbatch
        nfram = self.nfram
        nbin = self.nbin
        nband = self.nband

        # shape definitions: (stateful=true requires batch_shape)
        Fd = Input(batch_shape=(nbatch, nfram, nbin), dtype=tf.complex64)                # shape = (nbatch, nfram, nbin)
        Fy = Input(batch_shape=(nbatch, nfram, nbin), dtype=tf.complex64)                # shape = (nbatch, nfram, nbin)
        Fs = Input(batch_shape=(nbatch, nfram, nbin), dtype=tf.complex64)                # shape = (nbatch, nfram, nbin)

        layer1 = Dense(units=nband, activation='tanh')
        layer2 = LSTM(units=nband, activation='tanh', return_sequences=True, stateful=True)
        layer3 = Dense(units=nbin, activation='sigmoid')

        Lyd, Lyds = Lambda(self.mix_doubletalk)([Fd, Fy, Fs])

        # predict erle
        Z = layer1(Lyd)                                                   # shape = (nbatch, nfram, nband)
        Z = layer2(Z)                                                     # shape = (nbatch, nfram, nband)
        p_erle = layer3(Z)                                                # shape = (nbatch, nfram, nbin)
        cost_erle, erle = Lambda(self.get_erle)([Fd, Fy, p_erle])

        # predict sdr
        Z = layer1(Lyds)                                                  # shape = (nbatch, nfram, nband)
        Z = layer2(Z)                                                     # shape = (nbatch, nfram, nband)
        p_sdr = layer3(Z)                                                 # shape = (nbatch, nfram, nbin)
        cost_sdr, sdr = Lambda(self.get_sdr)([Fd, Fy, Fs, p_sdr])

        cost = Lambda(lambda x: tf.reduce_mean(x))([cost_erle*0.001, cost_sdr])

        self.model = Model(inputs=[Fd, Fy, Fs], outputs=[p_erle, p_sdr, erle, sdr])
        self.model.add_loss(cost)
        self.model.compile(loss=None, optimizer='adam')

        # append metric: erle
        #self.model.metrics_tensors.append(erle)
        #self.model.metrics_names.append('erle')
        #self.model.add_metric(erle, name='erle')

        # append metric: sdr
        #self.model.metrics_tensors.append(sdr)
        #self.model.metrics_names.append('sdr')
        #self.model.add_metric(sdr, name='sdr')

        print(self.model.summary())
        
        
        try:
            self.model.load_weights(self.weights_file)
        except:
            print('error loading weights file: %s' % self.weights_file)
        


    #---------------------------------------------------------
    def save_prediction(self, Fd, Fy, Fs, p_erle, p_sdr):
       
        data = {
                'Fd': Fd[0,...].T,
                'Fy': Fy[0,...].T,
                'Fs': Fs[0,...].T,
                'p_erle': p_erle[0,...].T,
                'p_sdr': p_sdr[0,...].T,
                'erle': self.logger.erle[0:self.fgen.nfiles],
                'sdr': self.logger.sdr[0:self.fgen.nfiles],
               }
        save_numpy_to_mat(self.predictions_file, data)
        


    #---------------------------------------------------------
    def train_model(self):

        print('train the model')
        i = 0
        while (i<self.iterations) and (self.file_date == os.path.getmtime(sys.argv[0])):

            Fd, Fy, Fs = self.fgen.generate_batch(self.nbatch, shuffle=True)
            self.model.fit([Fd, Fy, Fs], None, batch_size=self.nbatch, epochs=1, verbose=0, callbacks=[self.logger])

            i += 1
            if (i%100)==0:
                self.model.save_weights(self.weights_file)
                self.test_model()



    #---------------------------------------------------------
    def test_model(self):

        Fd, Fy, Fs = self.fgen.generate_batch(self.nbatch, shuffle=False)
        p_erle, p_sdr, erle, sdr = self.model.predict([Fd, Fy, Fs], batch_size=self.nbatch)
        print('ERLE:', erle, ', SDR:', sdr)
        self.save_prediction(Fd, Fy, Fs, p_erle, p_sdr)




#---------------------------------------------------------
#---------------------------------------------------------
if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='nres')
    parser.add_argument('--nband', default=100)
    parser.add_argument('mode', help='mode: [train, test]', nargs='?', choices=('train', 'test'), default='train')
    args = parser.parse_args()



    dnn = NRES(args.nband)

    if args.mode == 'train':
        dnn.train_model()

    if args.mode == 'test':
        dnn.test_model()





