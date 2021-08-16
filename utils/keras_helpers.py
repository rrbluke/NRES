# -*- coding: utf-8 -*-
__author__ = "Lukas Pfeifenberger"


import numpy as np
import sys
import time

import keras.backend as K
import tensorflow as tf




#-----------------------------------------------------
class Logger(tf.keras.callbacks.Callback):

    def __init__(self, name):
        self.name = name
        self.iteration = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_time_start = time.time()
        self.losses = []
        self.erle = []
        self.sdr = []

    def on_batch_end(self, batch, logs=None):
        self.losses = np.append(self.losses, logs['loss'])
        #self.erle = np.append(self.erle, logs['erle'])
        #self.sdr = np.append(self.sdr, logs['sdr'])
        #print('end of batch: ', logs['loss'].shape)

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_time_end = time.time()
        duration = self.epoch_time_end-self.epoch_time_start
        self.iteration += 1
        #print('end of epoch: ', self.losses.shape)

        #print('model: %s, iteration: %d, epoch: %d, runtime: %.3fs, loss: %.3f, ERLE: %.3fdB, SDR: %.3fdB' % \
        #    (self.name, self.iteration, epoch, duration, np.mean(self.losses), np.mean(self.erle), np.mean(self.sdr)) )

        print('model: %s, iteration: %d, epoch: %d, runtime: %.3fs, loss: %.3f' % \
            (self.name, self.iteration, epoch, duration, np.mean(self.losses)) )



#-----------------------------------------------------
def Debug(name, x):

    # print the dynamic shape of tensor x during runtime
    print_op = tf.print(name, '.shape =', tf.shape(x), '.dtype=', x.dtype, '.value=', tf.reduce_mean(x))
    with tf.control_dependencies([print_op]):
        return tf.identity(x)


#-----------------------------------------------------
def log10(x):

    return tf.math.log(x) / 2.302585092994046


#-----------------------------------------------------
def pow10(x):

    return tf.math.exp(x * 2.302585092994046)



