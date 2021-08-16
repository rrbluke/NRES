# -*- coding: utf-8 -*-
__author__ = "Lukas Pfeifenberger"


import numpy as np
import tensorflow as tf





#-------------------------------------------------------------------
@tf.custom_gradient
def elementwise_abs(z):

    s = tf.abs(z)

    def grad(grad_s):

        grad_s = tf.cast(tf.math.real(grad_s), tf.complex64)
        az = tf.cast(tf.abs(z)+1e-6, tf.complex64)
        gs = tf.cast(tf.math.real(grad_s), tf.complex64)
        grad_z = gs*z/az

        return grad_z

    return s, grad



#-------------------------------------------------------------------
@tf.custom_gradient
def elementwise_abs2(z):

    s = tf.abs(z)**2

    def grad(grad_s):

        grad_s = tf.cast(tf.math.real(grad_s), tf.complex64)
        grad_z = 2*grad_s*z

        return grad_z

    return s, grad



