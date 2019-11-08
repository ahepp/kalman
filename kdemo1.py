#!/usr/bin/python3

import numpy

from Kalman import Kalman

initial_state = numpy.array([
    [0], # x
    [0], # x'
    [20],# y
    [0]  # y'
])
initial_covar = numpy.array([
    [0],
    [0],
    [0],
    [0]
])

acceleration = numpy.array([
    [0],
    [-9.8]
])

dt = 1

kfilter = Kalman(initial_state, initial_covar)

for i in range(0,10):
    print(kfilter.estimate(None, acceleration, dt))
